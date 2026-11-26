"""Shared external encoder interface, configurable LLM backbone, EE and QA heads."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .encoders import TinyImageCNN, PointNet, MeshGIN
from .attention import install, stable_attention

from .settings import ROOT,CONFIG,model_path,model_spec
from .runtime import model_dtype
BRANCHES=['vision','multiview','depth','segmentation','flow','points','mesh','speech','impact','timeseries','force','tactile','proprio']


class SensorEncoders(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.dim=dim
        self.branches=nn.ModuleDict({
            'vision':nn.Linear(512,dim),'multiview':nn.Linear(1024,dim),
            'depth':TinyImageCNN(1,dim),'segmentation':TinyImageCNN(1,dim),
            'flow':TinyImageCNN(2,dim),'points':PointNet(dim),'mesh':MeshGIN(dim),
            'speech':nn.Linear(384,dim),'impact':TinyImageCNN(1,dim),
            'timeseries':nn.Sequential(nn.Linear(22,dim),nn.GELU(),nn.Linear(dim,dim)),
            'force':nn.Sequential(nn.Linear(2,dim),nn.GELU(),nn.Linear(dim,dim)),
            'tactile':TinyImageCNN(2,dim),
            'proprio':nn.Sequential(nn.Linear(23,dim),nn.GELU(),nn.Linear(dim,dim))})
        self.temporal=nn.ModuleDict({n:nn.Conv1d(dim,dim,3,padding=1) for n in BRANCHES if n!='mesh'})

    def forward(self,sensors,names,tokens):
        output={}
        for name in names:
            if name=='vision':x=sensors['vision_front'].float()
            elif name=='multiview':x=torch.cat([sensors['vision_overhead'],sensors['vision_side']],dim=-1).float()
            elif name=='mesh':
                output[name]=self.branches[name](sensors['mesh_vertices'].float(),sensors['mesh_adjacency'].float())[:,None].expand(-1,tokens,-1)
                continue
            else:x=sensors[name].float()
            b,t=x.shape[:2]
            if name=='force':x=torch.sign(x)*torch.log1p(x.abs())/5
            if name=='flow':x=x.clamp(-8,8)
            if name in ['depth','segmentation','flow','impact','tactile']:
                x=self.branches[name](x.flatten(0,1)).reshape(b,t,-1)
            elif name=='points':
                x=self.branches[name](x[:,:,::3].flatten(0,1)).reshape(b,t,-1)
            else:x=self.branches[name](x)
            x=F.gelu(self.temporal[name](x.transpose(1,2)))
            output[name]=F.adaptive_avg_pool1d(x,tokens).transpose(1,2)
        return output


class BenchModel(nn.Module):
    def __init__(self,backbone,rank=None):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig,get_peft_model,TaskType
        rank=rank or CONFIG["network"]["lora_rank"]
        self.backbone_name=backbone
        self.encoders=SensorEncoders()
        path=model_path(backbone)
        architecture=AutoConfig.from_pretrained(path,local_files_only=True)
        if architecture.model_type=='qwen3_5':
            full=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=model_dtype(),attn_implementation='eager')
            language=full.model.language_model
            target=['q_proj','v_proj','in_proj_qkv','out_proj']
        else:
            full=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=model_dtype(),attn_implementation='eager')
            language=full.model;target=['q_proj','v_proj']
        self.lm_head=full.lm_head
        hidden=language.config.hidden_size
        language.requires_grad_(False);self.lm_head.requires_grad_(False)
        del full
        self.language=get_peft_model(language,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=0.,
                                                       target_modules=target,task_type=TaskType.FEATURE_EXTRACTION,bias='none'))
        if CONFIG['runtime'].get('gradient_checkpointing',False):
            self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.compatibility_patches=install(self.language)
        self.attention_precision_patches=stable_attention(self.language)
        self.register_buffer('native_embedding_rms',self.language.get_input_embeddings().weight[:4096].detach().float().square().mean().sqrt(),persistent=False)
        self.projectors=nn.ModuleDict({n:nn.Sequential(nn.LayerNorm(128),nn.Linear(128,hidden)) for n in BRANCHES})
        self.modality_embedding=nn.Parameter(torch.randn(len(BRANCHES),1,hidden)*.02)
        self.time_embedding=nn.Parameter(torch.randn(1,CONFIG['network']['qa_tokens'],hidden)*.02)
        self.action_query=nn.Parameter(torch.randn(1,1,hidden)*.02)
        self.action_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,256),nn.GELU(),nn.Linear(256,4))
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.tokenizer.padding_side='left'
        self.compact_prefix=True
        self.match_embedding_scale=True
        # One sensor token per EE branch and a configurable whole-episode QA budget.
        self.ee_tokens=CONFIG['network']['ee_tokens']
        self.action_bound_m=None

    def pack_prefix(self,features,fmask,embeds,textmask):
        if not self.compact_prefix:
            return torch.cat([features,embeds],1),torch.cat([fmask,textmask.bool()],1)
        # Padding belongs before the complete prefix. Interior masked holes alter
        # RoPE distances and recurrent DeltaNet state between sensors and words.
        rows=[torch.cat([features[i][fmask[i]],embeds[i][textmask[i].bool()]],0)
              for i in range(features.shape[0])]
        longest=max(len(row) for row in rows)
        x=torch.stack([F.pad(row,(0,0,longest-len(row),0)) for row in rows])
        mask=torch.stack([torch.arange(longest,device=x.device)>=longest-len(row) for row in rows])
        return x,mask

    def feature_tokens(self,sensors,branches,tokens,intervention=None):
        names=sorted(set(n for row in branches for n in row),key=BRANCHES.index)
        features=self.encoders(sensors,names,tokens)
        xs=[];masks=[]
        for name in names:
            z=features[name]
            if intervention=='zero':z=torch.zeros_like(z)
            x=self.projectors[name](z)+self.modality_embedding[BRANCHES.index(name)]+self.time_embedding[:,:tokens]
            if self.match_embedding_scale:
                x=F.layer_norm(x.float(),(x.shape[-1],))*self.native_embedding_rms
            valid=torch.tensor([name in row for row in branches],device=x.device,dtype=torch.bool)
            xs.append(x);masks.append(valid[:,None].expand(-1,tokens))
        return torch.cat(xs,1),torch.cat(masks,1)

    def forward(self,sensors,text,branches,answers=None,task='qa',intervention=None):
        device=next(self.parameters()).device
        features,fmask=self.feature_tokens(sensors,branches,self.ee_tokens if task=='ee' else CONFIG['network']['qa_tokens'],intervention)
        if task=='ee':
            strings=['Predict the next end-effector XYZ delta in meters and gripper opening. Instruction: '+s for s in text]
        else:strings=['Question: '+s+'\nAnswer:' for s in text]
        tok=self.tokenizer(strings,add_special_tokens=True,padding=True,return_tensors='pt',truncation=True,max_length=128).to(device)
        embeds=self.language.get_input_embeddings()(tok.input_ids)
        x,mask=self.pack_prefix(features,fmask,embeds,tok.attention_mask)
        answer_ids=None
        if task=='ee':
            query=F.layer_norm(self.action_query,(self.action_query.shape[-1],))*self.native_embedding_rms if self.match_embedding_scale else self.action_query
            x=torch.cat([x,query.expand(x.shape[0],-1,-1)],1)
            mask=torch.cat([mask,torch.ones(x.shape[0],1,device=device,dtype=torch.bool)],1)
        elif answers is not None:
            encoded=[self.tokenizer.encode(str(a),add_special_tokens=False)+[self.tokenizer.eos_token_id] for a in answers]
            maxlen=min(16,max(map(len,encoded)))
            answer_ids=torch.full((len(encoded),maxlen),self.tokenizer.pad_token_id,device=device,dtype=torch.long)
            amask=torch.zeros_like(answer_ids,dtype=torch.bool)
            for k,ids in enumerate(encoded):
                ids=ids[:maxlen];answer_ids[k,:len(ids)]=torch.tensor(ids,device=device);amask[k,:len(ids)]=True
            x=torch.cat([x,self.language.get_input_embeddings()(answer_ids)],1)
            mask=torch.cat([mask,amask],1)
        positions=(mask.long().cumsum(-1)-1).clamp_min(0)
        out=self.language(inputs_embeds=x.to(next(self.language.parameters()).dtype),attention_mask=mask.long(),position_ids=positions,use_cache=False,return_dict=True)
        if task=='ee':
            raw=self.action_head(out.last_hidden_state[:,-1].float())
            delta=raw[:,:3]*.1
            if self.action_bound_m is not None:
                delta=self.action_bound_m*torch.tanh(delta/self.action_bound_m)
            return {'action':torch.cat([delta,raw[:,3:].sigmoid()],-1),'gripper_logits':raw[:,3]}
        if answers is not None:
            # The hidden state before each answer token predicts that token.
            h=out.last_hidden_state[:,-answer_ids.shape[1]-1:-1]
            logits=self.lm_head(h.to(self.lm_head.weight.dtype)).float()
            labels=answer_ids.masked_fill(~amask,-100)
            return {'loss':F.cross_entropy(logits.flatten(0,1),labels.flatten()),'logits':logits}
        return {'next_logits':self.lm_head(out.last_hidden_state[:,-1:].to(self.lm_head.weight.dtype))[:,0].float()}

    @torch.no_grad()
    def answer(self,sensors,text,branches,intervention=None,max_tokens=12):
        # Recompute the prefix to avoid backend-specific recurrent-cache differences.
        device=next(self.parameters()).device
        f,fmask=self.feature_tokens(sensors,branches,CONFIG['network']['qa_tokens'],intervention)
        tok=self.tokenizer(['Question: '+s+'\nAnswer:' for s in text],padding=True,return_tensors='pt',truncation=True,max_length=128).to(device)
        x,mask=self.pack_prefix(f,fmask,self.language.get_input_embeddings()(tok.input_ids),tok.attention_mask)
        # First token must follow the actual final prompt token, not right-padding.
        last=torch.full((len(text),),x.shape[1]-1,device=device,dtype=torch.long)
        generated=[];done=torch.zeros(len(text),device=device,dtype=torch.bool)
        for step in range(max_tokens):
            positions=(mask.long().cumsum(-1)-1).clamp_min(0)
            h=self.language(inputs_embeds=x.to(next(self.language.parameters()).dtype),attention_mask=mask.long(),position_ids=positions,use_cache=False,return_dict=True).last_hidden_state
            selected=h[torch.arange(len(text),device=device),last] if step==0 else h[:,-1]
            ids=self.lm_head(selected.to(self.lm_head.weight.dtype)).argmax(-1)
            generated.append(ids);done|=ids==self.tokenizer.eos_token_id
            if bool(done.all()):break
            x=torch.cat([x,self.language.get_input_embeddings()(ids)[:,None]],1)
            mask=torch.cat([mask,torch.ones(len(text),1,device=device,dtype=torch.bool)],1)
        sequences=torch.stack(generated,1).cpu().tolist()
        sequences=[ids[:ids.index(self.tokenizer.eos_token_id)] if self.tokenizer.eos_token_id in ids else ids for ids in sequences]
        return self.tokenizer.batch_decode(sequences,skip_special_tokens=True)

    def trainable_state(self):
        names={k for k,p in self.named_parameters() if p.requires_grad}
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if k in names}
