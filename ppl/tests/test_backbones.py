"""Exercise both teammate architectures using local, randomly initialized fixtures."""
import copy
import pytest
import torch


@pytest.fixture(params=['qwen2','llama'])
def tiny_model(request,tmp_path,monkeypatch):
    monkeypatch.setenv('OMNIBENCH_DEVICE','cpu')
    from transformers import Qwen2Config,Qwen2ForCausalLM,LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from omnibench.settings import CONFIG
    original=copy.deepcopy(CONFIG)
    folder=tmp_path/'model';folder.mkdir()
    vocabulary={'[PAD]':0,'[UNK]':1,'[EOS]':2,'blue':3,'red':4,'yes':5,'no':6}
    tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]').save_pretrained(folder)
    cls,modelcls=(Qwen2Config,Qwen2ForCausalLM) if request.param=='qwen2' else (LlamaConfig,LlamaForCausalLM)
    config=cls(vocab_size=len(vocabulary),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024)
    modelcls(config).save_pretrained(folder)
    CONFIG['models']['tiny']={'name':'tiny','hf_id':'local/tiny','directory':'tiny','path':str(folder)}
    CONFIG['network']['lora_rank']=2
    CONFIG['runtime']['dtype']='float32'
    from omnibench.model import BenchModel
    model=BenchModel('tiny');torch.set_num_threads(2)
    yield model
    CONFIG.clear();CONFIG.update(original)


def observations():
    b,t=1,2
    return {'vision_front':torch.randn(b,t,512),'vision_overhead':torch.randn(b,t,512),'vision_side':torch.randn(b,t,512),
            'depth':torch.randn(b,t,1,32,32),'segmentation':torch.randn(b,t,1,32,32),'flow':torch.randn(b,t,2,32,32),
            'points':torch.randn(b,t,12,3),'mesh_vertices':torch.randn(b,8,3),'mesh_adjacency':torch.eye(8)[None],
            'speech':torch.randn(b,t,384),'impact':torch.randn(b,t,1,64,32),'timeseries':torch.randn(b,t,22),
            'force':torch.rand(b,t,2),'tactile':torch.rand(b,t,2,16,16),'proprio':torch.rand(b,t,23)}


def test_action_and_qa_backpropagate(tiny_model,tmp_path):
    from omnibench.model import BRANCHES
    from omnibench.training import save_stage
    from omnibench.evaluation import load_model
    from omnibench.settings import CONFIG
    model=tiny_model;data=observations()
    ee=model(data,['pick the object'],[BRANCHES],task='ee')
    assert ee['action'].shape==(1,4)
    loss=ee['action'].square().mean()+model(data,['What color?'],[BRANCHES],answers=['blue'],task='qa')['loss']
    assert torch.isfinite(loss)
    loss.backward()
    for module in [model.encoders,model.projectors,model.language,model.action_head]:
        gradients=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0
    target=tmp_path/'checkpoint'
    save_stage(model,target,2,{'seed':42,'backbone':'tiny','ee_tokens':CONFIG['network']['ee_tokens']},{})
    restored,state=load_model(target,torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(data,['pick the object'],[BRANCHES],task='ee')['action'],restored(data,['pick the object'],[BRANCHES],task='ee')['action'])
        assert len(restored.answer(data,['What color?'],[BRANCHES],max_tokens=2))==1


def test_training_tracks(tiny_model,tmp_path,monkeypatch):
    """Run real optimizer updates, validation and checkpoint selection on fixtures."""
    if tiny_model.language.config.model_type!='qwen2':pytest.skip('Shared trainer already covered by the Qwen fixture')
    import csv,json,sys
    from omnibench.settings import CONFIG
    from omnibench.training import save_stage,main
    from omnibench import data
    folder=tmp_path/'dataset';folder.mkdir();cache=folder/'cache';cache.mkdir();qa=folder/'qa';qa.mkdir()
    aligned=folder/'aligned';(aligned/'episodes').mkdir(parents=True)
    manifest=folder/'episodes.csv'
    records=[];questions=[];manifest_rows=[]
    for eid,split in [('T1_0000','train'),('T1_0060','val')]:
        (aligned/'episodes'/eid).mkdir()
        sensor={k:v[0] for k,v in observations().items()}
        sensor.update(action_valid=torch.ones(2,dtype=torch.bool),action_target=torch.tensor([[.01,0.,0.,1.],[0.,.01,0.,0.]]),
                      ee_position=torch.zeros(2,3),next_observed_delta=torch.zeros(2,3))
        torch.save({'metadata':{'episode_id':eid,'task_id':'T1','instruction':'pick blue'},'sensors':sensor},cache/(eid+'.pt'))
        records.append({'episode_id':eid,'split':split,'task_id':'T1','primary_qa_eligible':True})
        questions.append({'episode_id':eid,'question_id':eid+'_Q1','split':split,'task_id':'T1','question':'What color?',
                          'input_modalities':['vision'],'expert_answer':'blue'})
        manifest_rows.append({'episode_id':eid,'owner':'Lee','task_id':'T1','split':split,'train_reference':int(split=='train'),
                              'validation_reference':int(split=='val'),'validation_execution':0,'test_reference':0,'test_execution':0})
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest_rows[0]));writer.writeheader();writer.writerows(manifest_rows)
    for name,rows in [('episodes.jsonl',records),('qa_primary.jsonl',questions),('feature_controls_grade_only.jsonl',[])]:
        (qa/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (qa/'checksums.json').write_text('{}')
    CONFIG['paths'].update(manifest=str(manifest),cache=str(cache),qa=str(qa),outputs=str(tmp_path/'outputs'))
    CONFIG['aligned_roots']=[str(aligned)];CONFIG['data']['strict_snapshot']=False
    CONFIG['training'].update(global_ee_batch=4,global_qa_batch=2,ee_micro_batch=1,qa_micro_batch=1,validation_batch=1)
    monkeypatch.setattr(data,'CACHE',cache);monkeypatch.setattr(data,'QA',qa);data.episode.cache_clear()
    from omnibench import training
    monkeypatch.setattr(training,'QA',qa)
    initial=tmp_path/'initial';save_stage(tiny_model,initial,0,{'backbone':'tiny','seed':42},{})
    for mode in ['action','qa','joint']:
        monkeypatch.setattr(sys,'argv',['train','--backbone','tiny','--mode',mode,'--start-device','0','--steps','2',
            '--initial-checkpoint',str(initial),'--run-name','fixture','--validation-every','1','--validation-batch-size','1',
            '--qa-microbatch-size','1','--ee-global-batch-size','2','--save-all-bests','--adaptive-joint-balance','--balance-every','1'])
        main()
        result=tmp_path/'outputs/checkpoints/fixture/tiny'/mode/'seed42'
        assert (result/'best/final.pt').is_file()
        assert (result/'latest.pt').is_file()
        assert json.loads((result/'config.json').read_text())['global_batch_by_task']['ee']==2
        assert json.loads((result/'completion.json').read_text())['steps']==2
    data.episode.cache_clear()
