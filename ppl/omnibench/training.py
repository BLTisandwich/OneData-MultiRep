"""Action-only, QA-only and internal joint-stage optimization."""
import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from .model import ROOT
from . import data
from .data import QADataset,collate,move,read_jsonl
from .evaluation import load_model
from .runtime import seed_all,iterator,get_device,autocast,setup_distributed,barrier,reduce_sum,broadcast,close_distributed,peak_memory,capture_rng,restore_rng
from .settings import CONFIG,path
from .action_data import AlignedFrames,losses as ee_losses

BASE=path('aligned')
QA=path('qa')


def task_loss(model,batch,task,device):
    if task=='ee':return ee_losses(model,batch,device)[0]
    with autocast(device):
        return model(move(batch,device),batch['text'],batch['branches'],answers=batch['answer'],task='qa')['loss']


def backward_task(model,batch,task,device,weight=1.,qa_microbatch=0):
    count=len(batch['text']);size=(qa_microbatch or count) if task=='qa' else CONFIG['training']['ee_micro_batch']
    tokenizer=(model.module if isinstance(model,DDP) else model).tokenizer
    token_counts=[min(16,len(tokenizer.encode(str(a),add_special_tokens=False))+1) for a in batch['answer']] if task=='qa' else [1]*count
    denominator=sum(token_counts);total=0.
    for start in range(0,count,size):
        end=min(start+size,count)
        micro={k:({name:tensor[start:end] for name,tensor in value.items()} if k=='sensors' else value[start:end]) for k,value in batch.items()}
        fraction=sum(token_counts[start:end])/denominator
        loss=task_loss(model,micro,task,device)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite '+task+' loss')
        (loss*(weight*fraction)).backward();total+=float(loss.detach())*fraction
    return total


@torch.no_grad()
def validate(model,sets,rank,world,device,batch_size=16):
    model.eval();results={}
    for task,dataset in sets.items():
        indices=list(range(rank,len(dataset),world));total=torch.zeros(3,device=device)
        for start in range(0,len(indices),batch_size):
            rows=[dataset[i] for i in indices[start:start+batch_size]];batch=collate(rows)
            if task=='ee':
                parts=ee_losses(model,batch,device);loss,metric=parts[0],parts[3]
            else:loss=task_loss(model,batch,task,device);metric=loss
            total[0]+=loss*len(rows);total[1]+=metric*len(rows);total[2]+=len(rows)
        reduce_sum(total)
        results[task]={'loss':float(total[0]/total[2]),'mae_mm' if task=='ee' else 'teacher_ce':float(total[1]/total[2]),'samples':int(total[2])}
    model.train();return results


def calibrate_joint(model,loaders,samplers,device,world,qa_microbatch=0):
    """Four train batches per task, actual globally averaged projector gradients."""
    values={task:[] for task in ['ee','qa']};iters={t:iterator(loaders[t],samplers[t]) for t in values}
    for _ in range(4):
        for task in values:
            model.zero_grad(set_to_none=True);backward_task(model,next(iters[task]),task,device,qa_microbatch=qa_microbatch)
            squared=torch.zeros((),device=device)
            for parameter in model.projectors.parameters():
                gradient=parameter.grad.float().clone() if parameter.grad is not None else torch.zeros_like(parameter,dtype=torch.float32)
                reduce_sum(gradient);gradient/=world;squared+=gradient.square().sum()
            values[task].append(float(squared.sqrt()))
    model.zero_grad(set_to_none=True)
    mean=lambda x:sum(x)/len(x)
    alpha=max(.001,min(1000.,mean(values['ee'])/max(mean(values['qa']),1e-12)))
    return {'qa_weight':alpha,'shared_projector_gradient_norms':values,'scope':'Four train batches per task, no optimizer updates and no validation/test labels.'}


def save_stage(model,folder,step,config,validation):
    folder.mkdir(parents=True,exist_ok=True)
    state={'model':model.trainable_state(),'step':step,'config':{**config,'steps':step},'validation':validation,
           'selection':config.get('checkpoint_selection','Lowest validation score') if folder.name in ['best','best_qa','best_action'] else 'Fixed update snapshot; validation-selected weights stored separately in best/'}
    torch.save(state,folder/'final.partial');(folder/'final.partial').replace(folder/'final.pt')
    (folder/'final_weights.json').write_text(json.dumps({'step':step,'sha256':hashlib.sha256((folder/'final.pt').read_bytes()).hexdigest()},indent=2)+'\n')
    (folder/'config.json').write_text(json.dumps(state['config'],indent=2)+'\n')
    (folder/'completion.json').write_text(json.dumps({'completed':True,'steps':step,'scope':'Aligned QA/Joint refinement'},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--mode',choices=['qa','joint','action'],required=True)
    p.add_argument('--start-device',type=int,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--run-name',default='onedata');p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--extend-from',type=Path,help='Continue an immutable prior run with optimizer/RNG, into a new namespace')
    p.add_argument('--validation-every',type=int,default=200);p.add_argument('--early-stopping-patience',type=int,default=0)
    p.add_argument('--minimum-steps',type=int,default=0)
    p.add_argument('--min-improvement',type=float,default=0.,help='Meaningful validation improvement required to reset patience')
    p.add_argument('--selection-task',choices=['ee','qa'])
    p.add_argument('--save-all-bests',action='store_true')
    p.add_argument('--adaptive-joint-balance',action='store_true')
    p.add_argument('--balance-every',type=int,default=100)
    p.add_argument('--qa-microbatch-size',type=int,default=0)
    p.add_argument('--ee-global-batch-size',type=int,default=None)
    p.add_argument('--validation-batch-size',type=int,default=16)
    p.add_argument('--save-every-validation',action='store_true');p.add_argument('--monitor-qa',action='store_true');args=p.parse_args()
    if not args.run_name.replace('_','').isalnum():raise ValueError('Invalid run namespace')
    assert args.validation_every>0 and args.steps>0 and args.min_improvement>=0
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=get_device(args.start_device+local);torch.set_num_threads(CONFIG['runtime']['threads'])
    setup_distributed(device);seed_all(CONFIG['seed'])
    from .dataset import check_dataset
    check_dataset(check_files=True)
    data.CACHE=path('cache');data.QA=QA;data.episode.cache_clear()
    records=[{**r,'accepted':r['primary_qa_eligible']} for r in read_jsonl(QA/'episodes.jsonl')]
    tasks=['qa'] if args.mode=='qa' else ['ee'] if args.mode=='action' else ['ee','qa']
    trainsets={t:AlignedFrames(records,'train',augment=True) if t=='ee' else QADataset('train',view='mixed') for t in tasks}
    valsets={t:AlignedFrames(records,'val') if t=='ee' else QADataset('val',view='mixed') for t in tasks}
    if args.monitor_qa:valsets['qa']=QADataset('val',view='mixed')
    selection_task=args.selection_task or ('qa' if args.mode=='qa' else 'ee')
    assert selection_task in valsets
    def selection_score(scores):return scores['qa']['loss'] if selection_task=='qa' else scores['ee']['mae_mm']
    samplers={t:DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=CONFIG['seed']) for t,ds in trainsets.items()}
    batch={'ee':args.ee_global_batch_size or CONFIG['training']['global_ee_batch'],'qa':CONFIG['training']['global_qa_batch']}
    assert all(value>0 and value%world==0 for value in batch.values()),'Global batches must divide WORLD_SIZE'
    loaders={t:DataLoader(ds,batch_size=batch[t]//world,sampler=samplers[t],collate_fn=collate,num_workers=0,drop_last=True) for t,ds in trainsets.items()}
    if any(len(loader)==0 for loader in loaders.values()):raise ValueError('Global batch exceeds the available task examples')
    extension=args.extend_from.resolve() if args.extend_from else None
    if extension:
        extension_config=json.loads((extension/'config.json').read_text())
        assert extension_config['backbone']==args.backbone and extension_config['mode']==args.mode
        assert extension_config['world_size']==world
        initial=Path(extension_config['initial_checkpoint'])
    else:
        if args.initial_checkpoint is None:raise ValueError('An explicit initialization checkpoint is required; use the run command')
        initial=args.initial_checkpoint.resolve()
    model,_=load_model(initial,device);model.train();model.ee_tokens=CONFIG['network']['ee_tokens'];model.action_bound_m=None
    folder=path('outputs')/'checkpoints'/args.run_name/args.backbone/args.mode/f'seed{CONFIG["seed"]}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():raise RuntimeError('Completed run is immutable; choose a new namespace')
    if (folder/'latest.pt').exists() and not args.resume:raise RuntimeError('Existing run requires --resume')
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':[p for n,p in named if 'lora_' in n],'lr':CONFIG['training']['lora_lr']},
                                {'params':[p for n,p in named if 'lora_' not in n],'lr':CONFIG['training']['encoder_head_lr']}],weight_decay=1e-5)
    start=0;saved=None;rng_source=folder
    if args.resume and (folder/'latest.pt').exists():
        saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['config']['training_target_steps']==args.steps
        model.load_state_dict(saved['model'],strict=False);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        calibration=saved['config']['joint_calibration']
    elif extension:
        assert extension!=folder.resolve()
        saved=torch.load(extension/'latest.pt',map_location='cpu',weights_only=False)
        assert 0<saved['step']<args.steps and saved['config']['schedule_horizon']==CONFIG['training']['schedule_horizon']
        model.load_state_dict(saved['model'],strict=False);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        calibration=saved['config']['joint_calibration'];rng_source=extension
        if rank==0:
            if not (folder/'best').exists():shutil.copytree(extension/'best',folder/'best')
            for old in extension.glob('validation_step*.json'):
                if not (folder/old.name).exists():shutil.copyfile(old,folder/old.name)
        barrier()
    else:calibration=calibrate_joint(model,loaders,samplers,device,world,args.qa_microbatch_size) if args.mode=='joint' else {'qa_weight':1.,'scope':'QA-only objective' if args.mode=='qa' else 'EE-only objective; no QA loss'}
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'seed':CONFIG['seed'],'training_target_steps':args.steps,'schedule_horizon':CONFIG['training']['schedule_horizon'],'history':CONFIG['network']['ee_history'],'ee_tokens':CONFIG['network']['ee_tokens'],
            'global_batch_by_task':batch,'train_examples':{t:len(ds) for t,ds in trainsets.items()},'joint_calibration':calibration,
            'initial_checkpoint':str(initial),'initial_checkpoint_sha256':hashlib.sha256((initial/'final.pt').read_bytes()).hexdigest(),
            'warm_start':'Optimizer/RNG continuation' if extension else 'Explicit shared initialization or previous internal stage',
            'optimizer_policy':'Preserved AdamW moments, LR trajectory, RNG and data cursor from extension source' if extension else 'Fresh AdamW and matched warmup for each new run; resume restores its optimizer',
            'continued_from_step':start,'extension_source':str(extension) if extension else None,
            'checkpoint_selection':'Lowest full validation QA CE' if selection_task=='qa' else 'Lowest full validation EE MAE; QA retention diagnostic only',
            'test_data_used':False,
            'world_size':world,'resolved_settings':CONFIG.copy(),'qa_manifest_sha256':hashlib.sha256((QA/'checksums.json').read_bytes()).hexdigest(),
            'dataset_manifest_sha256':hashlib.sha256(path('manifest').read_bytes()).hexdigest(),
            'code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['model.py','data.py','training.py']}}
    if saved:
        for key in ['qa_manifest_sha256','dataset_manifest_sha256','world_size','global_batch_by_task','backbone','mode']:
            assert saved['config'][key]==config[key],f'Resume changed {key}'
    if rank==0:(folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if start:
        rng=torch.load(rng_source/f'rng_step{start}_rank{rank}.pt',map_location='cpu',weights_only=False)
        restore_rng(rng,device)
    else:seed_all(CONFIG['seed']+rank)
    wrapped=DDP(model,device_ids=[device.index] if device.type!='cpu' else None,broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
    iters={t:iterator(loader,samplers[t],start) for t,loader in loaders.items()};began=time.time()
    log=(folder/'training.jsonl').open('a' if start else 'w') if rank==0 else None
    best_score=saved.get('best_score',float('inf')) if saved else float('inf')
    bad_validations=saved.get('bad_validations',0) if saved else 0
    best_step=saved.get('best_step',start) if saved else 0
    patience_reference=saved.get('patience_reference_score',best_score) if saved else best_score
    auxiliary=saved.get('auxiliary_bests',{}) if saved else {}
    def save_auxiliary(scores,step):
        for task,values in scores.items():
            value=values['loss'] if task=='qa' else values['mae_mm']
            if value<auxiliary.get(task,{}).get('score',float('inf')):
                auxiliary[task]={'score':value,'step':step}
                if rank==0 and args.save_all_bests:
                    label='best_qa' if task=='qa' else 'best_action'
                    description='Lowest validation QA CE' if task=='qa' else 'Lowest validation EE MAE'
                    save_stage(model,folder/label,step,{**config,'checkpoint_selection':description},scores)
    if start==0:
        baseline=validate(model,valsets,rank,world,device,args.validation_batch_size)
        best_score=selection_score(baseline)
        patience_reference=best_score
        if rank==0:
            (folder/'validation_step0.json').write_text(json.dumps(baseline,indent=2)+'\n')
            save_stage(model,folder/'best',0,config,baseline)
        save_auxiliary(baseline,0)
    scores={};stop_reason='fixed budget'
    for step in range(start+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True);metrics={}
        alpha=calibration['qa_weight'];denominator=1+alpha if args.mode=='joint' else 1
        measure_balance=args.mode=='joint' and args.adaptive_joint_balance and (step==1 or step%args.balance_every==0)
        shared=[p for n,p in named if not n.startswith('action_head.') and n!='action_query'] if measure_balance else []
        def shared_gradient():return torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p,dtype=torch.float32)).flatten() for p in shared])
        for task in tasks:
            weight=(alpha if task=='qa' else 1)/denominator
            metrics[task+'_loss']=backward_task(wrapped,next(iters[task]),task,device,weight,args.qa_microbatch_size)
            if measure_balance and task=='ee':ee_gradient=shared_gradient().clone()
        if measure_balance:
            qa_gradient=shared_gradient()-ee_gradient
            en=ee_gradient.norm();qn=qa_gradient.norm()
            ratio=float(qn/en.clamp_min(1e-12));cosine=float((ee_gradient*qa_gradient).sum()/(en*qn).clamp_min(1e-12))
            if not math.isfinite(ratio) or not math.isfinite(cosine):raise RuntimeError('Nonfinite joint balance diagnostics')
            correction=max(.5,min(2.,math.sqrt(float(en/qn.clamp_min(1e-12)))))
            next_alpha=max(1e-4,min(100.,alpha*correction))
            value=torch.tensor(next_alpha,device=device);broadcast(value)
            calibration['qa_weight']=float(value)
            if rank==0:
                with (folder/'joint_balance.jsonl').open('a') as stream:stream.write(json.dumps({'step':step,'qa_weight_used':alpha,'qa_weight_next':calibration['qa_weight'],
                    'weighted_qa_to_ee_gradient_ratio':ratio,'shared_gradient_cosine':cosine,'scope':'Actual training gradients before clipping; all shared parameters'})+'\n')
            del ee_gradient,qa_gradient
        gradient=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
        if not bool(torch.isfinite(gradient)):raise RuntimeError('Nonfinite gradient')
        if step==start+1:
            gradients={name:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                       for name,module in [('encoders',model.encoders),('projectors',model.projectors),('language_lora',model.language),('action_head',model.action_head)]}
            required=['encoders','projectors','language_lora']+(['action_head'] if args.mode in ['joint','action'] else [])
            if any(not math.isfinite(gradients[name]) or gradients[name]<=0 for name in required):
                raise RuntimeError(f'Missing/nonfinite training gradient: {gradients}')
            if rank==0:(folder/'gradient_validation.json').write_text(json.dumps(gradients,indent=2)+'\n')
        factor=min(1.,step/CONFIG['training']['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*min(step/CONFIG['training']['schedule_horizon'],1.))))
        for group,lr in zip(optimizer.param_groups,[CONFIG['training']['lora_lr'],CONFIG['training']['encoder_head_lr']]):group['lr']=lr*factor
        optimizer.step()
        if step==1 or step%10==0:
            if rank==0:
                row={'step':step,**metrics,'grad_norm':float(gradient),'elapsed_seconds':time.time()-began,'peak_memory_bytes':peak_memory()}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        if step%args.validation_every==0 or step==args.steps:
            scores=validate(model,valsets,rank,world,device,args.validation_batch_size)
            score=selection_score(scores)
            improved=score<best_score
            if improved:best_score=score;best_step=step
            if score<patience_reference-args.min_improvement:
                patience_reference=score;bad_validations=0
            else:bad_validations+=1
            save_auxiliary(scores,step)
            torch.save(capture_rng(device),folder/f'rng_step{step}_rank{rank}.pt')
            barrier()
            if rank==0:
                state={'model':model.trainable_state(),'optimizer':optimizer.state_dict(),'step':step,'config':config,'validation':scores,
                       'best_score':best_score,'best_step':best_step,'bad_validations':bad_validations,'patience_reference_score':patience_reference,'auxiliary_bests':auxiliary}
                torch.save(state,folder/'latest.partial');(folder/'latest.partial').replace(folder/'latest.pt')
                (folder/f'validation_step{step}.json').write_text(json.dumps(scores,indent=2)+'\n')
                if improved:save_stage(model,folder/'best',step,config,scores)
                if step==2000 or step==args.steps or args.save_every_validation:save_stage(model,folder/'stages'/f'step{step}',step,config,scores)
                (folder/'selection.json').write_text(json.dumps({'best_step':best_step,'best_score':best_score,'bad_validations':bad_validations,
                       'selection':config['checkpoint_selection'],'test_data_used':False},indent=2)+'\n')
                print(json.dumps({'event':'validation','step':step,'scores':scores}),flush=True)
            barrier()
            if args.early_stopping_patience and step>=args.minimum_steps and bad_validations>=args.early_stopping_patience:
                stop_reason='validation patience';break
    if rank==0:
        log.close();save_stage(model,folder,step,config,scores)
        (folder/'completion.json').write_text(json.dumps({'completed':True,'steps':step,'target_steps':args.steps,'stop_reason':stop_reason,
             'best_step':best_step,'best_score':best_score,'selection':config['checkpoint_selection'],'test_data_used':False},indent=2)+'\n')
    barrier();close_distributed()


if __name__=='__main__':main()
