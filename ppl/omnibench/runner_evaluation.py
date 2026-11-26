"""Frozen-checkpoint predictions and actual model-driven execution."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .settings import CONFIG,path
from .runtime import get_device
from .manifest import rows,raw_episode
from . import evaluation


def context(checkpoint,track):
    checkpoint=Path(checkpoint).resolve()
    weights=json.loads((checkpoint/'final_weights.json').read_text())
    assert hashlib.sha256((checkpoint/'final.pt').read_bytes()).hexdigest()==weights['sha256']
    state=torch.load(checkpoint/'final.pt',map_location='cpu',weights_only=False)
    key=state['config']['backbone'];rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    output=path('outputs')/'results'/CONFIG['experiment']/key/track/f'step{weights["step"]}'
    output.mkdir(parents=True,exist_ok=True)
    protocol={'model':key,'track':track,'step':weights['step'],'checkpoint_sha256':weights['sha256'],
              'world_size':world,'manifest_sha256':hashlib.sha256(path('manifest').read_bytes()).hexdigest(),
              'selection':'Validation-only checkpoint selection','geometry_penalty':False}
    metadata=output/'protocol.json'
    if metadata.exists():assert json.loads(metadata.read_text())==protocol,'Changed checkpoint, manifest or evaluation world size'
    elif rank==0:metadata.write_text(json.dumps(protocol,indent=2)+'\n')
    return checkpoint,weights,rank,world,output,protocol


def predict(checkpoint,track):
    checkpoint,weights,rank,world,output,_=context(checkpoint,track)
    device=get_device();torch.set_num_threads(CONFIG['runtime']['threads'])
    model,state=evaluation.load_model(checkpoint,device);assert state['step']==weights['step']
    batch=CONFIG['evaluation']['batch_size'];ablate=track=='joint_action' and CONFIG['evaluation']['modality_ablations']
    if track in ['action','joint_action']:evaluation.evaluate_ee(model,rank,world,device,output,batch,ablate)
    if track in ['qa','joint_action']:evaluation.evaluate_qa(model,rank,world,device,output,batch,ablate)
    (output/f'prediction_rank{rank}.complete').write_text('complete\n')
    if rank==0:
        while not all((output/f'prediction_rank{i}.complete').exists() for i in range(world)):time.sleep(2)
        subprocess.run([sys.executable,'-m','omnibench','--config',CONFIG['_config_file'],'_summarize','--directory',str(output)],check=True)
        (output/'predictions.complete').write_text('complete\n')


def wilson(successes,total):
    if total==0:return None
    import math
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def execute(checkpoint,track):
    from .rollout import ModelController,run_one
    from .scoring import completion_audit
    from .encoders import PretrainedVision
    checkpoint,weights,rank,world,output,protocol=context(checkpoint,track)
    assert not CONFIG['evaluation']['geometry_penalty'],'The released protocol does not penalize penetration'
    device=get_device();torch.set_num_threads(CONFIG['runtime']['threads'])
    model,_=evaluation.load_model(checkpoint,device);model.execution_checkpoint_sha256=weights['sha256']
    vision=PretrainedVision(path('vision')).to(device).eval()
    selected=[r for r in rows() if r['test_execution']]
    for index,row in enumerate(selected[rank::world]):
        directory=output/'rollouts'/row['episode_id'];filename=directory/'rollout_result.json'
        if filename.exists():
            result=json.loads(filename.read_text());assert result['checkpoint_sha256']==weights['sha256']
        else:
            source=raw_episode(row)
            result=run_one(source,directory,ModelController(source,model,vision,device,history=CONFIG['network']['ee_history']))
        assert result['actions']==96 and result['controller']=='ModelController'
        audit=completion_audit(directory,result)
        with np.load(directory/'robot_object_state.npz',allow_pickle=False) as z:height=z['object_position_world_m'][:,2]
        grade={'episode_id':row['episode_id'],'task_id':row['task_id'],'split':row['split'],
               'success':audit['goal_and_release_audit'],'source_goal_success':bool(result['success']),
               'checkpoint_sha256':weights['sha256'],'release_audit':audit,
               'geometry_diagnostic':bool(height.min()<-.01 or height[-1]<0),'geometry_penalty':False}
        (directory/'evaluation.json').write_text(json.dumps(grade,indent=2)+'\n')
        print(json.dumps({'rank':rank,'completed':index+1,'episode_id':row['episode_id'],'success':grade['success']}),flush=True)
    (output/f'execution_rank{rank}.complete').write_text('complete\n')
    if rank==0:
        while not all((output/f'execution_rank{i}.complete').exists() for i in range(world)):time.sleep(2)
        results=[json.loads((output/'rollouts'/r['episode_id']/'evaluation.json').read_text()) for r in selected]
        assert len({r['episode_id'] for r in results})==len(selected)
        assert all(r['checkpoint_sha256']==weights['sha256'] and r['task_id']!='T7' for r in results)
        groups=defaultdict(list)
        for row in results:
            groups['all'].append(row);groups[row['split']].append(row);groups[row['task_id']].append(row)
        summary={'protocol':protocol,'scope':'Assisted simulation execution; actual model actions',
                 'execution':{k:{'episodes':len(v),'successes':sum(r['success'] for r in v),
                    'success_rate':sum(r['success'] for r in v)/len(v),
                    'wilson_ci95':wilson(sum(r['success'] for r in v),len(v))} for k,v in groups.items()}}
        (output/'execution_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        (output/'execution.complete').write_text('complete\n')


def report(model,tracks):
    root=path('outputs')/'results'/CONFIG['experiment']/model
    result={'model':model,'tracks':{}}
    for track in tracks:
        folders=list((root/track).glob('step*'))
        for folder in folders:
            if not (folder/'predictions.complete').exists():continue
            item={'protocol':json.loads((folder/'protocol.json').read_text()),
                  'offline_metrics':json.loads((folder/'metrics.json').read_text())}
            if track!='qa':
                if not (folder/'execution.complete').exists():continue
                item['execution']=json.loads((folder/'execution_metrics.json').read_text())['execution']
            result['tracks'][track]=item
    root.mkdir(parents=True,exist_ok=True);(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {model}: measured results','', '| Track | Execution success |','|---|---:|']
    for track,item in result['tracks'].items():
        if 'execution' in item:
            r=item['execution']['all'];lines.append(f'| {track} | {r["successes"]}/{r["episodes"]} ({r["success_rate"]:.1%}) |')
    lines+=['','Detailed split, task, QA and modality-intervention metrics are in summary.json.','']
    (root/'SUMMARY.md').write_text('\n'.join(lines))
