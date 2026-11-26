"""Rebuild the frozen aligned reference, encoder cache and grounded QA locally."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .settings import ROOT,CONFIG,path
from .manifest import rows,reference_rows,raw_episode,reference_episode


def align_one(row):
    from .rollout import ExpertReplay,run_one
    from .sim_helpers import Config,canonical_qa,validate_episode,checksum_manifest
    source=raw_episode(row);destination=path('aligned')/'episodes'/row['episode_id']
    if (destination/'_COMPLETE.json').exists():return row['episode_id']
    spec=json.loads((source/'task.json').read_text())
    assert spec['episode_id']==row['episode_id'] and spec['split']==row['split']
    result=run_one(source,destination,ExpertReplay(source))
    (destination/'execution_preview.mp4').replace(destination/'expert_preview.mp4')
    (destination/'model_action_log.jsonl').replace(destination/'expert_action_log.jsonl')
    with np.load(source/'actions.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
    with np.load(destination/'actions.npz',allow_pickle=False) as z:actions={k:z[k] for k in z.files}
    current=actions['executed_ee_position_world_m']
    actions['expert_next_ee_target_world_m']=original['expert_next_ee_target_world_m']
    actions['expert_next_ee_delta_from_current_world_m']=original['expert_next_ee_target_world_m']-current
    actions['gripper_next_command_m']=original['gripper_next_command_m']
    actions['commanded_delta_world_m']=actions['commanded_ee_target_world_m']-current
    actions['expert_replay_requested_delta_world_m']=actions.pop('model_predicted_delta_world_m')
    np.savez_compressed(destination/'actions.npz',**actions)
    with np.load(destination/'physics_240hz.npz',allow_pickle=False) as z:fast={k:z[k] for k in z.files}
    qa_result={**result,'phase_list':['controlled_expert']*len(current),'grasp_assisted':spec['task_id'] in ['T1','T3','T4','T7','T8'],
               'actual_ee_first':current[0].tolist(),'actual_ee_last':current[-1].tolist(),'contact_peak_N':float(fast['robot_object_contact_N'].max())}
    questions=canonical_qa(spec,qa_result)
    for i,q in enumerate(questions):q['question_id']=f'{row["episode_id"]}_ALIGN_Q{i+1:02d}'
    (destination/'qa.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
    _,checks=validate_episode(destination,spec,qa_result,Config(owner=row['owner']))
    # Valid empty JSON event arrays are legitimate observations.
    required=['actions.npz','robot_object_state.npz','physics_240hz.npz','multiview_rgb_depth_seg_flow.npz','pointcloud_world.npz',
              'tactile_proxy.npz','spoken_instruction.wav','impact_audio_SYNTHETIC.wav','scene_and_replay.json','task.json',
              'events.json','qa.jsonl','expert_preview.mp4','target_cube_local.obj']
    checks['files_present']=all((destination/f).is_file() and (destination/f).stat().st_size>0 for f in required)
    checks['events_valid_json_array']=isinstance(json.loads((destination/'events.json').read_text()),list)
    if not all(checks.values()):raise ValueError(f'Reference {row["episode_id"]} did not validate: {checks}')
    xyz=fast['object_xyz_world_m']
    if not np.isfinite(xyz).all() or xyz[:,2].min()<-.01 or xyz[-1,2]<0:
        raise ValueError(f'Regenerated reference {row["episode_id"]} differs from the frozen reference snapshot; check simulator versions')
    (destination/'validation.json').write_text(json.dumps({'all_pass':True,'checks':checks},indent=2)+'\n')
    (destination/'_COMPLETE.json').write_text(json.dumps({'dataset_version':'aligned-final-v1','sha256':checksum_manifest(destination),'verified':True},indent=2)+'\n')
    return row['episode_id']


def encode_static(source,speech,extractor,device):
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    vertices=[];faces=[]
    for line in (source/'target_cube_local.obj').read_text().splitlines():
        if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
        if line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    adjacency=torch.eye(len(vertices))
    for tri in faces:
        for i,j in [(0,1),(1,2),(2,0)]:adjacency[tri[i],tri[j]]=adjacency[tri[j],tri[i]]=1
    waveform,rate=sf.read(source/'spoken_instruction.wav',dtype='float32')
    if rate!=16000:
        factor=gcd(rate,16000);waveform=resample_poly(waveform,16000//factor,rate//factor)
    features=extractor(waveform,sampling_rate=16000,return_tensors='pt').input_features
    with torch.no_grad():
        z=speech.encoder(features.to(device),return_dict=True).last_hidden_state
        valid=min(z.shape[1],max(1,int(np.ceil(len(waveform)/320))))
        encoded=F.adaptive_avg_pool1d(z[:,:valid].transpose(1,2),16)[0].T.cpu().half()
    task=json.loads((source/'task.json').read_text())
    return {'metadata':{k:task[k] for k in ['episode_id','task_id','split','seed','instruction']},
            'sensors':{'speech':encoded,'mesh_vertices':torch.tensor(vertices),'mesh_adjacency':adjacency}}


def prepare(stage='all',workers=4):
    selected=reference_rows();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if stage in ['all','align']:
        if world!=1:raise ValueError('Alignment uses CPU workers; run it without torchrun, then launch distributed cache preparation')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for eid in pool.map(align_one,selected):print(json.dumps({'stage':'align','episode':eid}),flush=True)
    if stage in ['all','cache','static']:
        import torch
        from transformers import WhisperFeatureExtractor
        from .runtime import get_device
        from .encoders import PretrainedVision,PretrainedSpeech
        from .cache import prepare as build_cache,atomic_save
        device=get_device();torch.set_num_threads(CONFIG['runtime']['threads'])
        speech=PretrainedSpeech(path('speech')).to(device).eval()
        extractor=WhisperFeatureExtractor.from_pretrained(path('speech'),local_files_only=True)
        if stage!='static':
            vision=PretrainedVision(path('vision')).to(device).eval()
            for row in selected[rank::world]:
                destination=path('cache')/(row['episode_id']+'.pt')
                if not destination.exists():atomic_save(build_cache(reference_episode(row['episode_id']),vision,speech,extractor,device),destination)
                print(json.dumps({'stage':'cache','episode':row['episode_id']}),flush=True)
            del vision
        execution=[r for r in rows() if r['test_execution'] or r['validation_execution']]
        for row in execution[rank::world]:
            destination=path('static_cache')/(row['episode_id']+'.pt')
            if not destination.exists():atomic_save(encode_static(raw_episode(row),speech,extractor,device),destination)
    if stage in ['all','qa']:
        if world!=1:raise ValueError('QA generation is CPU-only; run without torchrun')
        from .qa_builder import main as build_qa
        wrappers=path('outputs')/'dataset_sources';arguments=['--output',str(path('qa'))]
        for owner in CONFIG['raw_roots']:
            directory=wrappers/owner;(directory/'episodes').mkdir(parents=True,exist_ok=True)
            own=[r for r in selected if r['owner']==owner]
            for row in own:
                link=directory/'episodes'/row['episode_id'];source=reference_episode(row['episode_id'])
                if not link.exists():link.symlink_to(source,target_is_directory=True)
            (directory/'task_manifest.json').write_text(json.dumps({'owner':owner,'rows':[{k:r[k] for k in ['episode_id','owner','task_id','split']} for r in own]},indent=2)+'\n')
            arguments+=['--source',owner+'='+str(directory)]
        build_qa(arguments)
    if stage in ['all','qa']:return check_dataset(check_files=True)
    return check_dataset(check_files=False)


def check_dataset(check_files=False):
    selected=reference_rows();expected=Counter(r['split'] for r in selected)
    snapshot=json.loads((ROOT/'metadata/snapshot.json').read_text())
    if CONFIG['data'].get('strict_snapshot',True):
        assert len(rows())==800 and expected==Counter(train=418,val=66,test_iid=61,test_compositional=67,test_visual_ood=66)
        assert hashlib.sha256(path('manifest').read_bytes()).hexdigest()==snapshot['manifest_sha256']
    result={'reference_counts':dict(expected),'execution_counts':dict(Counter(r['split'] for r in rows() if r['test_execution']))}
    if check_files:
        required=['qa_primary.jsonl','episodes.jsonl','feature_controls_grade_only.jsonl','checksums.json']
        missing=[str(path('qa')/f) for f in required if not (path('qa')/f).is_file()]
        missing += [str(path('cache')/(r['episode_id']+'.pt')) for r in selected if not (path('cache')/(r['episode_id']+'.pt')).is_file()]
        if missing:raise FileNotFoundError('Prepare or configure dataset artifacts first. Missing examples: '+', '.join(missing[:8]))
        episode_rows=[json.loads(s) for s in (path('qa')/'episodes.jsonl').read_text().splitlines() if s]
        known={r['episode_id']:r['split'] for r in selected}
        assert {r['episode_id'] for r in episode_rows}==set(known)
        assert all(r['split']==known[r['episode_id']] for r in episode_rows)
        questions=[json.loads(s) for s in (path('qa')/'qa_primary.jsonl').read_text().splitlines() if s]
        assert all(q['episode_id'] in known and q['split']==known[q['episode_id']] for q in questions)
        if CONFIG['data'].get('strict_snapshot',True):assert Counter(q['split'] for q in questions)==Counter(snapshot['primary_qa_counts'])
        result['qa_counts']=dict(Counter(q['split'] for q in questions))
    return result
