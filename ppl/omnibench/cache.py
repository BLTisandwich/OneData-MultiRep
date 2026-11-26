"""Frozen encoder cache construction; raw observations and labels remain separate."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .audio import logmel

def resize(x, size=32):
    return F.interpolate(torch.as_tensor(x.copy()).float(), size=(size,size), mode='bilinear', align_corners=False)

def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def atomic_save(obj, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    torch.save(obj,tmp); tmp.replace(path)

def prepare(path, vision, speech, extractor, device):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    task=json.loads((path/'task.json').read_text())
    a=load(path/'multiview_rgb_depth_seg_flow.npz')
    s=load(path/'robot_object_state.npz')
    p=load(path/'pointcloud_world.npz')
    physics=load(path/'physics_240hz.npz')
    actions=load(path/'actions.npz')
    touch=load(path/'tactile_proxy.npz')
    arrays={}
    # Preserve every observed frame; vision temporal aggregation occurs in training.
    with torch.no_grad():
        for view in ['front','overhead','side']:
            rgb=resize(a[view+'_rgb'].transpose(0,3,1,2),64)/255
            z=[]
            for chunk in rgb.split(32):
                z.append(vision.image_features(chunk.to(device)).cpu())
            arrays['vision_'+view]=torch.cat(z).half()
    arrays['depth']=resize(a['front_depth_mm'][:,None].astype(np.float32)/1000).half()
    arrays['flow']=resize(a['front_optical_flow_rgb_estimated_px'].astype(np.float32).transpose(0,3,1,2)).half()/16
    # Segmentation remains an explicitly identified input family; IDs are not phases.
    seg=(a['front_segmentation_int32'] % 256).astype(np.float32)/255
    arrays['segmentation']=F.interpolate(torch.tensor(seg[:,None]),size=(32,32),mode='nearest').half()
    arrays['points']=torch.tensor(np.concatenate([p[v+'_world_xyz_m'] for v in ['front','overhead','side']],axis=1)).half()
    verts=[]; faces=[]
    for line in (path/'target_cube_local.obj').read_text().splitlines():
        if line.startswith('v '):verts.append([float(x) for x in line.split()[1:4]])
        if line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    arrays['mesh_vertices']=torch.tensor(verts)
    adj=torch.eye(len(verts))
    for tri in faces:
        for i,j in [(0,1),(1,2),(2,0)]:adj[tri[i],tri[j]]=adj[tri[j],tri[i]]=1
    arrays['mesh_adjacency']=adj
    wave,sr=sf.read(path/'spoken_instruction.wav',dtype='float32')
    if sr!=16000:
        g=gcd(sr,16000);wave=resample_poly(wave,16000//g,sr//g)
    features=extractor(wave,sampling_rate=16000,return_tensors='pt').input_features
    with torch.no_grad():
        z=speech.encoder(features.to(device),return_dict=True).last_hidden_state
        valid=min(z.shape[1],max(1,int(np.ceil(len(wave)/320))))
        arrays['speech']=F.adaptive_avg_pool1d(z[:,:valid].transpose(1,2),16)[0].T.cpu().half()
    wave,sr=sf.read(path/'impact_audio_SYNTHETIC.wav',dtype='float32')
    # QA uses all windows. EE indexes only the audio ending at its current timestamp.
    arrays['impact']=torch.stack([logmel(wave[max(0,int(t*sr)-int(sr*.25)):int(t*sr)],sr=sr)
                                   for t in s['timestamp_s']]).half()
    arrays['tactile']=torch.tensor(touch['pressure_2pads_PROXY']).half()
    arrays['timeseries']=torch.tensor(np.concatenate([s['ee_position_world_m'],s['ee_velocity_fd_world_m_s'],s['joint_position_rad'],s['joint_velocity_rad_s'],s['finger_position_m']],axis=-1))
    arrays['proprio']=torch.tensor(np.concatenate([s['ee_position_world_m'],s['ee_quaternion_world_xyzw'],s['joint_position_rad'],s['joint_velocity_rad_s'],s['finger_position_m']],axis=-1))
    arrays['force_240hz']=torch.tensor(physics['robot_object_contact_N'][:,None])
    # Each causal observation contains only current/past physical samples, no event log.
    causal=[]
    for t in s['timestamp_s']:
        take=(physics['time_s']<=t+1e-6)&(physics['time_s']>t-1/12-1e-6)
        v=physics['robot_object_contact_N'][take]
        causal.append([float(v.mean()) if len(v) else 0.,float(v.max()) if len(v) else 0.])
    arrays['force']=torch.tensor(causal)
    arrays['timestamps']=torch.tensor(s['timestamp_s'])
    arrays['action_target']=torch.tensor(np.concatenate([actions['expert_next_ee_delta_from_current_world_m'],(actions['gripper_next_command_m']/.04)[:,None]],axis=-1))
    arrays['action_valid']=torch.tensor(actions['action_valid'])
    arrays['ee_position']=torch.tensor(s['ee_position_world_m'])
    arrays['next_observed_delta']=torch.tensor(actions['executed_next_ee_delta_world_m'])
    # Simulator object state is grade-only; never include it in the sensor tensor dictionary.
    severe=bool(physics['object_xyz_world_m'][:,2].min()<-.01 or physics['object_xyz_world_m'][-1,2]<0)
    metadata={k:task[k] for k in ['episode_id','task_id','split','seed','instruction']}
    metadata.update(eligible=not severe,source_path=str(path),schema='omnibench-sensors-v3',
                    complete_manifest_sha256=hashlib.sha256((path/'_COMPLETE.json').read_bytes()).hexdigest())
    for name,x in arrays.items():
        if x.is_floating_point() and not torch.isfinite(x).all():raise ValueError(f'Nonfinite cache {path.name}:{name}')
    return {'metadata':metadata,'sensors':arrays}
