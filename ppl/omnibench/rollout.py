"""Execute predicted actions in the source-derived assisted simulator."""
import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .simulator import controlled_episode
from .sim_helpers import Config
from .cache import resize
from .model import BRANCHES
from .data import static_episode
from .audio import logmel
from .runtime import autocast
PROTOCOL="source-derived-executed-state-assistance-v1"

class Controller:
    def __init__(self,source):self.source=source;self.records=[]
    def record_execution_limit(self,requested,executed):
        self.records[-1].update(requested_target_world_m=np.asarray(requested).tolist(),
                               clipped_target_world_m=np.asarray(executed).tolist(),
                               action_clipped=bool(np.max(abs(requested-executed))>1e-7))
    def record(self,raw,action):
        self.records.append(dict(timestamp_s=raw['timestamp_s'],actual_ee_world_m=raw['ee'].tolist(),
                                 predicted_delta_m=np.asarray(action[:3]).tolist(),predicted_gripper_open=float(action[3])))
        return action

class ExpertReplay(Controller):
    def __init__(self,source):
        super().__init__(source)
        with np.load(source/'actions.npz') as z:
            self.targets=z['expert_next_ee_target_world_m'];self.grips=z['gripper_next_command_m']/.04
    def __call__(self,raw):
        i=min(len(self.records),len(self.targets)-1)
        return self.record(raw,np.r_[self.targets[i]-raw['ee'],self.grips[i]])

class ModelController(Controller):
    def __init__(self,source,model,vision,device,history=4):
        super().__init__(source);self.model=model;self.vision=vision;self.device=device
        self.history_length=history
        eid=source.name;cached=static_episode(source)
        self.static={k:cached['sensors'][k] for k in ['speech','mesh_vertices','mesh_adjacency']}
        self.instruction=cached['metadata']['instruction'];self.seed=cached['metadata']['seed']
        self.scene=json.loads((source/'scene_and_replay.json').read_text())
        self.history=[];self.previous_gray=None;self.previous_ee=None

    @torch.no_grad()
    def __call__(self,raw):
        import cv2
        encoded={}
        images=resize(np.stack([raw[v]['rgb'] for v in ['front','overhead','side']]).transpose(0,3,1,2),64)/255
        z=self.vision.image_features(images.to(self.device)).cpu().half()
        for index,view in enumerate(['front','overhead','side']):encoded['vision_'+view]=z[index]
        encoded['depth']=resize(raw['front']['depth_mm'][None,None].astype(np.float32)/1000)[0].half()
        seg=(raw['front']['seg']%256).astype(np.float32)/255
        encoded['segmentation']=F.interpolate(torch.tensor(seg[None,None]),size=(32,32),mode='nearest')[0].half()
        gray=cv2.cvtColor(raw['front']['rgb'],cv2.COLOR_RGB2GRAY)
        flow=np.zeros((*gray.shape,2),np.float32) if self.previous_gray is None else cv2.calcOpticalFlowFarneback(self.previous_gray,gray,None,.5,2,13,2,5,1.1,0)
        self.previous_gray=gray
        encoded['flow']=resize(flow.transpose(2,0,1)[None])[0].half()/16
        points=[]
        for camera_index,view in enumerate(['front','overhead','side']):
            calibration=self.scene['camera_calibration'][view]
            V=np.asarray(calibration['view_col_major']).reshape(4,4,order='F')
            P=np.asarray(calibration['projection_col_major']).reshape(4,4,order='F')
            d=raw[view]['depth_mm'].astype(np.float64)/1000;seg=raw[view]['seg']
            ids=np.where(seg<0,-1,seg&((1<<24)-1));ys,xs=np.where((ids>=0)&(d<2.95)&(d>.04))
            if not len(xs):points.append(np.zeros((256,3),np.float32));continue
            rng=np.random.default_rng(self.seed+len(self.records)*3+camera_index)
            ix=rng.choice(len(xs),256,replace=len(xs)<256);px=xs[ix];py=ys[ix];depth=d[py,px]
            zb=(3.-3.*.04/depth)/(3.-.04)
            clip=np.stack([2*(px+.5)/192-1,1-2*(py+.5)/144,2*zb-1,np.ones_like(depth)],-1)
            world=clip@np.linalg.inv(P@V).T;points.append((world[:,:3]/world[:,3:4]).astype(np.float32))
        encoded['points']=torch.tensor(np.concatenate(points)).half()
        ee_vel=np.zeros(3) if self.previous_ee is None else (raw['ee']-self.previous_ee)*12
        self.previous_ee=raw['ee'].copy()
        encoded['timeseries']=torch.tensor(np.r_[raw['ee'],ee_vel,raw['q'],raw['dq'],raw['fingers']]).float()
        encoded['proprio']=torch.tensor(np.r_[raw['ee'],raw['ee_quat'],raw['q'],raw['dq'],raw['fingers']]).float()
        encoded['force']=torch.tensor(raw['force']).float()
        encoded['impact']=logmel(raw['impact_wave']).half()
        encoded['tactile']=torch.tensor(raw['tactile']).half()
        self.history.append(encoded)
        history=([self.history[0]]*max(0,self.history_length-len(self.history))+self.history)[-self.history_length:]
        inputs={k:torch.stack([h[k] for h in history])[None].to(self.device) for k in encoded}
        inputs.update({k:v[None].to(self.device) for k,v in self.static.items()})
        with autocast(self.device):
            action=self.model(inputs,[self.instruction],[BRANCHES],task='ee')['action'][0].float().cpu().numpy()
        return self.record(raw,action)

def run_one(source,output,controller):
    spec=json.loads((source/'task.json').read_text());output.mkdir(parents=True,exist_ok=True)
    cfg=Config(owner=spec['owner'],output_root=str(output.parent),enable_api=False)
    real_which=shutil.which;real_run=subprocess.run
    def which(name,*args,**kwargs):
        return 'omnibench-existing-speech' if name in ['espeak','espeak-ng'] else real_which(name,*args,**kwargs)
    def run(command,*args,**kwargs):
        if isinstance(command,list) and command and command[0]=='omnibench-existing-speech':
            shutil.copyfile(source/'spoken_instruction.wav',command[command.index('-w')+1]);return subprocess.CompletedProcess(command,0,stdout='',stderr='')
        return real_run(command,*args,**kwargs)
    shutil.which=which;subprocess.run=run;os.environ['IMAGEIO_FFMPEG_EXE']=shutil.which('ffmpeg') or __import__('imageio_ffmpeg').get_ffmpeg_exe()
    started=time.time()
    try:result=controlled_episode(spec,cfg,output,controller)
    finally:shutil.which=real_which;subprocess.run=real_run
    video_name='model_execution.mp4' if isinstance(controller,ModelController) else 'execution_preview.mp4'
    (output/'expert_preview.mp4').replace(output/video_name)
    # These are model command records, never expert supervision or expert success.
    with np.load(output/'actions.npz') as z:arrays={k:z[k] for k in z.files if not k.startswith('expert_')}
    arrays['model_predicted_delta_world_m']=np.asarray([r['predicted_delta_m'] for r in controller.records],np.float32)
    np.savez_compressed(output/'actions.npz',**arrays)
    scene=json.loads((output/'scene_and_replay.json').read_text());scene.pop('waypoints',None)
    scene['source_waypoint_ik_preflight_max_error_m']=scene.pop('max_ik_position_error_m')
    endpoint_errors=[float(np.linalg.norm(np.asarray(later['actual_ee_world_m'])-np.asarray(now['clipped_target_world_m'])))
                     for now,later in zip(controller.records[:-1],controller.records[1:])]
    scene.update(controller_protocol=PROTOCOL,expert_phase_or_clock_used_for_interactions=False,
                 model_step_endpoint_max_error_m=max(endpoint_errors) if endpoint_errors else None,
                 source_commit='c0494a6076b024d9964d358ac39356483470a029',
                 action_limits={'delta_per_axis_m':.10,'workspace_min_m':[.20,-.50,.025],'workspace_max_m':[.85,.50,.60]},
                 point_sampling='same depth unprojection, deterministic per-frame/per-camera seed; no object oracle filtering')
    (output/'scene_and_replay.json').write_text(json.dumps(scene,indent=2))
    (output/'model_action_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in controller.records))
    summary={'episode_id':spec['episode_id'],'task_id':spec['task_id'],'split':spec['split'],
             'controller':type(controller).__name__,'protocol':PROTOCOL,'success':result['success'],
             'final_error_xy_m':result['final_error_xy_m'],'final_pos':result['final_pos'],
             'actions':len(controller.records),'clipped_actions':sum(r['action_clipped'] for r in controller.records),
             'events':result['events'],'seconds':time.time()-started,
             'execution_video':video_name,
             'mean_step_target_tracking_error_m':float(np.mean(endpoint_errors)) if endpoint_errors else None,
             'assisted_kinematic':True,'real_robot_or_unassisted_physical_success':False}
    if isinstance(controller,ModelController):
        summary['checkpoint_sha256']=controller.model.execution_checkpoint_sha256
    (output/'rollout_result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return summary
