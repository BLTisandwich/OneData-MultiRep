"""Scene helpers derived from the pinned OneData-MultiRep notebook. See NOTICE.md."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, subprocess, time, wave
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
DATASET_VERSION = "onedata-omnibench-v1.0"

@dataclass
class Config:
    owner: str = 'Lee'
    output_root: str = 'data/aligned'
    fps: int = 12
    physics_hz: int = 240
    duration_s: float = 8.0
    width: int = 192
    height: int = 144
    camera_names: tuple = ('front', 'overhead', 'side')
    sample_points: int = 256
    tactile_resolution: int = 16
    max_attempts: int = 14
    api_key: str = ''
    api_chat_url: str = ''
    api_model: str = ''
    enable_api: bool = False
    limit_per_task: int = 100
    save_demo_every: int = 10

def to_json(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, tuple): return list(obj)
    raise TypeError(type(obj))

def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, default=to_json, indent=2, ensure_ascii=False), encoding='utf-8')

def make_instruction(spec):
    tid=spec['task_id']; style=spec['style']
    color=style['target_name'];goal=style['goal_name']
    other=style['other_name']
    phrases={
        'T1':f'Pick up the {color} cube and place it on the {goal} target marker.',
        'T2':f'Push the {color} cube across the table to the {goal} marker.',
        'T3':f'Stack the {color} cube on top of the {other} stationary support.',
        'T4':f'Pick up the {color} cube and drop it inside the {other} open container.',
        'T5':f'Press the {color} button until it activates.',
        'T6':f'Tap the {color} cube once and then retract the end-effector.',
        'T7':f'Select the {color} cube, not the {other} distractors, '
             f'and move it onto the {goal} target marker.',
        'T8':f'Grip the {color} cube and drag it over the tabletop to the {goal} marker.'}
    return phrases[tid]

def waypoints(spec):
    bx,by,bz=spec['initial_xyz']; gx,gy,_=spec['target_xyz'];
    tid=spec['task_id']; zc=.09
    def wp(t,x,y,z,g,phase):return (float(t),[float(x),float(y),float(z)],float(g),phase)
    # 8s total, per-finger opening in meters. All commanded poses are world frame.
    if tid in ('T1','T3','T4','T7','T8'):
        release_z=.09
        if tid=='T3':release_z=.16 # above support cube
        if tid=='T4':release_z=.19 # above open container rim
        if tid=='T8':release_z=.09 # keep object touching table
        lift_z=.10 if tid=='T8' else .28
        return [wp(0,bx-.09,by,.31,.04,'approach'),wp(1.0,bx,by,.22,.04,'approach'),
                wp(2.0,bx,by,zc,.04,'descend'),wp(2.55,bx,by,zc,0,'grasp'),
                wp(3.0,bx,by,zc,0,'grasp_hold'),wp(4.0,bx,by,lift_z,0,'lift'),
                wp(5.4,gx,gy,lift_z,0,'transfer'),wp(6.55,gx,gy,release_z,0,'lower'),
                wp(6.85,gx,gy,release_z,.04,'release'),wp(7.75,gx,gy,.30,.04,'retreat'),
                wp(8.0,gx,gy,.30,.04,'done')]
    if tid=='T2':
        return [wp(0,bx,by-.15,.25,.04,'approach'),wp(1.5,bx,by-.095,.07,.04,'align'),
                wp(2.3,bx,by-.045,.06,.04,'contact'),wp(5.6,gx,gy-.04,.06,.04,'push'),
                wp(6.4,gx,gy-.04,.06,.04,'settle'),wp(7.6,gx,gy-.08,.28,.04,'retreat'),
                wp(8,gx,gy-.08,.28,.04,'done')]
    if tid in ('T5','T6'):
        # Contact events use geometric tool proximity; verify episode outcome separately.
        contact_z=.048 if tid=='T5' else .052
        return [wp(0,bx-.08,by,.27,.04,'approach'),wp(1.3,bx,by,.20,.04,'align'),
                wp(2.6,bx,by,.085,.04,'descend'),wp(3.2,bx,by,contact_z,.04,'press' if tid=='T5' else 'tap'),
                wp(4.2,bx,by,contact_z,.04,'hold' if tid=='T5' else 'tap_hold'),
                wp(5.1,bx,by,.23,.04,'retreat'),wp(7.7,bx-.08,by,.27,.04,'retreat'),
                wp(8,bx-.08,by,.27,.04,'done')]
    raise KeyError(tid)

def canonical_qa(spec,episode_result):
    """Answers are determined from saved simulator state, not guessed by an LLM."""
    sp=spec;result=episode_result; start=sp['initial_xyz']; goal=sp['target_xyz'];
    success=result['success'];events=result['events']; phases=result['phase_list'];
    firstevent=events[0]['time_s'] if events else None
    def qa(cat,q,a,typ,mods,source,interval=None):
        return {'category':cat,'canonical_question':q,'question':q,
                'expert_answer':a,'answer_type':typ,'evidence_modalities':mods,
                'answer_source':source,'evidence_time_range_s':interval,
                'api_paraphrase':None,'api_paraphrase_verified':False}
    name=sp['task_name'].replace('_',' ')
    qas=[
       qa('visual', 'What manipulation task is shown in this episode?',name,'string',
          ['video','task_instruction'],'task_spec'),
       qa('spatial','Is the target x-coordinate smaller than the start x-coordinate?',
          'yes' if goal[0]<start[0] else 'no','yes_no',['depth','pointcloud','object_pose'], 'scene_spec'),
       qa('temporal','At what time (in seconds) did the first recorded interaction event occur? Answer none if no event occurred.',
          f"{firstevent:.2f}" if firstevent is not None else 'none',
          'decimal_or_none',['events','robot_timeseries','audio'],'recorded_event_timestamps'),
       qa('contact','Was the gripper commanded to close in this episode?',
          'yes' if result['grasp_assisted'] else 'no','yes_no',
          ['robot_state','gripper'],'recorded_gripper_commands'),
       qa('audio','How many recorded event cues drive the synthetic impact audio?',
          str(len(events)),'integer',['impact_audio','events'],'event_log'),
       qa('ee_action','Does the end-effector finish higher than it starts?',
          'yes' if result['actual_ee_last'][2]>result['actual_ee_first'][2] else 'no',
          'yes_no',['ee_pose','time_series'],'measured_ee_poses'),
       qa('outcome','Does the manipulated object finish higher than its initial height?',
          'yes' if result['final_pos'][2]>start[2]+.03 else 'no',
          'yes_no',['object_pose','video'],'actual_object_start_and_end_states'),
       qa('cross_modal','Does the episode contain a nonzero PyBullet robot-object contact force?',
          'yes' if result['contact_peak_N']>0 else 'no','yes_no',
          ['force','robot_state','video'],'pybullet_contact_forces')]
    return qas

def validate_episode(path,spec,result,cfg):
    pth=Path(path)
    required=['actions.npz','robot_object_state.npz','physics_240hz.npz','multiview_rgb_depth_seg_flow.npz',
              'pointcloud_world.npz','tactile_proxy.npz','spoken_instruction.wav',
              'impact_audio_SYNTHETIC.wav','scene_and_replay.json','task.json','events.json',
              'qa.jsonl','expert_preview.mp4','target_cube_local.obj']
    checks={'files_present':all((pth/f).is_file() and (pth/f).stat().st_size>64 for f in required),
            'task_success':bool(result['success'])}
    a=np.load(pth/'actions.npz');v=np.load(pth/'multiview_rgb_depth_seg_flow.npz')
    s=np.load(pth/'robot_object_state.npz');pc=np.load(pth/'pointcloud_world.npz')
    tactile=np.load(pth/'tactile_proxy.npz'); fast=np.load(pth/'physics_240hz.npz')
    T=int(cfg.duration_s*cfg.fps); t=a['timestamp_s'];
    checks['aligned_timestamps']=bool(np.allclose(t,v['timestamp_s']) and
       np.allclose(t,s['timestamp_s']) and np.allclose(t,pc['timestamp_s']) and
       np.allclose(t,tactile['timestamp_s']))
    checks['frame_count']=bool(len(t)==T)
    checks['physics_length']=bool(len(fast['time_s'])==int(cfg.physics_hz*cfg.duration_s))
    checks['finite_actions']=bool(np.isfinite(a['commanded_ee_target_world_m']).all())
    checks['last_mask']=bool(np.all(a['action_valid'][:-1]) and not a['action_valid'][-1])
    checks['next_actual_delta']=bool(np.allclose(a['executed_next_ee_delta_world_m'][:-1],
                   np.diff(a['executed_ee_position_world_m'],axis=0),atol=1e-5))
    checks['causal_expert_action']=bool(np.allclose(a['expert_next_ee_target_world_m'][:-1],
                   a['commanded_ee_target_world_m'][1:],atol=1e-5) and
                   np.allclose(a['expert_next_ee_delta_from_current_world_m'][:-1],
                   a['commanded_ee_target_world_m'][1:]-a['executed_ee_position_world_m'][:-1],atol=1e-5))
    checks['all_camera_modalities']=all(v[k+'_rgb'].shape==(T,cfg.height,cfg.width,3)
                                    and v[k+'_depth_mm'].shape==(T,cfg.height,cfg.width)
                                    and v[k+'_segmentation_int32'].shape==(T,cfg.height,cfg.width)
                                    and pc[k+'_world_xyz_m'].shape==(T,cfg.sample_points,3)
                                    for k in cfg.camera_names)
    checks['optical_flow']=bool(v['front_optical_flow_rgb_estimated_px'].shape==(T,cfg.height,cfg.width,2))
    checks['tactile']=bool(tactile['pressure_2pads_PROXY'].shape ==
                          (T,2,cfg.tactile_resolution,cfg.tactile_resolution))
    checks['qa_count']=len((pth/'qa.jsonl').read_text().splitlines())==8
    checks['valid_audio']=all((pth/f).stat().st_size>1000 for f in ['spoken_instruction.wav','impact_audio_SYNTHETIC.wav'])
    checks['no_leaked_api_key']=cfg.api_key not in (pth/'task.json').read_text() if cfg.api_key else True
    dump_json(pth/'validation.json',{'all_pass':bool(all(checks.values())),'checks':checks})
    return all(checks.values()),checks

def checksum_manifest(folder):
    ans={}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_file() and p.name!='_COMPLETE.json':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(2**20),b''):h.update(block)
            ans[str(p.relative_to(folder))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    return ans
