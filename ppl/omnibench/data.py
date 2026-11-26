"""Episode-disjoint datasets and sensor-only field allowlist."""
import json
from functools import lru_cache
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from .model import ROOT,BRANCHES
from .settings import CONFIG,path
from .manifest import reference_episode,reference_rows

CACHE=path('cache')
QA=path('qa')
SENSOR_KEYS=['vision_front','vision_overhead','vision_side','depth','segmentation','flow','points',
             'mesh_vertices','mesh_adjacency','speech','impact','timeseries','force','tactile','proprio']


def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def causal_tactile(state):
    """Match source proxy, but virtual-event boosts may only start after the event.

    The source offline generator uses abs(event_time-frame_time)<0.15, which
    puts a virtual press/tap boost in an earlier frame. This EE-only view uses
    current contact/attachment and already-active virtual state instead.
    """
    contact=state['robot_object_physical_contact_N'].astype(np.float32)
    proxy=np.where(state['scripted_attachment_active'],5.,contact)
    ts=state['timestamp_s']
    active=state['virtual_button_active']|state['virtual_tap_detected']
    seen=np.flatnonzero(active)
    if len(seen):
        onset=ts[seen[0]]
        allowed=(ts>=onset)&(ts-onset<.15)&active.astype(bool)
        proxy[allowed]=np.maximum(proxy[allowed],3.)
    yy,xx=np.mgrid[:16,:16];pad=np.exp(-((xx-7.5)**2+(yy-7.5)**2)/8)
    return torch.tensor(np.broadcast_to(proxy[:,None,None,None]*pad[None,None],(len(ts),2,16,16)).copy()).half()


@lru_cache(maxsize=CONFIG['data']['episode_cache_size'])
def episode(eid):
    result=torch.load(CACHE/(eid+'.pt'),map_location='cpu',weights_only=True)
    result['metadata']['source_path']=str(reference_episode(eid))
    if result['metadata']['task_id'] in ['T5','T6']:
        with np.load(Path(result['metadata']['source_path'])/'robot_object_state.npz',allow_pickle=False) as z:
            state={k:z[k] for k in ['timestamp_s','robot_object_physical_contact_N','scripted_attachment_active','virtual_button_active','virtual_tap_detected']}
        result['sensors']['tactile_causal']=causal_tactile(state)
    else:result['sensors']['tactile_causal']=result['sensors']['tactile']
    return result


def sensors(eid,frame=None,history=None):
    history=history or CONFIG['network']['ee_history']
    raw=episode(eid)['sensors']
    out={k:raw[k] for k in SENSOR_KEYS}
    if frame is not None:
        out['tactile']=raw['tactile_causal']
        indices=torch.tensor([max(0,i) for i in range(frame-history+1,frame+1)])
        out={k:(v if k in ['mesh_vertices','mesh_adjacency','speech'] else v[indices]) for k,v in out.items()}
    return out


class EEDataset(Dataset):
    def __init__(self,split='train'):
        self.rows=[]
        self.episodes=[]
        allowed={r['episode_id'] for r in reference_rows(split)}
        for row in read_jsonl(QA/'episodes.jsonl'):
            if row['episode_id'] not in allowed or row['split']!=split or not row['primary_qa_eligible']:continue
            # User decision: retain T7 training, exclude from all final tests.
            if split!='train' and row['task_id']=='T7':continue
            eid=row['episode_id'];self.episodes.append(eid)
            valid=episode(eid)['sensors']['action_valid']
            self.rows.extend((eid,t) for t in range(len(valid)) if bool(valid[t]))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        eid,t=self.rows[index];ep=episode(eid)
        return dict(sensors=sensors(eid,t),text=ep['metadata']['instruction'],branches=BRANCHES,
                    target=ep['sensors']['action_target'][t],episode_id=eid,frame=t,
                    ee_position=ep['sensors']['ee_position'][t],
                    observed_delta=ep['sensors']['next_observed_delta'][t])


class QADataset(Dataset):
    def __init__(self,split='train',view='isolated',rows=None):
        self.rows=[q for q in (rows if rows is not None else read_jsonl(QA/'qa_primary.jsonl')) if q['split']==split]
        if split.startswith('test_'):
            self.rows=[q for q in self.rows if q['task_id']!='T7']
        allowed={r['episode_id'] for r in reference_rows(split)}
        self.rows=[q for q in self.rows if q['episode_id'] in allowed]
        self.view=view
    def __len__(self):return len(self.rows)*(2 if self.view=='mixed' else 1)
    def __getitem__(self,index):
        q=self.rows[index//2 if self.view=='mixed' else index];eid=q['episode_id']
        # Mixed training includes both full-observation QA and isolated probes.
        full=self.view=='full' or (self.view=='mixed' and index%2==0)
        return dict(sensors=sensors(eid),text=q['question'],branches=BRANCHES if full else q['input_modalities'],
                    answer=q['expert_answer'],episode_id=eid,question_id=q['question_id'],row=q)


def collate(rows):
    out={'sensors':{k:torch.stack([r['sensors'][k] for r in rows]) for k in SENSOR_KEYS},
         'text':[r['text'] for r in rows],'branches':[r['branches'] for r in rows],
         'episode_id':[r['episode_id'] for r in rows]}
    for key in ['target','ee_position','observed_delta']:
        if key in rows[0]:out[key]=torch.stack([r[key] for r in rows])
    for key in ['answer','question_id','frame','row']:
        if key in rows[0]:out[key]=[r[key] for r in rows]
    return out


def move(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch['sensors'].items()}


@lru_cache(maxsize=64)
def static_episode(source):
    source=Path(source);filename=path('static_cache')/(source.name+'.pt')
    if not filename.exists():raise FileNotFoundError(f'Static encoder cache for {source.name} missing; run dataset prepare --stage static')
    return torch.load(filename,map_location='cpu',weights_only=True)
