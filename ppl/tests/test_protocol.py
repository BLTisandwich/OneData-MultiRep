import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pytest


def test_frozen_manifest():
    from omnibench.manifest import rows,reference_rows
    from omnibench.settings import path
    allrows=rows()
    assert len(allrows)==800
    assert len({r['episode_id'] for r in allrows})==800
    assert Counter(r['split'] for r in reference_rows())==Counter(train=418,val=66,test_iid=61,test_compositional=67,test_visual_ood=66)
    assert Counter(r['split'] for r in allrows if r['test_execution'])==Counter(test_iid=69,test_compositional=70,test_visual_ood=70)
    assert sum(r['task_id']=='T7' for r in reference_rows('train'))==34
    assert all(r['task_id']!='T7' for r in allrows if r['validation_reference'] or r['validation_execution'] or r['test_reference'] or r['test_execution'])
    snapshot=json.loads((Path(__file__).parents[1]/'metadata/snapshot.json').read_text())
    assert hashlib.sha256(path('manifest').read_bytes()).hexdigest()==snapshot['manifest_sha256']


def test_answer_scoring():
    from omnibench.evaluation import answer_correct
    q={'answer_type':'number','expert_answer':.5,'numeric_tolerance':.02}
    assert answer_correct(q,'0.51 seconds')
    assert not answer_correct(q,'0.54')
    q={'answer_type':'string','expert_answer':'blue'}
    assert answer_correct(q,'It is blue.')
    assert not answer_correct(q,'blue or red')
    assert not answer_correct(q,'green')


def test_goal_requires_release(tmp_path):
    from omnibench.scoring import completion_audit
    result={'task_id':'T1','success':True,'events':[{'name':'assisted_grasp_start'}]}
    np.savez(tmp_path/'robot_object_state.npz',scripted_attachment_active=np.array([True]))
    assert not completion_audit(tmp_path,result)['goal_and_release_audit']
    np.savez(tmp_path/'robot_object_state.npz',scripted_attachment_active=np.array([False]))
    result['events'].append({'name':'assisted_release'})
    assert completion_audit(tmp_path,result)['goal_and_release_audit']


def test_causal_tactile():
    from omnibench.data import causal_tactile
    state={'robot_object_physical_contact_N':np.zeros(4), 'scripted_attachment_active':np.zeros(4,dtype=bool),
           'timestamp_s':np.array([0,.08,.16,.24]),'virtual_button_active':np.array([False,False,True,True]),
           'virtual_tap_detected':np.zeros(4,dtype=bool)}
    result=causal_tactile(state)
    assert result[:2].max()==0
    assert result[2:].max()>0


def test_iterator_resume_cursor():
    import torch
    from torch.utils.data import DataLoader,DistributedSampler,TensorDataset
    from omnibench.runtime import iterator
    dataset=TensorDataset(torch.arange(20));sampler=DistributedSampler(dataset,num_replicas=1,rank=0,seed=42)
    loader=DataLoader(dataset,batch_size=4,sampler=sampler)
    original=iterator(loader,sampler)
    for _ in range(7):next(original)
    expected=next(original)[0]
    resumed=iterator(loader,sampler,7)
    assert torch.equal(expected,next(resumed)[0])


def test_interval_boundaries():
    from omnibench.runner_evaluation import wilson
    assert wilson(0,0) is None
    assert wilson(10,10)[0]<1
    assert wilson(0,10)[1]>0
