"""Held-out teacher-forced EE, generated QA and encoder intervention predictions.

Independent ranks shard samples without DistributedSampler padding/duplication.
Gold and donor mappings are used solely to grade or construct interventions.
"""
import argparse
import json
import os
import re
from pathlib import Path
import torch
from .model import BenchModel,ROOT,BRANCHES
from .data import EEDataset,QADataset,collate,move,episode,sensors,read_jsonl,QA
from .runtime import seed_all,autocast
from .settings import CONFIG

FAMILIES={'visual':['vision','multiview','depth','segmentation','flow'],
          '3d':['points','mesh'],'language':['speech'],'time_series':['timeseries'],
          'force':['force'],'tactile':['tactile'],'proprio':['proprio'],'environment_audio':['impact']}
SPLITS=['test_iid','test_compositional','test_visual_ood']


def answer_correct(q,prediction):
    if q['answer_type']=='number':
        values=re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',prediction)
        return bool(values and abs(float(values[0])-float(q['expert_answer']))<=q['numeric_tolerance']+1e-8)
    clean=prediction.lower().strip().strip('.!,:;\"\'')
    expected=str(q['expert_answer']).lower().strip()
    if clean==expected:return True
    # Accept an unambiguous structured value inside a short natural answer;
    # reject outputs naming competing colors/directions or both yes and no.
    vocabularies=[['blue','green','orange','magenta','red','yellow','gray','grey'],
                  ['left','right','up','down','still'],['yes','no']]
    for vocabulary in vocabularies:
        if expected in vocabulary:
            values=set(re.findall(r'\b(?:'+'|'.join(vocabulary)+r')\b',clean))
            return values=={expected}
    if expected.isdigit():
        numbers=re.findall(r'(?<![\d.])\d+(?![\d.])',clean)
        return len(numbers)==1 and int(numbers[0])==int(expected)
    return False


def load_model(checkpoint,device):
    state=torch.load(checkpoint/'final.pt',map_location='cpu',weights_only=False)
    seed_all(state['config']['seed'])
    model=BenchModel(state['config']['backbone']).to(device)
    model.ee_tokens=state['config'].get('ee_tokens',CONFIG['network']['ee_tokens'])
    assert {k for k,p in model.named_parameters() if p.requires_grad}<=set(state['model'])
    model.load_state_dict(state['model'],strict=False)
    return model.eval(),state


def apply_family(rows,family):
    if family is None:return rows
    for row in rows:
        row['branches']=[n for n in row['branches'] if n not in FAMILIES[family]]
        if family=='language':row['text']='[instruction withheld]'
    return rows


@torch.no_grad()
def evaluate_ee(model,rank,world,device,output,batch_size,ablate):
    conditions=['full']+(['without_'+f for f in FAMILIES] if ablate else [])
    for split in SPLITS:
        dataset=EEDataset(split)
        indices=list(range(rank,len(dataset),world))
        for condition in conditions:
            dest=output/f'ee_{split}_{condition}_rank{rank}.jsonl'
            if dest.exists() and dest.with_suffix('.complete').exists():continue
            with dest.with_suffix('.partial').open('w') as stream:
                for offset in range(0,len(indices),batch_size):
                    rows=[dataset[k] for k in indices[offset:offset+batch_size]]
                    family=condition.removeprefix('without_') if condition!='full' else None
                    batch=collate(apply_family(rows,family))
                    with autocast(device):
                        pred=model(move(batch,device),batch['text'],batch['branches'],task='ee')['action'].float().cpu()
                    for row,action in zip(rows,pred):
                        error=(action[:3]-row['target'][:3]).abs()
                        observed=(action[:3]-row['observed_delta']).abs()
                        record=dict(episode_id=row['episode_id'],task_id=row['episode_id'].split('_')[0],split=split,
                                    condition=condition,frame=row['frame'],delta_mae_cm=float(error.mean()*100),
                                    absolute_target_mae_cm=float(error.mean()*100),
                                    next_observed_mae_cm=float(observed.mean()*100),
                                    target_l2_cm=float(error.square().sum().sqrt()*100),
                                    predicted_delta_m=action[:3].tolist(),expert_delta_m=row['target'][:3].tolist(),
                                    current_ee_world_m=row['ee_position'].tolist(),
                                    predicted_gripper_open=bool(action[3]>.5),gold_gripper_open=bool(row['target'][3]>.5))
                        stream.write(json.dumps(record)+'\n')
            dest.with_suffix('.partial').replace(dest);dest.with_suffix('.complete').write_text('complete\n')
            print(json.dumps({'event':'ee_evaluated','rank':rank,'split':split,'condition':condition,'samples':len(indices)}),flush=True)


@torch.no_grad()
def evaluate_qa(model,rank,world,device,output,batch_size,ablate):
    allrows=[q for q in read_jsonl(QA/'qa_primary.jsonl') if q['task_id']!='T7']
    donors={r['question_id']:r for r in read_jsonl(QA/'feature_controls_grade_only.jsonl')}
    byid={q['question_id']:q for q in allrows}
    conditions=['isolated_matched','isolated_zero','isolated_swapped','full_matched']
    if ablate:conditions+=['full_without_'+f for f in FAMILIES]
    for split in SPLITS:
        for condition in conditions:
            subset=[q for q in allrows if q['split']==split and (condition!='isolated_swapped' or q['question_id'] in donors)]
            local=subset[rank::world]
            dest=output/f'qa_{split}_{condition}_rank{rank}.jsonl'
            if dest.exists() and dest.with_suffix('.complete').exists():continue
            with dest.with_suffix('.partial').open('w') as stream:
                for offset in range(0,len(local),batch_size):
                    qs=local[offset:offset+batch_size];batchrows=[];golds=[]
                    for q in qs:
                        donor=donors[q['question_id']] if condition=='isolated_swapped' else None
                        source=donor['donor_episode_id'] if donor else q['episode_id']
                        branches=BRANCHES if condition.startswith('full') else q['input_modalities']
                        if condition.startswith('full_without_'):
                            family=condition.removeprefix('full_without_')
                            branches=[n for n in branches if n not in FAMILIES[family]]
                        batchrows.append(dict(sensors=sensors(source),text=q['question'],branches=branches,episode_id=q['episode_id']))
                        golds.append(byid[donor['donor_question_id']] if donor else q)
                    batch=collate(batchrows)
                    with autocast(device):
                        answers=model.answer(move(batch,device),batch['text'],batch['branches'],
                                             intervention='zero' if condition=='isolated_zero' else None,max_tokens=CONFIG['evaluation']['qa_max_tokens'])
                    for q,gold,pred in zip(qs,golds,answers):
                        record=dict(question_id=q['question_id'],episode_id=q['episode_id'],task_id=q['task_id'],split=split,
                                    encoder_branch=q['encoder_branch'],condition=condition,answer_type=q['answer_type'],
                                    prediction=pred,expected_answer=gold['expert_answer'],numeric_tolerance=q['numeric_tolerance'],
                                    correct=answer_correct(gold,pred),original_answer=q['expert_answer'],
                                    agrees_original_answer=answer_correct(q,pred))
                        stream.write(json.dumps(record)+'\n')
            dest.with_suffix('.partial').replace(dest);dest.with_suffix('.complete').write_text('complete\n')
            print(json.dumps({'event':'qa_evaluated','rank':rank,'split':split,'condition':condition,'samples':len(local)}),flush=True)
