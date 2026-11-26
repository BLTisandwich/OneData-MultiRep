"""Aggregate measured predictions with episode-cluster bootstrap intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .settings import ROOT


def read(paths):
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]


def interval(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    rng=np.random.default_rng(20261124)
    means=np.asarray([rng.choice(a,len(a),replace=True).mean() for _ in range(1000)])
    return np.quantile(means,[.025,.975]).tolist()


def ee_metrics(rows):
    episodes=defaultdict(list)
    for r in rows:episodes[r['episode_id']].append(r)
    means=[np.mean([r['delta_mae_cm'] for r in ep]) for ep in episodes.values()]
    tp=sum(r['predicted_gripper_open'] and r['gold_gripper_open'] for r in rows)
    fp=sum(r['predicted_gripper_open'] and not r['gold_gripper_open'] for r in rows)
    fn=sum(not r['predicted_gripper_open'] and r['gold_gripper_open'] for r in rows)
    return {'frames':len(rows),'episodes':len(episodes),
            'absolute_target_mae_cm':float(np.mean(means)),
            'delta_mae_cm':float(np.mean(means)),
            'delta_mae_episode_bootstrap_ci95_cm':interval(means),
            'next_observed_mae_cm':float(np.mean([r['next_observed_mae_cm'] for r in rows])),
            'gripper_accuracy':float(np.mean([r['predicted_gripper_open']==r['gold_gripper_open'] for r in rows])),
            'gripper_open_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'gripper_positive_definition':'finger opening > 0.02 m (open)'}


def qa_metrics(rows):
    branch=defaultdict(list);episodes=defaultdict(list)
    for r in rows:branch[r['encoder_branch']].append(r);episodes[r['episode_id']].append(r)
    category={k:float(np.mean([r['correct'] for r in v])) for k,v in branch.items()}
    # Cluster resampling preserves all questions in each sampled episode.
    rng=np.random.default_rng(20261124);eids=sorted(episodes);boots=[]
    for _ in range(1000):
        sample=rng.choice(eids,len(eids),replace=True)
        sample_branch=defaultdict(list)
        for eid in sample:
            for r in episodes[eid]:sample_branch[r['encoder_branch']].append(r['correct'])
        boots.append(np.mean([np.mean(v) for v in sample_branch.values()]))
    return {'questions':len(rows),'episodes':len(episodes),'micro_accuracy':float(np.mean([r['correct'] for r in rows])),
            'encoder_macro_accuracy':float(np.mean(list(category.values()))),
            'encoder_macro_episode_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist(),
            'per_encoder_accuracy':category,'per_encoder_count':{k:len(v) for k,v in branch.items()},
            'agrees_original_answer_fraction':float(np.mean([r['agrees_original_answer'] for r in rows]))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    folder=args.directory.resolve()
    results={'scope':'Measured held-out predictions only; expert success flags are not rollout success.',
             'ee':{},'qa':{}}
    for task,metric in [('ee',ee_metrics),('qa',qa_metrics)]:
        rows=read(sorted(folder.glob(task+'_test_*_rank*.jsonl')))
        if any(row['task_id']=='T7' for row in rows):
            raise ValueError('T7 must be excluded before generating formal test predictions')
        groups=defaultdict(list)
        for row in rows:
            groups[(row['split'],row['condition'],'all')].append(row)
            groups[(row['split'],row['condition'],row['task_id'])].append(row)
        for key,subset in groups.items():
            results[task]['/'.join(key)]=metric(subset)
        if task=='qa':
            indexed={(r['condition'],r['question_id']):r for r in rows}
            paired=defaultdict(list)
            for swapped in rows:
                if swapped['condition']!='isolated_swapped':continue
                qid=swapped['question_id']
                matched=indexed.get(('isolated_matched',qid));zero=indexed.get(('isolated_zero',qid))
                if matched is None or zero is None:continue
                record={'question_id':qid,'matched_correct':matched['correct'],'zero_correct':zero['correct'],
                        'swapped_donor_correct':swapped['correct'],'swapped_original_correct':swapped['agrees_original_answer']}
                paired[(swapped['split'],'all')].append(record)
                paired[(swapped['split'],swapped['encoder_branch'])].append(record)
            results['paired_interventions']={'/'.join(k):{'questions':len(v),
                **{name:float(np.mean([r[name] for r in v])) for name in
                   ['matched_correct','zero_correct','swapped_donor_correct','swapped_original_correct']}}
                for k,v in paired.items()}
    dest=folder/'metrics.json';dest.write_text(json.dumps(results,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
