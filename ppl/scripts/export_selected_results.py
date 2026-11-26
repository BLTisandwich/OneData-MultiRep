"""Export aggregate measurements from a configured local experiment directory.

This utility copies no observations, gold answers, model weights or host paths.
"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/selected_results.json'))
    args=parser.parse_args();root=args.source
    args.output.parent.mkdir(parents=True,exist_ok=True)
    profiles=[
        ('Qwen3.5-2B','joint_action',5500,'omnibench_final_v1','joint_to_action',2000),
        ('Llama-3.2-3B-Instruct','action',4000,'omnibench_final_v1','action',4000),
        ('Qwen3.5-4B','action',5800,'omnibench_qwen4b_v1','action',800),
        ('Qwen3.5-4B','joint_action',4500,'omnibench_qwen4b_v1','staged',800),
    ]
    result={'snapshot':'aligned-final-v1','selection_scope':'Selected completed configurations; not a complete ablation table or a matched-budget model-size comparison.',
            'execution_protocol':'Actual model actions, 96 steps, assisted kinematic simulation; task goal and required release; penetration diagnostic only.',
            'qa_protocol':'Generated QA; full_matched; encoder-category macro accuracy. Separate QA-only checkpoint.',
            'profiles':[],'pending':[]}
    for model,track,step,run,source_track,qa_step in profiles:
        folder=root/'results'/run/model/source_track/f'step{step}'
        metrics_file=folder/'execution_metrics.json'
        if not metrics_file.exists():metrics_file=folder/'metrics.json'
        measured=json.loads(metrics_file.read_text());protocol=json.loads((folder/'evaluation_protocol.json').read_text())
        qa=root/'results'/run/model/'qa'/f'step{qa_step}'
        qa_measured=json.loads((qa/'metrics.json').read_text())
        item={'model':model,'execution_track':track,'execution_phase_step':step,
              'execution_checkpoint_sha256':protocol['checkpoint_sha256'],
              'execution':measured['execution'],'ee':measured.get('ee',{}),
              'qa_track':'qa','qa_phase_step':qa_step,
              'qa':{key:value for key,value in qa_measured['qa'].items() if '/full_matched/' in key}}
        for name in ['prediction_protocol.json','evaluation_protocol.json']:
            if (qa/name).exists():
                value=json.loads((qa/name).read_text())
                if 'checkpoint_sha256' in value:item['qa_checkpoint_sha256']=value['checkpoint_sha256']
        if model=='Qwen3.5-4B' and track=='joint_action':
            offline=json.loads((folder/'metrics.json').read_text())
            item['qa_retention_same_execution_checkpoint']={key:value for key,value in offline['qa'].items() if '/full_matched/' in key}
            item['training']={'joint_completed_step':1300,'joint_selected_step':800,'action_completed_step':4800,
                              'action_selected_step':4500,'action_global_batch':512,'joint_global_ee_batch':256,
                              'global_qa_batch':128,'world_size':16,'seed':42,'selection':'Validation only',
                              'stop_reason':'Validation patience','execution_test_note':protocol.get('experiment_track')}
            detail={'model':model,'track':'joint_action','checkpoint_sha256':protocol['checkpoint_sha256'],
                    'training':item['training'],'execution':measured['execution'],'offline':offline}
            (args.output.parent/'qwen35_4b_joint_action.json').write_text(json.dumps(detail,indent=2)+'\n')
        result['profiles'].append(item)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
