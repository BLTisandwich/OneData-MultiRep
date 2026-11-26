"""Completed measurement and stage-specific recipe consistency checks."""
import copy
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).parents[1]


def test_completed_results():
    result=json.loads((ROOT/'results/selected_results.json').read_text())
    assert result['pending']==[]
    staged=next(p for p in result['profiles'] if p['model']=='Qwen3.5-4B' and p['execution_track']=='joint_action')
    execution=staged['execution']
    assert (execution['all']['successes'],execution['all']['episodes'])==(132,209)
    assert [(execution[s]['successes'],execution[s]['episodes']) for s in ['test_iid','test_compositional','test_visual_ood']]==[(67,69),(39,70),(26,70)]
    assert sum(execution[s]['successes'] for s in ['test_iid','test_compositional','test_visual_ood'])==execution['all']['successes']
    detailed=json.loads((ROOT/'results/qwen35_4b_joint_action.json').read_text())
    assert detailed['checkpoint_sha256']==staged['execution_checkpoint_sha256']
    assert detailed['training']['action_global_batch']==512
    recipe=yaml.safe_load((ROOT/'configs/qwen35_4b_complete.yaml').read_text())
    assert recipe['training']['global_ee_batch']==256
    assert recipe['training']['joint_action']['action_global_ee_batch']==512


def test_stage_batch_routing(tmp_path,monkeypatch):
    from omnibench.settings import CONFIG
    from omnibench import orchestrator,runner_evaluation
    original=copy.deepcopy(CONFIG);commands=[]
    try:
        CONFIG['paths']['outputs']=str(tmp_path/'outputs')
        CONFIG['training']['joint_action']['action_global_ee_batch']=512
        monkeypatch.setattr(orchestrator.subprocess,'run',lambda *args,**kwargs:None)
        monkeypatch.setattr(orchestrator.subprocess,'call',lambda args,**kwargs:commands.append(args) or 0)
        monkeypatch.setattr(runner_evaluation,'report',lambda *args:None)
        orchestrator.run(CONFIG['model'],16,['joint_action'])
        training=[args for args in commands if '_train' in args]
        joint=next(args for args in training if args[args.index('--mode')+1]=='joint')
        action=next(args for args in training if args[args.index('--mode')+1]=='action')
        assert '--ee-global-batch-size' not in joint
        assert action[action.index('--ee-global-batch-size')+1]=='512'
    finally:
        CONFIG.clear();CONFIG.update(original)
