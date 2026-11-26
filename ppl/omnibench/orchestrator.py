"""Three public tracks: Action-only, QA-only and Joint-to-Action."""
import json
from pathlib import Path
import subprocess
import sys
from .settings import CONFIG,path


def checkpoint(experiment,model,mode):
    return path('outputs')/'checkpoints'/experiment/model/mode/f'seed{CONFIG["seed"]}'


def initialize(model):
    from .model import BenchModel
    from .training import save_stage
    from .runtime import seed_all
    destination=checkpoint(CONFIG['experiment'],model,'initial')
    if (destination/'completion.json').exists():return destination
    seed_all(CONFIG['seed']);network=BenchModel(model)
    save_stage(network,destination,0,{'backbone':model,'seed':CONFIG['seed'],'mode':'initial',
               'history':CONFIG['network']['ee_history'],'ee_tokens':CONFIG['network']['ee_tokens'],'resolved_settings':CONFIG.copy()}, {})
    return destination


def run(model,devices,tracks):
    output=path('outputs');output.mkdir(parents=True,exist_ok=True);logs=output/'logs'/CONFIG['experiment']/model;logs.mkdir(parents=True,exist_ok=True)
    config=CONFIG['_config_file'];experiment=CONFIG['experiment'];settings=CONFIG['training']
    def launch(label,command):
        state={'phase':label,'model':model,'status':'running'}
        (logs/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        args=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={devices}',
              '-m','omnibench','--config',config,'--model',model,*command]
        with (logs/f'{label}.log').open('a') as stream:code=subprocess.call(args,stdout=stream,stderr=subprocess.STDOUT)
        if code:
            state.update(status='stopped',exit_code=code);(logs/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            raise SystemExit(code)
    subprocess.run([sys.executable,'-m','omnibench','--config',config,'--model',model,'_initialize'],check=True,env={**__import__('os').environ,'OMNIBENCH_DEVICE':'cpu'})
    initial=checkpoint(experiment,model,'initial')
    def train(label,mode,scope,start,recipe,extra=()):
        folder=checkpoint(scope,model,mode)
        if not (folder/'completion.json').exists():
            command=['_train','--backbone',model,'--mode',mode,'--start-device','0','--initial-checkpoint',str(start),
                     '--run-name',scope,'--steps',str(recipe['max_steps']),'--minimum-steps',str(recipe['minimum_steps']),
                     '--early-stopping-patience',str(recipe['patience']),'--min-improvement',str(recipe['min_improvement']),
                     '--validation-every',str(settings['validation_every']),'--validation-batch-size',str(settings['validation_batch']),
                     '--qa-microbatch-size',str(settings['qa_micro_batch']),'--save-every-validation','--save-all-bests','--resume',*extra]
            launch(label,command)
        return folder
    def evaluate(track,weights):
        launch(track+'_predictions',['_predict','--track',track,'--checkpoint',str(weights)])
        if track!='qa':launch(track+'_execution',['_execute','--track',track,'--checkpoint',str(weights)])
    if 'qa' in tracks:
        folder=train('qa','qa',experiment,initial,settings['qa']);evaluate('qa',folder/'best')
    if 'action' in tracks:
        folder=train('action','action',experiment,initial,settings['action']);evaluate('action',folder/'best')
    if 'joint_action' in tracks:
        options=settings['joint_action']
        recipe={k:options['joint_'+k] for k in ['max_steps','minimum_steps','patience','min_improvement']}
        joint=train('joint_action_representation','joint',experiment+'_staged',initial,recipe,
                    ['--selection-task','qa','--adaptive-joint-balance','--balance-every',str(options['balance_every'])])
        final=train('joint_action_specialization','action',experiment+'_staged',joint/'best_qa',settings['action'],['--monitor-qa','--ee-global-batch-size',str(options.get('action_global_ee_batch',settings['global_ee_batch']))])
        evaluate('joint_action',final/'best')
    from .runner_evaluation import report
    report(model,tracks)
    (logs/'status.json').write_text(json.dumps({'phase':'complete','model':model,'status':'complete'},indent=2)+'\n')
