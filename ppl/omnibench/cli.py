"""Portable command line interface for preparation, training and evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path


def fetch(model,encoders=False):
    from huggingface_hub import HfApi,snapshot_download
    from .settings import CONFIG,model_spec,model_path,path
    _,spec=model_spec(model)
    repositories=[(spec['hf_id'],model_path(model))]
    if encoders:repositories.append(('openai/whisper-tiny',path('speech')))
    receipts=[]
    for repository,destination in repositories:
        revision=HfApi().model_info(repository).sha
        snapshot_download(repository,revision=revision,local_dir=destination)
        receipts.append({'repository':repository,'revision':revision})
    if encoders:
        import torch
        from torchvision.models import ResNet18_Weights
        destination=path('vision');destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            state=torch.hub.load_state_dict_from_url(ResNet18_Weights.IMAGENET1K_V1.url,check_hash=True,map_location='cpu')
            torch.save(state,destination)
        receipts.append({'encoder':'ResNet18','weights':'IMAGENET1K_V1'})
    destination=path('outputs')/'asset_receipts';destination.mkdir(parents=True,exist_ok=True)
    (destination/(model+'.json')).write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts,indent=2))


def main(argv=None):
    from .settings import configure,CONFIG,model_spec
    arguments=list(sys.argv[1:] if argv is None else argv)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--config');common.add_argument('--model')
    options,arguments=common.parse_known_args(arguments)
    configure(options.config)
    if options.model:CONFIG['model']=model_spec(options.model)[0]
    # Worker flags belong to the tested trainer and aggregator parsers.
    if arguments and arguments[0] in ['_train','_summarize']:
        command=arguments.pop(0)
        sys.argv=[command,*arguments]
        if command=='_train':from .training import main as target
        else:from .summary import main as target
        return target()
    parser=argparse.ArgumentParser(description='OneData-OmniBench: Action-only, QA-only and Joint-to-Action. Global options: --config FILE --model KEY')
    commands=parser.add_subparsers(dest='command',required=True,metavar='{doctor,fetch,dataset,run}')
    commands.add_parser('doctor',help='Inspect the runtime and configured assets')
    download=commands.add_parser('fetch',help='Download a model and optionally the shared encoders')
    download.add_argument('--encoders',action='store_true')
    dataset=commands.add_parser('dataset',help='Inspect, prepare or validate the frozen dataset')
    dataset.add_argument('operation',choices=['inspect','prepare','check'])
    dataset.add_argument('--stage',choices=['all','align','cache','static','qa'],default='all')
    dataset.add_argument('--workers',type=int,default=4)
    run=commands.add_parser('run',help='Train and evaluate the selected public tracks')
    run.add_argument('--devices',type=int);run.add_argument('--tracks',nargs='+',choices=['action','qa','joint_action'],default=['action','qa','joint_action'])
    commands.add_parser('_initialize',help=argparse.SUPPRESS)
    for name in ['_predict','_execute']:
        worker=commands.add_parser(name,help=argparse.SUPPRESS)
        worker.add_argument('--track',choices=['action','qa','joint_action'],required=True)
        worker.add_argument('--checkpoint',type=Path,required=True)
    commands._choices_actions=[choice for choice in commands._choices_actions if not choice.dest.startswith('_')]
    args=parser.parse_args(arguments)
    if args.command=='doctor':
        from .runtime import get_device
        from .settings import path,model_path
        import torch
        print(json.dumps({'device':str(get_device()),'torch':torch.__version__,'model':CONFIG['model'],
                         'assets_present':{k:path(k).exists() for k in ['vision','speech','cache','qa','static_cache']},
                         'model_present':model_path(CONFIG['model']).exists()},indent=2))
    elif args.command=='fetch':fetch(CONFIG['model'],args.encoders)
    elif args.command=='dataset':
        from .dataset import check_dataset,prepare
        result=prepare(args.stage,args.workers) if args.operation=='prepare' else check_dataset(check_files=args.operation=='check')
        print(json.dumps(result,indent=2))
    elif args.command=='run':
        from .orchestrator import run
        devices=args.devices or CONFIG['runtime']['devices']
        if devices<1:parser.error('--devices must be positive')
        run(CONFIG['model'],devices,args.tracks)
    elif args.command=='_initialize':
        from .orchestrator import initialize
        initialize(CONFIG['model'])
    elif args.command=='_predict':
        from .runner_evaluation import predict
        predict(args.checkpoint,args.track)
    elif args.command=='_execute':
        from .runner_evaluation import execute
        execute(args.checkpoint,args.track)
