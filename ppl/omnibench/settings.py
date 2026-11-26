"""Configuration paths are relative to an explicit project root, never a host."""
import os
from pathlib import Path
import yaml

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
CONFIG={}
ROOT=PACKAGE_ROOT


def configure(filename=None):
    global ROOT
    filename=Path(filename or os.environ.get('OMNIBENCH_CONFIG',PACKAGE_ROOT/'configs/default.yaml')).resolve()
    values=yaml.safe_load(filename.read_text())
    ROOT=(filename.parent/values.get('project_root','..')).resolve()
    CONFIG.clear();CONFIG.update(values);CONFIG['_config_file']=str(filename)
    os.environ['OMNIBENCH_CONFIG']=str(filename)
    return CONFIG


def path(name):
    value=Path(CONFIG['paths'][name]).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def model_spec(key):
    models=CONFIG['models']
    if key in models:return key,models[key]
    for name,spec in models.items():
        if key in [spec['name'],spec['hf_id'],spec['directory']]:return name,spec
    raise ValueError(f'Unknown model {key!r}; add its entry to the models configuration')


def model_path(key):
    _,spec=model_spec(key)
    if spec.get('path'):
        p=Path(spec['path']);return p if p.is_absolute() else ROOT/p
    return path('models')/spec['directory']


configure()
