"""Frozen episode-level partition; observations and gold labels are not bundled."""
import csv
import json
from .settings import ROOT,path,CONFIG


def rows():
    filename=path('manifest')
    with filename.open(newline='') as stream:
        result=list(csv.DictReader(stream))
    flags=['train_reference','validation_reference','validation_execution','test_reference','test_execution']
    for row in result:
        for field in flags:row[field]=bool(int(row[field]))
        if row['task_id']=='T7' and any(row[k] for k in flags[1:]):raise ValueError('T7 must be train-only')
    return result


def reference_rows(split=None):
    selected=[r for r in rows() if r['train_reference'] or r['validation_reference'] or r['test_reference']]
    return [r for r in selected if split is None or r['split']==split]


def raw_episode(row):
    root=CONFIG['raw_roots'][row['owner']]
    from pathlib import Path
    root=Path(root);root=root if root.is_absolute() else ROOT/root
    return root/'episodes'/row['episode_id']


def reference_episode(eid):
    from pathlib import Path
    for value in CONFIG.get('aligned_roots',[CONFIG['paths']['aligned']]):
        root=Path(value);root=root if root.is_absolute() else ROOT/root
        candidate=root/'episodes'/eid
        if candidate.exists():return candidate
    raise FileNotFoundError(f'Aligned episode {eid} unavailable; configure aligned_roots or prepare the dataset')
