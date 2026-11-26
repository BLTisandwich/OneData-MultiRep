"""Check a source release for accidental runtime artifacts or host-specific content."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','__pycache__','.pytest_cache','data','assets','outputs','build','dist'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(
        part in EXCLUDED or part.startswith('.test_tmp') or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts))


def check():
    patterns=[r'/inspire/',r'/home/[^\s]+',r'/opt/[^\s]+',r'/usr/local/[^\s]+',r'sk-proj-[A-Za-z0-9_-]{20,}',r'hf_[A-Za-z0-9]{20,}',r'[\u4e00-\u9fff]']
    forbidden={'.pt','.pth','.npz','.mp4','.wav','.zip','.pyc'}
    problems=[]
    for filename in files():
        if filename.suffix in forbidden:problems.append(f'Runtime artifact: {filename.relative_to(ROOT)}');continue
        content=filename.read_text(encoding='utf-8')
        # This scanner intentionally contains detection expressions, not secrets.
        if filename==Path(__file__).resolve():continue
        for pattern in patterns:
            if re.search(pattern,content):problems.append(f'Unexpected host/credential/non-English content: {filename.relative_to(ROOT)}')
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Release audit passed: {len(files())} source/documentation/metadata files')


if __name__=='__main__':check()
