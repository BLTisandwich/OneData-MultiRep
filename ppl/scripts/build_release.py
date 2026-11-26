"""Build a GitHub-ready source ZIP with a per-file integrity manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from check_release import ROOT,files,check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'OneData-OmniBench.zip')
    args=parser.parse_args();check()
    manifest=ROOT/'RELEASE_FILES.json'
    included=[p for p in files() if p!=manifest]
    manifest.write_text(json.dumps({'version':'0.1.1','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in included}},indent=2)+'\n')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for filename in included+[manifest]:archive.write(filename,Path(ROOT.name)/filename.relative_to(ROOT))
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(digest+'  '+args.output.name+'\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(included)+1
    print(json.dumps({'archive':args.output.name,'files':len(included)+1,'bytes':args.output.stat().st_size,'sha256':digest},indent=2))


if __name__=='__main__':main()
