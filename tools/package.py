"""Package the standalone repository, omitting Git internals and bytecode."""
import hashlib
import json
import zipfile
from common import ROOT


def included(path):
    return path.is_file() and not any(x in {'.git','__pycache__','.cache'} for x in path.relative_to(ROOT).parts) and path.suffix not in {'.pyc','.zip'}


def main():
    files=sorted(p for p in ROOT.rglob('*') if included(p))
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    output=ROOT.parent/'llm-mllm-interviews-2026-10-02.zip'
    files=sorted(p for p in ROOT.rglob('*') if included(p))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,'llm-mllm-interviews/'+str(p.relative_to(ROOT)).replace('\\','/'))
        archive.writestr('llm-mllm-interviews/manifest-sha256.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'output':str(output),'files':len(files),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=='__main__':
    main()
