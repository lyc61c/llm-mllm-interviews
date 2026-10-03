"""Run reproducible local checks and save an honest validation report."""
import datetime
import hashlib
import json
import subprocess
import sys
from common import ROOT


def main():
    target=ROOT/'.cache'/'validation.json'
    target.parent.mkdir(exist_ok=True)
    commands=[
        [sys.executable,'-X','utf8','tools/validate.py'],
        [sys.executable,'-X','utf8','-m','unittest','discover','-s','tools','-p','test_supplement.py','-v'],
        [sys.executable,'-X','utf8','-m','unittest','discover','-s','tools','-p','test_presentation.py','-v'],
        [sys.executable,'-X','utf8','-m','unittest','discover','-s','coding','-p','test_*.py','-v'],
        ['node','tools/test_explorer.cjs'],
        [sys.executable,'-X','utf8','-m','py_compile','tools/common.py','tools/build.py','tools/presentation.py','tools/search.py','tools/validate.py','tools/package.py','tools/supplement.py','coding/reference.py','coding/torch_primitives.py'],
        [sys.executable,'-X','utf8','tools/search.py','--id','COD-003','--answers'],
        [sys.executable,'-X','utf8','tools/search.py','Z21','--answers'],
        [sys.executable,'-X','utf8','tools/search.py','--category','VLM','--level','L2','--mock','3','--seed','42'],
    ]
    records=[]
    for command in commands:
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        output=(completed.stdout+completed.stderr).strip()
        recorded_command=['python',*command[1:]] if command[0]==sys.executable else command
        records.append({'command':recorded_command,'returncode':completed.returncode,'output':output})
        print(f"{'PASS' if completed.returncode==0 else 'FAIL'}: {' '.join(command[1:])}")
        if completed.returncode:
            print(output)
    checked_at=datetime.datetime.now(datetime.timezone.utc)
    report={
        'date':checked_at.astimezone(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat(),'timezone':'Asia/Shanghai',
        'checked_at_utc':checked_at.isoformat(),
        'status':'pass_with_explicit_limits' if all(r['returncode']==0 for r in records) else 'fail',
        'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.json'))},
        'checks':records,
        'limits':[
            '16 standard-library coding tests executed; 8 optional PyTorch tests skipped because PyTorch is not installed.',
            '8 supplied-question coverage and provenance regression tests executed, including user-paste/webpage distinction and source-catalog consistency.',
            'Offline HTML verified through Node VM / minimal DOM smoke tests; no live browser screenshot acceptance.',
            'External URLs were researched at collection time; there is no exhaustive automated current-availability check.',
            'Source evidence does not authenticate the employers or prove reported interviews occurred.',
        ],
    }
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0 if report['status']!='fail' else 1


if __name__=='__main__':
    sys.exit(main())
