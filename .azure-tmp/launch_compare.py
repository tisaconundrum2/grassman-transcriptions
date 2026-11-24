import time
from huggingface_hub import HfApi, get_token
import os

import sys
SHA = "5a4f4c7e7634ba4bc0e2390bea505cfe21ca48c6"
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else "diag_resume.py"
ARGS = sys.argv[2:]
code = (
    "import urllib.request,runpy,sys; "
    f"urllib.request.urlretrieve('https://raw.githubusercontent.com/tisaconundrum2/grassman-transcriptions/{SHA}/scripts/{SCRIPT}','/tmp/s.py'); "
    f"sys.argv={[SCRIPT] + ARGS!r}; "
    "runpy.run_path('/tmp/s.py', run_name='__main__')"
)
api = HfApi()
for attempt in range(8):
    try:
        job = api.run_job(
            image="ghcr.io/tisaconundrum2/grassmann-ablation:548fb917de67fa6d87d8a3fc2b17f230a5ac4100",
            command=["python", "-c", code],
            env={"CODE_ROOT": "/app", "VAL_SHARDS": "2"},
            secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
            flavor="t4-small",
            timeout="1h",
        )
        print("JOB", job.id)
        break
    except Exception as e:
        print("retry", attempt, type(e).__name__)
        time.sleep(3)
