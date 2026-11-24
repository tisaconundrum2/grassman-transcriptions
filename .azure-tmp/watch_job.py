import re
import sys
import time

from huggingface_hub import HfApi

JOB = sys.argv[1]
api = HfApi()
stage = "?"
for _ in range(80):
    try:
        stage = api.inspect_job(job_id=JOB).status.stage
        lines = list(api.fetch_job_logs(job_id=JOB))
        n = sum(1 for l in lines if "RESULT" in l)
        print(f"{time.strftime('%H:%M:%S')} {stage} results={n}", flush=True)
        if stage in ("COMPLETED", "ERROR", "CANCELED"):
            for l in lines:
                if re.search(r"Cached|scoring on|RESULTS|param_checksum|optimizer tracks|requires_grad|start_epoch=|step |Error|Traceback", l):
                    print(l.rstrip())
            break
    except Exception as e:
        print("retry", type(e).__name__, flush=True)
    time.sleep(30)
