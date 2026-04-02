import os
from pathlib import Path

root = Path(r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro\runs\cost_function_sweep\trpo")
if root.exists():
    for path in root.rglob("training_log.jsonl"):
        with open(path, "r") as f:
            lines = f.readlines()
            if len(lines) > 5:
                print(f"{path}: {len(lines)} lines")
else:
    print(f"Path does not exist: {root}")
