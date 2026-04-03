import os
import subprocess
import json
import shutil
from pathlib import Path
import sys

def test_resume_out_sync():
    test_dir = Path("tmp_test_resume")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    wrong_dir = Path("WRONG_DIR")
    if wrong_dir.exists():
        shutil.rmtree(wrong_dir)
        
    print(f"Testing synchronization of args.out with args.resume...")
    
    # Create a dummy args.json that says out is somewhere else
    args_payload = {
        "alpha": "2.2",
        "gamma": "pi/2",
        "out": str(wrong_dir),
        "iterations": 1,
        "episodes_per_batch": 1,
        "num_workers": 1,
        "seed": 1,
        "cost_chi": 1.0,
        "cost_beta": 1.0,
        "cost_mu": 0.1,
        "cost_kappa": 0.1,
        "dt_ns": 2.0,
        "max_time_ns": 10.0,
        "runtime_norm_ns": 60.0,
        "termination_cost": 0.15,
        "init_log_std": -1.0,
        "max_kl": 0.005,
        "hidden_sizes": [16]
    }
    (test_dir / "args.json").write_text(json.dumps(args_payload, indent=2))
    
    # Run the script with --resume test_dir --iterations 1
    # We use sys.executable to ensure we use the same python
    cmd = [
        sys.executable, "train_trpo_single_target.py",
        "--resume", str(test_dir),
        "--iterations", "1",
        "--episodes-per-batch", "1"
    ]
    print(f"Running: {' '.join(cmd)}")
    
    # We run it synchronously
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if a new args.json was written to test_dir
    new_args_path = test_dir / "args.json"
    if not new_args_path.exists():
        print("FAILED: args.json not found in resume directory")
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        return
    
    new_args = json.loads(new_args_path.read_text())
    # Note: we compare with the string version since that's what's in JSON
    if new_args["out"] == str(test_dir):
        print("SUCCESS: args.out was correctly synchronized to args.resume")
    else:
        print(f"FAILED: args.out is still {new_args['out']}, expected {test_dir}")

    # Also check if training_log exists in test_dir
    if (test_dir / "training_log.jsonl").exists():
        print("SUCCESS: training_log.jsonl written to resume directory")
    else:
        print("FAILED: training_log.jsonl NOT found in resume directory")

    if wrong_dir.exists():
        print("FAILED: WRONG_DIR was created!")
        # Clean up
        shutil.rmtree(wrong_dir)
    else:
        print("SUCCESS: WRONG_DIR was NOT created")
        
    # Clean up test_dir
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_resume_out_sync()
