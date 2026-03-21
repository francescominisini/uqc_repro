from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast thesis-oriented experiment suite for useful results in minutes, not months.")
    parser.add_argument("--alpha", type=str, default="2.2")
    parser.add_argument("--gamma", type=str, default="pi/2")
    parser.add_argument("--transition-targets", type=str, default="2.0:pi/2,2.4:pi/2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    adam_dir = out / "adam_fast"
    trpo_dir = out / "trpo_fast"
    transition_dir = out / "transition_fast"

    adam_cmd = [
        sys.executable,
        str(root / "train_adam_baseline.py"),
        "--alpha", args.alpha,
        "--gamma", args.gamma,
        "--dt-ns", "10",
        "--runtime-norm-ns", "60",
        "--lr", "0.03",
        "--adam-iters", "40",
        "--horizons-ns", "60,90,120",
        "--seed", str(args.seed),
        "--out", str(adam_dir),
    ]
    run_cmd(adam_cmd)

    trpo_cmd = [
        sys.executable,
        str(root / "train_trpo_single_target.py"),
        "--alpha", args.alpha,
        "--gamma", args.gamma,
        "--noise-optimized",
        "--iterations", "5",
        "--episodes-per-batch", "32",
        "--eval-every", "1",
        "--save-every", "1",
        "--dt-ns", "10",
        "--max-time-ns", "60",
        "--runtime-norm-ns", "60",
        "--termination-cost", "0.15",
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--episodes-per-task", str(args.episodes_per_task),
        "--out", str(trpo_dir),
    ]
    run_cmd(trpo_cmd)

    best_ckpt = trpo_dir / "best_agent.pt"
    transition_cmd = [
        sys.executable,
        str(root / "train_transition_experiments.py"),
        "--source-checkpoints", str(best_ckpt),
        "--source-labels", f"alpha_{args.alpha}",
        "--targets", args.transition_targets,
        "--include-scratch",
        "--noise-optimized",
        "--iterations", "2",
        "--episodes-per-batch", "16",
        "--eval-every", "1",
        "--save-every", "1",
        "--dt-ns", "10",
        "--max-time-ns", "60",
        "--runtime-norm-ns", "60",
        "--termination-cost", "0.15",
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--episodes-per-task", str(max(1, args.episodes_per_task // 2)),
        "--out", str(transition_dir),
    ]
    run_cmd(transition_cmd)

    manifest = {
        "alpha": args.alpha,
        "gamma": args.gamma,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "episodes_per_task": args.episodes_per_task,
        "adam_summary": str(adam_dir / "summary.json"),
        "trpo_summary": str(trpo_dir / "summary.json"),
        "transition_summary": str(transition_dir / "transition_summary.csv"),
    }
    with open(out / "fast_suite_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
