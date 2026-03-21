from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List

from uqc.eval import ControlPlan, robustness_metrics
from uqc.physics import GmonSystem, GmonSystemConfig
from uqc.utils import ensure_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate control plans under the paper's average-fidelity metric.")
    parser.add_argument("--plans", type=str, nargs="+", required=True, help="List of control plan .npz files")
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--noise-min", type=float, default=0.1)
    parser.add_argument("--noise-max", type=float, default=3.5)
    parser.add_argument("--noise-step", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.out)
    labels = args.labels if args.labels is not None and len(args.labels) == len(args.plans) else [os.path.splitext(os.path.basename(p))[0] for p in args.plans]

    rows: List[dict] = []
    sigmas = []
    cur = args.noise_min
    while cur <= args.noise_max + 1e-12:
        sigmas.append(round(cur, 10))
        cur += args.noise_step

    for plan_path, label in zip(args.plans, labels):
        plan = ControlPlan.load(plan_path)
        system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
        for idx, sigma in enumerate(sigmas):
            metrics = robustness_metrics(system, plan, sigma_mhz=sigma, num_samples=args.samples, seed=args.seed + idx)
            metrics["label"] = label
            metrics["plan_path"] = plan_path
            rows.append(metrics)
            print(json.dumps(metrics))

    csv_path = os.path.join(args.out, "robustness.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "plan_path", "sigma_mhz", "num_samples", "average_fidelity", "average_gate_fidelity", "fidelity_variance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved robustness metrics to {csv_path}")


if __name__ == "__main__":
    main()
