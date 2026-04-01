from __future__ import annotations

import argparse
import csv
import json
import os

from uqc.baseline_adam import AdamBaselineConfig, TorchGmonObjective
from uqc.eval import ControlPlan, simulate_nominal_plan
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.utils import ensure_dir, parse_angle_expr, set_seeds



def main() -> None:
    parser = argparse.ArgumentParser(description="Adam baseline for the UFO paper reproduction.")
    parser.add_argument("--alpha", type=str, required=True)
    parser.add_argument("--gamma", type=str, default="pi/2")
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--adam-iters", type=int, default=400)
    parser.add_argument("--horizons-ns", type=str, default="60,90,120,150,180,210,240")
    parser.add_argument("--train-noise-std", type=float, default=0.0)
    parser.add_argument("--train-noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cost-chi", type=float, default=10.0, help="Weight for fidelity cost")
    parser.add_argument("--cost-beta", type=float, default=10.0, help="Weight for leakage cost")
    parser.add_argument("--cost-mu", type=float, default=0.2, help="Weight for boundary cost")
    parser.add_argument("--cost-kappa", type=float, default=0.1, help="Weight for time cost")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    alpha = parse_angle_expr(args.alpha)
    gamma = parse_angle_expr(args.gamma)
    horizons = tuple(float(x) for x in args.horizons_ns.split(",") if x.strip())

    set_seeds(args.seed)
    ensure_dir(args.out)
    weights = UFOCostWeights(
        chi=args.cost_chi,
        beta=args.cost_beta,
        mu=args.cost_mu,
        kappa=args.cost_kappa,
    )
    baseline_cfg = AdamBaselineConfig(
        dt_ns=args.dt_ns,
        runtime_norm_ns=args.runtime_norm_ns,
        lr=args.lr,
        adam_iters=args.adam_iters,
        horizons_ns=horizons,
        train_noise_std_mhz=args.train_noise_std,
        train_noise_samples=args.train_noise_samples,
        seed=args.seed,
        cost_weights=weights,
    )
    objective = TorchGmonObjective(baseline_cfg)
    result = objective.optimize(alpha, gamma)
    best = result["best"]

    plan = ControlPlan(
        controls_mhz_and_phase=best["controls"],
        target_alpha=alpha,
        target_gamma=gamma,
        dt_ns=args.dt_ns,
        runtime_norm_ns=args.runtime_norm_ns,
        cost_weights=weights,
        note="Adam baseline",
    )
    plan_path = os.path.join(args.out, "best_control_plan.npz")
    plan.save(plan_path)

    system = GmonSystem(GmonSystemConfig(dt_ns=args.dt_ns, runtime_norm_ns=args.runtime_norm_ns))
    nominal = simulate_nominal_plan(system, plan)

    with open(os.path.join(args.out, "horizon_search.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon_ns", "best_cost"])
        writer.writeheader()
        for row in result["all_results"]:
            writer.writerow({"horizon_ns": row["horizon_ns"], "best_cost": row["best_cost"]})

    summary = {
        "alpha": alpha,
        "gamma": gamma,
        "plan_path": plan_path,
        "best_horizon_ns": best["horizon_ns"],
        "objective_cost": best["best_cost"],
        "nominal_cost": nominal["cost"],
        "nominal_fidelity": nominal["fidelity"],
        "nominal_leakage": nominal["leakage"],
        "nominal_time_ns": nominal["time_ns"],
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
