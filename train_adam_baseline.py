from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

from uqc.baseline_adam import AdamBaselineConfig, TorchGmonObjective
from uqc.eval import ControlPlan, robustness_metrics, simulate_nominal_plan
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.utils import JsonlLogger, ensure_dir, parse_angle_expr, set_seeds


def parse_float_csv(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def sigma_tag(sigma: float) -> str:
    return str(float(sigma)).replace("-", "m").replace(".", "p")


def compute_robustness_suite(
    system: GmonSystem,
    plan: ControlPlan,
    *,
    sigmas_mhz: List[float],
    num_samples: int,
    base_seed: int,
) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for idx, sigma in enumerate(sigmas_mhz):
        tag = f"sigma_{sigma_tag(sigma)}"
        metrics = robustness_metrics(
            system=system,
            plan=plan,
            sigma_mhz=float(sigma),
            num_samples=int(num_samples),
            seed=int(base_seed + idx),
        )
        out[tag] = {
            "sigma_mhz": float(metrics["sigma_mhz"]),
            "num_samples": int(metrics["num_samples"]),
            "average_fidelity": float(metrics["average_fidelity"]),
            "average_gate_fidelity": float(metrics["average_gate_fidelity"]),
            "fidelity_variance": float(metrics["fidelity_variance"]),
        }
    return out


def flatten_robustness_for_log(
    robustness: Dict[str, Dict[str, float | int]],
) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for sigma_key, metrics in robustness.items():
        for metric_name in ("average_fidelity", "average_gate_fidelity", "fidelity_variance"):
            value = metrics.get(metric_name)
            if value is not None:
                flat[f"robustness_{sigma_key}_{metric_name}"] = float(value)
    return flat


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adam baseline for the UFO paper reproduction.")
    parser.add_argument("--alpha", type=str, required=True)
    parser.add_argument("--gamma", type=str, default="pi/2")
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--max-time-ns", type=float, default=600.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--termination-cost", type=float, default=0.15)
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
    parser.add_argument(
        "--robustness-sigmas",
        type=str,
        default="1.0",
        help="Comma-separated sigma values in MHz for robustness eval. Empty string disables.",
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=60,
        help="Monte Carlo samples per robustness evaluation.",
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    alpha = parse_angle_expr(args.alpha)
    gamma = parse_angle_expr(args.gamma)
    horizons = tuple(float(x) for x in args.horizons_ns.split(",") if x.strip())
    robustness_sigmas = parse_float_csv(args.robustness_sigmas)

    set_seeds(args.seed)
    ensure_dir(args.out)

    # ---- Save args.json (matching TRPO) ----
    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    weights = UFOCostWeights(
        chi=args.cost_chi,
        beta=args.cost_beta,
        mu=args.cost_mu,
        kappa=args.cost_kappa,
    )

    # ---- Setup directories (matching TRPO structure) ----
    robustness_dir = os.path.join(args.out, "robustness")
    ensure_dir(robustness_dir)

    logger = JsonlLogger(os.path.join(args.out, "training_log.jsonl"))

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

    system = GmonSystem(
        GmonSystemConfig(
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            bandwidth_mhz=baseline_cfg.filter_bandwidth_mhz,
        )
    )

    objective = TorchGmonObjective(baseline_cfg)
    overall_start = time.time()

    # ================================================================
    # Optimize each horizon with detailed logging
    # ================================================================
    all_horizon_results: List[Dict[str, Any]] = []

    for horizon_idx, horizon_ns in enumerate(horizons):
        print(f"\n{'='*60}", flush=True)
        print(f"Horizon {horizon_idx + 1}/{len(horizons)}: {horizon_ns} ns", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        result = objective.optimize_for_horizon(alpha, gamma, horizon_ns)
        elapsed = time.time() - t0

        # ---- Per-Adam-step log records ----
        best_cost_so_far = float("inf")
        for entry in result["history"]:
            cost_val = float(entry["cost"])
            best_cost_so_far = min(best_cost_so_far, cost_val)
            record: Dict[str, Any] = {
                "type": "adam_step",
                "horizon_ns": float(horizon_ns),
                "horizon_index": int(horizon_idx),
                "adam_iter": int(entry["iter"]),
                "cost": cost_val,
                "best_cost_so_far": float(best_cost_so_far),
                "alpha": float(alpha),
                "gamma": float(gamma),
            }
            logger.write(record)

        # ---- Build ControlPlan for this horizon's best ----
        plan = ControlPlan(
            controls_mhz_and_phase=result["controls"],
            target_alpha=alpha,
            target_gamma=gamma,
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            cost_weights=weights,
            note=f"Adam baseline horizon {horizon_ns} ns",
        )

        # ---- Full nominal evaluation (matching TRPO eval) ----
        nominal = simulate_nominal_plan(system, plan)

        # ---- Horizon summary record ----
        horizon_summary: Dict[str, Any] = {
            "type": "horizon_summary",
            "horizon_ns": float(horizon_ns),
            "horizon_index": int(horizon_idx),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "max_time_ns": float(args.max_time_ns),
            "termination_cost": float(args.termination_cost),
            "time_steps": int(result["controls"].shape[0]),
            "adam_iters": int(args.adam_iters),
            "lr": float(args.lr),
            "elapsed_seconds": round(elapsed, 2),
            # Optimizer result
            "optimizer_cost": float(result["best_cost"]),
            # Full nominal evaluation breakdown
            "nominal_cost": float(nominal["cost"]),
            "nominal_fidelity": float(nominal["fidelity"]),
            "nominal_leakage": float(nominal["leakage"]),
            "nominal_leakage_boundary": float(nominal["leakage_boundary"]),
            "nominal_leakage_integral": float(nominal["leakage_integral"]),
            "nominal_boundary_cost": float(nominal["boundary_cost"]),
            "nominal_time_cost": float(nominal["time_cost"]),
            "nominal_time_ns": float(nominal["time_ns"]),
        }

        logger.write(horizon_summary)
        print(json.dumps(horizon_summary, indent=2), flush=True)

        all_horizon_results.append({
            "horizon_ns": horizon_ns,
            "result": result,
            "nominal": nominal,
            "plan": plan,
            "summary": horizon_summary,
        })

    # ================================================================
    # Find best horizon
    # ================================================================
    best_entry = min(all_horizon_results, key=lambda x: x["result"]["best_cost"])
    best = best_entry["result"]
    best_plan: ControlPlan = best_entry["plan"]
    best_nominal: Dict[str, Any] = best_entry["nominal"]

    # Save best control plan
    plan_path = os.path.join(args.out, "best_control_plan.npz")
    best_plan.save(plan_path)

    # ---- Horizon search CSV (enriched) ----
    with open(os.path.join(args.out, "horizon_search.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "horizon_ns", "optimizer_cost", "nominal_cost", "nominal_fidelity",
            "nominal_leakage", "nominal_leakage_boundary", "nominal_leakage_integral",
            "nominal_boundary_cost", "nominal_time_cost", "nominal_time_ns",
        ])
        writer.writeheader()
        for entry in all_horizon_results:
            writer.writerow({
                "horizon_ns": entry["horizon_ns"],
                "optimizer_cost": entry["result"]["best_cost"],
                "nominal_cost": entry["nominal"]["cost"],
                "nominal_fidelity": entry["nominal"]["fidelity"],
                "nominal_leakage": entry["nominal"]["leakage"],
                "nominal_leakage_boundary": entry["nominal"]["leakage_boundary"],
                "nominal_leakage_integral": entry["nominal"]["leakage_integral"],
                "nominal_boundary_cost": entry["nominal"]["boundary_cost"],
                "nominal_time_cost": entry["nominal"]["time_cost"],
                "nominal_time_ns": entry["nominal"]["time_ns"],
            })

    # ================================================================
    # Robustness evaluation on best plan
    # ================================================================
    robustness_data: Dict[str, Any] | None = None
    robustness_path: str | None = None

    if robustness_sigmas:
        print(f"\n{'='*60}", flush=True)
        print("Computing robustness metrics for best plan...", flush=True)
        print(f"  Sigmas: {robustness_sigmas} MHz", flush=True)
        print(f"  Samples: {args.robustness_samples}", flush=True)
        print(f"{'='*60}", flush=True)

        robustness = compute_robustness_suite(
            system=system,
            plan=best_plan,
            sigmas_mhz=robustness_sigmas,
            num_samples=args.robustness_samples,
            base_seed=args.seed,
        )
        robustness_data = robustness

        robustness_payload: Dict[str, Any] = {
            "alpha": float(alpha),
            "gamma": float(gamma),
            "best_horizon_ns": float(best["horizon_ns"]),
            "nominal_cost": float(best_nominal["cost"]),
            "nominal_fidelity": float(best_nominal["fidelity"]),
            "nominal_leakage": float(best_nominal["leakage"]),
            "nominal_time_ns": float(best_nominal["time_ns"]),
            "plan_path": plan_path,
            "robustness": robustness,
        }
        robustness_path = os.path.join(robustness_dir, "best_robustness.json")
        save_json(robustness_path, robustness_payload)

        # Write robustness record to training log
        robustness_record: Dict[str, Any] = {
            "type": "robustness",
            "alpha": float(alpha),
            "gamma": float(gamma),
            "best_horizon_ns": float(best["horizon_ns"]),
            "robustness": robustness,
            "robustness_path": robustness_path,
            **flatten_robustness_for_log(robustness),
        }
        logger.write(robustness_record)
        print(json.dumps(robustness_record, indent=2), flush=True)

    total_elapsed = time.time() - overall_start

    # ================================================================
    # Comprehensive summary.json
    # ================================================================
    summary: Dict[str, Any] = {
        # Configuration
        "alpha": float(alpha),
        "gamma": float(gamma),
        "dt_ns": float(args.dt_ns),
        "max_time_ns": float(args.max_time_ns),
        "runtime_norm_ns": float(args.runtime_norm_ns),
        "termination_cost": float(args.termination_cost),
        "lr": float(args.lr),
        "adam_iters": int(args.adam_iters),
        "seed": int(args.seed),
        "horizons_ns": list(horizons),
        "cost_weights": {
            "chi": float(weights.chi),
            "beta": float(weights.beta),
            "mu": float(weights.mu),
            "kappa": float(weights.kappa),
        },
        "train_noise_std_mhz": float(args.train_noise_std),
        "train_noise_samples": int(args.train_noise_samples),
        # Best result
        "best_horizon_ns": float(best["horizon_ns"]),
        "objective_cost": float(best["best_cost"]),
        # Nominal evaluation of best plan (full decomposition)
        "nominal_cost": float(best_nominal["cost"]),
        "nominal_fidelity": float(best_nominal["fidelity"]),
        "nominal_leakage": float(best_nominal["leakage"]),
        "nominal_leakage_boundary": float(best_nominal["leakage_boundary"]),
        "nominal_leakage_integral": float(best_nominal["leakage_integral"]),
        "nominal_boundary_cost": float(best_nominal["boundary_cost"]),
        "nominal_time_cost": float(best_nominal["time_cost"]),
        "nominal_time_ns": float(best_nominal["time_ns"]),
        # Paths
        "plan_path": plan_path,
        "robustness_path": robustness_path,
        # Robustness config
        "robustness_sigmas": robustness_sigmas,
        "robustness_samples": int(args.robustness_samples),
        # Timing
        "total_elapsed_seconds": round(total_elapsed, 2),
        # Per-horizon results
        "per_horizon": [
            {
                "horizon_ns": float(e["horizon_ns"]),
                "optimizer_cost": float(e["result"]["best_cost"]),
                "nominal_cost": float(e["nominal"]["cost"]),
                "nominal_fidelity": float(e["nominal"]["fidelity"]),
                "nominal_leakage": float(e["nominal"]["leakage"]),
                "nominal_leakage_boundary": float(e["nominal"]["leakage_boundary"]),
                "nominal_leakage_integral": float(e["nominal"]["leakage_integral"]),
                "nominal_boundary_cost": float(e["nominal"]["boundary_cost"]),
                "nominal_time_cost": float(e["nominal"]["time_cost"]),
                "nominal_time_ns": float(e["nominal"]["time_ns"]),
                "elapsed_seconds": e["summary"]["elapsed_seconds"],
            }
            for e in all_horizon_results
        ],
    }

    # Add robustness to summary
    if robustness_data is not None:
        summary["robustness"] = robustness_data
        summary.update(flatten_robustness_for_log(robustness_data))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
