from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="Adam baseline sweep for Fig. 3 style runtime curves.")
    parser.add_argument("--gammas", type=str, default="pi/2,pi/6,pi/3")
    parser.add_argument("--alpha-start", type=str, default="0.0")
    parser.add_argument("--alpha-stop", type=str, default="pi")
    parser.add_argument("--alpha-step", type=float, default=0.1)
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
    parser.add_argument("--resume", action="store_true", help="Resume from the existing outdir.")
    args = parser.parse_args()

    ensure_dir(args.out)
    if args.resume:
        args_file = os.path.join(args.out, "args.json")
        if os.path.exists(args_file):
            with open(args_file, "r") as f:
                saved_args = json.load(f)
            for k, v in saved_args.items():
                if k not in ["resume", "out"]:
                    setattr(args, k, v)
    else:
        with open(os.path.join(args.out, "args.json"), "w") as f:
            save_args = {k: v for k, v in vars(args).items() if k != "resume"}
            json.dump(save_args, f, indent=2)

    set_seeds(args.seed)
    alpha_start = parse_angle_expr(args.alpha_start)
    alpha_stop = parse_angle_expr(args.alpha_stop)
    gamma_values = [parse_angle_expr(x) for x in args.gammas.split(",") if x.strip()]
    horizons = tuple(float(x) for x in args.horizons_ns.split(",") if x.strip())
    robustness_sigmas = parse_float_csv(args.robustness_sigmas)

    weights = UFOCostWeights(
        chi=args.cost_chi,
        beta=args.cost_beta,
        mu=args.cost_mu,
        kappa=args.cost_kappa,
    )

    system = GmonSystem(
        GmonSystemConfig(
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            bandwidth_mhz=10.0,
        )
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

    alpha_values: List[float] = []
    alpha = alpha_start
    while alpha <= alpha_stop + 1e-12:
        alpha_values.append(float(alpha))
        if abs(alpha - alpha_stop) < 1e-12:
            break
        next_alpha = round(alpha + args.alpha_step, 10)
        if next_alpha > alpha_stop and abs(next_alpha - alpha_stop) < args.alpha_step + 1e-12:
            next_alpha = alpha_stop
        if next_alpha <= alpha + 1e-12:
            break
        alpha = next_alpha

    summary_rows: List[Dict[str, object]] = []
    total_gammas = len(gamma_values)
    total_alphas = len(alpha_values)

    for gamma_idx, gamma in enumerate(gamma_values, start=1):
        gamma_dir = os.path.join(args.out, f"gamma_{gamma:.6f}")
        ensure_dir(gamma_dir)
        logger = JsonlLogger(os.path.join(gamma_dir, "training_log.jsonl"))

        for alpha_idx, alpha in enumerate(alpha_values, start=1):
            phase_dir = os.path.join(gamma_dir, f"alpha_{alpha:.6f}")
            summary_path = os.path.join(phase_dir, "summary.json")
            
            if args.resume and os.path.exists(summary_path):
                print(f"[Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas}] Skipping completed phase gamma={gamma:.6f}, alpha={alpha:.6f}", flush=True)
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_rows.append(json.load(f))
                continue

            ensure_dir(phase_dir)
            print(f"\n--- [Progress] Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas} ---", flush=True)

            all_horizon_results = []
            for horizon_idx, horizon_ns in enumerate(horizons):
                print(f"  Horizon {horizon_idx + 1}/{len(horizons)}: {horizon_ns} ns", flush=True)
                t0 = time.time()
                result = objective.optimize_for_horizon(alpha, gamma, horizon_ns)
                elapsed = time.time() - t0

                # Per-Adam-step log records (optionally too noisy, but consistent with baseline)
                best_cost_so_far = float("inf")
                for entry in result["history"]:
                    cost_val = float(entry["cost"])
                    best_cost_so_far = min(best_cost_so_far, cost_val)
                    # We only log every 10 steps to reduce file size in the sweep if it's too much,
                    # but baseline logs every step. Let's stick to baseline style for now.
                    record: Dict[str, Any] = {
                        "type": "adam_step",
                        "alpha": float(alpha),
                        "gamma": float(gamma),
                        "horizon_ns": float(horizon_ns),
                        "adam_iter": int(entry["iter"]),
                        "cost": cost_val,
                        "best_cost_so_far": float(best_cost_so_far),
                    }
                    logger.write(record)

                plan = ControlPlan(
                    controls_mhz_and_phase=result["controls"],
                    target_alpha=alpha,
                    target_gamma=gamma,
                    dt_ns=args.dt_ns,
                    runtime_norm_ns=args.runtime_norm_ns,
                    cost_weights=weights,
                    note=f"Adam sweep gamma={gamma:.6f} alpha={alpha:.6f} horizon={horizon_ns}ns",
                )
                nominal = simulate_nominal_plan(system, plan)
                
                horizon_summary = {
                    "type": "horizon_summary",
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "horizon_ns": float(horizon_ns),
                    "elapsed_seconds": round(elapsed, 2),
                    "optimizer_cost": float(result["best_cost"]),
                    "nominal_cost": float(nominal["cost"]),
                    "nominal_fidelity": float(nominal["fidelity"]),
                    "nominal_leakage": float(nominal["leakage"]),
                    "nominal_time_ns": float(nominal["time_ns"]),
                }
                logger.write(horizon_summary)
                all_horizon_results.append({
                    "horizon_ns": horizon_ns,
                    "result": result,
                    "nominal": nominal,
                    "plan": plan,
                })

            # Identify best horizon for this alpha/gamma
            best_entry = min(all_horizon_results, key=lambda x: x["result"]["best_cost"])
            best_plan = best_entry["plan"]
            best_nominal = best_entry["nominal"]
            
            plan_path = os.path.join(phase_dir, "best_control_plan.npz")
            best_plan.save(plan_path)

            # Robustness on best
            robustness_data = None
            robustness_path = None
            if robustness_sigmas:
                robustness_dir = os.path.join(phase_dir, "robustness")
                ensure_dir(robustness_dir)
                robustness = compute_robustness_suite(
                    system=system,
                    plan=best_plan,
                    sigmas_mhz=robustness_sigmas,
                    num_samples=args.robustness_samples,
                    base_seed=args.seed + int(alpha * 1000) + int(gamma * 10000),
                )
                robustness_data = robustness
                robustness_path = os.path.join(robustness_dir, "best_robustness.json")
                save_json(robustness_path, {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "best_horizon_ns": float(best_entry["horizon_ns"]),
                    "robustness": robustness,
                })
                
                robustness_record = {
                    "type": "robustness",
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "best_horizon_ns": float(best_entry["horizon_ns"]),
                    "robustness": robustness,
                    **flatten_robustness_for_log(robustness),
                }
                logger.write(robustness_record)

            summary_row = {
                "alpha": float(alpha),
                "gamma": float(gamma),
                "best_cost": float(best_nominal["cost"]),
                "best_fidelity": float(best_nominal["fidelity"]),
                "best_leakage": float(best_nominal["leakage"]),
                "runtime_ns": float(best_nominal["time_ns"]),
                "best_horizon_ns": float(best_entry["horizon_ns"]),
                "control_plan": plan_path,
                "robustness_path": robustness_path,
            }
            if robustness_data:
                summary_row.update(flatten_robustness_for_log(robustness_data))
            
            summary_rows.append(summary_row)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_row, f, indent=2, sort_keys=True)

    csv_path = os.path.join(args.out, "runtime_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    final_summary = {
        "runtime_summary_csv": csv_path,
        "num_rows": len(summary_rows),
        "horizons_tested": list(horizons),
        "robustness_sigmas": robustness_sigmas,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, sort_keys=True)
    print(f"Saved Adam runtime summary to {csv_path}")


if __name__ == "__main__":
    main()
