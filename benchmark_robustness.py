import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from uqc.env import EnvConfig, QuantumControlEnv
from uqc.eval import ControlPlan, robustness_metrics, simulate_nominal_plan
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.trpo import TRPOAgent, TRPOConfig
from uqc.utils import ensure_dir, parse_angle_expr, set_seeds


def load_plan_from_checkpoint(ckpt_path: str) -> Tuple[ControlPlan, Dict[str, Any]]:
    base_dir = os.path.dirname(ckpt_path)
    args_json = os.path.join(base_dir, "args.json")
    summary_json = os.path.join(base_dir, "summary.json")

    args: Dict[str, Any] = {}
    if os.path.exists(args_json):
        with open(args_json, "r") as f:
            args = json.load(f)
    elif os.path.exists(summary_json):
        with open(summary_json, "r") as f:
            args = json.load(f)

    alpha_val = args.get("alpha")
    gamma_val = args.get("gamma", "pi/2")
    if isinstance(alpha_val, float):
        alpha = alpha_val
    else:
        alpha = parse_angle_expr(str(alpha_val)) if alpha_val else 2.2
    
    if isinstance(gamma_val, float):
        gamma = gamma_val
    else:
        gamma = parse_angle_expr(str(gamma_val))
    
    dt_ns = float(args.get("dt_ns", 2.0))
    runtime_norm_ns = float(args.get("runtime_norm_ns", 60.0))
    max_time_ns = float(args.get("max_time_ns", 600.0))
    cost_chi = float(args.get("cost_chi", 10.0))
    cost_beta = float(args.get("cost_beta", 10.0))
    cost_mu = float(args.get("cost_mu", 0.2))
    cost_kappa = float(args.get("cost_kappa", 0.1))

    # Checkpoint payload might have alpha, gamma embedded
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "alpha" in state:
            alpha = float(state["alpha"])
        if "gamma" in state:
            gamma = float(state["gamma"])

    system = GmonSystem(
        GmonSystemConfig(
            dt_ns=dt_ns,
            runtime_norm_ns=runtime_norm_ns,
            bandwidth_mhz=10.0,
        )
    )
    weights = UFOCostWeights(
        chi=cost_chi,
        beta=cost_beta,
        mu=cost_mu,
        kappa=cost_kappa,
    )
    eval_env = QuantumControlEnv(
        system,
        EnvConfig(
            target_alpha=alpha,
            target_gamma=gamma,
            max_time_ns=max_time_ns,
            dt_ns=dt_ns,
            noise_optimized=False,
            reward_mode=args.get("reward_mode", "dense_current_cost"),
            termination_cost=args.get("termination_cost", 0.15),
            runtime_norm_ns=runtime_norm_ns,
            cost_weights=weights,
            seed=args.get("seed", 1),
        ),
    )

    trpo_cfg = TRPOConfig()
    agent = TRPOAgent(eval_env.observation_dim, eval_env.action_dim, config=trpo_cfg)
    
    if isinstance(state, dict):
        if "agent_state" in state:
            agent.load_state_dict(state["agent_state"])
        elif "policy" in state:
            agent.load_state_dict(state)
        
    rollout = eval_env.rollout(agent.get_action, deterministic=True)

    plan = ControlPlan(
        controls_mhz_and_phase=np.asarray(rollout["nominal_controls"], dtype=np.float64),
        target_alpha=alpha,
        target_gamma=gamma,
        dt_ns=dt_ns,
        runtime_norm_ns=runtime_norm_ns,
        cost_weights=weights,
        note=f"Reconstructed from {ckpt_path}",
    )
    
    metadata = {
        "source_path": ckpt_path,
        "input_type": "checkpoint",
        "num_steps": len(rollout["nominal_controls"]),
    }
    return plan, metadata


def format_label_from_path(path: str) -> str:
    name = os.path.basename(path.rstrip('\\/'))
    return name.replace(".npz", "").replace(".pt", "")


def parse_and_resolve_inputs(inputs: List[str], labels: Optional[List[str]]) -> List[Tuple[ControlPlan, str, Dict[str, Any]]]:
    resolved = []
    
    for i, path in enumerate(inputs):
        label = labels[i] if labels and i < len(labels) else format_label_from_path(path)
        
        if os.path.isdir(path):
            plan_path = os.path.join(path, "best_control_plan.npz")
            ckpt_path = os.path.join(path, "best_agent.pt")
            final_ckpt = os.path.join(path, "final_agent.pt")
            
            if os.path.exists(plan_path):
                plan = ControlPlan.load(plan_path)
                meta = {
                    "source_path": plan_path,
                    "input_type": "plan (auto-directory)",
                    "num_steps": len(plan.controls_mhz_and_phase)
                }
                resolved.append((plan, label, meta))
            elif os.path.exists(ckpt_path):
                plan, meta = load_plan_from_checkpoint(ckpt_path)
                meta["input_type"] = "checkpoint (auto-directory)"
                resolved.append((plan, label, meta))
            elif os.path.exists(final_ckpt):
                plan, meta = load_plan_from_checkpoint(final_ckpt)
                meta["input_type"] = "checkpoint final (auto-directory)"
                resolved.append((plan, label, meta))
            else:
                raise ValueError(f"Directory {path} doesn't contain a recognizable .npz or .pt file.")
                
        elif path.endswith(".npz"):
            plan = ControlPlan.load(path)
            meta = {
                "source_path": path,
                "input_type": "plan",
                "num_steps": len(plan.controls_mhz_and_phase)
            }
            resolved.append((plan, label, meta))
            
        elif path.endswith(".pt"):
            plan, meta = load_plan_from_checkpoint(path)
            resolved.append((plan, label, meta))
            
        else:
            raise ValueError(f"Unsupported file format for input: {path}")

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate control plans for Figure 4 datasets.")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, 
                        help="List of paths to .npz files, .pt checkpoints, or directories.")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="List of labels matching --inputs. If omitted, derives from paths.")
    parser.add_argument("--out", type=str, required=True, 
                        help="Output directory where each method gets its own subfolder.")
    parser.add_argument("--noise-min", type=float, default=0.1)
    parser.add_argument("--noise-max", type=float, default=3.5)
    parser.add_argument("--noise-step", type=float, default=0.1)
    parser.add_argument("--sigmas", type=str, default=None, 
                        help="Comma-separated explicit sigma list, overrides min/max/step if provided.")
    parser.add_argument("--samples", type=int, default=60, help="Monte Carlo samples per noise level.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots (Figure 4).")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        print("Error: --labels must have the same length as --inputs")
        sys.exit(1)

    ensure_dir(args.out)

    if args.sigmas:
        sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    else:
        sigmas = []
        cur = args.noise_min
        while cur <= args.noise_max + 1e-12:
            sigmas.append(round(cur, 10))
            cur += args.noise_step

    set_seeds(args.seed)

    print("Resolving inputs...")
    resolved_plans = parse_and_resolve_inputs(args.inputs, args.labels)
    
    for plan, label, meta in resolved_plans:
        method_out_dir = os.path.join(args.out, label)
        ensure_dir(method_out_dir)
        print(f"\nEvaluating '{label}' -> {method_out_dir}")
        print(f"Source: {meta['source_path']} ({meta['input_type']})")
        
        system = GmonSystem(GmonSystemConfig(dt_ns=plan.dt_ns, runtime_norm_ns=plan.runtime_norm_ns))
        
        # Calculate nominal metrics
        nominal_info = simulate_nominal_plan(system, plan)
        
        # Gather info for summary
        summary: Dict[str, Any] = {
            "method_label": label,
            "source_path": meta["source_path"],
            "input_type": meta["input_type"],
            "target_alpha": plan.target_alpha,
            "target_gamma": plan.target_gamma,
            "num_steps": meta["num_steps"],
            "dt_ns": plan.dt_ns,
            "runtime_norm_ns": plan.runtime_norm_ns,
            "seed": args.seed,
            "num_samples": args.samples,
            "sigmas_evaluated": sigmas,
            "nominal": {
                "fidelity": float(nominal_info.get("fidelity", np.nan)),
                "leakage": float(nominal_info.get("leakage", np.nan)),
                "cost": float(nominal_info.get("cost", np.nan)),
                "time_ns": float(nominal_info.get("time_ns", np.nan)),
            }
        }
        
        rows: List[Dict[str, Any]] = []
        
        # Robustness sweep
        for idx, sigma in enumerate(sigmas):
            metrics = robustness_metrics(system, plan, sigma_mhz=sigma, num_samples=args.samples, seed=args.seed + idx)
            # Add context for csv
            metrics["method"] = label
            rows.append(metrics)
            
            print(f"  sigma={sigma:5.2f} MHz | F={metrics['average_fidelity']:.5f} | Gate F={metrics['average_gate_fidelity']:.5f} | Var={metrics['fidelity_variance']:.3e}")
        
        # Dump summary
        summary_path = os.path.join(method_out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            
        # Dump CSV
        csv_path = os.path.join(method_out_dir, "robustness_curve.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "sigma_mhz", "num_samples", "average_fidelity", "average_gate_fidelity", "fidelity_variance"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                
        print(f"Finished method '{label}'. Saved to {method_out_dir}")

    if args.plot:
        print("\nGenerating comparison plots...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # Consistent colors for comparison
        colors = plt.cm.tab10(np.linspace(0, 1, len(resolved_plans)))
        
        for (plan, label, meta), color in zip(resolved_plans, colors):
            csv_path = os.path.join(args.out, label, "robustness_curve.csv")
            if not os.path.exists(csv_path):
                continue
            
            data = np.genfromtxt(csv_path, delimiter=',', names=True)
            # Handle single row case
            if data.size == 1:
                data = np.array([data])
                
            ax1.plot(data['sigma_mhz'], data['average_fidelity'], label=label, color=color, linewidth=2)
            ax2.plot(data['sigma_mhz'], data['fidelity_variance'], label=label, color=color, linewidth=2)

        # Panel A: Average Fidelity
        ax1.set_ylabel("Average Fidelity", fontsize=12)
        ax1.set_title("Panel A: Average Fidelity vs Noise Strength", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Panel B: Fidelity Variance
        ax2.set_ylabel("Fidelity Variance", fontsize=12)
        ax2.set_xlabel(r"Noise Std Dev $\sigma$ (MHz)", fontsize=12)
        ax2.set_title("Panel B: Fidelity Variance vs Noise Strength", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(args.out, "robustness_comparison.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plots saved to {plot_path}")

    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()
