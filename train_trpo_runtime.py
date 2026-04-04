from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

# Keep Torch CPU-only and single-threaded in this environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from uqc.env import EnvConfig, QuantumControlEnv
from uqc.eval import ControlPlan, robustness_metrics
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.parallel import ParallelBatchCollector
from uqc.trpo import TRPOAgent, TRPOConfig
from uqc.utils import JsonlLogger, ensure_dir, parse_angle_expr, set_seeds


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass



def collect_batch(env: QuantumControlEnv, agent: TRPOAgent, episodes_per_batch: int) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    transitions: List[Dict[str, object]] = []
    finals: List[Dict[str, float]] = []
    for _ in range(episodes_per_batch):
        rollout = env.rollout(agent.get_action, deterministic=False)
        for tr in rollout["transitions"]:
            transitions.append(
                {
                    "obs": tr["obs"],
                    "raw_action": tr["raw_action"],
                    "reward": tr["reward"],
                    "mask": tr["mask"],
                }
            )
        finals.append(rollout["final_info"])
    return transitions, finals



def summarize(finals: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ["cost", "fidelity", "leakage", "time_ns", "boundary_cost", "time_cost", "min_cost", "min_cost_time_ns"]:
        vals = [float(x.get(key, np.nan)) for x in finals]
        out[f"avg_{key}"] = float(np.nanmean(vals))
    return out


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



def evaluate(agent: TRPOAgent, env: QuantumControlEnv) -> Dict[str, object]:
    rollout = env.rollout(agent.get_action, deterministic=True)
    final = dict(rollout["final_info"])
    final["nominal_controls"] = rollout["nominal_controls"]
    return final



def save_checkpoint(agent: TRPOAgent, path: str, *, alpha: float, gamma: float, phase_iter: int, note: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    payload = {
        "agent_state": agent.state_dict(),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "phase_iter": int(phase_iter),
        "note": str(note),
    }
    torch.save(payload, path)



def load_checkpoint_into_agent(agent: TRPOAgent, checkpoint_path: str) -> Dict[str, object]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "agent_state" in state:
        agent.load_state_dict(state["agent_state"])
        return state
    if isinstance(state, dict) and "policy" in state:
        agent.load_state_dict(state)
        return {"agent_state": state}
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")



def save_eval_plan(
    eval_info: Dict[str, object],
    *,
    alpha: float,
    gamma: float,
    dt_ns: float,
    runtime_norm_ns: float,
    weights: UFOCostWeights,
    path: str,
    note: str,
) -> None:
    plan = ControlPlan(
        controls_mhz_and_phase=np.asarray(eval_info["nominal_controls"], dtype=np.float64),
        target_alpha=alpha,
        target_gamma=gamma,
        dt_ns=dt_ns,
        runtime_norm_ns=runtime_norm_ns,
        cost_weights=weights,
        note=note,
    )
    plan.save(path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum TRPO sweep for Fig. 3 style runtime curves.")
    parser.add_argument("--gammas", type=str, default="pi/2,pi/6,pi/3")
    parser.add_argument("--alpha-start", type=str, default="0.0")
    parser.add_argument("--alpha-stop", type=str, default="pi")
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--noise-optimized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-noise-std", type=float, default=1.0)
    parser.add_argument("--episodes-per-batch", type=int, default=20000)
    parser.add_argument("--max-iters-per-alpha", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--max-time-ns", type=float, default=600.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--termination-cost", type=float, default=0.15)
    parser.add_argument("--advance-cost-threshold", type=float, default=0.15)
    parser.add_argument("--advance-train-cost-threshold", type=float, default=None)
    parser.add_argument("--advance-min-cost-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reward-mode", type=str, default="dense_current_cost", choices=["dense_current_cost", "terminal_ufo"])
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--gamma-rl", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-epochs", type=int, default=50)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32, 32])
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional checkpoint to initialize the first phase.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel rollout workers for batch collection.")
    parser.add_argument("--episodes-per-task", type=int, default=0, help="Episodes per worker task; 0 chooses automatically.")
    parser.add_argument(
        "--robustness-sigmas",
        type=str,
        default="1.0",
        help="Comma-separated sigma values in MHz for robustness eval, e.g. '0.5,1.0,2.0'. Empty string disables robustness eval.",
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=60,
        help="Monte Carlo samples per robustness evaluation.",
    )
    parser.add_argument("--eval-robustness-every", type=int, default=1, help="Run robustness evaluation every N iterations.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the existing outdir.")
    args = parser.parse_args()

    ensure_dir(args.out)
    if getattr(args, "resume", False):
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

    advance_train_cost_threshold = args.advance_train_cost_threshold if args.advance_train_cost_threshold is not None else float("inf")
    advance_min_cost_threshold = args.advance_min_cost_threshold if args.advance_min_cost_threshold is not None else float("inf")

    robustness_sigmas = parse_float_csv(args.robustness_sigmas)

    set_seeds(args.seed)
    alpha_start = parse_angle_expr(args.alpha_start)
    alpha_stop = parse_angle_expr(args.alpha_stop)
    gamma_values = [parse_angle_expr(x) for x in args.gammas.split(",") if x.strip()]

    system = GmonSystem(
        GmonSystemConfig(
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            bandwidth_mhz=10.0,
        )
    )
    weights = UFOCostWeights()
    trpo_cfg = TRPOConfig(
        gamma=args.gamma_rl,
        lam=args.lam,
        max_kl=args.max_kl,
        damping=args.damping,
        cg_iters=args.cg_iters,
        value_lr=args.value_lr,
        value_epochs=args.value_epochs,
        init_log_std=args.init_log_std,
        hidden_sizes=tuple(args.hidden_sizes),
    )

    summary_rows: List[Dict[str, object]] = []

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

    import glob

    total_gammas = len(gamma_values)
    total_alphas = len(alpha_values)

    for gamma_idx, gamma in enumerate(gamma_values, start=1):
        gamma_dir = os.path.join(args.out, f"gamma_{gamma:.6f}")
        ensure_dir(gamma_dir)
        ensure_dir(os.path.join(gamma_dir, "curriculum_checkpoints"))
        logger = JsonlLogger(os.path.join(gamma_dir, "training_log.jsonl"))
        agent = None
        last_curriculum_ckpt = args.init_checkpoint

        for alpha_idx, alpha in enumerate(alpha_values, start=1):
            phase_dir = os.path.join(gamma_dir, f"alpha_{alpha:.6f}")
            phase_ckpt_dir = os.path.join(phase_dir, "checkpoints")
            phase_plan_dir = os.path.join(phase_dir, "plans")
            phase_robustness_dir = os.path.join(phase_dir, "robustness")
            
            summary_path = os.path.join(phase_dir, "summary.json")
            if getattr(args, "resume", False) and os.path.exists(summary_path):
                print(f"[Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas}] Skipping completed phase gamma={gamma:.6f}, alpha={alpha:.6f}", flush=True)
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_rows.append(json.load(f))
                last_curriculum_ckpt = os.path.join(gamma_dir, "curriculum_checkpoints", f"after_alpha_{alpha:.6f}.pt")
                continue

            ensure_dir(phase_dir)
            ensure_dir(phase_ckpt_dir)
            ensure_dir(phase_plan_dir)
            ensure_dir(phase_robustness_dir)
            train_env = QuantumControlEnv(
                system,
                EnvConfig(
                    target_alpha=alpha,
                    target_gamma=gamma,
                    max_time_ns=args.max_time_ns,
                    dt_ns=args.dt_ns,
                    noise_optimized=bool(args.noise_optimized),
                    train_noise_std_mhz=args.train_noise_std,
                    reward_mode=args.reward_mode,
                    termination_cost=args.termination_cost,
                    runtime_norm_ns=args.runtime_norm_ns,
                    cost_weights=weights,
                    seed=args.seed,
                ),
            )
            eval_env = QuantumControlEnv(
                system,
                EnvConfig(
                    target_alpha=alpha,
                    target_gamma=gamma,
                    max_time_ns=args.max_time_ns,
                    dt_ns=args.dt_ns,
                    noise_optimized=False,
                    reward_mode=args.reward_mode,
                    termination_cost=args.termination_cost,
                    runtime_norm_ns=args.runtime_norm_ns,
                    cost_weights=weights,
                    seed=args.seed,
                ),
            )
            best_eval_cost = float("inf")
            best_eval = None
            best_train_stats = {}
            phase_iters_used = 0
            start_iteration = 0

            if agent is None:
                agent = TRPOAgent(train_env.observation_dim, train_env.action_dim, config=trpo_cfg)
                
                if getattr(args, "resume", False):
                    ckpt_files = glob.glob(os.path.join(phase_ckpt_dir, "iter_*.pt"))
                    if ckpt_files:
                        latest_ckpt_path = max(ckpt_files, key=lambda p: int(os.path.basename(p).replace("iter_", "").replace(".pt", "")))
                        start_iteration = int(os.path.basename(latest_ckpt_path).replace("iter_", "").replace(".pt", ""))
                        print(f"[Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas}] Resuming phase gamma={gamma:.6f}, alpha={alpha:.6f} from iter {start_iteration}", flush=True)
                        load_checkpoint_into_agent(agent, latest_ckpt_path)
                        
                        log_path = logger.path
                        if os.path.exists(log_path):
                            with open(log_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    if not line.strip(): continue
                                    try:
                                        r = json.loads(line)
                                        if abs(r.get("gamma", -1) - gamma) < 1e-5 and abs(r.get("alpha", -1) - alpha) < 1e-5:
                                            if r.get("phase_iter", 0) <= start_iteration:
                                                c = r.get("eval_cost", float("inf"))
                                                if c < best_eval_cost:
                                                    best_eval_cost = c
                                                    best_eval = {
                                                        "cost": c,
                                                        "fidelity": r.get("eval_fidelity", 0.0),
                                                        "leakage": r.get("eval_leakage", 0.0),
                                                        "time_ns": r.get("eval_time_ns", 0.0),
                                                        "min_cost": r.get("eval_min_cost", c),
                                                        "min_cost_time_ns": r.get("eval_min_cost_time_ns", r.get("eval_time_ns", 0.0)),
                                                    }
                                                    best_train_stats = {
                                                        "avg_fidelity": r.get("avg_fidelity", np.nan),
                                                        "avg_cost": r.get("avg_cost", np.nan),
                                                    }
                                                    phase_iters_used = r.get("phase_iter", 0)
                                    except Exception: pass
                    elif last_curriculum_ckpt and os.path.exists(last_curriculum_ckpt):
                        print(f"[Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas}] Starting phase gamma={gamma:.6f}, alpha={alpha:.6f} from curriculum ckpt", flush=True)
                        init_meta = load_checkpoint_into_agent(agent, last_curriculum_ckpt)
                elif last_curriculum_ckpt and os.path.exists(last_curriculum_ckpt):
                    init_meta = load_checkpoint_into_agent(agent, last_curriculum_ckpt)

            collector = ParallelBatchCollector(
                system_config=system.config,
                env_config=train_env.config,
                trpo_config=trpo_cfg,
                num_workers=args.num_workers,
                episodes_per_task=(None if args.episodes_per_task <= 0 else args.episodes_per_task),
                seed=args.seed + int(round(alpha * 1000.0)) + int(round(gamma * 1000.0)) * 10000,
            )
            with collector:
                for phase_iter in range(start_iteration + 1, args.max_iters_per_alpha + 1):
                    print(f"\n--- [Progress] Gamma {gamma_idx}/{total_gammas} | Alpha {alpha_idx}/{total_alphas} | Iter {phase_iter}/{args.max_iters_per_alpha} ---", flush=True)
                    transitions, finals = (collector.collect(agent, args.episodes_per_batch) if args.num_workers > 1 else collect_batch(train_env, agent, args.episodes_per_batch))
                    update_info = agent.update(transitions)
                    train_stats = summarize(finals)
                    eval_info = evaluate(agent, eval_env)
                    phase_iters_used = phase_iter
    
                    iter_ckpt_path = os.path.join(phase_ckpt_dir, f"iter_{phase_iter:06d}.pt")
                    if args.save_every > 0 and (phase_iter % args.save_every == 0 or phase_iter == args.max_iters_per_alpha):
                        save_checkpoint(
                            agent,
                            iter_ckpt_path,
                            alpha=alpha,
                            gamma=gamma,
                            phase_iter=phase_iter,
                            note=f"runtime sweep gamma={gamma:.6f} alpha={alpha:.6f} iter={phase_iter}",
                        )
                    else:
                        iter_ckpt_path = ""
    
                    iter_plan_path = os.path.join(phase_plan_dir, f"iter_{phase_iter:06d}_control_plan.npz")
                    save_eval_plan(
                        eval_info,
                        alpha=alpha,
                        gamma=gamma,
                        dt_ns=args.dt_ns,
                        runtime_norm_ns=args.runtime_norm_ns,
                        weights=weights,
                        path=iter_plan_path,
                        note=f"TRPO runtime sweep gamma={gamma:.6f} alpha={alpha:.6f} iter={phase_iter}",
                    )
    
                    if float(eval_info["cost"]) < best_eval_cost:
                        best_eval_cost = float(eval_info["cost"])
                        best_eval = eval_info
                        best_train_stats = train_stats
                        save_eval_plan(
                            eval_info,
                            alpha=alpha,
                            gamma=gamma,
                            dt_ns=args.dt_ns,
                            runtime_norm_ns=args.runtime_norm_ns,
                            weights=weights,
                            path=os.path.join(phase_dir, "best_control_plan.npz"),
                            note="TRPO curriculum runtime sweep best plan",
                        )
                        save_checkpoint(
                            agent,
                            os.path.join(phase_dir, "best_agent.pt"),
                            alpha=alpha,
                            gamma=gamma,
                            phase_iter=phase_iter,
                            note="TRPO curriculum runtime sweep best checkpoint",
                        )
    
                    record = {
                        "alpha": alpha,
                        "gamma": gamma,
                        "phase_iter": phase_iter,
                        "num_workers": int(args.num_workers),
                        **train_stats,
                        "eval_cost": float(eval_info["cost"]),
                        "eval_min_cost": float(eval_info.get("min_cost", eval_info["cost"])),
                        "eval_fidelity": float(eval_info["fidelity"]),
                        "eval_leakage": float(eval_info["leakage"]),
                        "eval_time_ns": float(eval_info["time_ns"]),
                        "eval_min_cost_time_ns": float(eval_info.get("min_cost_time_ns", eval_info["time_ns"])),
                        "eval_plan_path": iter_plan_path,
                        **{f"update_{k}": float(v) for k, v in update_info.items()},
                    }

                    if robustness_sigmas and (phase_iter % args.eval_robustness_every == 0 or phase_iter == args.max_iters_per_alpha):
                        iter_plan = ControlPlan(
                            controls_mhz_and_phase=np.asarray(eval_info["nominal_controls"], dtype=np.float64),
                            target_alpha=alpha,
                            target_gamma=gamma,
                            dt_ns=args.dt_ns,
                            runtime_norm_ns=args.runtime_norm_ns,
                            cost_weights=weights,
                            note=f"TRPO runtime sweep iteration {phase_iter}",
                        )
                        robustness = compute_robustness_suite(
                            system=system,
                            plan=iter_plan,
                            sigmas_mhz=robustness_sigmas,
                            num_samples=args.robustness_samples,
                            base_seed=args.seed + 10000 * phase_iter,
                        )

                        iter_robustness_path = os.path.join(phase_robustness_dir, f"iter_{phase_iter:06d}_robustness.json")
                        robustness_payload: Dict[str, Any] = {
                            "phase_iter": int(phase_iter),
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "eval_cost": float(eval_info["cost"]),
                            "eval_fidelity": float(eval_info["fidelity"]),
                            "eval_leakage": float(eval_info["leakage"]),
                            "eval_time_ns": float(eval_info["time_ns"]),
                            "eval_plan_path": iter_plan_path,
                            "robustness": robustness,
                        }
                        save_json(iter_robustness_path, robustness_payload)

                        record["robustness"] = robustness
                        record["robustness_path"] = iter_robustness_path
                        record.update(flatten_robustness_for_log(robustness))

                    if iter_ckpt_path:
                        record["checkpoint_path"] = iter_ckpt_path
                    logger.write(record)
                    print(json.dumps(record), flush=True)
                    if (float(eval_info["cost"]) <= args.advance_cost_threshold and 
                        float(train_stats["avg_cost"]) <= advance_train_cost_threshold and
                        float(eval_info.get("min_cost", float("inf"))) <= advance_min_cost_threshold):
                        break

            if best_eval is None:
                best_eval = evaluate(agent, eval_env)
                best_train_stats = train_stats if 'train_stats' in locals() else {}
            plan_path = os.path.join(phase_dir, "best_control_plan.npz")
            if not os.path.exists(plan_path):
                save_eval_plan(
                    best_eval,
                    alpha=alpha,
                    gamma=gamma,
                    dt_ns=args.dt_ns,
                    runtime_norm_ns=args.runtime_norm_ns,
                    weights=weights,
                    path=plan_path,
                    note="TRPO curriculum runtime sweep fallback plan",
                )
            last_ckpt_path = os.path.join(phase_dir, "last_agent.pt")
            save_checkpoint(
                agent,
                last_ckpt_path,
                alpha=alpha,
                gamma=gamma,
                phase_iter=phase_iters_used,
                note="TRPO curriculum runtime sweep final checkpoint for this phase",
            )
            
            best_ckpt_path = os.path.join(phase_dir, "best_agent.pt")
            if os.path.exists(best_ckpt_path):
                load_checkpoint_into_agent(agent, best_ckpt_path)

            curriculum_ckpt_path = os.path.join(gamma_dir, "curriculum_checkpoints", f"after_alpha_{alpha:.6f}.pt")
            save_checkpoint(
                agent,
                curriculum_ckpt_path,
                alpha=alpha,
                gamma=gamma,
                phase_iter=phase_iters_used,
                note="TRPO curriculum checkpoint carried to next alpha",
            )
            summary_row = {
                "alpha": float(alpha),
                "gamma": float(gamma),
                "best_cost": float(best_eval["cost"]),
                "best_min_cost": float(best_eval.get("min_cost", best_eval["cost"])),
                "best_min_cost_time_ns": float(best_eval.get("min_cost_time_ns", best_eval["time_ns"])),
                "best_fidelity": float(best_eval["fidelity"]),
                "best_leakage": float(best_eval["leakage"]),
                "runtime_ns": float(best_eval["time_ns"]),
                "train_avg_fidelity": float(best_train_stats.get("avg_fidelity", np.nan)),
                "iterations_used": float(phase_iters_used),
                "control_plan": plan_path,
                "best_checkpoint": os.path.join(phase_dir, "best_agent.pt") if os.path.exists(os.path.join(phase_dir, "best_agent.pt")) else None,
                "last_checkpoint": last_ckpt_path,
                "curriculum_checkpoint": curriculum_ckpt_path,
            }
            summary_rows.append(summary_row)
            with open(os.path.join(phase_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_row, f, indent=2, sort_keys=True)

    csv_path = os.path.join(args.out, "runtime_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "gamma",
                "best_cost",
                "best_min_cost",
                "best_min_cost_time_ns",
                "best_fidelity",
                "best_leakage",
                "runtime_ns",
                "train_avg_fidelity",
                "iterations_used",
                "control_plan",
                "best_checkpoint",
                "last_checkpoint",
                "curriculum_checkpoint",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    summary = {
        "runtime_summary_csv": csv_path,
        "num_rows": len(summary_rows),
        "init_checkpoint": args.init_checkpoint,
        "robustness_sigmas": robustness_sigmas,
        "robustness_samples": args.robustness_samples,
    }
    if init_meta is not None:
        summary["init_checkpoint_alpha"] = float(init_meta.get("alpha", np.nan)) if isinstance(init_meta.get("alpha", np.nan), (int, float)) else None
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved runtime summary to {csv_path}")


if __name__ == "__main__":
    main()
