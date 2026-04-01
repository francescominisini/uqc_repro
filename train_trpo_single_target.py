from __future__ import annotations

import argparse
import glob
import json
import os
import sys
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


def collect_batch(
    env: QuantumControlEnv,
    agent: TRPOAgent,
    episodes_per_batch: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
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


def summarize_final_infos(finals: List[Dict[str, float]]) -> Dict[str, float]:
    if not finals:
        return {
            "avg_cost": float("nan"),
            "avg_fidelity": float("nan"),
            "avg_leakage": float("nan"),
            "avg_time_ns": float("nan"),
            "avg_boundary_cost": float("nan"),
            "avg_time_cost": float("nan"),
            "avg_min_cost": float("nan"),
            "avg_min_cost_time_ns": float("nan"),
            "std_min_cost_time_ns": float("nan"),
        }
    keys = ["cost", "fidelity", "leakage", "time_ns", "boundary_cost", "time_cost", "min_cost", "min_cost_time_ns"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(info.get(key, np.nan)) for info in finals]
        out[f"avg_{key}"] = float(np.nanmean(vals))
        if key == "min_cost_time_ns":
            out[f"std_{key}"] = float(np.nanstd(vals))
    return out


def evaluate(agent: TRPOAgent, env: QuantumControlEnv) -> Dict[str, object]:
    rollout = env.rollout(agent.get_action, deterministic=True)
    final = dict(rollout["final_info"])
    final["nominal_controls"] = rollout["nominal_controls"]
    final["rewards"] = rollout["rewards"]
    final["costs"] = rollout["costs"]
    final["fidelities"] = rollout["fidelities"]
    return final


def _checkpoint_payload(
    agent: TRPOAgent,
    *,
    iteration: int,
    alpha: float,
    gamma: float,
    note: str = "",
) -> Dict[str, object]:
    return {
        "agent_state": agent.state_dict(),
        "iteration": int(iteration),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "note": str(note),
    }


def save_checkpoint(
    agent: TRPOAgent,
    path: str,
    *,
    iteration: int,
    alpha: float,
    gamma: float,
    note: str = "",
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(_checkpoint_payload(agent, iteration=iteration, alpha=alpha, gamma=gamma, note=note), path)


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


def parse_float_csv(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def sigma_tag(sigma: float) -> str:
    return str(float(sigma)).replace("-", "m").replace(".", "p")


def build_control_plan(
    eval_info: Dict[str, object],
    *,
    alpha: float,
    gamma: float,
    dt_ns: float,
    runtime_norm_ns: float,
    weights: UFOCostWeights,
    note: str,
) -> ControlPlan:
    return ControlPlan(
        controls_mhz_and_phase=np.asarray(eval_info["nominal_controls"], dtype=np.float64),
        target_alpha=float(alpha),
        target_gamma=float(gamma),
        dt_ns=float(dt_ns),
        runtime_norm_ns=float(runtime_norm_ns),
        cost_weights=weights,
        note=note,
    )


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
    parser = argparse.ArgumentParser(description="Train a single-target TRPO controller for the UFO experiment.")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment folder to resume from")
    parser.add_argument("--alpha", type=str, required=True, help="Target alpha, e.g. '2.2' or 'pi/2'")
    parser.add_argument("--gamma", type=str, default="pi/2", help="Target gamma, e.g. 'pi/2'")
    parser.add_argument("--noise-optimized", action="store_true", help="Train in the stochastic 1 MHz environment.")
    parser.add_argument("--train-noise-std", type=float, default=1.0)
    parser.add_argument("--episodes-per-batch", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N iterations.")
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--max-time-ns", type=float, default=600.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--termination-cost", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cost-chi", type=float, default=10.0, help="Weight for fidelity cost")
    parser.add_argument("--cost-beta", type=float, default=10.0, help="Weight for leakage cost")
    parser.add_argument("--cost-mu", type=float, default=0.2, help="Weight for boundary cost")
    parser.add_argument("--cost-kappa", type=float, default=0.1, help="Weight for time cost")
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
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional checkpoint to initialize or resume from.")
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
    parser.add_argument("--out", type=str, required=True)

    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            resume_path = sys.argv[idx + 1]
            args_file = os.path.join(resume_path, "args.json")
            if os.path.exists(args_file):
                with open(args_file, "r") as f:
                    saved_args = json.load(f)
                for action in parser._actions:
                    if action.dest in saved_args:
                        action.required = False
                parser.set_defaults(**saved_args)
            else:
                print(f"Warning: --resume provided but args.json not found in {resume_path}")

    args = parser.parse_args()

    alpha = parse_angle_expr(args.alpha)
    gamma = parse_angle_expr(args.gamma)
    robustness_sigmas = parse_float_csv(args.robustness_sigmas)
    set_seeds(args.seed)
    ensure_dir(args.out)
    
    with open(os.path.join(args.out, "args.json"), "w") as f:
        save_args = {k: v for k, v in vars(args).items() if k != "resume"}
        json.dump(save_args, f, indent=2)

    logger = JsonlLogger(os.path.join(args.out, "training_log.jsonl"))
    checkpoints_dir = os.path.join(args.out, "checkpoints")
    plans_dir = os.path.join(args.out, "plans")
    robustness_dir = os.path.join(args.out, "robustness")
    ensure_dir(checkpoints_dir)
    ensure_dir(plans_dir)
    ensure_dir(robustness_dir)

    system = GmonSystem(
        GmonSystemConfig(
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            bandwidth_mhz=10.0,
        )
    )
    weights = UFOCostWeights(
        chi=args.cost_chi,
        beta=args.cost_beta,
        mu=args.cost_mu,
        kappa=args.cost_kappa,
    )
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
    agent = TRPOAgent(train_env.observation_dim, train_env.action_dim, config=trpo_cfg)

    start_iteration = 0
    if args.resume:
        checkpoints_dir_res = os.path.join(args.resume, "checkpoints")
        if os.path.isdir(checkpoints_dir_res):
            ckpt_files = glob.glob(os.path.join(checkpoints_dir_res, "iter_*.pt"))
            if ckpt_files:
                latest_ckpt_path = max(ckpt_files, key=lambda p: int(os.path.basename(p).replace("iter_", "").replace(".pt", "")))
                last_iter = int(os.path.basename(latest_ckpt_path).replace("iter_", "").replace(".pt", ""))
                print(f"Resuming from checkpoint: {latest_ckpt_path} at iteration {last_iter}")
                args.init_checkpoint = latest_ckpt_path
                start_iteration = last_iter

    init_meta: Dict[str, object] | None = None
    if args.init_checkpoint:
        init_meta = load_checkpoint_into_agent(agent, args.init_checkpoint)

    best_eval_cost = float("inf")
    best_eval: Dict[str, object] | None = None
    if args.resume:
        summary_path = os.path.join(args.resume, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                try:
                    old_summary = json.load(f)
                    best_eval_cost = float(old_summary.get("best_eval_cost", float("inf")))
                except:
                    pass

    best_plan_path = os.path.join(args.out, "best_control_plan.npz")
    best_ckpt_path = os.path.join(args.out, "best_agent.pt")
    last_eval_info: Dict[str, object] | None = None
    last_robustness_path: str | None = None

    collector = ParallelBatchCollector(
        system_config=system.config,
        env_config=train_env.config,
        trpo_config=trpo_cfg,
        num_workers=args.num_workers,
        episodes_per_task=(None if args.episodes_per_task <= 0 else args.episodes_per_task),
        seed=args.seed,
    )

    with collector:
        for iteration in range(start_iteration + 1, args.iterations + 1):
            if args.num_workers > 1:
                transitions, finals = collector.collect(agent, args.episodes_per_batch)
            else:
                transitions, finals = collect_batch(train_env, agent, args.episodes_per_batch)

            update_info = agent.update(transitions)
            train_stats = summarize_final_infos(finals)

            iter_ckpt_path = os.path.join(checkpoints_dir, f"iter_{iteration:06d}.pt")
            if args.save_every > 0 and (iteration % args.save_every == 0 or iteration == args.iterations):
                note = f"single-target iter {iteration}"
                save_checkpoint(agent, iter_ckpt_path, iteration=iteration, alpha=alpha, gamma=gamma, note=note)
            else:
                iter_ckpt_path = ""

            record: Dict[str, Any] = {
                "iteration": int(iteration),
                "alpha": float(alpha),
                "gamma": float(gamma),
                "num_workers": int(args.num_workers),
                **train_stats,
                **{f"update_{k}": float(v) for k, v in update_info.items()},
            }
            if iter_ckpt_path:
                record["checkpoint_path"] = iter_ckpt_path

            if iteration % args.eval_every == 0:
                eval_info = evaluate(agent, eval_env)
                last_eval_info = eval_info

                iter_plan_path = os.path.join(plans_dir, f"iter_{iteration:06d}_control_plan.npz")
                save_eval_plan(
                    eval_info,
                    alpha=alpha,
                    gamma=gamma,
                    dt_ns=args.dt_ns,
                    runtime_norm_ns=args.runtime_norm_ns,
                    weights=weights,
                    path=iter_plan_path,
                    note=f"TRPO single target iteration {iteration}",
                )

                record.update(
                    {
                        "eval_cost": float(eval_info["cost"]),
                        "eval_fidelity": float(eval_info["fidelity"]),
                        "eval_leakage": float(eval_info["leakage"]),
                        "eval_time_ns": float(eval_info["time_ns"]),
                        "eval_min_cost": float(eval_info["min_cost"]),
                        "eval_min_cost_time_ns": float(eval_info["min_cost_time_ns"]),
                        "eval_plan_path": iter_plan_path,
                    }
                )

                if robustness_sigmas:
                    iter_plan = build_control_plan(
                        eval_info,
                        alpha=alpha,
                        gamma=gamma,
                        dt_ns=args.dt_ns,
                        runtime_norm_ns=args.runtime_norm_ns,
                        weights=weights,
                        note=f"TRPO single target iteration {iteration}",
                    )
                    robustness = compute_robustness_suite(
                        system=system,
                        plan=iter_plan,
                        sigmas_mhz=robustness_sigmas,
                        num_samples=args.robustness_samples,
                        base_seed=args.seed + 10000 * iteration,
                    )

                    iter_robustness_path = os.path.join(robustness_dir, f"iter_{iteration:06d}_robustness.json")
                    robustness_payload: Dict[str, Any] = {
                        "iteration": int(iteration),
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
                    last_robustness_path = iter_robustness_path

                    record["robustness"] = robustness
                    record["robustness_path"] = iter_robustness_path
                    record.update(flatten_robustness_for_log(robustness))

                if float(eval_info["cost"]) < best_eval_cost:
                    best_eval_cost = float(eval_info["cost"])
                    best_eval = eval_info
                    save_eval_plan(
                        eval_info,
                        alpha=alpha,
                        gamma=gamma,
                        dt_ns=args.dt_ns,
                        runtime_norm_ns=args.runtime_norm_ns,
                        weights=weights,
                        path=best_plan_path,
                        note=f"TRPO single target best at iteration {iteration}",
                    )
                    save_checkpoint(
                        agent,
                        best_ckpt_path,
                        iteration=iteration,
                        alpha=alpha,
                        gamma=gamma,
                        note="best single-target agent",
                    )

            logger.write(record)
            print(json.dumps(record), flush=True)

    final_ckpt_path = os.path.join(args.out, "final_agent.pt")
    save_checkpoint(
        agent,
        final_ckpt_path,
        iteration=args.iterations,
        alpha=alpha,
        gamma=gamma,
        note="final single-target agent",
    )

    final_plan_path = None
    if last_eval_info is not None:
        final_plan_path = os.path.join(args.out, "final_control_plan.npz")
        save_eval_plan(
            last_eval_info,
            alpha=alpha,
            gamma=gamma,
            dt_ns=args.dt_ns,
            runtime_norm_ns=args.runtime_norm_ns,
            weights=weights,
            path=final_plan_path,
            note="final single-target deterministic rollout",
        )

    summary: Dict[str, Any] = {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "noise_optimized": bool(args.noise_optimized),
        "best_eval_cost": float(best_eval_cost),
        "best_plan_path": best_plan_path if os.path.exists(best_plan_path) else None,
        "best_ckpt_path": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
        "final_ckpt_path": final_ckpt_path if os.path.exists(final_ckpt_path) else None,
        "final_plan_path": final_plan_path if final_plan_path and os.path.exists(final_plan_path) else None,
        "iterations": int(args.iterations),
        "episodes_per_batch": int(args.episodes_per_batch),
        "init_checkpoint": args.init_checkpoint,
        "robustness_sigmas": robustness_sigmas,
        "robustness_samples": int(args.robustness_samples),
        "last_robustness_path": last_robustness_path if last_robustness_path and os.path.exists(last_robustness_path) else None,
    }

    if init_meta is not None:
        raw_iter = init_meta.get("iteration", -1)
        summary["init_checkpoint_iteration"] = int(raw_iter) if isinstance(raw_iter, (int, float)) else None

    if best_eval is not None:
        summary.update(
            {
                "best_eval_fidelity": float(best_eval["fidelity"]),
                "best_eval_leakage": float(best_eval["leakage"]),
                "best_eval_time_ns": float(best_eval["time_ns"]),
                "best_eval_min_cost": float(best_eval["min_cost"]),
                "best_eval_min_cost_time_ns": float(best_eval["min_cost_time_ns"]),
            }
        )

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()