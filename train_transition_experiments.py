from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Keep Torch CPU-only and single-threaded in this environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from uqc.env import EnvConfig, QuantumControlEnv
from uqc.eval import ControlPlan
from uqc.physics import GmonSystem, GmonSystemConfig, UFOCostWeights
from uqc.parallel import ParallelBatchCollector
from uqc.trpo import TRPOAgent, TRPOConfig
from uqc.utils import JsonlLogger, ensure_dir, parse_angle_expr, set_seeds


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


@dataclass(frozen=True)
class SourceSpec:
    label: str
    checkpoint: str | None


@dataclass(frozen=True)
class TargetSpec:
    alpha: float
    gamma: float

    @property
    def name(self) -> str:
        return f"alpha_{self.alpha:.6f}__gamma_{self.gamma:.6f}"



def parse_target_specs(expr: str, default_gamma: float) -> List[TargetSpec]:
    specs: List[TargetSpec] = []
    for chunk in expr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            alpha_expr, gamma_expr = chunk.split(":", 1)
            alpha = parse_angle_expr(alpha_expr)
            gamma = parse_angle_expr(gamma_expr)
        else:
            alpha = parse_angle_expr(chunk)
            gamma = default_gamma
        specs.append(TargetSpec(alpha=alpha, gamma=gamma))
    if not specs:
        raise ValueError("No target specifications were parsed.")
    return specs



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



def summarize_final_infos(finals: List[Dict[str, float]]) -> Dict[str, float]:
    if not finals:
        return {"avg_cost": float("nan"), "avg_fidelity": float("nan"), "avg_leakage": float("nan"), "avg_time_ns": float("nan")}
    keys = ["cost", "fidelity", "leakage", "time_ns", "boundary_cost", "time_cost"]
    out = {}
    for key in keys:
        vals = [float(info.get(key, np.nan)) for info in finals]
        out[f"avg_{key}"] = float(np.nanmean(vals))
    return out



def evaluate(agent: TRPOAgent, env: QuantumControlEnv) -> Dict[str, object]:
    rollout = env.rollout(agent.get_action, deterministic=True)
    final = dict(rollout["final_info"])
    final["nominal_controls"] = rollout["nominal_controls"]
    return final



def save_checkpoint(agent: TRPOAgent, path: str, *, source_label: str, alpha: float, gamma: float, iteration: int, note: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    payload = {
        "agent_state": agent.state_dict(),
        "source_label": str(source_label),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "iteration": int(iteration),
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



def build_source_specs(checkpoints: Sequence[str] | None, labels: Sequence[str] | None, include_scratch: bool) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    checkpoints = list(checkpoints or [])
    labels = list(labels or [])
    if checkpoints and labels and len(checkpoints) != len(labels):
        raise ValueError("--source-labels must have the same length as --source-checkpoints")
    for idx, path in enumerate(checkpoints):
        label = labels[idx] if idx < len(labels) else os.path.splitext(os.path.basename(path))[0]
        specs.append(SourceSpec(label=label, checkpoint=path))
    if include_scratch or not specs:
        specs.insert(0, SourceSpec(label="scratch", checkpoint=None))
    return specs



def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint transfer / transition experiments across target gates.")
    parser.add_argument("--source-checkpoints", type=str, nargs="*", default=None)
    parser.add_argument("--source-labels", type=str, nargs="*", default=None)
    parser.add_argument("--include-scratch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--targets", type=str, required=True, help="Comma-separated alpha[:gamma] specs, e.g. '0.5:pi/2,1.0:pi/2,2.2:pi/2'")
    parser.add_argument("--default-gamma", type=str, default="pi/2")
    parser.add_argument("--noise-optimized", action="store_true", help="Train in the stochastic 1 MHz environment.")
    parser.add_argument("--train-noise-std", type=float, default=1.0)
    parser.add_argument("--episodes-per-batch", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--max-time-ns", type=float, default=600.0)
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--termination-cost", type=float, default=0.15)
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
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel rollout workers for batch collection.")
    parser.add_argument("--episodes-per-task", type=int, default=0, help="Episodes per worker task; 0 chooses automatically.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    set_seeds(args.seed)
    ensure_dir(args.out)

    default_gamma = parse_angle_expr(args.default_gamma)
    targets = parse_target_specs(args.targets, default_gamma=default_gamma)
    sources = build_source_specs(args.source_checkpoints, args.source_labels, include_scratch=bool(args.include_scratch))

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

    for source_idx, source in enumerate(sources):
        source_dir = os.path.join(args.out, source.label)
        ensure_dir(source_dir)
        for target_idx, target in enumerate(targets):
            target_dir = os.path.join(source_dir, target.name)
            ckpt_dir = os.path.join(target_dir, "checkpoints")
            plan_dir = os.path.join(target_dir, "plans")
            ensure_dir(target_dir)
            ensure_dir(ckpt_dir)
            ensure_dir(plan_dir)
            logger = JsonlLogger(os.path.join(target_dir, "training_log.jsonl"))

            train_env = QuantumControlEnv(
                system,
                EnvConfig(
                    target_alpha=target.alpha,
                    target_gamma=target.gamma,
                    max_time_ns=args.max_time_ns,
                    dt_ns=args.dt_ns,
                    noise_optimized=bool(args.noise_optimized),
                    train_noise_std_mhz=args.train_noise_std,
                    reward_mode=args.reward_mode,
                    termination_cost=args.termination_cost,
                    runtime_norm_ns=args.runtime_norm_ns,
                    cost_weights=weights,
                    seed=args.seed + 1000 * source_idx + 100 * target_idx,
                ),
            )
            eval_env = QuantumControlEnv(
                system,
                EnvConfig(
                    target_alpha=target.alpha,
                    target_gamma=target.gamma,
                    max_time_ns=args.max_time_ns,
                    dt_ns=args.dt_ns,
                    noise_optimized=False,
                    reward_mode=args.reward_mode,
                    termination_cost=args.termination_cost,
                    runtime_norm_ns=args.runtime_norm_ns,
                    cost_weights=weights,
                    seed=args.seed + 1000 * source_idx + 100 * target_idx,
                ),
            )
            agent = TRPOAgent(train_env.observation_dim, train_env.action_dim, config=trpo_cfg)
            init_meta = None
            if source.checkpoint is not None:
                init_meta = load_checkpoint_into_agent(agent, source.checkpoint)

            zero_shot = evaluate(agent, eval_env)
            zero_shot_plan_path = os.path.join(target_dir, "zero_shot_control_plan.npz")
            save_eval_plan(
                zero_shot,
                alpha=target.alpha,
                gamma=target.gamma,
                dt_ns=args.dt_ns,
                runtime_norm_ns=args.runtime_norm_ns,
                weights=weights,
                path=zero_shot_plan_path,
                note=f"zero-shot from {source.label}",
            )
            save_checkpoint(
                agent,
                os.path.join(target_dir, "initial_agent.pt"),
                source_label=source.label,
                alpha=target.alpha,
                gamma=target.gamma,
                iteration=0,
                note="initial checkpoint before target adaptation",
            )

            best_eval = zero_shot
            best_eval_cost = float(zero_shot["cost"])
            best_ckpt_path = os.path.join(target_dir, "best_agent.pt")
            best_plan_path = os.path.join(target_dir, "best_control_plan.npz")
            save_checkpoint(
                agent,
                best_ckpt_path,
                source_label=source.label,
                alpha=target.alpha,
                gamma=target.gamma,
                iteration=0,
                note="best checkpoint initialized from zero-shot",
            )
            save_eval_plan(
                zero_shot,
                alpha=target.alpha,
                gamma=target.gamma,
                dt_ns=args.dt_ns,
                runtime_norm_ns=args.runtime_norm_ns,
                weights=weights,
                path=best_plan_path,
                note=f"best plan initialized from zero-shot {source.label}",
            )

            last_eval = zero_shot
            collector = ParallelBatchCollector(
                system_config=system.config,
                env_config=train_env.config,
                trpo_config=trpo_cfg,
                num_workers=args.num_workers,
                episodes_per_task=(None if args.episodes_per_task <= 0 else args.episodes_per_task),
                seed=args.seed + 1000 * source_idx + 100 * target_idx,
            )
            with collector:
                for iteration in range(1, args.iterations + 1):
                    transitions, finals = (collector.collect(agent, args.episodes_per_batch) if args.num_workers > 1 else collect_batch(train_env, agent, args.episodes_per_batch))
                    update_info = agent.update(transitions)
                    train_stats = summarize_final_infos(finals)
                    record = {
                        "source_label": source.label,
                        "source_checkpoint": source.checkpoint,
                        "target_alpha": target.alpha,
                        "target_gamma": target.gamma,
                        "iteration": iteration,
                        "num_workers": int(args.num_workers),
                        **train_stats,
                        **{f"update_{k}": float(v) for k, v in update_info.items()},
                    }

                    iter_ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:06d}.pt")
                    if args.save_every > 0 and (iteration % args.save_every == 0 or iteration == args.iterations):
                        save_checkpoint(
                            agent,
                            iter_ckpt_path,
                            source_label=source.label,
                            alpha=target.alpha,
                            gamma=target.gamma,
                            iteration=iteration,
                            note=f"transition source={source.label} target={target.name} iter={iteration}",
                        )
                        record["checkpoint_path"] = iter_ckpt_path
    
                    if iteration % args.eval_every == 0:
                        eval_info = evaluate(agent, eval_env)
                        last_eval = eval_info
                        iter_plan_path = os.path.join(plan_dir, f"iter_{iteration:06d}_control_plan.npz")
                        save_eval_plan(
                            eval_info,
                            alpha=target.alpha,
                            gamma=target.gamma,
                            dt_ns=args.dt_ns,
                            runtime_norm_ns=args.runtime_norm_ns,
                            weights=weights,
                            path=iter_plan_path,
                            note=f"transition source={source.label} target={target.name} iter={iteration}",
                        )
                        record.update(
                            {
                                "eval_cost": float(eval_info["cost"]),
                                "eval_fidelity": float(eval_info["fidelity"]),
                                "eval_leakage": float(eval_info["leakage"]),
                                "eval_time_ns": float(eval_info["time_ns"]),
                                "eval_plan_path": iter_plan_path,
                            }
                        )
                        if float(eval_info["cost"]) < best_eval_cost:
                            best_eval_cost = float(eval_info["cost"])
                            best_eval = eval_info
                            save_checkpoint(
                                agent,
                                best_ckpt_path,
                                source_label=source.label,
                                alpha=target.alpha,
                                gamma=target.gamma,
                                iteration=iteration,
                                note="best transition checkpoint",
                            )
                            save_eval_plan(
                                eval_info,
                                alpha=target.alpha,
                                gamma=target.gamma,
                                dt_ns=args.dt_ns,
                                runtime_norm_ns=args.runtime_norm_ns,
                                weights=weights,
                                path=best_plan_path,
                                note="best transition plan",
                            )
    
                    logger.write(record)
                    print(json.dumps(record), flush=True)
    
            final_ckpt_path = os.path.join(target_dir, "final_agent.pt")
            save_checkpoint(
                agent,
                final_ckpt_path,
                source_label=source.label,
                alpha=target.alpha,
                gamma=target.gamma,
                iteration=args.iterations,
                note="final transition checkpoint",
            )
            final_plan_path = os.path.join(target_dir, "final_control_plan.npz")
            save_eval_plan(
                last_eval,
                alpha=target.alpha,
                gamma=target.gamma,
                dt_ns=args.dt_ns,
                runtime_norm_ns=args.runtime_norm_ns,
                weights=weights,
                path=final_plan_path,
                note="final transition plan",
            )

            summary = {
                "source_label": source.label,
                "source_checkpoint": source.checkpoint,
                "target_alpha": float(target.alpha),
                "target_gamma": float(target.gamma),
                "zero_shot_cost": float(zero_shot["cost"]),
                "zero_shot_fidelity": float(zero_shot["fidelity"]),
                "zero_shot_leakage": float(zero_shot["leakage"]),
                "zero_shot_time_ns": float(zero_shot["time_ns"]),
                "best_eval_cost": float(best_eval["cost"]),
                "best_eval_fidelity": float(best_eval["fidelity"]),
                "best_eval_leakage": float(best_eval["leakage"]),
                "best_eval_time_ns": float(best_eval["time_ns"]),
                "final_eval_cost": float(last_eval["cost"]),
                "final_eval_fidelity": float(last_eval["fidelity"]),
                "final_eval_leakage": float(last_eval["leakage"]),
                "final_eval_time_ns": float(last_eval["time_ns"]),
                "zero_shot_plan_path": zero_shot_plan_path,
                "best_plan_path": best_plan_path,
                "best_ckpt_path": best_ckpt_path,
                "final_ckpt_path": final_ckpt_path,
                "final_plan_path": final_plan_path,
                "iterations": int(args.iterations),
            }
            if init_meta is not None:
                summary["init_meta_iteration"] = int(init_meta.get("iteration", -1)) if isinstance(init_meta.get("iteration", -1), (int, float)) else None
            with open(os.path.join(target_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            summary_rows.append(summary)

    csv_path = os.path.join(args.out, "transition_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            extrasaction="ignore",
            fieldnames=[
                "source_label",
                "source_checkpoint",
                "target_alpha",
                "target_gamma",
                "zero_shot_cost",
                "zero_shot_fidelity",
                "zero_shot_leakage",
                "zero_shot_time_ns",
                "best_eval_cost",
                "best_eval_fidelity",
                "best_eval_leakage",
                "best_eval_time_ns",
                "final_eval_cost",
                "final_eval_fidelity",
                "final_eval_leakage",
                "final_eval_time_ns",
                "zero_shot_plan_path",
                "best_plan_path",
                "best_ckpt_path",
                "final_ckpt_path",
                "final_plan_path",
                "iterations",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved transition summary to {csv_path}")


if __name__ == "__main__":
    main()
