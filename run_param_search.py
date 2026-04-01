from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(text: str, cast=str) -> List[Any]:
    parts = [x.strip() for x in text.split(",") if x.strip()]
    return [cast(x) for x in parts]


def slugify_value(x: Any) -> str:
    s = str(x)
    return (
        s.replace("/", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace(" ", "")
        .replace(".", "p")
        .replace("-", "m")
    )


def config_to_name(config: Dict[str, Any], keys: Iterable[str]) -> str:
    chunks = []
    for k in keys:
        v = config[k]
        chunks.append(f"{k}-{slugify_value(v)}")
    return "__".join(chunks)


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: List[Dict[str, Any]],
    group_keys: List[str],
    metric_keys: List[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        agg: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        agg["num_runs"] = len(group)

        for mk in metric_keys:
            vals = []
            for g in group:
                v = g.get(mk)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                agg[f"{mk}_mean"] = sum(vals) / len(vals)
                agg[f"{mk}_min"] = min(vals)
                agg[f"{mk}_max"] = max(vals)
                agg[f"{mk}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        out.append(agg)

    return out


def sort_rows(rows: List[Dict[str, Any]], primary_metric: str) -> List[Dict[str, Any]]:
    def key_fn(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        primary = float(r.get(f"{primary_metric}_mean", float("inf")))
        fidelity = -float(
            r.get("best_eval_fidelity_mean", r.get("nominal_fidelity_mean", float("-inf")))
        )
        leakage = float(
            r.get("best_eval_leakage_mean", r.get("nominal_leakage_mean", float("inf")))
        )
        time_ns = float(
            r.get("best_eval_time_ns_mean", r.get("nominal_time_ns_mean", float("inf")))
        )
        return (primary, fidelity, leakage, time_ns)

    return sorted(rows, key=key_fn)


def build_trpo_jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    configs: List[Dict[str, Any]] = []
    keys = ["alpha", "gamma", "max_kl", "termination_cost", "init_log_std", "dt_ns", "max_time_ns", "cost_chi", "cost_beta", "cost_mu", "cost_kappa"]

    alphas = parse_csv_list(args.alphas, str)
    gammas = parse_csv_list(args.gammas, str)
    max_kls = parse_csv_list(args.max_kls, float)
    termination_costs = parse_csv_list(args.termination_costs, float)
    init_log_stds = parse_csv_list(args.init_log_stds, float)
    dt_vals = parse_csv_list(args.dt_values, float)
    max_time_vals = parse_csv_list(args.max_time_values, float)
    chis = parse_csv_list(args.cost_chis, float)
    betas = parse_csv_list(args.cost_betas, float)
    mus = parse_csv_list(args.cost_mus, float)
    kappas = parse_csv_list(args.cost_kappas, float)

    for alpha, gamma, max_kl, term_cost, init_log_std, dt_ns, max_time_ns, chi, beta, mu, kappa in itertools.product(
        alphas, gammas, max_kls, termination_costs, init_log_stds, dt_vals, max_time_vals, chis, betas, mus, kappas
    ):
        configs.append(
            {
                "alpha": alpha,
                "gamma": gamma,
                "max_kl": max_kl,
                "termination_cost": term_cost,
                "init_log_std": init_log_std,
                "dt_ns": dt_ns,
                "max_time_ns": max_time_ns,
                "cost_chi": chi,
                "cost_beta": beta,
                "cost_mu": mu,
                "cost_kappa": kappa,
            }
        )
    return configs, keys


def build_adam_jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    configs: List[Dict[str, Any]] = []
    keys = ["alpha", "gamma", "lr", "adam_iters", "horizons_ns", "dt_ns", "cost_chi", "cost_beta", "cost_mu", "cost_kappa"]

    alphas = parse_csv_list(args.alphas, str)
    gammas = parse_csv_list(args.gammas, str)
    lrs = parse_csv_list(args.lrs, float)
    adam_iters_list = parse_csv_list(args.adam_iters_list, int)
    horizons_list = parse_csv_list(args.horizons_list, str)
    dt_vals = parse_csv_list(args.dt_values, float)
    chis = parse_csv_list(args.cost_chis, float)
    betas = parse_csv_list(args.cost_betas, float)
    mus = parse_csv_list(args.cost_mus, float)
    kappas = parse_csv_list(args.cost_kappas, float)

    for alpha, gamma, lr, adam_iters, horizons_ns, dt_ns, chi, beta, mu, kappa in itertools.product(
        alphas, gammas, lrs, adam_iters_list, horizons_list, dt_vals, chis, betas, mus, kappas
    ):
        configs.append(
            {
                "alpha": alpha,
                "gamma": gamma,
                "lr": lr,
                "adam_iters": adam_iters,
                "horizons_ns": horizons_ns,
                "dt_ns": dt_ns,
                "cost_chi": chi,
                "cost_beta": beta,
                "cost_mu": mu,
                "cost_kappa": kappa,
            }
        )
    return configs, keys


def build_trpo_command(
    python_exec: str,
    root: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_exec,
        str(root / "train_trpo_single_target.py"),
        "--alpha", str(cfg["alpha"]),
        "--gamma", str(cfg["gamma"]),
        "--iterations", str(args.trpo_iterations),
        "--episodes-per-batch", str(args.trpo_episodes_per_batch),
        "--eval-every", "1",
        "--save-every", "1",
        "--dt-ns", str(cfg["dt_ns"]),
        "--max-time-ns", str(cfg["max_time_ns"]),
        "--runtime-norm-ns", str(args.runtime_norm_ns),
        "--termination-cost", str(cfg["termination_cost"]),
        "--seed", str(seed),
        "--max-kl", str(cfg["max_kl"]),
        "--init-log-std", str(cfg["init_log_std"]),
        "--num-workers", str(args.num_workers),
        "--cost-chi", str(cfg["cost_chi"]),
        "--cost-beta", str(cfg["cost_beta"]),
        "--cost-mu", str(cfg["cost_mu"]),
        "--cost-kappa", str(cfg["cost_kappa"]),
        "--out", str(out_dir),
    ]
    if args.noise_optimized:
        cmd.append("--noise-optimized")
        cmd.extend(["--train-noise-std", str(args.train_noise_std)])
    if args.episodes_per_task > 0:
        cmd.extend(["--episodes-per-task", str(args.episodes_per_task)])
    return cmd


def build_adam_command(
    python_exec: str,
    root: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_exec,
        str(root / "train_adam_baseline.py"),
        "--alpha", str(cfg["alpha"]),
        "--gamma", str(cfg["gamma"]),
        "--dt-ns", str(cfg["dt_ns"]),
        "--runtime-norm-ns", str(args.runtime_norm_ns),
        "--lr", str(cfg["lr"]),
        "--adam-iters", str(cfg["adam_iters"]),
        "--horizons-ns", str(cfg["horizons_ns"]),
        "--seed", str(seed),
        "--cost-chi", str(cfg["cost_chi"]),
        "--cost-beta", str(cfg["cost_beta"]),
        "--cost-mu", str(cfg["cost_mu"]),
        "--cost-kappa", str(cfg["cost_kappa"]),
        "--out", str(out_dir),
    ]
    if args.adam_train_noise_std > 0:
        cmd.extend(["--train-noise-std", str(args.adam_train_noise_std)])
        cmd.extend(["--train-noise-samples", str(args.adam_train_noise_samples)])
    return cmd


def extract_metrics(mode: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    if mode == "trpo":
        return {
            "primary_metric": float(summary.get("best_eval_cost", float("inf"))),
            "best_eval_cost": float(summary.get("best_eval_cost", float("inf"))),
            "best_eval_fidelity": float(summary.get("best_eval_fidelity", float("-inf"))),
            "best_eval_leakage": float(summary.get("best_eval_leakage", float("inf"))),
            "best_eval_time_ns": float(summary.get("best_eval_time_ns", float("inf"))),
        }
    if mode == "adam":
        return {
            "primary_metric": float(summary.get("nominal_cost", float("inf"))),
            "nominal_cost": float(summary.get("nominal_cost", float("inf"))),
            "nominal_fidelity": float(summary.get("nominal_fidelity", float("-inf"))),
            "nominal_leakage": float(summary.get("nominal_leakage", float("inf"))),
            "nominal_time_ns": float(summary.get("nominal_time_ns", float("inf"))),
        }
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Light parameter search for TRPO or Adam baseline.")

    parser.add_argument("--mode", choices=["trpo", "adam"], required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--alphas", type=str, default="2.2")
    parser.add_argument("--gammas", type=str, default="pi/2")
    parser.add_argument("--runtime-norm-ns", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="Rerun jobs even if summary.json exists.")

    # Shared compute knobs
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=0)

    # Cost space weights
    parser.add_argument("--cost-chis", type=str, default="10.0")
    parser.add_argument("--cost-betas", type=str, default="10.0")
    parser.add_argument("--cost-mus", type=str, default="0.2")
    parser.add_argument("--cost-kappas", type=str, default="0.1")

    # TRPO search space
    parser.add_argument("--noise-optimized", action="store_true")
    parser.add_argument("--train-noise-std", type=float, default=1.0)
    parser.add_argument("--trpo-iterations", type=int, default=30)
    parser.add_argument("--trpo-episodes-per-batch", type=int, default=2000)
    parser.add_argument("--max-kls", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--termination-costs", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--init-log-stds", type=str, default="-1.0,-0.5,0.0")
    parser.add_argument("--dt-values", type=str, default="2.0")
    parser.add_argument("--max-time-values", type=str, default="600.0")

    # Adam search space
    parser.add_argument("--lrs", type=str, default="0.01,0.03,0.1")
    parser.add_argument("--adam-iters-list", type=str, default="200,400,800")
    parser.add_argument("--horizons-list", type=str, default="60,90,120,150,180,210,240")
    parser.add_argument("--adam-train-noise-std", type=float, default=0.0)
    parser.add_argument("--adam-train-noise-samples", type=int, default=1)

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_root = Path(args.out)
    ensure_dir(out_root)

    seeds = parse_csv_list(args.seeds, int)

    if args.mode == "trpo":
        configs, config_keys = build_trpo_jobs(args)
    else:
        configs, config_keys = build_adam_jobs(args)

    raw_rows: List[Dict[str, Any]] = []

    for cfg_idx, cfg in enumerate(configs, start=1):
        cfg_name = config_to_name(cfg, config_keys)
        for seed in seeds:
            run_dir = out_root / args.mode / cfg_name / f"seed_{seed}"
            ensure_dir(run_dir)
            summary_path = run_dir / "summary.json"

            if summary_path.exists() and not args.force:
                print(f"[skip] {summary_path} already exists", flush=True)
            else:
                if args.mode == "trpo":
                    if (run_dir / "args.json").exists() and not args.force:
                        print(f"[resume] Resuming interrupted run at {run_dir}", flush=True)
                        cmd = [
                            sys.executable,
                            str(root / "train_trpo_single_target.py"),
                            "--resume", str(run_dir)
                        ]
                    else:
                        cmd = build_trpo_command(
                            python_exec=sys.executable,
                            root=root,
                            out_dir=run_dir,
                            cfg=cfg,
                            seed=seed,
                            args=args,
                        )
                else:
                    cmd = build_adam_command(
                        python_exec=sys.executable,
                        root=root,
                        out_dir=run_dir,
                        cfg=cfg,
                        seed=seed,
                        args=args,
                    )
                run_cmd(cmd)

            if not summary_path.exists():
                print(f"[warn] Missing summary.json for {run_dir}", flush=True)
                continue

            summary = read_json(summary_path)
            row: Dict[str, Any] = {
                "mode": args.mode,
                "config_id": cfg_name,
                "seed": seed,
                "run_dir": str(run_dir),
            }
            row.update(cfg)
            row.update(extract_metrics(args.mode, summary))
            raw_rows.append(row)

            print(
                f"[done] {cfg_idx}/{len(configs)} | seed={seed} | "
                f"config={cfg_name} | primary={row['primary_metric']}",
                flush=True,
            )

    raw_csv = out_root / f"{args.mode}_raw_results.csv"
    write_csv(raw_csv, raw_rows)

    if args.mode == "trpo":
        metric_keys = [
            "primary_metric",
            "best_eval_cost",
            "best_eval_fidelity",
            "best_eval_leakage",
            "best_eval_time_ns",
        ]
        primary_metric_name = "primary_metric"
    else:
        metric_keys = [
            "primary_metric",
            "nominal_cost",
            "nominal_fidelity",
            "nominal_leakage",
            "nominal_time_ns",
        ]
        primary_metric_name = "primary_metric"

    aggregated = aggregate_rows(
        rows=raw_rows,
        group_keys=["mode", "config_id"] + config_keys,
        metric_keys=metric_keys,
    )
    aggregated = sort_rows(aggregated, primary_metric_name)

    agg_csv = out_root / f"{args.mode}_aggregated_results.csv"
    write_csv(agg_csv, aggregated)

    best = aggregated[0] if aggregated else {}
    best_json = out_root / f"{args.mode}_best_config.json"
    write_json(best_json, best)

    manifest = {
        "mode": args.mode,
        "num_configs": len(configs),
        "num_runs": len(raw_rows),
        "raw_results_csv": str(raw_csv),
        "aggregated_results_csv": str(agg_csv),
        "best_config_json": str(best_json),
    }
    write_json(out_root / f"{args.mode}_search_manifest.json", manifest)

    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()