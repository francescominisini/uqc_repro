from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd



def gamma_label(g: float) -> str:
    if abs(g - math.pi / 2) < 1e-6:
        return r"$\gamma=\pi/2$"
    if abs(g - math.pi / 6) < 1e-6:
        return r"$\gamma=\pi/6$"
    if abs(g - math.pi / 3) < 1e-6:
        return r"$\gamma=\pi/3$"
    return rf"$\gamma={g:.3f}$"



def plot_runtime(runtime_csv: str, out_path: str) -> None:
    df = pd.read_csv(runtime_csv)
    plt.figure(figsize=(8, 5))
    for gamma, sub in df.groupby("gamma"):
        sub = sub.sort_values("alpha")
        plt.plot(sub["alpha"], sub["runtime_ns"] / 60.0, marker="o", label=gamma_label(float(gamma)))
    plt.axhline(215.0 / 60.0, linestyle="--", label="optimal gate synthesis")
    plt.xlabel(r"target $\alpha$")
    plt.ylabel("time / 60ns")
    plt.title("Gate run time of N(alpha, alpha, gamma)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_robustness(robustness_csv: str, out_prefix: str) -> None:
    df = pd.read_csv(robustness_csv)

    plt.figure(figsize=(8, 5))
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("sigma_mhz")
        plt.plot(sub["sigma_mhz"] / 10.0, sub["average_fidelity"], marker="o", label=label)
    plt.xlabel("noise / 10MHz")
    plt.ylabel("average fidelity")
    plt.title("Average fidelity vs Gaussian control noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_average_fidelity.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("sigma_mhz")
        plt.plot(sub["sigma_mhz"] / 10.0, sub["fidelity_variance"], marker="o", label=label)
    plt.yscale("log")
    plt.xlabel("noise / 10MHz")
    plt.ylabel("fidelity variance")
    plt.title("Fidelity variance vs Gaussian control noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_variance.png", dpi=300)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style runtime and robustness figures.")
    parser.add_argument("--runtime-csv", type=str, default=None)
    parser.add_argument("--robustness-csv", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.runtime_csv:
        plot_runtime(args.runtime_csv, os.path.join(args.out, "figure3_runtime.png"))
    if args.robustness_csv:
        plot_robustness(args.robustness_csv, os.path.join(args.out, "figure4"))


if __name__ == "__main__":
    main()
