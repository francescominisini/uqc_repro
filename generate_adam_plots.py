import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set aesthetic parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Define colors
colors = {
    "cost": "#2E86AB",
    "fidelity": "#D62828",
    "leakage": "#F77F00",
    "boundary": "#8338EC",
    "time": "#3A5A40",
    "highlight": "#FFD700",
    "optimizer": "#000000"
}

def generate_figure_1():
    print("Generating Figure 1...")
    csv_path = "final_results/adam_noise/horizon_search.csv"
    df = pd.read_csv(csv_path)
    
    # Export data to figures_data
    df.to_csv("final_results/figures_data/adam_horizon_sweep.csv", index=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # (a) Nominal Total Cost vs Horizon
    ax0 = axes[0]
    ax0.plot(df["horizon_ns"], df["nominal_cost"], marker="o", color=colors["cost"], label="Nominal Total Cost")
    
    # Highlight minimum at 70ns
    min_cost_row = df[df["horizon_ns"] == 70.0]
    if not min_cost_row.empty:
        ax0.scatter(70.0, min_cost_row["nominal_cost"].values[0], 
                    s=150, facecolors="none", edgecolors=colors["highlight"], 
                    linewidths=2, zorder=5, label="Nominal Minimum (70 ns)")
    
    # Mark optimizer-selected 60ns
    opt_selected_row = df[df["horizon_ns"] == 60.0]
    if not opt_selected_row.empty:
        ax0.scatter(60.0, opt_selected_row["nominal_cost"].values[0], 
                    marker="x", s=100, color=colors["optimizer"], 
                    zorder=6, label="Optimizer-Selected (60 ns)")
    
    ax0.set_ylabel("Nominal Total Cost")
    ax0.set_title("(a) Nominal Total Cost vs Horizon")
    ax0.legend()
    ax0.grid(True)
    
    # (b) Nominal Fidelity and Total Leakage vs Horizon
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    
    ln1 = ax1.plot(df["horizon_ns"], df["nominal_fidelity"], marker="s", color=colors["fidelity"], label="Nominal Fidelity")
    ln2 = ax1_twin.plot(df["horizon_ns"], df["nominal_leakage"], marker="^", color=colors["leakage"], label="Total Leakage")
    
    ax1.set_ylabel("Nominal Fidelity", color=colors["fidelity"])
    ax1_twin.set_ylabel("Total Leakage", color=colors["leakage"])
    ax1.set_title("(b) Nominal Fidelity and Total Leakage")
    
    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.grid(True)
    
    # (c) Boundary Penalty and Time Penalty vs Horizon
    ax2 = axes[2]
    ax2.plot(df["horizon_ns"], df["nominal_boundary_cost"], marker="d", color=colors["boundary"], label="Boundary Penalty")
    ax2.plot(df["horizon_ns"], df["nominal_time_cost"], marker="v", color=colors["time"], label="Time Penalty")
    
    ax2.set_xlabel("Horizon (ns)")
    ax2.set_ylabel("Penalty Cost")
    ax2.set_title("(c) Boundary and Time Penalties")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("fig_adam_single_target_horizon_sweep.png", dpi=300)
    plt.close()

def generate_figure_2():
    print("Generating Figure 2...")
    csv_path = "final_results/adam_runtime_sweep/runtime_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Export data to figures_data
    df.to_csv("final_results/figures_data/adam_family_sweep.csv", index=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Top: Best selected horizon vs alpha as a step plot
    ax0 = axes[0]
    ax0.step(df["alpha"], df["best_horizon_ns"], where="post", color=colors["optimizer"], label="Selected Horizon")
    ax0.set_ylabel("Best Horizon (ns)")
    ax0.set_title("Selected Horizon vs $\\alpha$")
    ax0.set_yticks([40, 50, 60, 70, 80])
    ax0.grid(True)
    
    # Middle: Best nominal fidelity and best nominal cost vs alpha
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    
    ax1.plot(df["alpha"], df["best_fidelity"], marker=".", color=colors["fidelity"], label="Best Nominal Fidelity")
    ax1_twin.plot(df["alpha"], df["best_cost"], marker=".", color=colors["cost"], label="Best Nominal Cost")
    
    ax1.set_ylabel("Best Nominal Fidelity", color=colors["fidelity"])
    ax1_twin.set_ylabel("Best Nominal Cost", color=colors["cost"])
    ax1.set_title("Best Nominal Performance vs $\\alpha$")
    ax1.grid(True)
    
    # Bottom: Noisy average fidelity at sigma=1MHz and variance vs alpha
    ax2 = axes[2]
    ax2_twin = ax2.twinx()
    
    ax2.errorbar(df["alpha"], df["robustness_sigma_1p0_average_fidelity"], 
                 yerr=np.sqrt(df["robustness_sigma_1p0_fidelity_variance"]), 
                 fmt="o", color=colors["time"], label="Robustness Fidelity ($\\sigma=1$ MHz)")
    ax2_twin.plot(df["alpha"], df["robustness_sigma_1p0_fidelity_variance"], 
                  marker="x", linestyle="None", color=colors["boundary"], label="Fidelity Variance")
    
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("Average Fidelity", color=colors["time"])
    ax2_twin.set_ylabel("Fidelity Variance", color=colors["boundary"])
    ax2.set_title("Robustness and Variance vs $\\alpha$")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("fig_adam_family_sweep_gamma_pi_over_2.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    print("Done!")
