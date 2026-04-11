import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    "highlight": "#FFD700"
}

def generate_nominal_trpo_training_curves():
    print("Generating Figure: Nominal TRPO Training Curves...")
    log_path = "final_results/nominal/training_log.jsonl"
    
    # Read the JSONL file
    df = pd.read_json(log_path, lines=True)
    
    # Subset required columns
    cols_to_keep = [
        "iteration", "eval_cost", "eval_fidelity", "eval_leakage", 
        "eval_time_ns", "eval_min_cost_time_ns"
    ]
    df_subset = df[cols_to_keep].copy()
    
    # Drop rows where eval metrics might not exist (e.g. if eval wasn't run every iteration)
    # Looking at the data, it seems eval_cost etc. are present.
    # We forward fill eval_min_cost_time_ns because it might be NaN early on
    df_subset['eval_min_cost_time_ns'] = df_subset['eval_min_cost_time_ns'].ffill()
    # If still NaNs at the beginning, use eval_time_ns
    df_subset['eval_min_cost_time_ns'] = df_subset['eval_min_cost_time_ns'].fillna(df_subset['eval_time_ns'])
    
    # Export data
    out_csv = "final_results/figures_data/nominal_trpo_training_curves.csv"
    df_subset.to_csv(out_csv, index=False)
    print(f"Exported data to {out_csv}")
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    iterations = df_subset["iteration"]
    
    # (a) Evaluation cost versus iteration
    ax0 = axes[0]
    ax0.plot(iterations, df_subset["eval_cost"], color=colors["cost"], label="Evaluation Cost")
    
    # Highlight the best checkpoint (minimum eval_cost)
    min_cost_idx = df_subset["eval_cost"].idxmin()
    best_iter = df_subset.loc[min_cost_idx, "iteration"]
    best_cost = df_subset.loc[min_cost_idx, "eval_cost"]
    
    ax0.scatter(best_iter, best_cost, s=150, facecolors="none", edgecolors=colors["highlight"], 
                linewidths=2, zorder=5, label=f"Best Checkpoint (Iter {int(best_iter)})")
    
    ax0.set_ylabel("Evaluation Cost")
    ax0.set_title("(a) Evaluation Cost vs Iteration")
    ax0.legend()
    ax0.grid(True)
    
    # (b) Evaluation fidelity and leakage versus iteration
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    
    ln1 = ax1.plot(iterations, df_subset["eval_fidelity"], color=colors["fidelity"], label="Evaluation Fidelity")
    ln2 = ax1_twin.plot(iterations, df_subset["eval_leakage"], color=colors["leakage"], label="Evaluation Leakage", alpha=0.8)
    
    ax1.set_ylabel("Evaluation Fidelity", color=colors["fidelity"])
    ax1_twin.set_ylabel("Evaluation Leakage", color=colors["leakage"])
    ax1.set_title("(b) Fidelity and Leakage vs Iteration")
    
    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.grid(True)
    
    # (c) Evaluation gate time and minimum-cost time vs iteration
    ax2 = axes[2]
    
    ax2.plot(iterations, df_subset["eval_time_ns"], color=colors["time"], label="Target Horizon $T$")
    ax2.plot(iterations, df_subset["eval_min_cost_time_ns"], color=colors["boundary"], linestyle="--", label=r"Min-Cost Time $t_{\min}$")
    
    ax2.set_xlabel("Policy Iteration")
    ax2.set_ylabel("Time (ns)")
    ax2.set_title("(c) Target Horizon and Min-Cost Time vs Iteration")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    out_img = "fig_nominal_trpo_training_curves.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved plot to {out_img}")

if __name__ == "__main__":
    generate_nominal_trpo_training_curves()
