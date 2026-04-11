import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Define colors
colors = {
    "Adam": "#4361EE",       # Blue
    "Nominal TRPO": "#EF233C" # Red
}

def generate_comparison_plots():
    print("Generating Figure: Adam vs Nominal TRPO Comparison...")
    
    # Paths to the summaries
    adam_summary_path = "final_results/adam_runtime_sweep/gamma_1.570796/alpha_2.200000/summary.json"
    nominal_summary_path = "final_results/nominal/summary.json"
    
    # Load data
    with open(adam_summary_path, "r") as f:
        adam_data = json.load(f)
        
    with open(nominal_summary_path, "r") as f:
        nominal_data = json.load(f)
        
    # Extract metrics
    metrics = {
        "Method": ["Adam", "Nominal TRPO"],
        "Total Cost": [adam_data["best_cost"], nominal_data["best_eval_cost"]],
        "Fidelity": [adam_data["best_fidelity"], nominal_data["best_eval_fidelity"]],
        "Leakage": [adam_data["best_leakage"], nominal_data["best_eval_leakage"]],
        "Runtime (ns)": [adam_data["runtime_ns"], nominal_data["best_eval_time_ns"]]
    }
    
    df = pd.DataFrame(metrics)
    
    # Ensure export directory exists
    os.makedirs("final_results/figures_data", exist_ok=True)
    out_csv = "final_results/figures_data/adam_vs_nominal_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Exported data to {out_csv}")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    methods = df["Method"]
    x = np.arange(len(methods))
    width = 0.6
    
    def plot_bars(ax, target_metric, ylabel, title, log_scale=False):
        values = df[target_metric]
        bars = ax.bar(x, values, width, color=[colors[m] for m in methods], edgecolor="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, axis="y")
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            label_text = f"{yval:.4f}" if yval < 10 else f"{yval:.1f}"
            if log_scale:
                # Add text slightly above the bar in log scale
                ax.text(bar.get_x() + bar.get_width()/2, yval * 1.15, label_text, ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, yval + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, label_text, ha='center', va='bottom', fontsize=10)

    # (a) Nominal Cost
    plot_bars(axes[0], "Total Cost", "Total Nominal Cost", "(a) Nominal Cost")
    
    # (b) Nominal Fidelity
    plot_bars(axes[1], "Fidelity", "Nominal Fidelity", "(b) Nominal Fidelity")
    axes[1].set_ylim([0.9, 1.01]) # Adjust for fidelity range
    
    # (c) Total Leakage (log scale)
    plot_bars(axes[2], "Leakage", "Total Leakage", "(c) Total Leakage (log-scale)", log_scale=True)
    axes[2].set_ylim([bottom:=df["Leakage"].min()*0.1, top:=df["Leakage"].max()*10])
    
    # (d) Runtime
    plot_bars(axes[3], "Runtime (ns)", "Runtime (ns)", "(d) Gate Time")
    axes[3].set_ylim([0, max(df["Runtime (ns)"])*1.2])

    plt.tight_layout()
    
    # The user asked to generate the figures inside nominal_results folder.
    # We create it if it doesn't exist to be safe.
    target_dir = "final_results"
    os.makedirs(target_dir, exist_ok=True)
    out_img = os.path.join(target_dir, "fig_single_target_adam_vs_nominal_trpo.png")
    
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_img}")

if __name__ == "__main__":
    generate_comparison_plots()
