import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def setup_aesthetic():
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams['figure.dpi'] = 300

def load_sweep_data(raw_csv_path, base_project_dir):
    print(f"Loading raw results from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    robustness_list = []
    
    # Iterate through all run dirs to fetch robustness
    for idx, row in df.iterrows():
        # The run_dir in raw results might be like 'runs\cost_function_sweep\trpo\a-2p2_...'
        # We need to map it to 'final_results\...'
        run_dir_rel = str(row['run_dir'])
        if run_dir_rel.startswith('runs\\') or run_dir_rel.startswith('runs/'):
            run_dir_rel = run_dir_rel.replace('runs\\', 'final_results\\').replace('runs/', 'final_results/')
            
        full_dir = os.path.join(base_project_dir, run_dir_rel)
        log_file = os.path.join(full_dir, 'training_log.jsonl')
        
        last_robustness = None
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'robustness_sigma_1p0_average_fidelity' in data:
                                last_robustness = data['robustness_sigma_1p0_average_fidelity']
                        except json.JSONDecodeError:
                            pass
        else:
            print(f"Warning: log file not found at {log_file}")
            
        robustness_list.append(last_robustness)

    df['robustness_sigma_1p0_average_fidelity'] = robustness_list
    
    # Drop rows without robustness data if any exist
    df_clean = df.dropna(subset=['robustness_sigma_1p0_average_fidelity']).copy()
    print(f"Loaded {len(df_clean)} complete runs with robustness data out of {len(df)}.")
    return df_clean

def plot_pareto(df, out_dir):
    # Scatter: Leakage vs Robustness
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # We will color code by cost_chi to see trade-offs
    scatter = ax.scatter(df['best_eval_leakage'], df['robustness_sigma_1p0_average_fidelity'], 
                         c=df['cost_chi'], cmap='coolwarm', s=60, alpha=0.8, edgecolor='w')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('cost_chi Weight')
    
    ax.set_xscale('log')
    ax.set_xlabel('Best Eval Leakage (Log Scale)')
    ax.set_ylabel('Robustness Avg Fidelity (Sigma=1.0)')
    ax.set_title('Pareto Front: Leakage vs Robustness')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_leakage_vs_robustness.png'))
    plt.close()

def plot_parameter_sensitivity(df, param_names, metric_col, title, filename_out, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, p in enumerate(param_names):
        ax = axes[idx]
        if p in df.columns:
            # Create a boxplot or grouped scatter for discrete parameters
            unique_vals = sorted(df[p].unique())
            groups = [df[df[p] == v][metric_col].values for v in unique_vals]
            
            ax.boxplot(groups, positions=range(len(unique_vals)), widths=0.4, patch_artist=True)
            # Add scatter for individual points
            for i, v in enumerate(unique_vals):
                y = df[df[p] == v][metric_col].values
                x = [i]*len(y)
                ax.scatter(x, y, alpha=0.5, color='tab:blue')
            
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels([f"{v:.2g}" for v in unique_vals])
            ax.set_xlabel(p)
            ax.set_ylabel(title)
            
    plt.suptitle(f'Sensitivity of {title} to Cost Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename_out))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro\final_results\cost_function_sweep",
                        help="Path containing raw run results.")
    parser.add_argument("--project_dir", type=str, default=r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro",
                        help="Root path of project to resolve local directories.")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.base_dir), "cost_function_sweep_results")
    os.makedirs(out_dir, exist_ok=True)
    setup_aesthetic()

    raw_csv = os.path.join(args.base_dir, "trpo_raw_results.csv")
    df = load_sweep_data(raw_csv, args.project_dir)

    if df.empty:
        print("No valid sweep data found.")
        return

    # Export merged features
    export_path = os.path.join(out_dir, "sweep_complete_metrics.csv")
    df.to_csv(export_path, index=False)
    print(f"Exported merged metrics to: {export_path}")

    # Generate visual diagnostics
    plot_pareto(df, out_dir)
    
    cost_params = ['cost_chi', 'cost_beta', 'cost_mu', 'cost_kappa']
    plot_parameter_sensitivity(df, cost_params, 'robustness_sigma_1p0_average_fidelity', 
                               'Robustness Fidelity', 'sensitivity_robustness.png', out_dir)
                               
    plot_parameter_sensitivity(df, cost_params, 'best_eval_leakage', 
                               'Best Eval Leakage', 'sensitivity_leakage.png', out_dir)

    print(f"Completed! Plots generated in {out_dir}")

if __name__ == "__main__":
    main()
