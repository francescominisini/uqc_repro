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
    df_clean = df.dropna(subset=['robustness_sigma_1p0_average_fidelity']).copy()
    print(f"Loaded {len(df_clean)} complete runs with robustness data out of {len(df)}.")
    return df_clean

def plot_boxplot_sensitivity(df, param_name, y_col, y_label, out_name, out_dir, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group data by the categorical parameter (hidden_sizes)
    unique_vals = sorted(df[param_name].unique())
    groups = [df[df[param_name] == v][y_col].values for v in unique_vals]
    
    # Boxplot
    box = ax.boxplot(groups, positions=range(len(unique_vals)), widths=0.5, 
                     patch_artist=True, boxprops=dict(facecolor='tab:cyan', alpha=0.6))
    
    # Overlay scatter to see individual seeds
    for i, v in enumerate(unique_vals):
        y = df[df[param_name] == v][y_col].values
        x = [i] * len(y)
        ax.scatter(x, y, alpha=0.7, color='tab:blue', edgecolor='k', s=60)

    ax.set_xticks(range(len(unique_vals)))
    ax.set_xticklabels([str(v) for v in unique_vals], rotation=15, ha='right')
    ax.set_xlabel(f'Neural Network Architecture ({param_name})')
    ax.set_ylabel(y_label)
    
    if log_y:
        ax.set_yscale('log')
        
    plt.title(f'{y_label} across Architectures')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close()

def plot_pareto(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    unique_archs = sorted(df['hidden_sizes'].unique())
    colors = plt.cm.tab10.colors
    
    for i, arch in enumerate(unique_archs):
        subset = df[df['hidden_sizes'] == arch]
        ax.scatter(subset['best_eval_leakage'], subset['robustness_sigma_1p0_average_fidelity'], 
                   label=arch, color=colors[i % len(colors)], s=80, alpha=0.8, edgecolor='w')
    
    ax.set_xscale('log')
    ax.set_xlabel('Best Eval Leakage (Log Scale)')
    ax.set_ylabel('Robustness Avg Fidelity (Sigma=1.0)')
    ax.set_title('Pareto Front: Leakage vs Robustness by Architecture')
    ax.legend(title='hidden_sizes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_leakage_vs_robustness.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro\final_results\nn_size_sweep",
                        help="Path containing raw run results.")
    parser.add_argument("--project_dir", type=str, default=r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro",
                        help="Root path of project to resolve local directories.")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.base_dir), "nn_size_sweep_results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    setup_aesthetic()

    raw_csv = os.path.join(args.base_dir, "trpo_raw_results.csv")
    df = load_sweep_data(raw_csv, args.project_dir)

    if df.empty:
        print("No valid sweep data found.")
        return

    # Export merged
    export_path = os.path.join(out_dir, "sweep_complete_metrics.csv")
    df.to_csv(export_path, index=False)
    print(f"Exported merged metrics to: {export_path}")

    # Plot Robustness and Leakage directly vs Architecture size
    plot_boxplot_sensitivity(df, 'hidden_sizes', 'robustness_sigma_1p0_average_fidelity', 
                             'Robustness Fidelity', 'architecture_vs_robustness.png', out_dir)
                             
    plot_boxplot_sensitivity(df, 'hidden_sizes', 'best_eval_leakage', 
                             'Best Eval Leakage', 'architecture_vs_leakage.png', out_dir, log_y=True)

    # Plot Pareto
    plot_pareto(df, out_dir)
    
    print(f"Completed! Plots generated in {out_dir}")

if __name__ == "__main__":
    main()
