import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob

def setup_aesthetic():
    # Set up modern style for plots
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

def load_data(base_path):
    print(f"Loading data from: {base_path}")
    
    # Locate summary file
    summary_path = os.path.join(base_path, "runtime_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: summary file not found at {summary_path}")
        df_summary = pd.DataFrame()
    else:
        df_summary = pd.read_csv(summary_path)

    # Locate jsonl log
    # It might be nested in a gamma_* directory
    jsonl_files = glob(os.path.join(base_path, "**", "training_log.jsonl"), recursive=True)
    if not jsonl_files:
        print(f"Error: No training_log.jsonl found in {base_path}")
        return df_summary, pd.DataFrame()
    
    log_path = jsonl_files[0]
    print(f"Found detailed log at: {log_path}")
    
    records = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                    
    df_log = pd.DataFrame(records)
    return df_summary, df_log

def generate_summaries(df_log, out_dir):
    # 1. Mega CSV with all iterations
    all_iterations_path = os.path.join(out_dir, "all_iterations_log.csv")
    df_log.to_csv(all_iterations_path, index=False)
    print(f"Saved complete iterations log to {all_iterations_path}")

    # 2. Phase final CSV (last recorded iteration for each alpha)
    # The logs are sequential, we can just drop duplicates keeping the last
    if 'alpha' in df_log.columns:
        df_final = df_log.drop_duplicates(subset=['alpha'], keep='last')
        phase_final_path = os.path.join(out_dir, "phase_final_results_log.csv")
        df_final.to_csv(phase_final_path, index=False)
        print(f"Saved phase final results log to {phase_final_path}")
        return df_final
    return pd.DataFrame()

def create_plots(df_log, df_final, df_summary, out_dir):
    setup_aesthetic()
    
    if df_log.empty:
        return

    # Add a global step index to make plotting easier
    df_log['global_step'] = range(len(df_log))
    
    # ---------------------------------------------------------
    # Plot 1: Convergence (Fidelity and Cost over global step)
    # ---------------------------------------------------------
    if 'avg_fidelity' in df_log.columns and 'avg_cost' in df_log.columns:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:blue'
        ax1.set_xlabel('Global Training Iteration')
        ax1.set_ylabel('Average Fidelity', color=color)
        ax1.plot(df_log['global_step'], df_log['avg_fidelity'], color=color, alpha=0.8, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Average Cost', color=color)  
        ax2.plot(df_log['global_step'], df_log['avg_cost'], color=color, alpha=0.6, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.spines['top'].set_visible(False)
        # Identify boundaries of alpha shifts if we have 'alpha'
        if 'alpha' in df_log.columns:
            alpha_shifts = df_log.drop_duplicates(subset=['alpha'], keep='first')
            for _, row in alpha_shifts.iterrows():
                # Avoid plotting line at step 0 to reduce clutter
                if row['global_step'] > 0:
                    ax1.axvline(x=row['global_step'], color='gray', linestyle='--', alpha=0.3)
                    
        plt.title('Training Convergence across Curriculum Phases')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_convergence.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 2: Alpha vs Fidelity & Robustness 
    # ---------------------------------------------------------
    if not df_final.empty and 'alpha' in df_final.columns:
        plt.figure(figsize=(8, 5))
        
        if 'eval_fidelity' in df_final.columns:
            plt.plot(df_final['alpha'], df_final['eval_fidelity'], marker='o', label='Eval Fidelity', linewidth=2)
            
        if 'robustness_sigma_1p0_average_fidelity' in df_final.columns:
            plt.plot(df_final['alpha'], df_final['robustness_sigma_1p0_average_fidelity'], 
                     marker='s', label='Robustness (Sigma=1.0) Avg Fidelity', linewidth=2)
            
        plt.xlabel(r'Curriculum Constraint Parameter ($\alpha$)')
        plt.ylabel('Fidelity')
        plt.title('Performance vs Constraint Alpha')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'alpha_vs_fidelity.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 3: Leakage and Boundary Costs Evolution
    # ---------------------------------------------------------
    if not df_final.empty and 'alpha' in df_final.columns:
        plt.figure(figsize=(8, 5))
        
        has_leakage = False
        if 'eval_leakage' in df_final.columns:
            plt.plot(df_final['alpha'], df_final['eval_leakage'], marker='^', label='Eval Leakage', color='purple')
            has_leakage = True
            
        if 'avg_boundary_cost' in df_final.columns:
            plt.plot(df_final['alpha'], df_final['avg_boundary_cost'], marker='x', label='Avg Boundary Cost', color='orange')
            has_leakage = True
            
        if has_leakage:
            plt.xlabel(r'Curriculum Constraint Parameter ($\alpha$)')
            plt.ylabel('Cost Value')
            plt.title('Constraint Satisfaction across Curriculum')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'alpha_vs_constraints.png'), dpi=300)
            plt.close()

    # ---------------------------------------------------------
    # Plot 4: Time Trajectory
    # ---------------------------------------------------------
    if not df_final.empty and 'alpha' in df_final.columns:
        plt.figure(figsize=(8, 5))
        has_time = False
        if 'eval_min_cost_time_ns' in df_final.columns:
            plt.plot(df_final['alpha'], df_final['eval_min_cost_time_ns'], marker='o', label='Minimum Cost Time (ns)')
            has_time = True
            
        # from summary we have runtime_ns
        if not df_summary.empty and 'runtime_ns' in df_summary.columns and 'alpha' in df_summary.columns:
            plt.plot(df_summary['alpha'], df_summary['runtime_ns'], marker='s', label='Runtime (ns)')
            has_time = True
            
        if has_time:
            plt.xlabel(r'Curriculum Constraint Parameter ($\alpha$)')
            plt.ylabel('Time (ns)')
            plt.title('Time Optimization Trajectory vs Constraint')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'alpha_vs_time_trajectory.png'), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze runtime evaluation logs.")
    parser.add_argument("--base_dir", type=str, default=r"c:\Users\Francerso\source_vsc\Tesi\uqc_repro\final_results\runtime",
                        help="Path containing runtime logs.")
    args = parser.parse_args()

    # Determine out directory inside final_results
    out_dir = os.path.join(os.path.dirname(args.base_dir), "runtime_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved in: {out_dir}")

    # Load
    df_summary, df_log = load_data(args.base_dir)
    
    if df_log.empty:
        print("No log data to process. Exiting.")
        return

    # Process and Export
    df_final = generate_summaries(df_log, out_dir)
    
    # Plot
    create_plots(df_log, df_final, df_summary, out_dir)
    print("Done!")

if __name__ == "__main__":
    main()
