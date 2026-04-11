import pandas as pd
import numpy as np
import os

def export_combined_data(base_dir, out_file, span_fid=50, span_var=80):
    methods = ['adam', 'nominal', 'noise']
    
    dfs = []
    for m in methods:
        path = os.path.join(base_dir, m, "robustness_curve.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Compute EWMA on individual arrays
            df[f'{m}_fidelity_raw'] = df['average_fidelity']
            df[f'{m}_fidelity_ewma'] = df['average_fidelity'].ewm(span=span_fid).mean()
            
            df[f'{m}_variance_raw'] = df['fidelity_variance']
            df[f'{m}_variance_ewma'] = df['fidelity_variance'].ewm(span=span_var).mean()
            
            df = df[['sigma_mhz', f'{m}_fidelity_raw', f'{m}_fidelity_ewma', f'{m}_variance_raw', f'{m}_variance_ewma']]
            df = df.set_index('sigma_mhz')
            dfs.append(df)
            
    if not dfs:
        print(f"No data found in {base_dir}")
        return
        
    combined = pd.concat(dfs, axis=1)
    
    # Sort just in case
    combined = combined.sort_index()
    
    combined.to_csv(out_file)
    print(f"Exported combined EWMA data to {out_file}")

if __name__ == "__main__":
    export_combined_data(
        "final_results/final_robustness_analysis", 
        "final_results/final_robustness_analysis/combined_ewma_data.csv"
    )
    export_combined_data(
        "final_results/final_robustness_analysis_2", 
        "final_results/final_robustness_analysis_2/combined_ewma_data.csv"
    )
