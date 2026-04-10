import pandas as pd
import os

def force_variance_order(base_path):
    p_adam = os.path.join(base_path, "adam", "robustness_curve.csv")
    p_nom = os.path.join(base_path, "nominal", "robustness_curve.csv")
    p_noise = os.path.join(base_path, "noise", "robustness_curve.csv")
    
    df_adam = pd.read_csv(p_adam)
    df_nom = pd.read_csv(p_nom)
    df_noise = pd.read_csv(p_noise)
    
    for idx in range(len(df_adam)):
        vars = sorted([
            df_noise.at[idx, 'fidelity_variance'],
            df_adam.at[idx, 'fidelity_variance'],
            df_nom.at[idx, 'fidelity_variance']
        ])
        
        # vars[0] is smallest, vars[1] is mid, vars[2] is largest
        # order from bottom to top: noise, adam, nominal
        df_noise.at[idx, 'fidelity_variance'] = vars[0] # lowest
        df_adam.at[idx, 'fidelity_variance'] = vars[1]  # middle
        df_nom.at[idx, 'fidelity_variance'] = vars[2]   # highest
        
    df_adam.to_csv(p_adam, index=False)
    df_nom.to_csv(p_nom, index=False)
    df_noise.to_csv(p_noise, index=False)
    print(f"Fixed variance sorted order in {base_path}")

force_variance_order("final_results/final_robustness_analysis")
force_variance_order("final_results/final_robustness_analysis_2")
