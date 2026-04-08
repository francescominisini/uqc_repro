import pandas as pd
import os

base = r"final_results\final_robustness_analysis"
df_adam = pd.read_csv(os.path.join(base, "adam", "robustness_curve.csv")).set_index('sigma_mhz')
df_nom = pd.read_csv(os.path.join(base, "nominal", "robustness_curve.csv")).set_index('sigma_mhz')
df_noise = pd.read_csv(os.path.join(base, "noise", "robustness_curve.csv")).set_index('sigma_mhz')

merged = pd.concat([df_adam['average_fidelity'], df_nom['average_fidelity'], df_noise['average_fidelity']], 
                   axis=1, keys=['adam', 'nominal', 'noise'])

merged['noise_v_adam'] = merged['noise'] - merged['adam']
merged['adam_v_nom'] = merged['adam'] - merged['nominal']
merged['noise_v_nom'] = merged['noise'] - merged['nominal']

print("\nDifferences after 1.0 (mean):")
print(merged[merged.index >= 1.0][['noise_v_adam', 'adam_v_nom', 'noise_v_nom']].mean())

neg_noise_v_adam = merged[(merged.index >= 1.0) & (merged['noise_v_adam'] < 0)]
neg_adam_v_nom = merged[(merged.index >= 1.0) & (merged['adam_v_nom'] < 0)]

print("\nFirst noise_v_adam crossover:")
print(neg_noise_v_adam.head(1))
print("\nFirst adam_v_nom crossover:")
print(neg_adam_v_nom.head(1))
