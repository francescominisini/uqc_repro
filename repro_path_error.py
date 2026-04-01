import os
import json

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

path = r'runs\cost_function_sweep\trpo\a-2p2__g-pi_2__kl-0p005__tc-0p15__std-m1p0__dt-2p0__mt-180p0__chi-5p0__beta-10p0__mu-0p1__kappa-0p05__hs-64m32m32\seed_1\robustness\iter_000001_robustness.json'
abs_path = os.path.abspath(path)
print(f"Absolute path length: {len(abs_path)}")
print(f"Absolute path: {abs_path}")

try:
    dir_name = os.path.dirname(abs_path)
    print(f"Creating directory: {dir_name}")
    ensure_dir(dir_name)
    print("Directory created or already exists.")
    
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump({"test": 123}, f)
    print("File written successfully.")
except Exception as e:
    print(f"Error: {e}")
