import pathlib
import sys

path = pathlib.Path("runs/cost_function_sweep/trpo")
results = []

for file in sorted(path.rglob("training_log.json*")):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            count = len(lines)
            if count != 5:
                results.append(f"{file}: {count}")
    except Exception as e:
        results.append(f"Error reading {file}: {e}")

if not results:
    print("All training logs have exactly 5 lines.")
else:
    print("The following files do NOT have 5 lines:")
    for r in results:
        print(r)
