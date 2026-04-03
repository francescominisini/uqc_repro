import pathlib

path = pathlib.Path("runs/cost_function_sweep/trpo")
all_files = sorted(list(path.rglob("training_log.jsonl")))

results = []
results.append(f"Checking {len(all_files)} files...")

for file in all_files:
    with open(file, 'r') as f:
        count = len(f.readlines())
        if count != 5:
            results.append(f"MALFORMED: {file} - {count} lines")

results.append("Check finished.")

with open("tmp/final_report.txt", "w") as f:
    f.write("\n".join(results))
