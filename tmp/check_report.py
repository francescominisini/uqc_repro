import pathlib

path = pathlib.Path("runs/cost_function_sweep/trpo")
all_files = sorted(list(path.rglob("training_log.jsonl")))

print(f"Checking {len(all_files)} files...")

for file in all_files:
    with open(file, 'r') as f:
        count = len(f.readlines())
        if count != 5:
            print(f"MALFORMED: {file} - {count} lines")
        else:
            # print(f"OK: {file}")
            pass

print("Check finished.")
