import pathlib

path = pathlib.Path("runs/cost_function_sweep/trpo")
results = []

for file in sorted(list(path.rglob("training_log.jsonl"))):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            count = len(lines)
            if count != 5:
                results.append(f"{file}: {count}")
    except Exception as e:
        results.append(f"Error reading {file}: {e}")

if not results:
    print("Tutti i training_log.jsonl hanno esattamente 5 linee.")
else:
    print("I seguenti file NON hanno 5 linee:")
    for r in results:
        print(r)
