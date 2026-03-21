# Pilot results and runtime checks

## Convergence checks run in this session

### Short-horizon single-target pilot
Command family: TRPO single target, `alpha=2.2`, `gamma=pi/2`, noise-optimized, `episodes_per_batch=20`, `iterations=5`, `dt_ns=10`, `max_time_ns=60`, `value_epochs=5`, `cg_iters=5`.

Observed eval trajectory:

- iter 1: cost 5.3192, fidelity 0.4820, leakage 1.78e-06
- iter 2: cost 4.8152, fidelity 0.5430, leakage 3.84e-06
- iter 3: cost 4.6309, fidelity 0.5650, leakage 4.89e-06
- iter 4: cost 4.4112, fidelity 0.5990, leakage 8.96e-06
- iter 5: cost 4.5319, fidelity 0.5870, leakage 1.09e-05

Best checkpoint: iteration 4.
Wall-clock duration: about 9 s.

### Paper-like horizon pilot
Command family: TRPO single target, `alpha=2.2`, `gamma=pi/2`, noise-optimized, `episodes_per_batch=2`, `iterations=5`, `dt_ns=2`, `max_time_ns=600`, `value_epochs=5`, `cg_iters=5`.

Observed eval trajectory:

- iter 1: cost 10.6291, fidelity 0.0438, leakage 1.17e-03
- iter 2: cost 11.0756, fidelity 0.0031, leakage 1.80e-03
- iter 3: cost 9.7041, fidelity 0.1474, leakage 2.05e-03
- iter 4: cost 11.1054, fidelity 0.0418, leakage 2.43e-03
- iter 5: cost 10.9037, fidelity 0.0297, leakage 2.00e-03

Best checkpoint: iteration 3.
Wall-clock duration: about 34 s.

This did not converge to the paper-quality regime; it is only a sanity-check that the training loop runs and updates under a realistic horizon.

## Runtime benchmarks used for scaling estimates

### Benchmark A
`episodes_per_batch=20`, `iterations=1`, `dt_ns=20`, `max_time_ns=40`, noise-optimized.

- wall-clock: about 9 s
- eval cost: 5.3898
- eval fidelity: 0.4716

### Benchmark B
`episodes_per_batch=20`, `iterations=1`, `dt_ns=10`, `max_time_ns=60`, noise-optimized.

- wall-clock: about 8 s
- eval cost: 5.2915
- eval fidelity: 0.4852

### Benchmark C
`episodes_per_batch=2`, `iterations=1`, `dt_ns=2`, `max_time_ns=600`, noise-optimized.

- wall-clock: about 13 s
- eval cost: 10.7572
- eval fidelity: 0.0278

## Transition experiment smoke test
Source checkpoint: best checkpoint from the short-horizon `alpha=2.2, gamma=pi/2` pilot.
Target gate: `alpha=2.4, gamma=pi/2`.
Short adaptation budget: `episodes_per_batch=2`, `iterations=2`, `dt_ns=20`, `max_time_ns=40`, noise-optimized.

### Scratch baseline
- zero-shot cost: 7.9613
- zero-shot fidelity: 0.2106
- best adapted cost: 7.9210
- best adapted fidelity: 0.2291

### Transfer from alpha=2.2 checkpoint
- zero-shot cost: 7.3312
- zero-shot fidelity: 0.3001
- best adapted cost: 7.1770
- best adapted fidelity: 0.3384

This smoke test shows the transition / transfer script is working and, at least on this tiny budget, transferred initialization is materially better than scratch.
