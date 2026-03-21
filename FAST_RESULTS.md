# Fast thesis-oriented results and speedups

## What was changed

The main bottleneck in the original reproduction scaffold was not the 9x9 propagator. It was the leakage-bound computation, which was recomputed by rescanning the full episode history at every step. I changed three things:

1. Vectorized the TSWT S1/S2 solves using a precomputed manifold-denominator matrix.
2. Made the leakage integral incremental, so each step is now O(1) instead of repeatedly rescanning the whole trajectory.
3. Replaced the generic matrix exponential in rollout propagation with a Hermitian eigendecomposition step and added persistent parallel batch collection for TRPO rollouts.

## Measured speedup

Benchmark target: `alpha=2.2`, `gamma=pi/2`, `dt=2 ns`, `max_time=600 ns`, 300 steps/episode.

- Original collector: 2 episodes in **18.92 s**
- New serial collector: 2 episodes in **0.56 s**
- Serial speedup: **34.0x**
- New parallel collector with 4 workers: 16 episodes in **2.14 s** after worker warm-up
- Throughput gain vs new serial collector: **2.08x**

Raw benchmark file: `/mnt/data/uqc_fast_benchmark.json`

## Fast useful result 1: Adam baseline

Command used:

```bash
python train_adam_baseline.py \
  --alpha 2.2 --gamma pi/2 \
  --dt-ns 10 --runtime-norm-ns 60 \
  --lr 0.03 --adam-iters 40 \
  --horizons-ns 60,90,120 \
  --out /mnt/data/adam_fast_timed
```

Wall time: about **11.2 s**.

Best nominal result:

- Best horizon: **90 ns**
- UFO cost: **0.2504**
- Gate fidelity: **0.999145**
- Leakage: **4.21e-05**

This is the fastest path I found to something thesis-usable right away.

## Fast useful result 2: TRPO pilot

Command used:

```bash
python train_trpo_single_target.py \
  --alpha 2.2 --gamma pi/2 \
  --noise-optimized \
  --iterations 5 --episodes-per-batch 32 \
  --eval-every 1 --save-every 1 \
  --dt-ns 10 --max-time-ns 60 --runtime-norm-ns 60 \
  --termination-cost 0.15 \
  --num-workers 4 --episodes-per-task 8 \
  --out /mnt/data/trpo_fastpilot
```

Wall time: about **10.7 s**.

Best eval result:

- Best eval cost: **4.5646**
- Best eval fidelity: **0.5861**
- Best eval leakage: **1.14e-05**

This is not close to the Adam result yet, but it is fast enough for ablations, transition tests, and policy-learning plots.

## Fast useful result 3: transfer / transition experiment

Seed checkpoint: `/mnt/data/trpo_fastpilot/best_agent.pt`

Target: `alpha=2.4`, `gamma=pi/2`

Command used:

```bash
python train_transition_experiments.py \
  --source-checkpoints /mnt/data/trpo_fastpilot/best_agent.pt \
  --source-labels alpha2p2_seed \
  --targets 2.4:pi/2 \
  --include-scratch \
  --noise-optimized \
  --iterations 2 --episodes-per-batch 16 \
  --eval-every 1 --save-every 1 \
  --dt-ns 10 --max-time-ns 60 --runtime-norm-ns 60 \
  --termination-cost 0.15 \
  --num-workers 4 --episodes-per-task 4 \
  --out /mnt/data/transition_fast
```

Wall time: about **12.1 s**.

Comparison:

- Scratch zero-shot cost: **7.9814**, fidelity **0.2119**
- Transfer zero-shot cost: **7.0322**, fidelity **0.3394**
- Scratch after 2 iters: **7.8469**, fidelity **0.2271**
- Transfer after 2 iters: **7.0010**, fidelity **0.3486**

So even in the fast setting, transfer is already clearly helpful.

## Practical thesis recommendation

If the goal is a bachelor-thesis-ready result set quickly, the best sequence is:

1. Use the Adam baseline to get high-fidelity nominal controls fast.
2. Use the fast TRPO setup only on one or two targets for policy-learning curves.
3. Use transition experiments to show transfer helps on neighboring gates.
4. Reserve the expensive `dt=2 ns` long-horizon runs for a small appendix or one final confirmation run, not the whole sweep.

## Convenience runner

A convenience script is included:

```bash
python train_fast_thesis_suite.py --out /path/to/output
```

It runs:

- a fast Adam baseline,
- a fast TRPO seed run,
- fast transition experiments from that checkpoint.

