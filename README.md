# Universal quantum control reproduction scaffold

This package is a paper-faithful rewrite of the initial prototype. It removes the hard dependency on `qutip` and `gymnasium`, implements the gmon Hamiltonian directly with NumPy/SciPy/Torch, and adds the missing pieces needed to reproduce the paper figures more closely:

- second-order TSWT leakage bound for the UFO cost
- exact double-exponential bandwidth filter from Appendix C
- noise training with per-step Gaussian perturbations on eta, g, delta_j, and f_j
- Appendix D average-fidelity evaluation from Monte Carlo samples
- curriculum runtime sweep over the target family `N(alpha, alpha, gamma)`
- Adam baseline for runtime comparison
- per-iteration checkpoint saving for TRPO experiments
- transition / transfer experiments across target gates

The TRPO scripts now save:

- `checkpoints/iter_XXXXXX.pt` at every iteration by default
- `best_agent.pt` and `final_agent.pt`
- `plans/iter_XXXXXX_control_plan.npz`, `best_control_plan.npz`, and `final_control_plan.npz`

## Scripts

Single-target TRPO training:

```bash
python train_trpo_single_target.py --alpha 2.2 --gamma pi/2 --noise-optimized --out runs/trpo_2p2_noise
```

Curriculum runtime sweep (Fig. 3 style):

```bash
python train_trpo_runtime.py --out runs/runtime_sweep
```

Transition / transfer experiments:

```bash
python train_transition_experiments.py \
  --source-checkpoints runs/trpo_2p2_noise/best_agent.pt \
  --source-labels from_2p2 \
  --targets 2.4:pi/2,2.6:pi/2 \
  --include-scratch \
  --out runs/transitions
```

Adam baseline:

```bash
python train_adam_baseline.py --alpha 2.2 --gamma pi/2 --out runs/adam_2p2
```

Robustness evaluation (Fig. 4 style):

```bash
python evaluate_controls.py \
  --plans runs/trpo_2p2_noise/best_control_plan.npz runs/trpo_2p2_plain/best_control_plan.npz runs/adam_2p2/best_control_plan.npz \
  --labels noise_optimized no_noise adam \
  --out runs/robustness
```

Plotting:

```bash
python plot_paper_repro.py --runtime-csv runs/runtime_sweep/runtime_summary.csv --robustness-csv runs/robustness/robustness.csv --out runs/figures
```

## Practical note for this environment

The TRPO / Adam scripts set CPU-only Torch environment variables automatically (`CUDA_VISIBLE_DEVICES=''`, single-threaded BLAS / OpenMP). That avoids the import stalls I saw in this container.

## Important assumptions

The paper clearly specifies the UFO cost weights, filter bandwidth, target family, transfer-learning curriculum, and robustness metric, but it does not fully specify every engineering detail required for an exact byte-for-byte reproduction. In particular, the following remain exposed as command-line arguments so you can tune them during replication:

- `runtime_norm_ns`: the time normalization used to make the `kappa T` term numerically commensurate with the other terms
- `termination_cost` / `advance_cost_threshold`: practical thresholds for early stopping and curriculum advancement
- TRPO optimizer settings such as `max_kl`, `lam`, `value_epochs`, and initial exploration scale
- time discretization `dt_ns` and maximum runtime budget `max_time_ns`

These are surfaced rather than hard-coded because the paper does not pin all of them down precisely.
