# Reproduction notes for `Universal Quantum Control.pdf`

## What had to change from the tentative draft

1. **Leakage term**
   - Replaced the raw computational-to-leakage Hamiltonian block with a numerical implementation of the **second-order TSWT** leakage bound.
   - Added `S1`, `S2`, and the residual block-off-diagonal Hamiltonian used in the paper's Appendix B.

2. **Units**
   - Kept Hamiltonians in `rad/ns` internally and converted the gmon gap `Delta ~ 200 MHz` to the same units before using it in the leakage bound.

3. **Noise model**
   - Added per-step Gaussian noise on `eta`, `g`, `delta1`, `delta2`, `f1`, and `f2`.
   - Left phases noise-free.

4. **Bandwidth filter**
   - Implemented the Appendix C double-exponential filter on piecewise-constant controls.

5. **Reward / stopping / curriculum**
   - Implemented dense reward as `-current_cost` by default.
   - Episode length is `min(runtime upper bound, time to satisfy cost threshold)`.
   - Runtime sweep now carries the same agent across `alpha` values and supports the paper's `gamma in {pi/2, pi/6, pi/3}`.

6. **Robustness metric**
   - Added Appendix D average-fidelity evaluation from sampled noisy trajectories.
   - Added the figure-4 style robustness sweep over noise variance.

7. **Baseline method**
   - Added a differentiable gradient baseline corresponding to the paper's SGD/Adam comparison.

8. **Practical engineering**
   - Removed the hard dependency on `qutip` and `gymnasium` so the experiment can run in a lighter environment.

## Main entry points

- `train_trpo_single_target.py`
- `train_trpo_runtime.py`
- `train_adam_baseline.py`
- `evaluate_controls.py`
- `plot_paper_repro.py`

## Remaining knobs

Some values are still exposed because the paper does not fully pin them down in implementation detail:

- `runtime_norm_ns`
- `termination_cost`
- `advance_cost_threshold`
- TRPO optimizer hyperparameters
- `dt_ns`
- `max_time_ns`

These are the parameters to tune if your curves are qualitatively right but still shifted relative to the paper.

## Additions after the first scaffold

9. **Checkpointing for later comparison**
   - Single-target and curriculum TRPO runs now save per-iteration checkpoints, best checkpoints, and final checkpoints.
   - Deterministic control plans are also saved per iteration so you can compare not only NN weights but the induced pulse sequence.

10. **Transition / transfer experiments**
   - Added `train_transition_experiments.py` to test zero-shot transfer and short fine-tuning from one gate checkpoint to another target gate.
   - The script can compare transferred initialization against scratch on the same target and saves a `transition_summary.csv` for side-by-side analysis.
