# Workspace File Overview

This document provides a description of each file in the repository, along with instructions on how to run the scripts and their associated parameters.

## Root Directory Scripts

| File | Description | Run Command | Parameters |
| :--- | :--- | :--- | :--- |
| `train_trpo_single_target.py` | Trains a single-target TRPO controller for the Universal Functional Optimization (UFO) experiment. | `python train_trpo_single_target.py --alpha <alpha> --gamma <gamma> --out <dir>` | `--alpha`: Target alpha (e.g., '2.2', 'pi/2'). <br> `--gamma`: Target gamma (default: 'pi/2'). <br> `--noise-optimized`: Train in stochastic environment. <br> `--iterations`: Number of iterations. <br> `--out`: Output directory. |
| `train_trpo_runtime.py` | Performs a curriculum TRPO sweep to generate runtime curves (similar to Figure 3 in the paper). | `python train_trpo_runtime.py --out <dir>` | `--gammas`: Comma-separated target gammas. <br> `--alpha-start/stop/step`: Alpha sweep range. <br> `--advance-cost-threshold`: Threshold to move to next alpha. <br> `--out`: Output directory. |
| `train_transition_experiments.py` | Conducts checkpoint transfer/transition experiments across different target gates to test adaptation. | `python train_transition_experiments.py --targets <specs> --out <dir>` | `--source-checkpoints`: Paths to starting checkpoints. <br> `--targets`: Comma-separated alpha[:gamma] specs. <br> `--include-scratch`: Include training from scratch for comparison. <br> `--out`: Output directory. |
| `train_adam_baseline.py` | Runs an Adam-based gradient optimization baseline for comparison with TRPO. | `python train_adam_baseline.py --alpha <alpha> --out <dir>` | `--alpha`: Target alpha. <br> `--lr`: Learning rate. <br> `--adam-iters`: Number of optimization steps. <br> `--horizons-ns`: Search horizons in nanoseconds. <br> `--out`: Output directory. |
| `evaluate_controls.py` | Evaluates control plans using the average-fidelity metric (similar to Figure 4 in the paper). | `python evaluate_controls.py --plans <files> --out <dir>` | `--plans`: List of `.npz` control plan files. <br> `--noise-min/max/step`: Robustness sweep range. <br> `--samples`: Monte Carlo samples per point. <br> `--out`: Output directory. |
| `plot_paper_repro.py` | Generates paper-style figures for runtime and robustness from CSV summaries. | `python plot_paper_repro.py --out <dir>` | `--runtime-csv`: Path to `runtime_summary.csv`. <br> `--robustness-csv`: Path to `robustness.csv`. <br> `--out`: Output directory for images. |
| `train_fast_thesis_suite.py` | A wrapper script that runs a "fast" suite of experiments (Adam, TRPO, and Transitions) in minutes. | `python train_fast_thesis_suite.py --out <dir>` | `--alpha`: Focal target alpha. <br> `--transition-targets`: Targets for adaptation test. <br> `--seed`: Random seed. <br> `--out`: Root output directory. |

## Library Components (`uqc/` directory)

| Component | Description |
| :--- | :--- |
| `uqc/env.py` | Implementation of the `QuantumControlEnv` (Gym-like interface for quantum control). |
| `uqc/physics.py` | Core physics logic, including the Gmon Hamiltonian, TSWT leakage bounds, and UFO cost functions. |
| `uqc/trpo.py` | Implementation of the Trust Region Policy Optimization (TRPO) agent and algorithm. |
| `uqc/baseline_adam.py`| Logic for the Adam gradient-based optimization baseline. |
| `uqc/eval.py` | Utilities for deterministic rollout, control plan management, and robustness evaluation. |
| `uqc/operators.py` | Definitions of quantum operators (Pauli matrices, projection operators). |
| `uqc/parallel.py` | Multi-processed batch collection for RL training. |
| `uqc/utils.py` | Logging, directory handling, and math expression parsing. |

## Documentation and Data Files

| File | Purpose |
| :--- | :--- |
| `README.md` | General project overview, installation, and example run commands. |
| `REPRO_NOTES.md` | Technical details on changes made from the original draft to ensure paper fidelity. |
| `FAST_RESULTS.md` | Summary of results obtained from the "fast" experiment suite. |
| `PILOT_RESULTS.md` | Preliminary results from pilot runs. |
| `requirements.txt` | List of Python dependencies (NumPy, SciPy, Torch, Matplotlib, Pandas). |
| `results.json` | A data file likely containing aggregated experimental results. |
