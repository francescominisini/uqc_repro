from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .physics import GmonSystem, TSWTLeakageEstimator, UFOCostWeights, evaluate_average_fidelity


@dataclass(frozen=True)
class ControlPlan:
    controls_mhz_and_phase: np.ndarray
    target_alpha: float
    target_gamma: float
    dt_ns: float
    runtime_norm_ns: float
    cost_weights: UFOCostWeights
    filter_bandwidth_mhz: float = 10.0
    note: str = ""

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            controls=self.controls_mhz_and_phase,
            target_alpha=self.target_alpha,
            target_gamma=self.target_gamma,
            dt_ns=self.dt_ns,
            runtime_norm_ns=self.runtime_norm_ns,
            chi=self.cost_weights.chi,
            beta=self.cost_weights.beta,
            mu=self.cost_weights.mu,
            kappa=self.cost_weights.kappa,
            filter_bandwidth_mhz=self.filter_bandwidth_mhz,
            note=self.note,
        )

    @classmethod
    def load(cls, path: str) -> "ControlPlan":
        data = np.load(path, allow_pickle=True)
        weights = UFOCostWeights(
            chi=float(data["chi"]),
            beta=float(data["beta"]),
            mu=float(data["mu"]),
            kappa=float(data["kappa"]),
        )
        filter_bandwidth_mhz = float(data["filter_bandwidth_mhz"]) if "filter_bandwidth_mhz" in data.files else 10.0
        if "note" in data.files:
            raw_note = data["note"]
            if isinstance(raw_note, np.ndarray) and raw_note.shape == ():
                note = str(raw_note.item())
            else:
                note = str(raw_note)
        else:
            note = ""
        return cls(
            controls_mhz_and_phase=np.asarray(data["controls"], dtype=np.float64),
            target_alpha=float(data["target_alpha"]),
            target_gamma=float(data["target_gamma"]),
            dt_ns=float(data["dt_ns"]),
            runtime_norm_ns=float(data["runtime_norm_ns"]),
            cost_weights=weights,
            filter_bandwidth_mhz=filter_bandwidth_mhz,
            note=note,
        )


def simulate_nominal_plan(
    system: GmonSystem,
    plan: ControlPlan,
    eta_base_mhz: float | None = None,
) -> Dict[str, object]:
    eta_mhz = float(system.config.eta_base_mhz if eta_base_mhz is None else eta_base_mhz)
    target_gate = system.target_gate(plan.target_alpha, plan.target_gamma)
    U = system.initial_unitary()
    leakage = TSWTLeakageEstimator(system)
    for controls in plan.controls_mhz_and_phase:
        H = system.hamiltonian(controls, eta_mhz)
        U = system.evolve_step(U, controls, eta_mhz)
        leakage.step(H, eta_mhz)

    fidelity = system.gate_fidelity(U, target_gate)
    leakage_total, leakage_parts = leakage.current_leakage_bound()
    c0 = plan.controls_mhz_and_phase[0]
    ct = plan.controls_mhz_and_phase[-1]
    boundary_raw = c0[6] ** 2 + c0[2] ** 2 + c0[4] ** 2 + ct[6] ** 2 + ct[2] ** 2 + ct[4] ** 2
    boundary_cost = plan.cost_weights.mu * boundary_raw
    time_ns = plan.controls_mhz_and_phase.shape[0] * plan.dt_ns
    time_cost = plan.cost_weights.kappa * (time_ns / plan.runtime_norm_ns)
    cost = plan.cost_weights.chi * (1.0 - fidelity) + plan.cost_weights.beta * leakage_total + boundary_cost + time_cost
    return {
        "U_full": U,
        "projected_unitary": system.projected_unitary(U),
        "fidelity": float(fidelity),
        "leakage": float(leakage_total),
        "leakage_boundary": leakage_parts["boundary"],
        "leakage_integral": leakage_parts["integral"],
        "boundary_cost": float(boundary_cost),
        "time_cost": float(time_cost),
        "time_ns": float(time_ns),
        "cost": float(cost),
    }



def noisy_projected_unitary_samples(
    system: GmonSystem,
    plan: ControlPlan,
    sigma_mhz: float,
    num_samples: int,
    seed: int | None = None,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    projected: List[np.ndarray] = []
    for _ in range(num_samples):
        U = system.initial_unitary()
        for controls in plan.controls_mhz_and_phase:
            noisy = np.asarray(controls, dtype=np.float64).copy()
            eta_mhz = system.config.eta_base_mhz + rng.normal(0.0, sigma_mhz)
            noise = rng.normal(0.0, sigma_mhz, size=5)
            noisy[0] += noise[0]
            noisy[1] += noise[1]
            noisy[2] += noise[2]
            noisy[4] += noise[3]
            noisy[6] += noise[4]
            noisy[[3, 5]] = np.mod(noisy[[3, 5]], 2.0 * np.pi)
            U = system.evolve_step(U, noisy, eta_mhz)
        projected.append(system.projected_unitary(U))
    return projected



def robustness_metrics(
    system: GmonSystem,
    plan: ControlPlan,
    sigma_mhz: float,
    num_samples: int = 60,
    seed: int | None = None,
) -> Dict[str, float]:
    target_gate = system.target_gate(plan.target_alpha, plan.target_gamma)
    projected = noisy_projected_unitary_samples(system, plan, sigma_mhz, num_samples, seed=seed)
    gate_fidelities = np.array([
        system.gate_fidelity_from_projected(K, target_gate) for K in projected
    ], dtype=np.float64)
    avg_gate_fidelity = float(np.mean(gate_fidelities))
    fidelity_variance = float(np.mean((gate_fidelities - avg_gate_fidelity) ** 2))
    avg_fidelity = evaluate_average_fidelity(system, target_gate, projected)
    return {
        "sigma_mhz": float(sigma_mhz),
        "num_samples": int(num_samples),
        "average_fidelity": float(avg_fidelity),
        "average_gate_fidelity": avg_gate_fidelity,
        "fidelity_variance": fidelity_variance,
    }
