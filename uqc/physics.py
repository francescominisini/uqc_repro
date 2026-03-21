from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.linalg

from .operators import destroy, paulis, qeye, tensor


@dataclass(frozen=True)
class UFOCostWeights:
    chi: float = 10.0
    beta: float = 10.0
    mu: float = 0.2
    kappa: float = 0.1


@dataclass(frozen=True)
class GmonSystemConfig:
    n_levels: int = 3
    dt_ns: float = 2.0
    eta_base_mhz: float = -200.0
    bandwidth_mhz: float = 10.0
    max_control_mhz: float = 20.0
    runtime_norm_ns: float = 60.0


class GmonSystem:
    """
    Two-transmon / gmon-style truncated model used in the paper.

    The RWA Hamiltonian follows Eq. (A20) in the paper:
        H_RWA = eta/2 * sum_j n_j (n_j - 1)
                + g(t) (a2^dag a1 + a1^dag a2)
                + sum_j delta_j(t) n_j
                + sum_j i f_j(t) (a_j e^{i phi_j} - a_j^dag e^{-i phi_j})

    We keep units in rad/ns internally.
    """

    def __init__(self, config: GmonSystemConfig):
        self.config = config
        self.n_levels = config.n_levels
        self.dt_ns = float(config.dt_ns)
        self.two_pi = 2.0 * np.pi
        self.mhz_to_rad_per_ns = 1e-3 * self.two_pi

        self.a1 = tensor(destroy(self.n_levels), qeye(self.n_levels))
        self.a2 = tensor(qeye(self.n_levels), destroy(self.n_levels))
        self.ad1 = self.a1.conj().T
        self.ad2 = self.a2.conj().T
        self.n1 = self.ad1 @ self.a1
        self.n2 = self.ad2 @ self.a2
        self.eye = tensor(qeye(self.n_levels), qeye(self.n_levels))
        self.op_coupling = self.ad2 @ self.a1 + self.ad1 @ self.a2
        self.drift_shape = 0.5 * (
            self.n1 @ (self.n1 - np.eye(self.n1.shape[0]))
            + self.n2 @ (self.n2 - np.eye(self.n2.shape[0]))
        )

        self.comp_indices = [0 * self.n_levels + 0, 0 * self.n_levels + 1, 1 * self.n_levels + 0, 1 * self.n_levels + 1]
        all_indices = list(range(self.n_levels ** 2))
        self.leak_indices = [i for i in all_indices if i not in self.comp_indices]
        self.subspaces = self._build_subspaces()
        self.block_mask = self._build_block_mask(self.subspaces).astype(bool)
        self.offdiag_mask = (~self.block_mask).astype(bool)

        self.h0_coeff_diag = np.diag(self.drift_shape).real
        self.subspace_coeffs = [float(np.mean(self.h0_coeff_diag[idxs])) for idxs in self.subspaces]

        self.basis_subspace_coeffs = np.zeros(self.eye.shape[0], dtype=np.float64)
        for coeff, idxs in zip(self.subspace_coeffs, self.subspaces):
            self.basis_subspace_coeffs[np.asarray(idxs, dtype=np.int64)] = coeff
        self.coeff_diff_matrix = self.basis_subspace_coeffs[None, :] - self.basis_subspace_coeffs[:, None]
        self.denom_mask = self.offdiag_mask & (~np.isclose(self.coeff_diff_matrix, 0.0))

    def _build_subspaces(self) -> List[List[int]]:
        coeffs = np.round(np.diag(self.drift_shape).real, 12)
        groups: Dict[float, List[int]] = {}
        for idx, coeff in enumerate(coeffs.tolist()):
            groups.setdefault(coeff, []).append(idx)

        ordered_coeffs = sorted(groups.keys())
        subspaces = [self.comp_indices]
        for coeff in ordered_coeffs:
            group = groups[coeff]
            if set(group) == set(self.comp_indices):
                continue
            if set(group).intersection(self.comp_indices):
                group = [i for i in group if i not in self.comp_indices]
                if not group:
                    continue
            subspaces.append(group)
        return subspaces

    def _build_block_mask(self, subspaces: Sequence[Sequence[int]]) -> np.ndarray:
        dim = self.eye.shape[0]
        mask = np.zeros((dim, dim), dtype=bool)
        for idxs in subspaces:
            mask[np.ix_(idxs, idxs)] = True
        return mask

    @property
    def dim(self) -> int:
        return self.eye.shape[0]

    def initial_unitary(self) -> np.ndarray:
        return self.eye.copy()

    def h0(self, eta_mhz: float) -> np.ndarray:
        return eta_mhz * self.mhz_to_rad_per_ns * self.drift_shape

    def hamiltonian(self, controls: Sequence[float], eta_mhz: float) -> np.ndarray:
        d1, d2, f1, phi1, f2, phi2, g = [float(x) for x in controls]
        scale = self.mhz_to_rad_per_ns
        eta = eta_mhz * scale
        d1_r = d1 * scale
        d2_r = d2 * scale
        f1_r = f1 * scale
        f2_r = f2 * scale
        g_r = g * scale

        h_drift = eta * self.drift_shape
        h_det = d1_r * self.n1 + d2_r * self.n2
        h_c = g_r * self.op_coupling
        h_drive = np.zeros_like(h_drift)
        if f1_r != 0.0:
            h_drive += 1j * f1_r * (self.a1 * np.exp(1j * phi1) - self.ad1 * np.exp(-1j * phi1))
        if f2_r != 0.0:
            h_drive += 1j * f2_r * (self.a2 * np.exp(1j * phi2) - self.ad2 * np.exp(-1j * phi2))
        return h_drift + h_det + h_c + h_drive

    def unitary_from_hamiltonian(self, H: np.ndarray) -> np.ndarray:
        Hh = 0.5 * (H + H.conj().T)
        eigvals, eigvecs = np.linalg.eigh(Hh)
        phases = np.exp(-1j * eigvals * self.dt_ns)
        return (eigvecs * phases) @ eigvecs.conj().T

    def evolve_step(self, U: np.ndarray, controls: Sequence[float], eta_mhz: float) -> np.ndarray:
        H = self.hamiltonian(controls, eta_mhz)
        return self.unitary_from_hamiltonian(H) @ U

    def projected_unitary(self, U: np.ndarray) -> np.ndarray:
        return U[np.ix_(self.comp_indices, self.comp_indices)]

    def embed_rho(self, rho_comp: np.ndarray) -> np.ndarray:
        rho_full = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho_full[np.ix_(self.comp_indices, self.comp_indices)] = rho_comp
        return rho_full

    def apply_projected_channel(self, U_full: np.ndarray, rho_comp: np.ndarray) -> np.ndarray:
        rho_full = self.embed_rho(rho_comp)
        out_full = U_full @ rho_full @ U_full.conj().T
        return out_full[np.ix_(self.comp_indices, self.comp_indices)]

    def target_gate(self, alpha: float, gamma: float = np.pi / 2.0) -> np.ndarray:
        sx, sy, sz = paulis()
        XX = np.kron(sx, sx)
        YY = np.kron(sy, sy)
        ZZ = np.kron(sz, sz)
        h_tgt = alpha * XX + alpha * YY + gamma * ZZ
        return scipy.linalg.expm(1j * h_tgt)

    @staticmethod
    def gate_fidelity_from_projected(projected_u: np.ndarray, target_u: np.ndarray) -> float:
        d = target_u.shape[0]
        tr = np.trace(projected_u.conj().T @ target_u)
        return float(np.abs(tr) ** 2 / (d ** 2))

    def gate_fidelity(self, U_full: np.ndarray, target_u: np.ndarray) -> float:
        return self.gate_fidelity_from_projected(self.projected_unitary(U_full), target_u)

    def pauli_basis_two_qubit(self) -> List[np.ndarray]:
        I = np.eye(2, dtype=np.complex128)
        sx, sy, sz = paulis()
        singles = [I, sx, sy, sz]
        return [np.kron(a, b) for a in singles for b in singles]

    def decompose_hamiltonian(self, H: np.ndarray, eta_mhz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h0 = self.h0(eta_mhz)
        h_rest = H - h0
        h1 = np.where(self.block_mask, h_rest, 0.0)
        h2 = h_rest - h1
        subspace_energies = np.array(self.subspace_coeffs, dtype=np.float64) * eta_mhz * self.mhz_to_rad_per_ns
        return h0, h1, h2, subspace_energies

    def blockdiag(self, M: np.ndarray) -> np.ndarray:
        return np.where(self.block_mask, M, 0.0)

    def offdiag(self, M: np.ndarray) -> np.ndarray:
        return np.where(self.offdiag_mask, M, 0.0)

    def tswt_denominator_matrix(self, eta_mhz: float) -> np.ndarray:
        return eta_mhz * self.mhz_to_rad_per_ns * self.coeff_diff_matrix


class TSWTLeakageEstimator:
    """
    Numerical second-order TSWT proxy consistent with Appendix B.

    We compute S1 and S2 from Eqs. (B13)-(B15) and use the residual
    third-order block-off-diagonal Hamiltonian from Eq. (B11):
        H_od^(3) = [H1, S2] + 1/3 [[H2, S1], S1] - i dS2/dt

    This version keeps the same bound structure but updates it incrementally,
    eliminating the O(T^2) history rescans that dominated rollout time.
    """

    def __init__(self, system: GmonSystem):
        self.system = system
        self.dt_ns = system.dt_ns
        self.reset()

    def reset(self) -> None:
        self.prev_s1: np.ndarray | None = None
        self.prev_s2: np.ndarray | None = None
        self.prev_hod_1: np.ndarray | None = None
        self.prev_hod_2: np.ndarray | None = None
        self.first_hod: np.ndarray | None = None
        self.last_hod: np.ndarray | None = None
        self.first_delta = 0.0
        self.last_delta = 0.0
        self.first_norm = 0.0
        self.last_norm = 0.0
        self.integral_accum = 0.0

    def _comm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B - B @ A

    def _solve_offdiag(self, numerator: np.ndarray, eta_mhz: float) -> np.ndarray:
        denom = self.system.tswt_denominator_matrix(eta_mhz)
        out = np.zeros_like(numerator)
        np.divide(numerator, denom, out=out, where=self.system.denom_mask)
        return self.system.offdiag(out)

    def _compute_s1(self, h2: np.ndarray, eta_mhz: float) -> np.ndarray:
        return self._solve_offdiag(h2, eta_mhz)

    def _compute_s2(self, h1: np.ndarray, s1: np.ndarray, ds1_dt: np.ndarray, eta_mhz: float) -> np.ndarray:
        rhs = self.system.offdiag(self._comm(h1, s1) - 1j * ds1_dt)
        return self._solve_offdiag(rhs, eta_mhz)

    def step(self, H: np.ndarray, eta_mhz: float) -> np.ndarray:
        _, h1, h2, _ = self.system.decompose_hamiltonian(H, eta_mhz)
        s1 = self._compute_s1(h2, eta_mhz)
        ds1_dt = np.zeros_like(s1) if self.prev_s1 is None else (s1 - self.prev_s1) / self.dt_ns
        s2 = self._compute_s2(h1, s1, ds1_dt, eta_mhz)
        ds2_dt = np.zeros_like(s2) if self.prev_s2 is None else (s2 - self.prev_s2) / self.dt_ns

        residual = self._comm(h1, s2) + (1.0 / 3.0) * self._comm(self._comm(h2, s1), s1) - 1j * ds2_dt
        hod_eff = self.system.offdiag(0.5 * (residual + residual.conj().T))
        delta_rad = abs(eta_mhz) * self.system.mhz_to_rad_per_ns
        current_norm = self.spectral_norm(hod_eff)

        if self.first_hod is None:
            self.first_hod = hod_eff
            self.first_delta = float(delta_rad)
            self.first_norm = float(current_norm)
        if self.prev_hod_2 is not None and self.prev_hod_1 is not None:
            d2 = (hod_eff - 2.0 * self.prev_hod_1 + self.prev_hod_2) / (self.dt_ns ** 2)
            self.integral_accum += self.spectral_norm(d2) * self.dt_ns / max(delta_rad ** 2, 1e-12)

        self.prev_s1 = s1
        self.prev_s2 = s2
        self.prev_hod_2 = self.prev_hod_1
        self.prev_hod_1 = hod_eff
        self.last_hod = hod_eff
        self.last_delta = float(delta_rad)
        self.last_norm = float(current_norm)
        return hod_eff

    @staticmethod
    def spectral_norm(M: np.ndarray) -> float:
        if M.size == 0:
            return 0.0
        vals = np.linalg.svd(M, compute_uv=False)
        return float(vals[0]) if vals.size else 0.0

    def current_leakage_bound(self) -> Tuple[float, Dict[str, float]]:
        if self.first_hod is None or self.last_hod is None:
            return 0.0, {"boundary": 0.0, "integral": 0.0}
        boundary = self.first_norm / max(self.first_delta, 1e-12) + self.last_norm / max(self.last_delta, 1e-12)
        total = boundary + self.integral_accum
        return float(total), {"boundary": float(boundary), "integral": float(self.integral_accum)}


def evaluate_average_fidelity(
    system: GmonSystem,
    target_gate: np.ndarray,
    noisy_projected_unitaries: Sequence[np.ndarray],
) -> float:
    r"""
    Evaluate the paper's Appendix D estimator from sampled noisy realizations.

    Eq. (D2):
        F_avg(E, U) = [sum_j Tr(U U_j^dagger U^dagger E(U_j)) + d^2] / [d^2 (d+1)]
    with d = 4 and E approximated by sample averaging over noisy projected unitaries.
    """
    d = target_gate.shape[0]
    paulis = system.pauli_basis_two_qubit()

    def channel_of(op: np.ndarray) -> np.ndarray:
        accum = np.zeros((d, d), dtype=np.complex128)
        for K in noisy_projected_unitaries:
            accum += K @ op @ K.conj().T
        return accum / max(len(noisy_projected_unitaries), 1)

    total = 0.0 + 0.0j
    for Uj in paulis:
        total += np.trace(target_gate @ Uj.conj().T @ target_gate.conj().T @ channel_of(Uj))
    favg = (total + d ** 2) / (d ** 2 * (d + 1))
    return float(np.real_if_close(favg))
