"""ED-3: Sensitivity of the single-step spectral description under
homogeneity violation.

Editor question: Theorem 3.3 gives an exact spectral shadow of the
Markovian bound under homogeneity (C1). Quantify how fast this spectral
description breaks down when (C1) is slightly violated, i.e. when
    Phi_n = Phi + eps_n
with a small drift. Identify the error threshold at which the
single-step spectral data become misleading compared to the
path-functional approach.

We implement the drift in TWO independent ways, to show the conclusion
is not an artefact of one particular perturbation model:

  Method A (Hamiltonian-kick / dilation-level):
    Perturb the dilating unitary at each step,
        U_n = U @ exp(-i eps H_n),
    with H_n a fresh random two-qubit Hamiltonian. The reduced channel
    Phi_n is then CPTP by construction (physical perturbation).

  Method B (PTM-level / channel-level):
    Perturb the affine (Pauli transfer) representation of the channel
    directly,
        T_n = T + eps G_n,   t_n = t + eps g_n,
    with G_n, g_n random. This is an abstract perturbation that need not
    be exactly CPTP; we monitor it as a complementary stress test.

Diagnostic:
  The "single-step spectral data" of Theorem 3.3 fixes the relaxation
  structure from a SINGLE propagator Phi. Under homogeneity this yields
  the homogeneous estimate
      D_spec(N) = N * delta(Phi),
  i.e. the prediction obtained by assuming every transaction is the same
  base channel Phi. The drift-aware path functional is
      C_N = sum_n delta(Phi_n),
  which uses the actual (drifting) sequence. The single-step spectral
  prediction becomes misleading precisely when D_spec departs from C_N;
  we track the relative departure
      |C_N - D_spec| / C_N
  as a function of the drift magnitude eps.

We find this departure vanishes at eps = 0 (homogeneity exact, machine
precision) and grows linearly, O(eps), for small eps, in both
perturbation models.

Outputs:
- outputs/homogeneity_violation_sensitivity.png
- outputs/homogeneity_violation_sensitivity.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_paper_style, label_panels
from common import (
    I2, X, Y, Z, PAULIS,
    ket0, ket1, dm,
    partial_swap,
    channel_from_unitary_and_env_state,
    qubit_channel_delta,
    trace_distance,
    partial_trace_two_qubits,
    kron,
    pauli_transfer_matrix,
    affine_from_channel,
    make_random_two_qubit_hamiltonian,
    unitary_from_hamiltonian,
    operator_norm,
)

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Channel-displacement helpers
# ----------------------------------------------------------------------
def delta_from_affine(T: np.ndarray, t: np.ndarray,
                      n_theta: int = 41, n_phi: int = 81) -> float:
    """Channel displacement of a qubit affine map (T, t) by Bloch sampling.

    A qubit state with Bloch vector r maps to r' = T r + t. The trace
    distance between two qubit states equals half the Euclidean distance
    between their Bloch vectors, so D = 0.5 * ||r' - r||.
    """
    best = 0.0
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    for th in thetas:
        for ph in phis:
            r = np.array([
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th),
            ])
            r_out = T @ r + t
            val = 0.5 * float(np.linalg.norm(r_out - r))
            if val > best:
                best = val
    return best


def affine_apply(T: np.ndarray, t: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply affine qubit map to a density matrix, return density matrix."""
    r = np.array([
        np.real(np.trace(X @ rho)),
        np.real(np.trace(Y @ rho)),
        np.real(np.trace(Z @ rho)),
    ])
    r_out = T @ r + t
    return 0.5 * (I2 + r_out[0] * X + r_out[1] * Y + r_out[2] * Z)


# ----------------------------------------------------------------------
# Method A: Hamiltonian-kick (dilation-level) drift
# ----------------------------------------------------------------------
def run_method_A(eps: float, n_steps: int, theta: float,
                 rho0: np.ndarray, mu: np.ndarray, seed: int):
    """Phi_n = reduced channel of U @ exp(-i eps H_n), H_n fresh random."""
    rng = np.random.default_rng(seed)
    U_base = partial_swap(theta)

    # Homogeneous reference: single base channel
    base_channel = channel_from_unitary_and_env_state(U_base, mu)
    delta_base = qubit_channel_delta(base_channel, n_theta=41, n_phi=81)

    rho = rho0.copy()
    deltas = []
    for _ in range(n_steps):
        H = make_random_two_qubit_hamiltonian(seed=int(rng.integers(0, 2**31)))
        U = U_base @ unitary_from_hamiltonian(H, eps)
        ch = channel_from_unitary_and_env_state(U, mu)
        d = qubit_channel_delta(ch, n_theta=41, n_phi=81)
        deltas.append(d)
        rho = ch(rho)

    C_N = float(np.sum(deltas))
    D_spec = n_steps * delta_base
    return C_N, D_spec, delta_base


# ----------------------------------------------------------------------
# Method B: PTM-level (channel-level) drift
# ----------------------------------------------------------------------
def run_method_B(eps: float, n_steps: int, theta: float,
                 rho0: np.ndarray, mu: np.ndarray, seed: int):
    """T_n = T + eps G_n, t_n = t + eps g_n, G_n/g_n random Gaussian."""
    rng = np.random.default_rng(seed)
    U_base = partial_swap(theta)
    base_channel = channel_from_unitary_and_env_state(U_base, mu)
    T0, t0, _ = affine_from_channel(base_channel)
    delta_base = delta_from_affine(T0, t0)

    deltas = []
    for _ in range(n_steps):
        G = rng.normal(size=(3, 3))
        g = rng.normal(size=3)
        G /= max(operator_norm(G), 1e-12)
        g /= max(np.linalg.norm(g), 1e-12)
        T = T0 + eps * G
        t = t0 + eps * g
        d = delta_from_affine(T, t)
        deltas.append(d)

    C_N = float(np.sum(deltas))
    D_spec = n_steps * delta_base
    return C_N, D_spec, delta_base


def main():
    n_steps = 30
    theta = 0.22
    rho0 = dm((ket0() + ket1()) / np.sqrt(2.0))
    mu = dm(ket0())

    eps_values = np.linspace(0.0, 0.40, 17)

    relerr_A = []
    relerr_B = []
    absdiff_A = []
    absdiff_B = []

    for eps in eps_values:
        C_A, Dspec_A, _ = run_method_A(eps, n_steps, theta, rho0, mu, seed=101)
        C_B, Dspec_B, _ = run_method_B(eps, n_steps, theta, rho0, mu, seed=202)

        # relative deviation of the homogeneous spectral estimate from C_N
        rA = abs(C_A - Dspec_A) / max(C_A, 1e-12)
        rB = abs(C_B - Dspec_B) / max(C_B, 1e-12)
        relerr_A.append(rA)
        relerr_B.append(rB)
        absdiff_A.append(abs(C_A - Dspec_A))
        absdiff_B.append(abs(C_B - Dspec_B))

    relerr_A = np.array(relerr_A)
    relerr_B = np.array(relerr_B)

    # linear fit on small-eps region (first 6 points, excluding eps=0)
    fitA = np.polyfit(eps_values[1:7], relerr_A[1:7], deg=1)
    fitB = np.polyfit(eps_values[1:7], relerr_B[1:7], deg=1)

    print("=== ED-3: Homogeneity violation sensitivity ===")
    print(f"n_steps={n_steps}, theta={theta}")
    print(f"\n[Method A: Hamiltonian-kick / dilation-level]")
    print(f"  rel err at eps=0.00: {relerr_A[0]:.4e}")
    print(f"  rel err at eps=0.10: {relerr_A[4]:.4f}")
    print(f"  rel err at eps=0.20: {relerr_A[8]:.4f}")
    print(f"  rel err at eps=0.40: {relerr_A[-1]:.4f}")
    print(f"  small-eps linear slope: {fitA[0]:.4f}")
    print(f"\n[Method B: PTM-level / channel-level]")
    print(f"  rel err at eps=0.00: {relerr_B[0]:.4e}")
    print(f"  rel err at eps=0.10: {relerr_B[4]:.4f}")
    print(f"  rel err at eps=0.20: {relerr_B[8]:.4f}")
    print(f"  rel err at eps=0.40: {relerr_B[-1]:.4f}")
    print(f"  small-eps linear slope: {fitB[0]:.4f}")

    # Figure
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(eps_values, relerr_A, marker="o", label="Method A (Hamiltonian-kick)")
    ax.plot(eps_values, relerr_B, marker="s", label="Method B (PTM-level)")
    ax.set_title(r"Spectral-estimate relative error vs drift $\varepsilon$")
    ax.set_xlabel(r"drift magnitude $\varepsilon$")
    ax.set_ylabel(r"$|\mathcal{C}_N - N\,\delta(\Phi)| / \mathcal{C}_N$")
    ax.legend(frameon=False)

    ax = axes[1]
    # log-log to expose the O(eps) scaling at small eps
    mask = eps_values > 0
    ax.loglog(eps_values[mask], relerr_A[mask], marker="o", label="Method A")
    ax.loglog(eps_values[mask], relerr_B[mask], marker="s", label="Method B")
    # reference O(eps) line
    ref = eps_values[mask]
    ax.loglog(ref, ref * (relerr_A[mask][0] / ref[0]), linestyle=":",
              color="gray", label=r"$O(\varepsilon)$ reference")
    ax.set_title(r"Small-$\varepsilon$ scaling (log-log)")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("relative error")
    ax.legend(frameon=False)

    label_panels(axes)
    fig.tight_layout()
    fig.savefig(OUTDIR / "homogeneity_violation_sensitivity.png",
                dpi=180, bbox_inches="tight")

    data = np.column_stack([eps_values, relerr_A, relerr_B,
                            np.array(absdiff_A), np.array(absdiff_B)])
    header = "epsilon,relerr_A_Hkick,relerr_B_PTM,absdiff_A,absdiff_B"
    np.savetxt(OUTDIR / "homogeneity_violation_sensitivity.csv", data,
               delimiter=",", header=header, comments="")
    print(f"\nSaved figure to: {OUTDIR / 'homogeneity_violation_sensitivity.png'}")


if __name__ == "__main__":
    main()
