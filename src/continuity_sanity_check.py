"""Continuity sanity check for dilation closeness vs channel displacement.

This script perturbs the identity dilation by a small two-qubit Hamiltonian:
    U_eps = exp(-i eps H),
with the environment initialized in |0><0|.
It then constructs the reduced channel Phi_eps and estimates
    delta(Phi_eps) = sup_rho D(Phi_eps(rho), rho)
by dense pure-state sampling on the Bloch sphere.

The point is not to prove a theorem, but to numerically confirm the expected
small-epsilon scaling behind continuity-based bounds.

Outputs:
- outputs/continuity_sanity_check.png
- outputs/continuity_sanity_check.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import (
    dm,
    ket0,
    channel_from_unitary_and_env_state,
    make_random_two_qubit_hamiltonian,
    unitary_from_hamiltonian,
    qubit_channel_delta,
    operator_norm,
)

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def run_experiment(eps_values: np.ndarray, seed: int = 5):
    H = make_random_two_qubit_hamiltonian(seed=seed)
    mu = dm(ket0())
    U0 = np.eye(4, dtype=complex)

    op_norm_diffs = []
    deltas = []

    for eps in eps_values:
        U = unitary_from_hamiltonian(H, eps)
        channel = channel_from_unitary_and_env_state(U, mu)
        delta = qubit_channel_delta(channel, n_theta=61, n_phi=121)
        op_norm_diff = operator_norm(U - U0)
        op_norm_diffs.append(op_norm_diff)
        deltas.append(delta)

    return np.array(op_norm_diffs), np.array(deltas)


def save_csv(eps_values: np.ndarray, op_norm_diffs: np.ndarray, deltas: np.ndarray, path: Path):
    data = np.column_stack([eps_values, op_norm_diffs, deltas, deltas / np.maximum(op_norm_diffs, 1e-15)])
    header = "epsilon,||Ueps-I||,delta(Phi_eps),delta_over_operator_norm"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def make_figure(eps_values: np.ndarray, op_norm_diffs: np.ndarray, deltas: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    ax.plot(op_norm_diffs, deltas, marker="o")
    ax.set_title(r"$\delta(\Phi_\varepsilon)$ vs $\|U_\varepsilon-I\|$")
    ax.set_xlabel(r"$\|U_\varepsilon - I\|$")
    ax.set_ylabel(r"$\delta(\Phi_\varepsilon)$")

    ax = axes[1]
    ax.plot(eps_values, deltas / np.maximum(op_norm_diffs, 1e-15), marker="s")
    ax.set_title("Ratio stability in the small-perturbation regime")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\delta(\Phi_\varepsilon) / \|U_\varepsilon-I\|$")

    fig.suptitle("Continuity sanity check: dilation perturbation vs reduced-channel displacement", y=1.02)
    fig.tight_layout()
    return fig


def main():
    eps_values = np.linspace(0.01, 0.45, 18)
    op_norm_diffs, deltas = run_experiment(eps_values, seed=13)

    coeff = np.polyfit(op_norm_diffs[:8], deltas[:8], deg=1)
    print("=== Continuity sanity check ===")
    print(f"Small-perturbation linear fit slope: {coeff[0]:.6f}")
    print(f"Small-perturbation linear fit intercept: {coeff[1]:.6e}")
    print(f"Max delta / ||U-I|| ratio: {np.max(deltas / op_norm_diffs):.6f}")
    print(f"Min delta / ||U-I|| ratio: {np.min(deltas / op_norm_diffs):.6f}")

    fig = make_figure(eps_values, op_norm_diffs, deltas)
    fig.savefig(OUTDIR / "continuity_sanity_check.png", dpi=180, bbox_inches="tight")
    save_csv(eps_values, op_norm_diffs, deltas, OUTDIR / "continuity_sanity_check.csv")
    print(f"Saved figure to: {OUTDIR / 'continuity_sanity_check.png'}")
    print(f"Saved CSV to: {OUTDIR / 'continuity_sanity_check.csv'}")


if __name__ == "__main__":
    main()
