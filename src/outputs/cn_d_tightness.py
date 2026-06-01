"""ED-1: Tightness analysis of the cumulative path functional C_N.

Editor question: C_N grows linearly with N via the triangle inequality,
while the actual deviation D(rho_N, rho_0) may saturate or oscillate.
Investigate the overestimation ratio C_N / D and identify under which
trajectory geometries the bound becomes tight.

This is an EXPLORATORY script: we do not assume a target behaviour in
advance. We probe three qualitatively different trajectory types and
observe the C_N / D(rho_N, rho_0) ratio as a function of N.

Trajectory types (labels describe observed behaviour; the control
parameter is the partial-swap angle theta, reported alongside):
  (1) slow-rotation      (theta=0.10): weak angle -> slow Bloch rotation
  (2) early-saturation   (theta=0.85): strong angle -> fast saturation of D
  (3) memory-oscillation (theta=0.22): hidden-memory collision model

Outputs:
- outputs/cn_d_tightness.png
- outputs/cn_d_tightness.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_paper_style, label_panels
from common import (
    ket0,
    ket1,
    dm,
    partial_swap,
    channel_from_unitary_and_env_state,
    qubit_channel_delta,
    trace_distance,
    partial_trace_two_qubits,
    kron,
)

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def step_memoryful(rho, mu, U):
    joint = kron(rho, mu)
    out = U @ joint @ U.conj().T
    rho_next = partial_trace_two_qubits(out, keep="A")
    mu_next = partial_trace_two_qubits(out, keep="B")
    return rho_next, mu_next


def step_memoryless(rho, mu_reset, U):
    joint = kron(rho, mu_reset)
    out = U @ joint @ U.conj().T
    return partial_trace_two_qubits(out, keep="A")


def run_homogeneous(theta, n_steps, rho0, mu_reset):
    """Homogeneous (memoryless) trajectory at fixed partial-swap angle."""
    U = partial_swap(theta)
    fixed_channel = channel_from_unitary_and_env_state(U, mu_reset)
    fixed_delta = qubit_channel_delta(fixed_channel, n_theta=51, n_phi=101)

    rho = rho0.copy()
    distances = [0.0]
    deltas = []
    for _ in range(n_steps):
        rho = step_memoryless(rho, mu_reset, U)
        deltas.append(fixed_delta)
        distances.append(trace_distance(rho, rho0))
    C = np.concatenate([[0.0], np.cumsum(deltas)])
    return np.array(distances), C


def run_memoryful(theta, n_steps, rho0, mu0):
    """Memory-bearing trajectory: hidden memory carried along."""
    U = partial_swap(theta)
    rho = rho0.copy()
    mu = mu0.copy()
    distances = [0.0]
    deltas = []
    for _ in range(n_steps):
        current_channel = channel_from_unitary_and_env_state(U, mu)
        current_delta = qubit_channel_delta(current_channel, n_theta=51, n_phi=101)
        rho, mu = step_memoryful(rho, mu, U)
        deltas.append(current_delta)
        distances.append(trace_distance(rho, rho0))
    C = np.concatenate([[0.0], np.cumsum(deltas)])
    return np.array(distances), C


def main():
    n_steps = 60
    rho0 = dm((ket0() + ket1()) / np.sqrt(2.0))
    mu_reset = dm(ket0())
    mu_memoryful = dm(ket1())

    # Probe three trajectory geometries. Labels are descriptive of the
    # OBSERVED behaviour and are a convenience only; the underlying control
    # parameter is the partial-swap angle theta, reported alongside.
    theta_slow = 0.10     # slow-rotation
    theta_early = 0.85    # early-saturation
    theta_mem = 0.22      # memory-oscillation (same as main example)

    D_slow, C_slow = run_homogeneous(theta_slow, n_steps, rho0, mu_reset)
    D_early, C_early = run_homogeneous(theta_early, n_steps, rho0, mu_reset)
    D_mem, C_mem = run_memoryful(theta_mem, n_steps, rho0, mu_memoryful)

    steps = np.arange(n_steps + 1)

    def ratio(C, D):
        return C / np.maximum(D, 1e-12)

    R_slow = ratio(C_slow, D_slow)
    R_early = ratio(C_early, D_early)
    R_mem = ratio(C_mem, D_mem)

    print("=== ED-1: C_N / D tightness exploration ===")
    for name, D, C, R in [
        ("slow-rotation     (theta=0.10)", D_slow, C_slow, R_slow),
        ("early-saturation  (theta=0.85)", D_early, C_early, R_early),
        ("memory-oscillation(theta=0.22)", D_mem, C_mem, R_mem),
    ]:
        print(f"\n[{name}]")
        print(f"  D range:   [{D[1:].min():.4f}, {D[1:].max():.4f}]")
        print(f"  C_N final: {C[-1]:.4f}")
        print(f"  D final:   {D[-1]:.4f}")
        print(f"  ratio C_N/D at N=10:  {R[10]:.3f}")
        print(f"  ratio C_N/D at N=30:  {R[30]:.3f}")
        print(f"  ratio C_N/D at N=60:  {R[60]:.3f}")

    # Figure
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(steps, D_slow, label=r"slow-rotation ($\theta=0.10$)")
    ax.plot(steps, D_early, label=r"early-saturation ($\theta=0.85$)")
    ax.plot(steps, D_mem, label=r"memory-oscillation ($\theta=0.22$)")
    ax.set_title(r"Final deviation $D(\rho_N,\rho_0)$")
    ax.set_xlabel("N")
    ax.set_ylabel("D")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(steps[1:], R_slow[1:], label=r"slow-rotation ($\theta=0.10$)")
    ax.plot(steps[1:], R_early[1:], label=r"early-saturation ($\theta=0.85$)")
    ax.plot(steps[1:], R_mem[1:], label=r"memory-oscillation ($\theta=0.22$)")
    ax.set_title(r"Overestimation ratio $\mathcal{C}_N / D(\rho_N,\rho_0)$")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\mathcal{C}_N / D$")
    ax.legend(frameon=False)

    label_panels(axes)
    fig.tight_layout()
    fig.savefig(OUTDIR / "cn_d_tightness.png", dpi=180, bbox_inches="tight")

    data = np.column_stack([steps, D_slow, C_slow, R_slow,
                            D_early, C_early, R_early,
                            D_mem, C_mem, R_mem])
    header = ("N,D_slow,C_slow,ratio_slow,"
              "D_early,C_early,ratio_early,"
              "D_mem,C_mem,ratio_mem")
    np.savetxt(OUTDIR / "cn_d_tightness.csv", data, delimiter=",",
               header=header, comments="")
    print(f"\nSaved figure to: {OUTDIR / 'cn_d_tightness.png'}")


if __name__ == "__main__":
    main()
