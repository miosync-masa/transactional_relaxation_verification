"""Memoryless vs memory-bearing collision model.

This script implements a minimal hidden-memory update rule:
    rho_{n+1} = Tr_M[ U (rho_n \otimes mu_n) U^\dagger ]
    mu_{n+1}  = Tr_S[ U (rho_n \otimes mu_n) U^\dagger ]

Two regimes are compared:
- memoryless: mu_n is reset to a fixed state at each step
- memory-bearing: mu_n is updated and carried along the trajectory

Outputs:
- outputs/collision_model_path_dependence.png
- outputs/collision_model_metrics.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import (
    I2,
    ket0,
    ket1,
    dm,
    partial_swap,
    channel_from_unitary_and_env_state,
    qubit_channel_delta,
    trace_distance,
    pauli_transfer_matrix,
    partial_trace_two_qubits,
    kron,
    operator_norm,
)

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def step_memoryful(rho: np.ndarray, mu: np.ndarray, U: np.ndarray):
    joint = kron(rho, mu)
    out = U @ joint @ U.conj().T
    rho_next = partial_trace_two_qubits(out, keep="A")
    mu_next = partial_trace_two_qubits(out, keep="B")
    return rho_next, mu_next


def step_memoryless(rho: np.ndarray, mu_reset: np.ndarray, U: np.ndarray):
    joint = kron(rho, mu_reset)
    out = U @ joint @ U.conj().T
    rho_next = partial_trace_two_qubits(out, keep="A")
    return rho_next


def simulate(n_steps: int = 40, theta: float = 0.22):
    U = partial_swap(theta)
    rho0 = dm((ket0() + ket1()) / np.sqrt(2.0))
    mu_reset = dm(ket0())
    mu_memoryful = dm(ket1())

    rho_memless = rho0.copy()
    rho_memful = rho0.copy()

    deltas_memless = []
    deltas_memful = []
    distances_memless = [0.0]
    distances_memful = [0.0]
    M_local = [0.0]
    mu_purity = [float(np.real(np.trace(mu_memoryful @ mu_memoryful)))]
    channels_memless = []
    channels_memful = []
    Rs_memful = []

    fixed_channel = channel_from_unitary_and_env_state(U, mu_reset)
    fixed_delta = qubit_channel_delta(fixed_channel, n_theta=51, n_phi=101)
    fixed_R = pauli_transfer_matrix(fixed_channel)

    for _ in range(n_steps):
        # Memoryless leg
        rho_memless = step_memoryless(rho_memless, mu_reset, U)
        deltas_memless.append(fixed_delta)
        distances_memless.append(trace_distance(rho_memless, rho0))
        channels_memless.append(fixed_channel)

        # Memoryful leg
        current_channel = channel_from_unitary_and_env_state(U, mu_memoryful)
        current_delta = qubit_channel_delta(current_channel, n_theta=51, n_phi=101)
        current_R = pauli_transfer_matrix(current_channel)
        rho_memful, mu_memoryful = step_memoryful(rho_memful, mu_memoryful, U)

        deltas_memful.append(current_delta)
        distances_memful.append(trace_distance(rho_memful, rho0))
        channels_memful.append(current_channel)
        Rs_memful.append(current_R)
        mu_purity.append(float(np.real(np.trace(mu_memoryful @ mu_memoryful))))

        if len(Rs_memful) == 1:
            M_local.append(operator_norm(Rs_memful[-1] - fixed_R))
        else:
            M_local.append(operator_norm(Rs_memful[-1] - Rs_memful[-2]))

    C_memless = np.cumsum(deltas_memless)
    C_memful = np.cumsum(deltas_memful)

    return {
        "steps": np.arange(n_steps + 1),
        "deltas_memless": np.array(deltas_memless),
        "deltas_memful": np.array(deltas_memful),
        "distances_memless": np.array(distances_memless),
        "distances_memful": np.array(distances_memful),
        "C_memless": np.concatenate([[0.0], C_memless]),
        "C_memful": np.concatenate([[0.0], C_memful]),
        "M_local": np.array(M_local),
        "mu_purity": np.array(mu_purity),
        "fixed_delta": fixed_delta,
    }


def save_csv(results: dict, path: Path):
    data = np.column_stack([
        results["steps"],
        results["C_memless"],
        results["C_memful"],
        results["distances_memless"],
        results["distances_memful"],
        results["M_local"],
        results["mu_purity"],
    ])
    header = "step,C_memless,C_memful,D_memless,D_memful,M_local,mu_purity"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def make_figure(results: dict):
    steps = results["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(steps[1:], results["deltas_memless"], label="memoryless")
    ax.plot(steps[1:], results["deltas_memful"], label="memory-bearing")
    ax.set_title(r"One-transaction displacement $\delta(\Phi_n)$")
    ax.set_xlabel("transaction n")
    ax.set_ylabel(r"$\delta(\Phi_n)$")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(steps, results["C_memless"], label=r"$\mathcal{C}_N$ memoryless")
    ax.plot(steps, results["C_memful"], label=r"$\mathcal{C}_N$ memory-bearing")
    ax.plot(steps, results["distances_memless"], linestyle="--", label=r"$D(\rho_N,\rho_0)$ memoryless")
    ax.plot(steps, results["distances_memful"], linestyle="--", label=r"$D(\rho_N,\rho_0)$ memory-bearing")
    ax.set_title(r"Path functional vs final deviation")
    ax.set_xlabel("transaction N")
    ax.set_ylabel("distance")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(steps, results["M_local"])
    ax.set_title(r"Local propagator variation proxy $\mathcal{M}_N$")
    ax.set_xlabel("transaction n")
    ax.set_ylabel(r"$\|\Phi_n - \Phi_{n-1}\|$")

    ax = axes[1, 1]
    ax.plot(steps, results["mu_purity"])
    ax.set_title("Hidden memory purity")
    ax.set_xlabel("transaction n")
    ax.set_ylabel(r"Tr$(\mu_n^2)$")

    fig.suptitle("Collision model: homogeneous vs memory-bearing transactional dynamics", y=1.02)
    fig.tight_layout()
    return fig


def main():
    results = simulate(n_steps=40, theta=0.22)

    print("=== Collision model path dependence ===")
    print(f"Fixed memoryless delta: {results['fixed_delta']:.6f}")
    print(f"Final memoryless C_N: {results['C_memless'][-1]:.6f}")
    print(f"Final memory-bearing C_N: {results['C_memful'][-1]:.6f}")
    print(f"Final memoryless D(rho_N, rho_0): {results['distances_memless'][-1]:.6f}")
    print(f"Final memory-bearing D(rho_N, rho_0): {results['distances_memful'][-1]:.6f}")
    print(f"Mean local propagator variation proxy: {np.mean(results['M_local'][1:]):.6f}")

    fig = make_figure(results)
    fig.savefig(OUTDIR / "collision_model_path_dependence.png", dpi=180, bbox_inches="tight")
    save_csv(results, OUTDIR / "collision_model_metrics.csv")
    print(f"Saved figure to: {OUTDIR / 'collision_model_path_dependence.png'}")
    print(f"Saved metrics CSV to: {OUTDIR / 'collision_model_metrics.csv'}")


if __name__ == "__main__":
    main()
