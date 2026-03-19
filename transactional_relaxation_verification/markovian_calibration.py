"""Markovian calibration for the transactional relaxation paper.

This script demonstrates three points for homogeneous CP-divisible Pauli semigroups:
1. The continuous-time MKC bound on relaxation rates holds for Pauli semigroups.
2. The exact discrete-time recovery uses eta_l = exp(-Gamma_l / nu), hence
      Gamma_l = -nu * log |eta_l|,
   rather than the linearized approximation Gamma_l \approx nu * (1 - eta_l).
3. The discrete geometric-mean form of the MKC bound agrees exactly with the
   continuous-time bound for one-step propagators.

Outputs:
- outputs/markovian_calibration.png
- outputs/markovian_calibration_summary.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import pauli_semigroup_effective_rates, pauli_one_step_eigenvalues

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def generate_random_pauli_semigroup_data(n_samples: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0.02, 0.6, size=(n_samples, 3))
    rates = np.array([pauli_semigroup_effective_rates(g) for g in gammas])
    return gammas, rates


def mkc_gap_for_qubit(rates: np.ndarray) -> np.ndarray:
    lhs = np.max(rates, axis=1)
    rhs = 0.5 * np.sum(rates, axis=1)
    return rhs - lhs


def exact_and_linearized_recovery_errors(rates: np.ndarray, dt_values: np.ndarray):
    exact_errors = []
    linear_errors = []
    for dt in dt_values:
        nu = 1.0 / dt
        eta = np.exp(-rates * dt)
        exact = -nu * np.log(np.clip(np.abs(eta), 1e-15, 1.0))
        linear = nu * (1.0 - eta)
        exact_rel_err = np.max(np.abs(exact - rates) / np.maximum(rates, 1e-15), axis=1)
        linear_rel_err = np.max(np.abs(linear - rates) / np.maximum(rates, 1e-15), axis=1)
        exact_errors.append(float(np.mean(exact_rel_err)))
        linear_errors.append(float(np.mean(linear_rel_err)))
    return np.array(exact_errors), np.array(linear_errors)


def discrete_mkc_gap(rates: np.ndarray, dt: float) -> np.ndarray:
    eta = np.exp(-rates * dt)
    lhs = np.max(-np.log(np.clip(np.abs(eta), 1e-15, 1.0)), axis=1)
    rhs = 0.5 * np.sum(-np.log(np.clip(np.abs(eta), 1e-15, 1.0)), axis=1)
    return rhs - lhs


def save_summary_csv(rates: np.ndarray, path: Path):
    header = "Gamma_x,Gamma_y,Gamma_z,mkc_gap"
    gaps = mkc_gap_for_qubit(rates)
    data = np.column_stack([rates, gaps])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def make_figure(rates: np.ndarray, dt_values: np.ndarray):
    gaps = mkc_gap_for_qubit(rates)
    exact_errors, linear_errors = exact_and_linearized_recovery_errors(rates, dt_values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.hist(gaps, bins=30)
    ax.set_title("MKC gap for random qubit Pauli semigroups")
    ax.set_xlabel(r"$\frac{1}{2}\sum_\ell \Gamma_\ell - \Gamma_{\max}$")
    ax.set_ylabel("count")

    ax = axes[1]
    ax.plot(dt_values, exact_errors, marker="o", label=r"exact: $-\nu\log|\eta_\ell|$")
    ax.plot(dt_values, linear_errors, marker="s", label=r"linearized: $\nu(1-\eta_\ell)$")
    ax.set_title("Recovery error vs one-step duration")
    ax.set_xlabel(r"$\Delta t = 1/\nu$")
    ax.set_ylabel("mean max relative error")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[2]
    chosen_dt = dt_values[len(dt_values) // 2]
    dgaps = discrete_mkc_gap(rates, chosen_dt)
    ax.hist(dgaps, bins=30)
    ax.set_title(r"Discrete MKC gap using $-\log|\eta_\ell|$")
    ax.set_xlabel(r"$\frac{1}{2}\sum_\ell[-\log|\eta_\ell|] - \max_\ell[-\log|\eta_\ell|]$")
    ax.set_ylabel("count")

    fig.suptitle("Markovian calibration: exact discrete recovery of the spectral bound", y=1.03)
    fig.tight_layout()
    return fig


def main():
    gammas, rates = generate_random_pauli_semigroup_data(n_samples=1000, seed=11)
    dt_values = np.linspace(0.02, 1.0, 12)

    gaps = mkc_gap_for_qubit(rates)
    print("=== Markovian calibration ===")
    print(f"Samples: {len(rates)}")
    print(f"Minimum continuous-time MKC gap: {gaps.min():.6e}")
    print(f"Maximum continuous-time MKC gap: {gaps.max():.6e}")
    print(f"Mean continuous-time MKC gap: {gaps.mean():.6e}")

    fig = make_figure(rates, dt_values)
    fig.savefig(OUTDIR / "markovian_calibration.png", dpi=180, bbox_inches="tight")
    save_summary_csv(rates, OUTDIR / "markovian_calibration_summary.csv")
    print(f"Saved figure to: {OUTDIR / 'markovian_calibration.png'}")
    print(f"Saved summary CSV to: {OUTDIR / 'markovian_calibration_summary.csv'}")


if __name__ == "__main__":
    main()
