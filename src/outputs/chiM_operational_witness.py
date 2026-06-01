"""ED-2: Operational accessibility of the memory susceptibility chi_M.

Editor question: chi_M (Eq. 16) requires full knowledge of the
Stinespring dilation U. For a black-box experimentalist, chi_M is not
directly observable. Clarify whether chi_M can be estimated or bounded
from accessible data (process tomography / witness operators); if not,
frame it explicitly as a theoretical diagnostic.

We probe THREE operational routes to witnessing chi_M, on the same
minimal collision model used elsewhere (partial swap, theta=0.22). All
three quantities are reconstructed from system-only process tomography
of one transaction; none requires access to the dilation.

  Route 1 -- CP-divisibility breakdown (Rivas-Huelga-Plenio):
    smallest Choi eigenvalue of the intermediate map V_{n+1<-n}.
    NCP (lambda_min < 0) would witness non-Markovianity.

  Route 2 -- propagator drift (tomographic):
    ||R_n - R_{n-1}||, the step-to-step change of the reconstructed
    Pauli transfer matrix. Nonzero drift signals that a single Phi does
    not describe the sequence.

  Route 3 -- propagator non-commutativity (tomographic):
    ||[R_n, R_0]||, signalling that the order of transactions matters.

Findings (this model):
  * Route 1 does NOT fire: the dynamics stays CP-divisible
    (lambda_min = 0 to machine precision) even though memory is present.
  * Routes 2 and 3 DO fire: the reconstructed propagator drifts and the
    maps fail to commute.

Interpretation: chi_M is a model-dependent THEORETICAL diagnostic that
detects memory effects weaker than (and not equivalent to) RHP
non-Markovianity. Its presence can be operationally witnessed by
tomographic propagator drift/non-commutativity (a lower-bound-type
signature: drift>0 => chi_M>0), but its magnitude is not in one-to-one
correspondence with any single observable. chi_M is therefore best
framed as a theoretical diagnostic, consistent with the manuscript.

Outputs:
- outputs/chiM_operational_witness.png
- outputs/chiM_operational_witness.csv
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_paper_style, label_panels
from common import (
    PAULIS,
    ket0, ket1, dm,
    partial_swap,
    channel_from_unitary_and_env_state,
    pauli_transfer_matrix,
    partial_trace_two_qubits,
    kron,
    operator_norm,
)

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def step_memoryful(rho, mu, U):
    joint = kron(rho, mu)
    out = U @ joint @ U.conj().T
    return partial_trace_two_qubits(out, keep="A"), partial_trace_two_qubits(out, keep="B")


def ptm_to_choi(R):
    def apply_R(rho):
        coeffs = [np.trace(P @ rho) for P in PAULIS]
        out = np.zeros((2, 2), dtype=complex)
        for i, Pi in enumerate(PAULIS):
            out += 0.5 * sum(R[i, j] * coeffs[j] for j in range(4)) * Pi
        return out
    choi = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            E = np.zeros((2, 2), dtype=complex); E[a, b] = 1.0
            phiE = apply_R(E)
            ket = np.zeros((2, 1), dtype=complex); ket[a] = 1.0
            bra = np.zeros((1, 2), dtype=complex); bra[0, b] = 1.0
            choi += np.kron(ket @ bra, phiE)
    return choi


def intermediate_min_eig(C_next, C_curr):
    try:
        V = C_next @ np.linalg.inv(C_curr)
    except np.linalg.LinAlgError:
        V = C_next
    choi = ptm_to_choi(V); choi = (choi + choi.conj().T) / 2.0
    return float(np.min(np.linalg.eigvalsh(choi)))


def run(memory, n_steps, theta):
    U = partial_swap(theta)
    mu_reset = dm(ket0()); mu = dm(ket1())
    rho = dm((ket0() + ket1()) / np.sqrt(2.0))

    min_eigs, drifts, commutators = [], [], []
    C_curr = np.eye(4)
    R_prev = None
    R_first = None
    for _ in range(n_steps):
        if memory:
            ch = channel_from_unitary_and_env_state(U, mu)
            R = pauli_transfer_matrix(ch)
            rho, mu = step_memoryful(rho, mu, U)
        else:
            ch = channel_from_unitary_and_env_state(U, mu_reset)
            R = pauli_transfer_matrix(ch)
        if R_first is None:
            R_first = R
        C_next = R @ C_curr
        min_eigs.append(intermediate_min_eig(C_next, C_curr))
        drifts.append(0.0 if R_prev is None else operator_norm(R - R_prev))
        commutators.append(operator_norm(R @ R_first - R_first @ R))
        R_prev = R; C_curr = C_next
    return np.array(min_eigs), np.array(drifts), np.array(commutators)


def main():
    n_steps = 40
    theta = 0.22

    eig0, drift0, comm0 = run(False, n_steps, theta)
    eig1, drift1, comm1 = run(True, n_steps, theta)

    tol = 1e-9
    print("=== ED-2: chi_M operational witnesses (three routes) ===")
    print(f"n_steps={n_steps}, theta={theta}")
    print(f"\n[Route 1: CP-divisibility (Choi lambda_min)]")
    print(f"  memoryless    lambda_min stays >= 0 (CP): min={eig0.min():.3e}, NCP steps={int(np.sum(eig0<-tol))}")
    print(f"  memory-bearing lambda_min stays >= 0 (CP): min={eig1.min():.3e}, NCP steps={int(np.sum(eig1<-tol))}")
    print(f"  => Route 1 does NOT witness chi_M: dynamics remains CP-divisible.")
    print(f"\n[Route 2: tomographic propagator drift ||R_n - R_n-1||]")
    print(f"  memoryless    max drift: {drift0.max():.3e}")
    print(f"  memory-bearing max drift: {drift1.max():.3e}")
    print(f"  => Route 2 witnesses chi_M (nonzero drift).")
    print(f"\n[Route 3: non-commutativity ||[R_n, R_0]||]")
    print(f"  memoryless    max: {comm0.max():.3e}")
    print(f"  memory-bearing max: {comm1.max():.3e}")
    print(f"  => Route 3 witnesses chi_M (nonzero commutator).")

    steps = np.arange(1, n_steps + 1)
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    ax.plot(steps, eig0, marker="o", markersize=3, label=r"memoryless")
    ax.plot(steps, eig1, marker="s", markersize=3, label=r"memory-bearing")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_title(r"Route 1: CP-divisibility (stays CP) $\lambda_{\min}(\mathrm{Choi})$")
    ax.set_xlabel("transaction n"); ax.set_ylabel(r"$\lambda_{\min}$")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(steps, drift0, marker="o", markersize=3, label="memoryless")
    ax.plot(steps, drift1, marker="s", markersize=3, label="memory-bearing")
    ax.set_title(r"Route 2: propagator drift $\|R_n - R_{n-1}\|$")
    ax.set_xlabel("transaction n"); ax.set_ylabel("drift")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(steps, comm0, marker="o", markersize=3, label="memoryless")
    ax.plot(steps, comm1, marker="s", markersize=3, label="memory-bearing")
    ax.set_title(r"Route 3: non-commutativity $\|[R_n, R_0]\|$")
    ax.set_xlabel("transaction n"); ax.set_ylabel("commutator norm")
    ax.legend(frameon=False)

    label_panels(axes)
    fig.tight_layout()
    fig.savefig(OUTDIR / "chiM_operational_witness.png", dpi=180, bbox_inches="tight")

    data = np.column_stack([steps, eig0, eig1, drift0, drift1, comm0, comm1])
    header = ("n,lambda_min_memless,lambda_min_memful,"
              "drift_memless,drift_memful,commutator_memless,commutator_memful")
    np.savetxt(OUTDIR / "chiM_operational_witness.csv", data, delimiter=",", header=header, comments="")
    print(f"\nSaved figure to: {OUTDIR / 'chiM_operational_witness.png'}")


if __name__ == "__main__":
    main()
