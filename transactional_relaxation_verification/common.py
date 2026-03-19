import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, norm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]


def ket0() -> np.ndarray:
    return np.array([[1.0], [0.0]], dtype=complex)


def ket1() -> np.ndarray:
    return np.array([[0.0], [1.0]], dtype=complex)


def dm(ket: np.ndarray) -> np.ndarray:
    return ket @ ket.conj().T


def bloch_state(theta: float, phi: float) -> np.ndarray:
    ket = np.array(
        [[np.cos(theta / 2.0)], [np.exp(1j * phi) * np.sin(theta / 2.0)]],
        dtype=complex,
    )
    return dm(ket)


def sample_pure_qubit_states(n_theta: int = 61, n_phi: int = 121):
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    states = []
    for theta in thetas:
        for phi in phis:
            states.append(bloch_state(theta, phi))
    return states


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = rho - sigma
    evals = np.linalg.eigvalsh((delta + delta.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def kron(*ops: ArrayLike) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def partial_trace_two_qubits(rho_ab: np.ndarray, keep: str = "A") -> np.ndarray:
    tensor = rho_ab.reshape(2, 2, 2, 2)
    if keep.upper() == "A":
        return np.einsum("abcb->ac", tensor)
    if keep.upper() == "B":
        return np.einsum("abac->bc", tensor)
    raise ValueError("keep must be 'A' or 'B'")


def swap_operator() -> np.ndarray:
    basis = np.eye(4, dtype=complex)
    swap = np.zeros((4, 4), dtype=complex)
    mapping = [0, 2, 1, 3]
    for i, j in enumerate(mapping):
        swap[j, i] = 1.0
    return swap


def partial_swap(theta: float) -> np.ndarray:
    S = swap_operator()
    return np.cos(theta) * np.eye(4, dtype=complex) - 1j * np.sin(theta) * S


def channel_from_unitary_and_env_state(U: np.ndarray, mu: np.ndarray):
    def channel(rho: np.ndarray) -> np.ndarray:
        joint = kron(rho, mu)
        out = U @ joint @ U.conj().T
        return partial_trace_two_qubits(out, keep="A")
    return channel


def qubit_channel_delta(channel, n_theta: int = 61, n_phi: int = 121) -> float:
    best = 0.0
    for rho in sample_pure_qubit_states(n_theta=n_theta, n_phi=n_phi):
        val = trace_distance(channel(rho), rho)
        if val > best:
            best = val
    return best


def pauli_transfer_matrix(channel) -> np.ndarray:
    R = np.zeros((4, 4), dtype=float)
    for i, Pi in enumerate(PAULIS):
        for j, Pj in enumerate(PAULIS):
            image = channel(Pj)
            R[i, j] = 0.5 * np.real(np.trace(Pi @ image))
    return R


def affine_from_channel(channel):
    R = pauli_transfer_matrix(channel)
    T = R[1:, 1:]
    t = R[1:, 0]
    return T, t, R


def pauli_semigroup_effective_rates(lindblad_rates_xyz: np.ndarray) -> np.ndarray:
    gx, gy, gz = lindblad_rates_xyz
    return np.array([
        2.0 * (gy + gz),
        2.0 * (gx + gz),
        2.0 * (gx + gy),
    ])


def pauli_one_step_eigenvalues(gamma_xyz: np.ndarray, dt: float) -> np.ndarray:
    G = pauli_semigroup_effective_rates(gamma_xyz)
    return np.exp(-G * dt)


def operator_norm(a: np.ndarray) -> float:
    return float(norm(a, 2))


def make_random_two_qubit_hamiltonian(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (mat + mat.conj().T) / 2.0
    H /= max(operator_norm(H), 1e-12)
    return H


def unitary_from_hamiltonian(H: np.ndarray, eps: float) -> np.ndarray:
    return expm(-1j * eps * H)


def identity_dilation_channel():
    mu = dm(ket0())
    U = np.eye(4, dtype=complex)
    return channel_from_unitary_and_env_state(U, mu), U, mu
