# Transactional Relaxation Verification Codes

This bundle contains three minimal Python scripts supporting the draft paper:

1. `markovian_calibration.py`
   - Homogeneous qubit Pauli semigroups
   - Checks the qubit version of the Muratore-Kimura-Chruscinski (MKC) spectral bound
   - Demonstrates why exact discrete recovery should use
     `Gamma_l = -nu * log|eta_l|`
     rather than the linearized approximation `Gamma_l ~ nu * (1 - eta_l)`

2. `collision_model_path_dependence.py`
   - Minimal hidden-memory collision model using a partial-swap interaction
   - Compares memoryless vs memory-bearing transactional dynamics
   - Computes one-step displacement `delta(Phi_n)`, cumulative path functional `C_N`,
     and a simple local propagator variation proxy

3. `continuity_sanity_check.py`
   - Perturbs the identity dilation by a small Hamiltonian kick
   - Compares `delta(Phi_eps)` against `||U_eps - I||`
   - Supports the continuity-bound discussion for the appendix

## Dependencies

Only standard scientific Python packages are required:

- `numpy`
- `scipy`
- `matplotlib`

## Run

From this directory:

```bash
python markovian_calibration.py
python collision_model_path_dependence.py
python continuity_sanity_check.py
```

All figures and CSV files are written to `outputs/`.

## Notes

- The scripts are intentionally minimal and transparent rather than optimized.
- `delta(Phi)` is estimated by dense pure-state sampling on the Bloch sphere.
  For the paper, this is sufficient as a numerical sanity check; if desired,
  it can later be upgraded to exact optimization or SDP-based norms.
- The collision-model script implements the hidden-memory update rule directly,
  avoiding heavier process-tensor reconstruction machinery.
