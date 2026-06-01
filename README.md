# Transactional Relaxation Verification Codes

This bundle contains the Python scripts supporting the manuscript
"Transactional relaxation constraints for open quantum dynamics:
A discrete framework encompassing the Markovian spectral bound".

The first three scripts reproduce the figures of the original
submission; the last three were added for the revision, addressing the
editor's questions on bound tightness, operational accessibility of the
memory susceptibility, and sensitivity to homogeneity violation.

## Core scripts (original submission)

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
     and a local propagator variation proxy `||R_n - R_{n-1}||`
     (operator norm of the step-to-step change of the Pauli transfer matrix)

3. `continuity_sanity_check.py`
   - Perturbs the identity dilation by a small Hamiltonian kick
   - Compares `delta(Phi_eps)` against `||U_eps - I||`
   - Supports the continuity-bound discussion for the appendix

## Revision scripts (added in response to the editor)

4. `cn_d_tightness.py`  (editor question 1)
   - Explores the overestimation ratio `C_N / D(rho_N, rho_0)` as a
     function of the number of transactions `N`
   - Probes three trajectory geometries (labels are descriptive of the
     observed behaviour; the control parameter is the partial-swap angle
     `theta`, reported alongside):
     slow-rotation (`theta=0.10`), early-saturation (`theta=0.85`),
     and memory-oscillation (`theta=0.22`)
   - Observation: the ratio approaches 1 for small `N` and grows roughly
     linearly once `D` saturates or oscillates

5. `chiM_operational_witness.py`  (editor question 2)
   - Probes three system-only (tomographically accessible) routes to
     witnessing a nonzero memory susceptibility `chi_M`:
       Route 1: CP-divisibility breakdown (smallest Choi eigenvalue of
                the intermediate map) -- does NOT fire (stays CP)
       Route 2: tomographic propagator drift `||R_n - R_{n-1}||` -- fires
       Route 3: propagator non-commutativity `||[R_n, R_0]||` -- fires
   - Supports framing `chi_M` as a model-dependent theoretical diagnostic
     whose presence can be operationally witnessed but whose magnitude is
     not in one-to-one correspondence with a single observable

6. `homogeneity_violation_sensitivity.py`  (editor question 3)
   - Introduces a small drift `Phi_n = Phi + eps_n` and measures the
     relative departure of the homogeneous single-step spectral estimate
     `N * delta(Phi)` from the drift-aware path functional `C_N`
   - Uses two independent perturbation models for robustness:
       Method A: Hamiltonian-kick / dilation-level (`U_n = U exp(-i eps H_n)`)
       Method B: PTM-level / channel-level (`T_n = T + eps G_n`)
   - Observation: the departure vanishes at `eps = 0` (machine precision)
     and grows as `O(eps)` for small `eps` in both models

## Dependencies

Only standard scientific Python packages are required:

- `numpy`
- `scipy`
- `matplotlib`

## Run

From the `src` directory:

```bash
# original-submission figures
python markovian_calibration.py
python collision_model_path_dependence.py
python continuity_sanity_check.py

# revision figures
python cn_d_tightness.py
python chiM_operational_witness.py
python homogeneity_violation_sensitivity.py
```

All figures and CSV files are written to `outputs/`.
Shared figure styling (panel labels (a), (b), ... and enlarged fonts) is
provided by `plot_style.py`.

## Notes

- The scripts are intentionally minimal and transparent rather than optimized.
- `delta(Phi)` is estimated by dense pure-state sampling on the Bloch sphere.
  For the paper, this is sufficient as a numerical sanity check; if desired,
  it can later be upgraded to exact optimization or SDP-based norms.
- The collision-model script implements the hidden-memory update rule directly,
  avoiding heavier process-tensor reconstruction machinery.
- The revision scripts are exploratory in spirit: they report observed
  behaviour rather than asserting a target outcome in advance.
