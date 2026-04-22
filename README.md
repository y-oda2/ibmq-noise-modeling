# Sparse non-Markovian Noise Modeling of Transmon-Based Multi-Qubit Operations
### Data and Code Repository

**Paper:** Oda, Schultz, Norris, Shehab, and Quiroz — *Sparse non-Markovian Noise Modeling of Transmon-Based Multi-Qubit Operations*, PRX Quantum (2026). DOI: 10.1103/lx8x-z29x

---

## Overview

This repository contains the Jupyter notebooks and data files used to generate all figures in the paper. The work presents a unified noise characterization and modeling framework for transmon-based multi-qubit operations on IBM Quantum Platform (IBMQP) devices, capturing both Markovian and non-Markovian (correlated dephasing, control noise, TLS, crosstalk) effects via the Lindblad master equation (LME).

---

## Directory Structure

```
zenodo_upload/
├── README.md                    ← this file
├── notebooks/
│   ├── imports_IBM_NM.py        ← shared Python module (required by most notebooks)
│   ├── fig_01_*.ipynb           ← Figure 1
│   ├── fig_02_*.ipynb           ← Figure 2
│   │   ...
│   └── fig_17_*.ipynb           ← Figure 17 / Appendix Figure 19
└── data/
    ├── markov_plot_data.p       ← Markovian fit results (ibm_algiers)
    ├── ps_exps_algiers_char.p   ← Characterization experiment results (ibm_algiers, all qubits)
    ├── ps_sims_FIG3.p           ← TLS simulation results for Fig 3 (ibm_algiers)
    ├── data_FIG4.p              ← Crosstalk experiment data (ibmq_lima)
    ├── data_FIG5.p              ← Correlated dephasing histogram data (ibm_algiers)
    ├── figdata-corr_deph-PSDs.p ← Reconstructed dephasing PSDs (ibmq_belem)
    ├── figdata-corr_deph-FTTPS_PSD.p  ← FTTPS PSD fit data (ibm_hanoi)
    ├── figdata-corr_deph-DD.p   ← CPMG/DD experiment data for correlated dephasing
    ├── fttps_corr.p             ← Correlated control noise FTTPS/R-FTTPS data (ibmq_lima)
    ├── CRp45_X_CRm45_X-Utom-circs-lagos.p  ← ECR gate tomography circuits (ibm_lagos)
    ├── data_FIG8.p              ← Multi-qubit DD experiment data (ibm_cairo)
    ├── VQE_exp.p                ← VQE experimental energies (ibm_algiers)
    ├── VQE_sim_IBM.p            ← VQE IBM noise model simulation energies
    ├── VQE_sim_NM.p             ← VQE our noise model simulation energies
    ├── VQE_H2_theta_opt.p       ← Optimal VQE rotation angles for H2
    ├── ps_1f0.p                 ← 1/f noise FTTPS data for correlated dephasing
    ├── gif_fttps_res.p          ← FTTPS resonance experiment data (ibm_algiers, qubit 13)
    ├── CDF_Ts.p                 ← T1/T2 coherence time data across IBM devices
    └── g_values.csv             ← Noise characterization fit parameters
```

---

## Figure-to-Notebook Mapping

| Notebook | Paper Figure | Description |
|----------|-------------|-------------|
| `fig_01_noise_characterization_overview.ipynb` | Fig 1 | Noise characterization protocol overview (ibmq_guadalupe) |
| `fig_02_markovian_experiments.ipynb` | Fig 2 | Markovian characterization experiments: T1, T2, Q, P, RB (ibm_algiers, qubit 8) |
| `fig_03_tls_ramsey.ipynb` | Fig 3 | TLS Ramsey experiments and simulation (ibm_algiers, qubits 0,5,8,9) |
| `fig_04_crosstalk.ipynb` | Fig 4 | Two-qubit crosstalk (XT) experiments (ibmq_lima) |
| `fig_05_correlated_dephasing.ipynb` | Fig 5 | FTTPS correlated dephasing characterization (ibm_hanoi, ibmq_belem, ibm_algiers) |
| `fig_06_correlated_control_noise.ipynb` | Fig 6 | Correlated control noise via FTTPS and R-FTTPS (ibmq_lima) |
| `fig_07_ecr_gate.ipynb` | Fig 7 | CR/ECR gate characterization and LME simulation (ibm_lagos) |
| `fig_08_multiQubit_dynamical_decoupling.ipynb` | Fig 8 | Multi-qubit dynamical decoupling XY4 — Type 1 & 2 experiments (ibm_cairo) |
| `fig_09_vqe_H2.ipynb` | Fig 9 | VQE for H2 molecule dissociation curve (ibm_algiers) |
| `fig_10_fttps_markovian_noise.ipynb` | Fig 10 (App. A) | FTTPS simulations with Markovian noise parameters — analytical vs simulation |
| `fig_11_fttps_filter_functions.ipynb` | Fig 11 (App. B) | FTTPS and R-FTTPS filter functions |
| `fig_12_gaussian_vs_constant_pulse.ipynb` | Fig 12 (App. C) | Gaussian vs constant pulse comparison for FPW experiments |
| `fig_13_t2_dynamical_decoupling.ipynb` | Fig 13 (App. E) | T2 with increasing DD pulses — evidence of correlated dephasing (ibmq_guadalupe) |
| `fig_14_markovian_rb_simulations.ipynb` | Fig 14 (App. F) | RB simulations with Markovian noise parameters |
| `fig_15_fttps_resonance.ipynb` | Fig 16 (App. H) | FTTPS resonance feature in ibm_algiers (qubit 13) |
| `fig_16_tls_and_xt_ramsey.ipynb` | Fig 17 (App. I) | TLS and crosstalk: Ramsey with XY4 DD on spectator qubits (ibm_cairo) |
| `fig_17_device_properties_cdf.ipynb` | Fig 19 (App. K) | Device connectivity diagrams and T1/T2 CDFs across IBM devices |

---

## Running the Notebooks

### Requirements

Most notebooks require the following Python packages:
- `numpy`, `scipy`, `matplotlib`
- `qiskit` (≥ 0.43) and `qiskit_ibm_provider` — for circuit construction and IBM backend interfaces
- `qutip` — for Lindblad master equation simulations (Fig 7)

Install via:
```bash
pip install numpy scipy matplotlib qiskit qiskit-ibm-provider qutip
```

The shared `imports_IBM_NM.py` module (included in `notebooks/`) must be present in the same directory as the notebooks when running them.

### Running from the notebooks/ directory

All notebooks should be run from inside the `notebooks/` directory so that relative paths to `data/` files resolve correctly:

```bash
cd notebooks/
jupyter notebook
```

### IBM Quantum Access

Most figures use pre-saved experimental data from the `data/` directory and can be reproduced without an IBM Quantum account. The following notebooks contain cells that originally retrieved results live from IBM Quantum jobs but rely on pre-saved data for reproduction:

- **Fig 2** (`fig_02_markovian_experiments.ipynb`): Experimental data loaded from `data/markov_plot_data.p`. The IBM job retrieval cells can be skipped.
- **Fig 7** (`fig_07_ecr_gate.ipynb`): The experimental data was obtained from IBM Quantum job ID `cmskyrvvpdgg008gb7ag` on `ibm_lagos`. This job may no longer be retrievable. The notebook also contains LME simulation code (using `qutip`) that runs standalone to produce the simulation curves. The ECR pulse schedule is obtained via `qiskit`'s `FakeLagos` backend, which does not require IBM account access.

---

## Notes

- **Pickle files** (`.p` extension) were created with Python's `pickle` module. Some files (e.g., `CRp45_X_CRm45_X-Utom-circs-lagos.p`) contain serialized Qiskit circuit objects and require `qiskit` to be installed for loading.
- **RB standard deviations** (Appendix G, Fig 15): This figure is generated within `fig_14_markovian_rb_simulations.ipynb` as a companion scatter plot to Fig 14.
- **Model time evolution** (Appendix J, Fig 18): Generated by the `24-02-10 - model time evolution non-markovian - osaka - PAPER FIG.ipynb` notebook in the original working directory; data is computed inline using the noise model.
- **Fig 1** contains both schematic diagram panels (created in presentation software) and experimental data panels generated by `fig_01_noise_characterization_overview.ipynb`.

---

## Citation

If you use this data or code, please cite:

> Y. Oda, K. Schultz, L. Norris, O. Shehab, and G. Quiroz, *Sparse non-Markovian Noise Modeling of Transmon-Based Multi-Qubit Operations*, PRX Quantum **0**, XXXXXX (2026). DOI: 10.1103/lx8x-z29x
