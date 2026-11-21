# Characterization and Calibration of a Superconducting Transmon Qubit Using QUA Pulse Sequences

## Abstract
This repository contains the QUA quantum-control execution script (`prog_exec.py`) and the data-analysis/fitting script (`data_proc.py`) used to calibrate and characterize a superconducting transmon qubit (`Q1`) on the `qolab` backend via `IQCC_Cloud` and OPX hardware[cite: 3, 4]. The scripts implement resonator and qubit spectroscopy, flux-bias and resonator-bias sweeps, DRAG Rabi duration sweeps, single-shot $I/Q$ state discrimination, consecutive $\pi$-pulse tests, coherence measurements ($T_1$, Ramsey $T_2^*$, and Hahn echo $T_2$), and non-selective driven Quantum Zeno Effect sequences[cite: 3, 4]. Running these scripts acquires or loads raw $I/Q$ measurements, performs statistical/curve fitting and error analysis, saves JSON results and fit reports, and generates the plots used in the paper[cite: 3, 4].

## File Descriptions

### `prog_exec.py`
* **Description:** Defines and executes QUA pulse sequences for qubit `Q1` (`Q1.xy`, `Q1.resonator`, `Q1.z`) on the `qolab` quantum computer via `IQCC_Cloud`, supporting `resonator_sweep`, `resonator_bias_sweep`, `bias_sweep`, `qubit_sweep`, `rabi_find`, `discrimination`, `consecutive_pulse`, `t1`, `ramsey`, `echo`, and `zeno` (selected via `EXPERIMENT`)[cite: 3].
* **Inputs:** `config/main_config.py` (`config` dictionary and XY/resonator LO frequencies), `data/token.txt` (API token), and `data/results.json` (previously saved calibration parameters such as `resonator_IF`, `drive_IF`, and `bias_amp`, when present)[cite: 3].
* **Outputs:** Writes raw `I`/`Q` streams and experiment setup metadata to `data/<name>.json` (`resonator_sweep.json`, `resonator_bias_sweep.json`, `bias_sweep.json`, `qubit_sweep.json`, `rabi_find.json`, `discrimination.json`, `consecutive_pulse.json`, `t1.json`, `ramsey.json`, `echo.json`, or `zeno.json`)[cite: 3].

### `data_proc.py`
* **Description:** Loads recorded experiment JSON files, computes propagated $I/Q$ SEM or Wilson binomial intervals, fits physical/phenomenological models (`signed_lorentzian`, `rabi_model`, `parity_model`, `exp_decay`, `ramsey_model`, and the Zeno differential suppression model), trains a linear $I/Q$ state discriminator, and generates fit, residual, diagnostic, and comparison plots (selected via `ANALYSIS`)[cite: 4].
* **Inputs:** `data/<name>.json` files produced by `prog_exec.py` (or `readout_amp_sweep.json`), `config/main_config.py` (via `get_lo_from_config` for XY LO frequency), and `data/results.json` / `data/fit_reports_234.json` when reading saved calibration or discriminator parameters[cite: 4].
* **Outputs:** Updates `data/results.json` and `data/fit_reports_234.json`, prints parameter/goodness-of-fit summaries to stdout, saves Zeno figures (`zeno.png`, `zeno_fit.png`, `zeno_residuals.png`, `zeno_controls.png`) under `data/zeno_figures/`, and displays interactive `matplotlib` figures for the selected analysis[cite: 4].
