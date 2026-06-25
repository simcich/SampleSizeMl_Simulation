# Sample Size Estimation for AI-Based Clinical Prediction Models

A simulation-based framework for estimating the training data requirements of binary clinical prediction models, accompanying the publication:

> **"Simulation-Based Sample Size Estimation for AI Clinical Prediction Models"**  
> *A flexible framework examining the effects of class imbalance, task complexity, and performance thresholds on required sample size for logistic regression and gradient boosting models.*

---

## Overview

Determining adequate sample size for machine learning-based clinical prediction models is non-trivial. Standard rules of thumb (e.g., events-per-variable) were designed for classical regression and do not generalise well to modern, high-dimensional settings.

This repository provides a flexible Python simulation framework that:

- Generates synthetic clinical populations with controlled characteristics
- Trains and evaluates **Logistic Regression** and **LightGBM** across a range of training set sizes
- Quantifies the minimum sample size needed to reach a user-defined performance threshold
- Exports publication-ready figures and summary tables

---

## Repository Structure

```
.
└── Experiments/
    ├── SampleExp.ipynb          # Main experiment notebook
    ├── figtOutput/              # Output directory for figures (PNG, 300 dpi)
    │   ├── UnbalancedClasses.png
    │   └── balancedClasses.png
    └── tableOutput/             # Output directory for result tables (DOCX)
        ├── results_table_weights_a.docx
        ├── results_table_weights_b.docx
        └── results_table_thresholds.docx
```

---

## Methods

### Synthetic Population Generation (`get_population`)

Populations are generated using `sklearn.datasets.make_classification`, with the following configurable parameters:

| Parameter | Description |
|---|---|
| `n_samples` | Total population size (default: 50 000) |
| `n_features` | Total number of features |
| `n_informative` | Features with a true predictive signal |
| `n_redundant` | Features that are linear combinations of informative ones |
| `flip_y` | Label noise fraction |
| `weights` | Class proportions, e.g. `[0.9, 0.1]` for 10 % minority class |
| `class_sep` | Class separability; values below 1.0 increase cluster overlap |

### Simulation Loop (`run_model_comparison_simulation`)

For each experiment, training subsets of increasing size (`n = 50–5000`) are repeatedly drawn from the population (default: 50 random draws per size). Both models are then evaluated on a held-out test split (default: 70 % of the population):

- **LightGBM** — trained with early stopping (patience = 15 rounds) on an internal 80/20 validation split
- **Logistic Regression** — trained on the full training sample with `max_iter=1000`

Performance is measured by **ROC AUC** and **Brier score**, reported as mean ± 1.96 SD across simulations.

### Threshold-Based Sample Size Estimation

The minimum *n* required to reach a fraction of the asymptotic performance is determined by linear interpolation:

- AUC target: `max(AUC) × threshold`
- Brier score target: `min(Brier) + (1 − threshold) × range(Brier)`

The default threshold is **0.98** (i.e., 98 % of the best observed performance).

---

## Experiments

### Simulation 1 — Imbalanced Classes
Minority class prevalence of 10 %, moderate noise (`flip_y = 0.1`), standard separability.

### Simulation 2 — Balanced Classes
Equal class prevalence (50/50), low noise (`flip_y = 0.01`), higher separability (`class_sep = 1.5`).

### Experiment: Class Imbalance Sweep (A & B)
The class weight ratio is varied across six scenarios — from balanced (`[0.5, 0.5]`) to heavily imbalanced (`[0.95, 0.05]`) — under two noise conditions:

- **A**: Low noise (`flip_y = 0.01`), high separability (`class_sep = 1.4`)
- **B**: Higher noise (`flip_y = 0.1`), standard separability (`class_sep = 1.0`)

### Experiment: Performance Threshold Sensitivity
Using a fixed balanced population, the performance threshold is varied across `[0.90, 0.95, 0.96, 0.97, 0.98, 0.99]` to characterise the non-linear relationship between required sample size and the stringency of the performance criterion.

---

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
lightgbm
scipy
matplotlib
python-docx
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm scipy matplotlib python-docx
```

> **Note:** The final cell uses `winsound.Beep` (Windows only) to signal experiment completion. This can be safely removed or replaced on macOS/Linux.

---

## Usage

Open `Experiments/SampleExp.ipynb` in Jupyter and run cells sequentially. Output figures are saved to `Experiments/figtOutput/` and result tables to `Experiments/tableOutput/`. Both directories must exist before running:

```bash
mkdir -p Experiments/figtOutput Experiments/tableOutput
```

To adjust an experiment, modify the relevant `population_config` and `simulation_config` dictionaries before running the corresponding section.

---

## Outputs

**Figures** (300 dpi PNG) show AUC and Brier score as a function of sample size for each model, with 95 % confidence bands and an annotated vertical line marking the threshold-crossing point.

**Tables** (DOCX) report the estimated minimum *n* and corresponding metric value for each model and scenario combination.

---

## Reproducibility

All experiments use a fixed global seed (`SEED = 42`). Per-simulation seeds are derived deterministically as `seed + n + s`, ensuring full reproducibility without fixing the random state globally across libraries.

---

## Citation

If you use this framework in your research, please cite the associated publication (details to be added upon acceptance).
