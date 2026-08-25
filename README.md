# BROL-DML reproducibility code

This repository contains the code for the BROL-DML analyses and their shared
utilities.

## Repository structure

- `scripts/exp01_semisynthetic_continuous_benchmark_50rep.py` runs the
  semi-synthetic continuous-treatment benchmark.
- `scripts/exp02_empirical_core_ablation_full_crossfit.py` runs the
  representation-specification stability analysis.
- `scripts/exp03_representation_support_diagnostics.py` runs the
  representation-support diagnostics.
- `scripts/exp04_treatment_margin_decomposition.py` runs the treatment-margin
  and temporal specifications.
- `scripts/exp05_orthogonal_rcs_dose_response.py` runs the orthogonal
  restricted-cubic-spline dose-response diagnostic.
- `scripts/exp06_within_year_permutation_placebo.py` runs the full-crossfit
  diagnostics and within-year permutation analysis.
- `scripts/_support/` contains shared implementation modules used by the
  experiment entry points.
- `src/` contains shared configuration, data preparation, causal-analysis,
  and output utilities.

## Environment

The code was tested with Python 3.11.15. Install the pinned dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Running the experiments

Run commands from the repository root:

```powershell
python scripts/exp01_semisynthetic_continuous_benchmark_50rep.py
python scripts/exp02_empirical_core_ablation_full_crossfit.py
python scripts/exp04_treatment_margin_decomposition.py
python scripts/exp05_orthogonal_rcs_dose_response.py
python scripts/exp06_within_year_permutation_placebo.py
python scripts/exp03_representation_support_diagnostics.py --analysis common-x
python scripts/exp03_representation_support_diagnostics.py --analysis k100
```

`exp01`, `exp04`, `exp05`, and the shared support pipeline use the exp02 core
implementation. Run exp02 before exp05 when the linear-comparator output is
needed. The exp03 diagnostics use outputs created by exp06; run its common-X
diagnostic before the k100 diagnostic because k100 uses its fold coordinates.

Generated files are written under `outputs/`.
