# Neighborhood-Constrained Variational Mode Decomposition (NCVMD)

**VMD**, **MVMD**, **NCVMD**, and **VMDMS** (decomposition step) in Python, with a built-in synthetic **toy example** for reconstruction tests.

No electric-load datasets or external data files are required.

## Author and license

**Copyright © Paolo Fazzini.** All code in this repository is authored by **Paolo Fazzini**, released under the **[MIT License](LICENSE)** (permissive use, modification, and redistribution with attribution and license notice).

The VMD core in `VMD_pers.py` and `VMD_extended.py` extends the Python implementation by Vinícius Rezende Carvalho (see file headers; cite Dragomiretskiy & Zosso, 2014, for standard VMD). **NCVMD**, **VMDMS**-related modules, **MVMD** integration, tests, and this repository layout are original contributions by Paolo Fazzini.

## NCVMD at a glance

Aligning mode frequencies across channels without penalising the channel that matters (VMD vs MVMD vs NCVMD):

![Aligning mode frequencies without penalising the channel that matters: VMD, MVMD, and NCVMD compared](figures/ncvmd_mode_alignment.png)

## Contents

| File | Role |
|------|------|
| `VMD_pers.py` | VMD (extended implementation used in the project) |
| `VMD_extended.py` | VMD with coupling term (used by NCVMD) |
| `mvmd.py` | MVMD |
| `ncvmd.py` | NCVMD |
| `vmdms.py` | VMDMS decomposition (independent VMD per channel; frequency partitioning for forecasting is not included here) |
| `toy_data.py` | Synthetic signals `Test_1D` and `Artificial2D_1` |
| `test_toy_example.py` | Reconstruction test on toy data |

## Requirements

- Python 3.8+
- `numpy` (see `requirements.txt`)

## Running the test

```bash
pip install -r requirements.txt
python test_toy_example.py
```

Univariate toy signal:

```bash
python test_toy_example.py --dataset Test_1D
```

The script prints per-channel reconstruction RMSE for each method. It does not require TensorFlow or external datasets.

## Citation

If you use code from this repository in academic work, please cite the relevant publication(s) below.

### NCVMD

If you use **NCVMD**, please cite:

> Fazzini, P., La Tona, G., Diez, M., Montuori, M. & Di Piazza, M. C. Multi-Channel Neighborhood-Constrained Variational Mode Decomposition. *Information* **17**(10), 970 (2026). https://doi.org/10.3390/info17100970

```bibtex
@Article{info17100970,
  AUTHOR   = {Fazzini, Paolo and La Tona, Giuseppe and Diez, Matteo and Montuori, Marco and Di Piazza, Maria Carmela},
  TITLE    = {Multi-Channel Neighborhood-Constrained Variational Mode Decomposition},
  JOURNAL  = {Information},
  VOLUME   = {17},
  YEAR     = {2026},
  NUMBER   = {10},
  ARTICLE-NUMBER = {970},
  URL      = {https://www.mdpi.com/2078-2489/17/10/970},
  ISSN     = {2078-2489},
  DOI      = {10.3390/info17100970}
}
```

### VMDMS

If you use **VMDMS**, please cite:

> Fazzini, P., La Tona, G., Diez, M. & Di Piazza, M. C. Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection. *Scientific Reports* **15**, 23941 (2025). https://doi.org/10.1038/s41598-025-06153-z

```bibtex
@article{Fazzini2025VMDMS,
  author  = {Fazzini, Paolo and La Tona, Giuseppe and Diez, Matteo and {Di Piazza}, Maria Carmela},
  title   = {Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {23941},
  year    = {2025},
  doi     = {10.1038/s41598-025-06153-z},
  url     = {https://doi.org/10.1038/s41598-025-06153-z}
}
```

Standard **VMD** and **MVMD** implementations here follow the literature cited in the module docstrings (Dragomiretskiy & Zosso, 2014, and the respective multivariate references). Cite those original papers when you rely on the base algorithms.

## Related work

- **VMDMS** (earlier public toy / forecasting example): [github.com/pfaz69/VMDMS](https://github.com/pfaz69/VMDMS)
- **VMDMS paper:** [Scientific Reports (2025)](https://doi.org/10.1038/s41598-025-06153-z) — Fazzini, La Tona, Diez, Di Piazza et al.

The full multivariate forecasting pipeline and experiments (datasets, partitioning, LaTeX outputs) are maintained in the authors’ private research codebase; contact **Paolo Fazzini** if you need access.
