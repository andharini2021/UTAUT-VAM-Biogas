# PLS-SEM for an integrated UTAUT2–VAM model of biogas adoption (East Java)

Python implementation (numpy/pandas/scipy only) of reflective PLS-SEM (Mode A, path-weighting scheme)
with case-resampling bootstrap, specific indirect effects, ANOVA with Tukey HSD / Games-Howell,
and a herd-size multigroup comparison (Keil et al., 2000). Companion code to the article
*Integrating UTAUT2 and the Value-Based Adoption Model to Explain Biogas Technology Adoption Among
Livestock Farmers in East Java, Indonesia*.

## Contents
| Path | Description |
|---|---|
| `pls_sem_utaut_vam.py` | Full analysis pipeline (command line) |
| `data/data_public_minimal.csv` | De-identified item-level data (40 Likert items, regency, herd size) |
| `expected_results/pls_sem_results_expected.xlsx` | Output of the run below, for checking your own run |
| `requirements.txt` | Dependencies |

## Run
```bash
pip install -r requirements.txt
python pls_sem_utaut_vam.py data/data_public_minimal.csv --boot 5000 --boot-group 2000 --seed 42 --out results
python pls_sem_utaut_vam.py --selftest      # parameter-recovery check on simulated data
```
Runtime: about 5 minutes for 5,000 bootstrap resamples. Use `--drop-invalid` to drop respondents
with Likert values outside 1–5. In Google Colab, upload the script and the CSV, then run the same
commands with `!python ...`.

## What to expect
Point estimates (loadings, reliability, AVE, path coefficients, R², f², HTMT) are deterministic and
should match `expected_results` to three decimals. Bootstrap SE, p-values and confidence intervals
vary slightly with the random seed and the number of resamples.

## Validation
The estimates were compared with SmartPLS 4 (path weighting, stop criterion 1e-7, 5,000 complete
bootstrap subsamples): identical to three decimals for all path coefficients, loadings, reliability
and validity indices, R² and f²; bootstrap SE differed by ≤ .002 and p-values by ≤ .01, with identical
significance decisions.

## Data note
The released file contains no names or direct identifiers. Respondent IDs were re-coded and
demographic variables other than regency and herd size were removed to limit re-identification risk,
so Table 1 (respondent profile) of the article cannot be reproduced from this file.
[Full demographic data: available from the corresponding author on reasonable request, subject to ethics approval conditions.]

## Not implemented
MICOM, permutation MGA, PLSpredict/CVPAT and SRMR (reported from SmartPLS 4 in the article).
