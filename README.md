# Hidden-state drought monitoring with recurrent precipitation models

Code, protocol and results for *"Can the hidden state of a recurrent precipitation model monitor
drought onset? A planned event-based evaluation of backward-coherence regularization"*
(Foo & Chang), prepared for *Stochastic Environmental Research and Risk Assessment*.
Repository: https://github.com/knight-ivan/rmrnn-drought-monitoring (private until acceptance).
This folder is also what is submitted to the journal as Online Resource 2.

Revisions of 2026-11-13 and 2026-11-14 (after two pre-submission reviews of the 2026-11-12
package): post hoc robustness checks added (PROTOCOL.md, Amendments 2 and 3), wording revised,
figures redrawn. No pre-specified result was changed; the monthly SPI-3 comparator is shown with a
corrected (dry-side) calibration in the main tables and with the pre-specified one in the supplement.

## Quick check (needs nothing outside this archive)
```
sh reproduce.sh          # = python3 code/serra/verify_from_csv.py
```
rebuilds every table and every number quoted in the paper from `results/` and compares them
with the published ones in `tex/tables/`.

## Files
- `PROTOCOL.md` — analysis plan written before training, its deviation log, Amendment 1
  (temperature-aware extension) and Amendments 2 and 3 (post hoc checks and one correction). The protocol is a
  dated file kept with the code; it was not lodged with an external registry.
- `code/serra/` — the analysis scripts of the paper. `code/common/` and
  `code/experiments_original/` — the original exploratory scripts, kept unchanged for the record
  (only `fetch_taiwan_hydromet.py` is used by the pipeline).
- `results/` — every result file the tables and figures are built from.
- `tex/` — manuscript, Online Resource 1, cover letter. `tex/tables/*.tex` and
  `tex/tables/numbers.tex` are **generated** by `code/serra/make_tables.py`; every number in the
  text is a macro taken from the result CSVs, so nothing is typed by hand, and the script stops if
  a descriptive sentence no longer matches the results.
- `figures/` — all figures (made by `code/serra/figures.py`).
- `logs/` — console logs of the original runs.
- `reproduce.sh` — the commands below in one file.
- `MANIFEST.sha256` — SHA-256 of every file in the archive.

## Reproduce (about 2 h on 16 cores for the full pipeline)
```
# 1. data
python3 code/fetch_ghcnd_long.py --region texas      # also germany, australia
python3 code/experiments_original/fetch_taiwan_hydromet.py --source raingel --stations <the 22 ids in COORD of code/build_taiwan.py, comma-separated> --years 1961-2024 --outdir data/taiwan_raingel
python3 code/build_taiwan.py
python3 code/fetch_power.py --region <each region>
# 2. precipitation-only analysis (pre-specified)
python3 code/serra/train_models.py
python3 code/serra/build_scores.py && python3 code/serra/add_spi30.py
python3 code/serra/evaluate_warning.py
python3 code/serra/forecast_eval.py
python3 code/serra/covariate_analysis.py
# 3. temperature-aware analysis (Amendment 1)
python3 code/build_temperature.py && python3 code/serra/train_models_mv.py && python3 code/serra/build_scores_mv.py
python3 code/serra/evaluate_warning.py --mv && python3 code/serra/forecast_eval.py --mv
python3 code/serra/covariate_analysis_mv.py && python3 code/serra/drift_mv.py
# 4. post hoc robustness checks (Amendment 2); need only results/panel*/
python3 code/serra/matching_checks.py && python3 code/serra/matching_checks.py --mv
python3 code/serra/incremental_forward.py && python3 code/serra/incremental_forward.py --mv
# 4b. further post hoc checks (Amendment 3); extract_intermediates.py needs results/scores*/ and data/*/power/
python3 code/serra/extract_intermediates.py
python3 code/serra/review_checks.py && python3 code/serra/review_checks.py --mv
python3 code/serra/incremental_forward.py --fixedref && python3 code/serra/incremental_forward.py --mv --fixedref
# 5. tables, numbers, figures, PDFs
python3 code/serra/text_checks.py && python3 code/serra/make_tables.py
python3 code/serra/figures.py && python3 code/serra/figures.py composite_mv
cd tex && pdflatex RM_Precipitation_SERRA_supplement && pdflatex RM_Precipitation_SERRA && bibtex RM_Precipitation_SERRA \
  && pdflatex RM_Precipitation_SERRA && pdflatex RM_Precipitation_SERRA_supplement \
  && pdflatex RM_Precipitation_SERRA && pdflatex RM_Precipitation_SERRA_supplement && pdflatex RM_Precipitation_SERRA
```
Steps 4 and 5 run from the stored score panels (`results/panel/*.npz`, `results/panel_mv/*.npz`,
about 200 MB) without any retraining; step 4 takes about 40 minutes on two cores.

## Random seeds
- Training (`train_models*.py`): seed = 1000*k + s for seed index k = 0..4, where s is the
  integer formed by the last four characters of the station id (0 if they are not digits);
  `torch.manual_seed(seed)` and `numpy.random.default_rng(seed)`.
- Bootstrap: 20261112 (`evaluate_warning.py`, `matching_checks.py`, `review_checks.py`; +1 for
  block resampling, +2 and +3 for the random-timing shifts, +4 for the forecast intervals),
  7 (`covariate_analysis.py`), 11 (`covariate_analysis_mv.py`), 1007 / 1011 (`incremental_forward.py`,
  second pass; 2007 / 2011 with `--fixedref`), 5 (`drift_mv.py`).

## Software
- Original runs: Python 3.12 with the packages in `requirements.txt`.
  To do before the public release: run `pip freeze > requirements-lock.txt` in that environment
  so that the exact versions are recorded.
- Post hoc checks and independent re-run (2026-11-13): Python 3.13.18, NumPy 2.5.3, pandas 3.0.5,
  SciPy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, Numba 0.68.1, Matplotlib 3.11.2.

## Independent re-run of the stored results (2026-11-13)
From the stored score panels, in the second environment above:
- `evaluate_warning.py` and `evaluate_warning.py --mv`: thresholds, event rows, station-year
  table, summary, paired differences, paired timing and operating curves **identical** to the
  stored files.
- `drift_mv.py`: identical. `text_checks.py`: identical.
- `covariate_analysis.py`, `covariate_analysis_mv.py`: within 2e-6 of the stored files
  (solver differences between library versions). The stored files are the originals.
- Not re-run: data download, training, `build_scores*.py`, `forecast_eval.py` (they need the raw
  data and the per-seed score files, about 2 GB).

## What the evidence says (as stated in the paper)
- Forecasts: the regularized network is equivalent to its λ=0 ablation and to a GRU within ±2%
  MSE (TOST), and its hidden-state path increment is 23.5% smaller.
- Warning (1,466 station-level soil-moisture drought onsets, 133 stations, 2001–2024, one alarm
  per station-year): the defect detects 0.665 of onsets, against 0.463 for randomly timed alarms
  and 0.763 for daily SPI-90 (difference −0.098 [−0.140, −0.056]); on events both detect it alarms
  a median 16 days earlier. Daily SPI-30 accounts for most of that speed, and the defect adds no
  detectable predictive information once SPI-30 is known (blocked CV and forward chaining).
  Combined with SPI-90 (post hoc) it keeps SPI-90's sensitivity and advances alarms by 5 days.
- Regularized vs unregularized defect: ΔPOD +0.010 [−0.009, +0.029]; no statistically detectable
  difference, which is not the same as equivalence.
- Temperature-aware extension (120 stations, 1,336 onsets): forecasts improve (MSESS 0.116 vs
  0.086); the defect detects 0.433 of onsets, the level of randomly timed alarms (0.433), against
  0.749 for SPEI-90.
- Robustness (Amendment 2): one-to-one matching, nine scoring windows, 2–4-year block bootstrap
  and forward-chaining validation leave these conclusions unchanged. The deficit to SPI-90 is
  larger when alarms before the onset are not credited (−0.194 at [0, +120]).
- Monthly SPI-3 (D4): the pre-specified grid search met the budget with a positive (wet-side)
  threshold in Germany and Australia. The main tables use the dry-side calibration (post hoc
  correction): SPI-3 detects 0.767 of onsets pooled (defect minus SPI-3: −0.102 [−0.139, −0.061]);
  the pre-specified calibration (0.510) is in the supplement.
- Detection by deadline (Amendment 3; counts every onset): by the onset date the defect has
  detected 0.257 of onsets against 0.194 for SPI-90 (+0.063 [+0.013, +0.119]); SPI-90 overtakes
  it after day 41. The D1+D3 combination is ahead of SPI-90 at 30 and 60 days.
- Fixed-reference events (percentiles from 1981–2000 only; 1,800 onsets): defect 0.655, SPI-90
  0.718 (−0.063 [−0.097, −0.028]); regularized minus unregularized +0.001 [−0.016, +0.017].
- Forecast equivalence holds with year-cluster intervals as well.

## Changes from the earlier exploratory work (all disclosed in the paper)
- Turkey dropped (no gauge has a 30-year reference period); SE Australia (Murray–Darling) added.
- Texas box widened to central Texas (40 long-record stations).
- Reference events come from MERRA-2 root-zone soil moisture (NASA POWER), not ERA5-Land,
  because of download rate limits.
- PyTorch re-implementation of the Elman network with early stopping; 5 seeds per station.

## What is in this archive beyond code and tables
Station lists (`data/<region>/station_meta.csv`, `temperature_meta.csv`), the reference events
(`results/panel*/<region>_events*.csv`), `results/checks/intermediates.npz` (raw MERRA-2 root-zone
soil wetness per station and station-year forecast errors) and the Natural Earth outlines.

## What is NOT in this archive (rebuilt by the scripts)
Raw GHCN `.dly` files, Raingel daily files, long-format daily inputs, MERRA-2 extracts,
per-seed score files (`results/scores*/`) and score panels (`results/panel*/*.npz`): about
2.2 GB in total. The score panels are enough to recompute every warning and covariate result;
they are kept with the authors and should be deposited (for example on Zenodo) with the public
release.
