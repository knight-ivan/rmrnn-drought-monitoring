# Hidden-state drought monitoring with recurrent precipitation models

Code, protocol and results for *"Can the hidden state of a recurrent precipitation model monitor
drought onset? A pre-specified event-based evaluation of backward-coherence regularisation"*
(Foo & Chang), prepared for *Stochastic Environmental Research and Risk Assessment* (2026-11-12).
Repository: https://github.com/knight-ivan/rmrnn-drought-monitoring (private until acceptance).

New study, not a revision of the JoH draft (JOHX56256, rejected by both reviewers).
All numbers come from new analyses run in this folder; nothing is carried over
from the earlier submission.

## Files
- `tex/RM_Precipitation_SERRA.pdf` — main paper (26 pp, Springer Nature `sn-jnl`,
  line numbers on).
- `tex/RM_Precipitation_SERRA_supplement.pdf` — Online Resource 1 (6 pp).
- `tex/tables/*.tex` and `tex/tables/numbers.tex` are **generated** by
  `code/serra/make_tables.py`; every number in the text is a macro taken from the
  result CSVs, so nothing is typed by hand.
- `PROTOCOL.md` — analysis plan written before training, with a log of every deviation.
- `results/` — all outputs (warning, forecast, covariates, panels, per-station scores).
- `figures/` — all figures (made by `code/serra/figures.py`).

## Reproduce (about 1.5 h on 16 cores)
```
python3 code/fetch_ghcnd_long.py --region texas      # also germany, australia
python3 code/experiments_original/fetch_taiwan_hydromet.py --source raingel --stations <list in build_taiwan.py> --years 1961-2024 --outdir data/taiwan_raingel
python3 code/build_taiwan.py
python3 code/fetch_power.py --region <each region>
python3 code/serra/train_models.py
python3 code/serra/build_scores.py && python3 code/serra/add_spi30.py
python3 code/serra/evaluate_warning.py
python3 code/serra/forecast_eval.py
python3 code/serra/covariate_analysis.py
python3 code/build_temperature.py && python3 code/serra/train_models_mv.py && python3 code/serra/build_scores_mv.py
python3 code/serra/evaluate_warning.py --mv && python3 code/serra/forecast_eval.py --mv
python3 code/serra/covariate_analysis_mv.py && python3 code/serra/drift_mv.py
python3 code/serra/text_checks.py && python3 code/serra/make_tables.py && python3 code/serra/figures.py && python3 code/serra/figures.py composite_mv
```

## What the new evidence says (stated in the paper)
- Forecasts: RMRNN is equivalent to its λ=0 ablation and to a GRU within ±2% MSE
  (TOST), and its hidden-state path increment is 23.5% smaller.
- Warning (1,466 soil-moisture drought onsets, 133 stations, 2001–2024, 1 alarm/yr):
  the defect detects fewer onsets than daily SPI-90 (0.665 vs 0.763), but on events
  both detect it alarms 16 days earlier. Daily SPI-30 accounts for most of that speed,
  and the defect adds no predictive information once SPI-30 is known.
  The regularisation does not change monitoring skill.
- Temperature-aware extension (PROTOCOL amendment 1; 120 stations with gauge Tmax/Tmin,
  1,336 onsets): forecasts improve (MSESS 0.116 vs 0.086) but the defect gets WORSE:
  POD 0.433 vs SPEI-90 0.749 (Δ −0.317 [−0.367, −0.266]); adds nothing beyond
  SPEI-90 + SPEI-30 + SPI-30 (ΔAUC −0.0015). Penalty still doesn't help monitoring.
- So the paper is an honest evaluation and framework paper. It does **not** claim
  that the defect gives early warning beyond SPI.

## Changes from the earlier work (all disclosed in the paper)
- Turkey dropped (no gauge has a 30-year reference period); SE Australia (Murray–Darling) added.
- Texas box widened to central Texas (40 long-record stations).
- Reference events come from MERRA-2 root-zone soil moisture (NASA POWER), not ERA5-Land,
  because of download rate limits.
- PyTorch re-implementation of the Elman RMRNN with early stopping; 5 seeds per station.

## Author checks before submission
1. The multivariate extension has been run (negative). Decide whether to submit
   this evaluation framing.
2. Co-author (Foo) review; CRediT roles are copied from the JoH version.
3. In the submission system: enter ORCIDs (Foo 0000-0003-4200-5849; Chang 0000-0002-4977-7721;
   both are also printed on the title pages), author contributions and competing interests
   (SERRA takes these from the interface).
4. On acceptance: make this repository public (`gh repo edit knight-ivan/rmrnn-drought-monitoring
   --visibility public --accept-visibility-change-consequences`).

## What is NOT in this repository (rebuilt by the scripts)
Raw GHCN `.dly` files, Raingel daily files, long-format daily inputs, MERRA-2 extracts,
trained-model score files (`results/scores*/`) and score panels (`results/panel*/*.npz`):
about 2.2 GB in total. They remain in the local Dropbox copy of this folder.

## SERRA guideline compliance (checked 2026-11-12 against the journal's submission guidelines)
Abstract 150–250 words; 6 keywords; Introduction within one printed page; author–year citations
with DOIs; "Statements and Declarations" (funding with grant number, competing interests, author
contributions, data and code availability) after the References; AI-tool use documented in
Methods (Section "Software"); figure captions without trailing punctuation; supplement cited as
"Online Resource 1"; dataset citations with DOI/URL; Springer Nature LaTeX template.
