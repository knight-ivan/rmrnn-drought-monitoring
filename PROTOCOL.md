# Pre-specified analysis protocol (written 2026-11-12, before any new results)

Target journal: *Stochastic Environmental Research and Risk Assessment* (SERRA).
This protocol responds to the JoH reviews (JOHX56256) and is fixed before the
new models are trained. Any later deviation is recorded at the end of this file.

## Data
- Daily gauge precipitation 1961-2024.
  - Texas (central, 29-33N, 101-96W), Germany, SE Australia (Murray-Darling,
    37-28S, 140-152E): NOAA GHCN-Daily, QC-flagged values set to missing,
    stations >= 90% complete in both 1961-1990 and 1991-2024, spatially thinned
    to <= 40 stations (greedy farthest point).
  - Taiwan: CWA synoptic stations (Raingel rebuild of CODiS), same completeness rule.
  - Turkey (original study) is dropped: no GHCN gauge in the box is >= 50%
    complete over 1961-1990, so a WMO-standard SPI-3 cannot be fitted.
- ERA5-Land daily covariates at each gauge (Open-Meteo archive): root-zone
  (28-100 cm) soil moisture, 7-28 cm soil moisture, 2-m temperature, FAO-56 ET0.

## Periods
- Reference/training: 1961-1990 (RNN training; SPI gamma fits; deseasonalisation).
  The last 3 years (1988-1990) are the early-stopping validation set.
- Calibration of alarm thresholds: 1991-2000.
- Evaluation: 2001-2024 (all reported warning and forecast results).

## Models (per station, precipitation-only input, 30-day windows, hidden 32)
- RMRNN: Elman (tanh) RNN + residual-MLP backward projector g; loss = next-day
  MSE + lambda * mean ||h_t - g(h_{t+1})||^2, lambda = 0 for 5 warm-up epochs,
  then 0.1 -> 0.01 geometrically.
- Ablation (lambda=0): identical backbone; projector trained on detached hidden
  states (same defect statistic, no regularisation of the trajectory).
- GRU: forecast baseline.
- 5 seeds each; detector scores are the seed-average of per-seed standardised scores.
- Missing days: input = day-of-year climatological mean of transformed precipitation
  plus a binary missingness channel (missing is never treated as 0 mm).

## Detectors (daily decision unless stated; all scores deseasonalised by
1961-1990 day-of-year mean/SD, smoothed +/-15 d)
- D1 RM defect, lambda>0: d_t = ||h_{t-1} - g(h_t)||, log-transformed. Downward CUSUM
  (direction pre-specified by the original study: dry spells lower the defect).
- D2 RM defect, lambda=0 ablation, same rule.
- D3 Daily SPI-90: 90-day total (>= 80 observed days, rescaled), zero-inflated gamma
  fitted per day of year (+/-15 d pooling) on 1961-1990, Phi^{-1}. Downward CUSUM.
- D4 SPI-3 (WMO standard, monthly): calendar-month zero-inflated gamma on 1961-1990;
  decision at month end; threshold rule SPI-3 <= c.
- D5 Forecast-residual CUSUM (y_t - yhat_t from RMRNN), downward.
- CUSUM: C_t = max(0, C_{t-1} - z_t - k), k = 0.5; alarm when C_t >= h; after an
  alarm, C is reset and alarms are suppressed for 90 days (refractory period).
- Missing score days: CUSUM is not updated.

## Matched alarm budget
Each detector's threshold (h, or c for D4) is chosen on 1991-2000 so that it
raises the same number of alarms per station-year (primary: 1.0 per year;
sensitivity: 0.5 and 2.0), pooled within region. Threshold-free comparison:
POD versus false alarms/year curves over a grid of thresholds on 2001-2024.

## Reference drought events (independent of all detectors)
- Root-zone soil-moisture percentile by day of year (+/-15 d window, all 1961-2024).
- Event: >= 30 consecutive days with percentile < 20; gaps < 30 days merged;
  onset = first day; at least 60 non-drought days before onset.
- Evaluated events: onset in 2001-2024.
- Hit: first alarm in [onset - 60 d, onset + 120 d]. Timing = alarm - onset.
- False alarm: alarm not in [onset - 60 d, event end] of any event (any onset year).
- Metrics: POD, false alarms per station-year, median timing among hits.
- Uncertainty: cluster bootstrap over (region, calendar year) blocks, 2000 resamples;
  paired detector differences computed on the same resamples.

## Physical association (Reviewer 2, comment 2)
- Monthly means of deseasonalised D1 score regressed on concurrent anomalies of SPI-3,
  root-zone soil moisture, T2m, ET0 (station fixed effects; cluster-robust SE by region-month).
- Incremental value: logistic model for "reference onset within next 60 days" using
  SPI-90 alone vs SPI-90 + D1 (+ D2), blocked-by-year cross-validation; report Delta AUC
  and Brier with cluster bootstrap CIs.

## Forecast evaluation (2001-2024)
- MSE, MAE (standardised log precipitation), AUC for exceedance of station P95.
- Paired station-level differences RMRNN - lambda0 and RMRNN - GRU: Wilcoxon
  signed-rank and TOST equivalence with margin +/-2% of the lambda0 MSE.
- Path increment Q_path within backbone (RMRNN vs lambda0).

## Deviations
(none yet)
- 2026-11-12 (before any detector or forecast evaluation): warm-up shortened
  5 -> 3 epochs, and RMRNN early-stopping checkpoints are eligible only once the
  penalty is active. Reason: a pilot on one Texas station showed early stopping
  otherwise selected a warm-up checkpoint, making RMRNN identical to the
  lambda=0 model for 3 of 5 seeds.
- 2026-11-12 (before any evaluation): covariate/soil-moisture source changed from
  ERA5-Land (Open-Meteo) to MERRA-2 via NASA POWER (GWETROOT root-zone soil
  wetness, T2M, EVPTRNS), 1981-2024, because the Open-Meteo rate limit allows only
  ~20 multi-decade station requests per day. Soil-moisture percentiles therefore
  use 1981-2024 as the day-of-year reference; ET0 is replaced by the
  evapotranspiration flux EVPTRNS.
- 2026-11-12 (AFTER seeing the primary warning results; exploratory, post hoc):
  added D6 = downward CUSUM on the re-standardised mean of the D1 and D3
  anomalies, motivated by the observed complementarity (D1 earlier, D3 more
  sensitive). Reported separately and labelled exploratory. Paired timing is
  now also reported for D6 vs D3 and D6 vs D1.
- Calibration implementation detail: the threshold is the grid value whose
  1991-2000 pooled alarm rate is closest to the budget; budgets a detector cannot
  attain are flagged (thresholds.csv, `attained`).
- 2026-11-12 (AFTER seeing the primary results; post hoc sensitivity): added D7 =
  daily SPI-30 (CUSUM), and SPI-30 as an added predictor in the incremental-value
  analysis, to test whether the defect's earlier alarms merely reflect the
  network's 30-day input window (a shorter accumulation period than SPI-90).

# AMENDMENT 1 — multivariate (temperature-aware) extension
Written 2026-11-12, after the univariate results above were known and BEFORE any
temperature data were processed or any multivariate model was trained.
Motivation: the univariate defect behaved as a short-memory precipitation summary.
Question: does a hidden state that also sees temperature and evaporative demand
carry drought-onset information beyond SPEI, the temperature-aware standard index?

## Data
- Daily Tmax/Tmin 1961-2024. GHCN regions: the rain gauge's own record if Tmax and
  Tmin are each >= 80% complete in 1961-1990 and in 1991-2024; otherwise the nearest
  GHCN station within 150 km meeting that rule (donor). Taiwan: the CWA station's
  own TxMaxAbs/TxMinAbs (Raingel). Stations without a qualifying record are excluded
  from the multivariate analysis; all multivariate comparisons (including the
  univariate detectors) are reported on this common subset.
- Values with Tmax < Tmin, or outside [-60, 60] C, are set missing.
- PET: Hargreaves-Samani, ET0 = 0.0023 * 0.408 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin),
  Ra = extraterrestrial radiation from latitude and day of year (FAO-56 eq. 21);
  Tmean = (Tmax + Tmin)/2.
- Soil moisture is NOT an input (the reference events are defined from it).

## Model (per station, 5 seeds, otherwise identical training protocol)
- Inputs: standardised log(1+P), standardised Tmax and Tmin anomalies, standardised
  PET anomaly, and missingness flags for P and temperature (missing filled with
  day-of-year climatology). Anomalies use 1961-1990 day-of-year mean/sd.
- Targets: next-day standardised log(1+P) and next-day standardised PET anomaly,
  equal weight (so that the state must encode the thermal regime).
- RMRNN-mv (penalty as before), lambda=0 ablation (detached projector), GRU-mv.

## Detectors (added to the evaluation; same CUSUM, k, refractory, budgets)
- D8 multivariate RM defect; D9 multivariate ablation defect;
- D10 daily SPEI-30 and D11 daily SPEI-90: 30/90-day sums of P - PET, three-parameter
  log-logistic distribution fitted by maximum likelihood per day of year (+/-15 d
  pooling) on 1961-1990, transformed to N(0,1) (Vicente-Serrano et al. 2010);
- D1, D3, D7 recomputed on the common subset for reference.
- Primary comparisons: D8 vs D11, D8 vs D10, D8 vs D9, D8 vs D1.
## Secondary
- Incremental value: base = SPEI-90 + SPEI-30 (+ SPI-30); added = 30-day mean D8.
- Association: D8 anomaly on SPI-3, soil wetness, T2m, ET (as before).
- Forecast: P-forecast equivalence of RMRNN-mv vs its ablation and GRU-mv.
- Amendment-1 deviation (2026-11-12, before any multivariate evaluation): SPEI
  log-logistic parameters are estimated by unbiased probability-weighted moments
  (the estimator of Vicente-Serrano et al. 2010) instead of maximum likelihood,
  because the MLE pilot on one station gave reference-period SPEI with mean 0.80
  and SD 0.48 (should be ~0 and ~1).

# AMENDMENT 2 — post hoc robustness checks
Written 2026-11-13, AFTER every result above was known and the manuscript had been drafted,
in response to a pre-submission review. Nothing in this amendment was pre-specified. No
threshold is re-tuned and no pre-specified result is replaced: the pre-specified analyses
remain the primary results, and every item below is reported and labelled post hoc.

## Clarification of the pre-specified scoring rule
- A hit is credited to an event when any alarm falls in [onset - 60 d, onset + 120 d]; the
  rule does not force a one-to-one correspondence. When the windows of two events at one
  station overlap (onsets <= 180 d apart), one alarm can be credited to both.
- An alarm after the end of a short event but before onset + 120 d counts both as a (late)
  hit and as a false alarm, because false alarms are alarms outside [onset - 60 d, end].
- Both conventions apply equally to every detector. Their frequency is tabulated.

## Event matching and scoring window (code/serra/matching_checks.py)
- One-to-one rule: every alarm is assigned to at most one event. Events are processed in
  chronological order and each takes the earliest unassigned alarm in its window (for windows
  of equal length this maximises the number of matched events). A false alarm is an unmatched
  alarm outside [onset - pre, end] of every event.
- Windows: pre in {60, 30, 0} d and post in {120, 90, 60} d, under both rules (the false-alarm
  window moves with pre). Thresholds are those calibrated for the primary budget.
- Random-timing reference: POD when each station's alarm sequence is shifted circularly within
  2001-2024 by a random offset (500 replicates), which keeps the number and spacing of alarms
  and removes their alignment with the events.
- Dependence: station-level onsets are grouped into regional episodes (onsets in a region linked
  when <= 30 d apart). The year-cluster bootstrap is repeated with circular blocks of 2, 3 and 4
  consecutive years within region (2000 resamples).
- SPI-3 (D4) calibration: the alarm count of the month-end down-crossing rule is not monotone
  in its threshold c, and the pre-specified grid search met the budget of 1.0 alarm per year with
  a positive threshold in Germany and Australia. D4 is re-calibrated with the grid restricted
  to c <= 0 (closest attainable rate) and re-scored by the pre-specified rule, at every budget.
- Check: with the pre-specified rule, window and year resampling, the script reproduces
  summary.csv, paired.csv and paired_timing.csv exactly.

## Predictive validation (code/serra/incremental_forward.py)
- The pre-specified five-fold blocked-by-year cross-validation trains on years that follow the
  test block; its AUC is described as retrospective discrimination.
- Forward chaining (primary post hoc scheme): weeks of year Y in 2001-2024 are predicted by a
  model fitted to weeks t < (1 January Y) - 60 d (expanding window from 1982, annual refit).
  Sensitivity: refit every six years; blocked folds with training weeks within 60 d of a test
  block removed; blocked scheme scored on 2001-2024 only.
- Same models, same region-year cluster bootstrap (1000 resamples), for the precipitation-only
  and the temperature-aware analyses.

## Other changes made with this amendment
- Terminology: detector D1 is called the "regularised defect" (previously "RM defect"); the
  penalty is described as backward-coherence regularisation inspired by reverse-martingale
  theory. No computation changes.
- Figures: operating curves and event composites redrawn as 2 x 2 panels for legibility;
  composite panel titles state how many onsets are averaged (onsets in the last 180 days of
  the record are left out because their window is incomplete).
- Registration: this protocol and its amendments are dated files kept with the code. They were
  not lodged with an external registry.
- Independent re-run (2026-11-13, Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn
  1.9, statsmodels 0.15, Numba 0.68): evaluate_warning.py (both panels) and drift_mv.py reproduce
  the stored result files exactly; covariate_analysis.py and covariate_analysis_mv.py reproduce
  them to within 2e-6. forecast_eval.py and the training scripts were not re-run.

# AMENDMENT 3 — further post hoc checks and one correction
Written 2026-11-14, AFTER every result above was known, in response to a second pre-submission
review. Nothing in this amendment was pre-specified. Pre-specified results remain reported.

## Correction: calibration of the monthly SPI-3 rule (D4)
- Amendment 2 showed that the pre-specified grid search can meet the alarm budget with a positive
  threshold c, at which the down-crossing rule alarms on returns from wet conditions. The tables
  of the main paper now use D4 with its threshold restricted to c <= 0 (closest attainable rate,
  calibrated on 1991-2000 as before). This is a correction of the comparator, made after the
  results were seen, and is labelled post hoc wherever it appears.
- The pre-specified calibration and its results are reported beside it in Online Resource 1.
  Where the pre-specified threshold is already negative the two coincide.

## Detection by deadline (code/serra/review_checks.py)
- Timing compared on events detected by both detectors conditions on joint detection. The
  unconditional summary is the fraction of all reference onsets for which a detector has alarmed
  by day d relative to onset (alarms counted from day -60), d = -60..+120; a missed onset never
  counts. Paired differences carry year-cluster bootstrap intervals (same resamples as the
  warning evaluation).

## Reference events from a fixed historical climatology (review_checks.py)
- The pre-specified soil-wetness percentiles pool all years 1981-2024, so the dating of an onset
  in an early evaluation year depends on later years. Sensitivity: percentile reference fixed to
  1981-2000 (+/-15-day pooling), applied unchanged to 2001-2024; same run rules; all detectors
  re-scored at their calibrated thresholds (calibration does not use the events).
- The forward-chaining incremental-information analysis is repeated with labels from these
  events (incremental_forward.py --fixedref).
- Check: with the all-years reference the script reproduces the stored events exactly.

## Forecast equivalence with year-cluster resampling (review_checks.py)
- The pre-specified equivalence test uses t intervals over stations. Sensitivity: 90% percentile
  intervals from resampling the 24 calendar years of 2001-2024 within region (2000 resamples),
  with every station's MSE recomputed on the resampled years.

## Other changes made with this amendment
- The statement that timing on jointly detected events avoids selection effects is withdrawn.
- The number of distinct MERRA-2 grid cells (0.5 x 0.625 degrees) containing the stations is
  reported; the reference events are described as defined independently of the detector inputs.
- The single-station trace figure moves to Online Resource 1; the pooled detection-by-deadline
  figure takes its place in the main paper. The operating-curve figure shows SPI-3 for c <= 0.
- code/serra/extract_intermediates.py writes results/checks/intermediates.npz (raw soil wetness,
  station-year forecast errors); code/serra/verify_from_csv.py rebuilds every table and number
  from the result files alone; make_tables.py caches the station-network statistics in
  results/checks/data_table.csv so that it no longer needs the daily inputs.
