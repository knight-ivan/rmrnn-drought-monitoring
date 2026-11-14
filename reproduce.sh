#!/bin/sh
# Exact command sequence for the paper (see README.md). Run from the repository root.
#   sh reproduce.sh            verify: rebuild every table and number from results/ and compare
#                              with the published ones (needs nothing outside this archive)
#   sh reproduce.sh tables     also redraw the figures and recompile the PDFs
#                              (figures need the score panels results/panel*/*.npz)
#   sh reproduce.sh posthoc    re-run the post hoc checks first (needs the score panels; ~1 h on 2 cores)
#   sh reproduce.sh all        everything from the raw downloads (network; ~2 h on 16 cores)
set -e
STAGE="${1:-verify}"

if [ "$STAGE" = "verify" ]; then
  python3 code/serra/verify_from_csv.py
  exit 0
fi

if [ "$STAGE" = "all" ]; then
  # 1. data
  for r in texas germany australia; do python3 code/fetch_ghcnd_long.py --region $r; done
  STATIONS=$(python3 -c "import sys; sys.path.insert(0, 'code'); import build_taiwan as b; print(','.join(b.COORD))")
  python3 code/experiments_original/fetch_taiwan_hydromet.py --source raingel --stations "$STATIONS" \
      --years 1961-2024 --outdir data/taiwan_raingel
  python3 code/build_taiwan.py
  for r in texas germany australia taiwan; do python3 code/fetch_power.py --region $r; done
  # 2. precipitation-only analysis (pre-specified)
  python3 code/serra/train_models.py
  python3 code/serra/build_scores.py && python3 code/serra/add_spi30.py
  python3 code/serra/evaluate_warning.py
  python3 code/serra/forecast_eval.py
  python3 code/serra/covariate_analysis.py
  # 3. temperature-aware analysis (Amendment 1)
  python3 code/build_temperature.py
  python3 code/serra/train_models_mv.py && python3 code/serra/build_scores_mv.py
  python3 code/serra/evaluate_warning.py --mv && python3 code/serra/forecast_eval.py --mv
  python3 code/serra/covariate_analysis_mv.py && python3 code/serra/drift_mv.py
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "posthoc" ]; then
  # 4. post hoc robustness checks (Amendment 2)
  python3 code/serra/matching_checks.py && python3 code/serra/matching_checks.py --mv
  python3 code/serra/incremental_forward.py && python3 code/serra/incremental_forward.py --mv
  # further post hoc checks (Amendment 3)
  [ -d results/scores ] && python3 code/serra/extract_intermediates.py
  python3 code/serra/review_checks.py && python3 code/serra/review_checks.py --mv
  python3 code/serra/incremental_forward.py --fixedref && python3 code/serra/incremental_forward.py --mv --fixedref
fi

# 5. tables, numbers, figures, PDFs
python3 code/serra/text_checks.py && python3 code/serra/make_tables.py
python3 code/serra/figures.py && python3 code/serra/figures.py composite_mv
cd tex
X="pdflatex -interaction=nonstopmode -halt-on-error"
$X RM_Precipitation_SERRA_supplement && $X RM_Precipitation_SERRA && bibtex RM_Precipitation_SERRA
$X RM_Precipitation_SERRA && $X RM_Precipitation_SERRA_supplement
$X RM_Precipitation_SERRA && $X RM_Precipitation_SERRA_supplement && $X RM_Precipitation_SERRA
$X Cover_Letter_SERRA
