"""
drive_cells.py  -- idempotent cell-by-cell driver
-------------------------------------------------

Runs remaining (table, lead, model) cells for Section 5 one at a time.
Checks <out>/<table>_results.json and skips cells already populated.
Meant to be invoked repeatedly from short-timeout shell calls; each
invocation runs as many cells as it can in `time_budget` seconds.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from run_section5 import run_cell

# Canonical cell catalog for Section 5
CATALOG = [
    # (table, lead, model)
    # Table 3: Tamsui
    *[("tamsui", L, m) for L in [1, 6]
      for m in ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]],
    # Table 4: Zhuoshui (subset baselines)
    *[("zhuoshui", L, m) for L in [1, 6]
      for m in ["gru", "rmrnn_ablation", "rmrnn"]],
    # Table 5: CHIRPS Taiwan
    *[("chirps_tw", L, m) for L in [1, 3]
      for m in ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]],
    # Table 6: CHIRPS Horn of Africa
    *[("chirps_hoa", L, m) for L in [1, 3]
      for m in ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]],
    # Table 7: ERA5-Land (multi-variable physics)
    *[("era5land", L, m) for L in [1, 6]
      for m in ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]],
]


def cell_done(out_dir: Path, table: str, lead: int, model: str) -> bool:
    p = out_dir / f"{table}_results.json"
    if not p.exists():
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    return model in d.get("table", {}).get(str(lead), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--time-budget", type=float, default=35.0,
                    help="Seconds to keep working before exiting")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pending = [c for c in CATALOG if not cell_done(out_dir, *c)]
    print(f"{len(pending)} / {len(CATALOG)} cells remaining")

    for table, lead, model in pending:
        elapsed = time.time() - t0
        if elapsed > args.time_budget:
            print(f"time budget reached ({elapsed:.1f}s); stopping")
            break
        t_start = time.time()
        try:
            run_cell(table, lead=lead, model=model,
                     reps=args.reps, n_epochs=args.epochs,
                     out_dir=out_dir)
        except Exception as e:
            print(f"FAILED {table} lead={lead} model={model}: {e}")
        print(f"    ({time.time() - t_start:.1f}s)")

    pending = [c for c in CATALOG if not cell_done(out_dir, *c)]
    print(f"done; {len(pending)} / {len(CATALOG)} cells still remaining")


if __name__ == "__main__":
    main()
