"""
drive_rho.py -- idempotent rho-sensitivity driver.

Runs the spatial-filtration-radius sweep one rho value per invocation.
Partial results are stored in <out>/rho_sensitivity_results.json.
Final plot is written to <figures_dir>/rho_sensitivity.pdf once all
rho values have been scored.
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np

from run_section5 import (climate_library, ClimateSpec, run_single,
                          aggregate)

RHO_KM = [5, 10, 15, 20, 25, 30, 40, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--time-budget", type=float, default=25.0)
    ap.add_argument("--figures-dir", type=str, default="figures")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "rho_sensitivity_results.json"

    if state_file.exists():
        state = json.load(open(state_file))
    else:
        state = {"rho_km": RHO_KM, "results": {}, "reps": args.reps}

    t0 = time.time()
    base_spec = climate_library()["tamsui"]
    for rho_km in RHO_KM:
        if str(rho_km) in state["results"]:
            continue
        if time.time() - t0 > args.time_budget:
            print(f"budget hit at rho={rho_km}, stopping")
            break
        k_sites = max(1, min(base_spec.n_sites, int(round(rho_km / 6.0))))
        spec_kwargs = asdict(base_spec)
        spec_kwargs["n_sites"] = k_sites
        sp = ClimateSpec(**spec_kwargs)
        metrics = []
        for rep in range(args.reps):
            m = run_single(sp, lead=1, seed=rep, model_name="rmrnn",
                           n_epochs=args.epochs, context=24, hidden=16,
                           tau=20.0, spi_window=72, domain_key="tamsui")
            metrics.append(m)
        agg = aggregate(metrics)
        state["results"][str(rho_km)] = {k: list(v) for k, v in agg.items()}
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  rho={rho_km} km  (k={k_sites})  CRPS="
              f"{agg['CRPS'][0]:.3f}+-{agg['CRPS'][1]:.3f}  "
              f"RMSE={agg['RMSE'][0]:.3f}+-{agg['RMSE'][1]:.3f}")

    remaining = [r for r in RHO_KM if str(r) not in state["results"]]
    if not remaining:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rho_arr = np.array(RHO_KM)
            crps_mean = np.array([state["results"][str(r)]["CRPS"][0]
                                  for r in RHO_KM])
            crps_sd = np.array([state["results"][str(r)]["CRPS"][1]
                                for r in RHO_KM])
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.errorbar(rho_arr, crps_mean, yerr=crps_sd, marker='o',
                        capsize=3, lw=1.2, color='C0')
            ax.set_xlabel(r"Filtration radius $\rho$ (km)")
            ax.set_ylabel("CRPS (mm / step)")
            ax.set_title(r"RMRNN CRPS vs spatial filtration radius $\rho$"
                         "\n" f"(Taiwan CWA synthetic pilot; reps = {args.reps})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = fig_dir / "rho_sensitivity.pdf"
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"saved {fig_path}")
        except Exception as e:
            print(f"plot failed: {e}")
    else:
        print(f"rho remaining: {remaining}")


if __name__ == "__main__":
    main()
