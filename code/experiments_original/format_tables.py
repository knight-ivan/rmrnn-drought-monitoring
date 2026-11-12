"""
format_tables.py -- emit Section-5 LaTeX rows from results JSON.

Writes <out>/section5_tables.tex with a collection of \newcommand{\TabTamsui}{...}
etc., one per table, ready to be \input into the paper. We only emit the
tabular body (rows between \toprule and \bottomrule); the surrounding
float wrapper stays in the paper.
"""
import json
from pathlib import Path


def fmt(v, sd, decimals=3):
    return f"{v:.{decimals}f}\\,$\\pm$\\,{sd:.{decimals}f}"


def load(out_dir: Path, name: str):
    return json.load(open(out_dir / f"{name}_results.json"))


def bold_best(values, lower_is_better=True):
    """Return a list of strings with the best value bolded."""
    if lower_is_better:
        best = min(values)
    else:
        best = max(values)
    return [("\\textbf{" + v + "}") if v == best else v for v in values]


def row(model_label, metrics, cols, decimals):
    cells = []
    for col, dec in zip(cols, decimals):
        mean, sd = metrics[col]
        cells.append(fmt(mean, sd, dec))
    return cells


def render_table(results, models, cols, decimals, leads):
    """Build a list of [lead_label, model_label, cell1, cell2, ...] rows."""
    rows = []
    for lead in leads:
        lead_key = str(lead)
        for m in models:
            if m not in results["table"][lead_key]:
                continue
            label = {
                "persistence": "Persistence",
                "climatology": "Climatology",
                "gru": "GRU",
                "rmrnn_ablation": "\\RMRNN$_{\\lambda=0}$",
                "rmrnn": "\\RMRNN",
            }[m]
            cells = row(m, results["table"][lead_key][m], cols, decimals)
            rows.append((str(lead) + "\\,h", label, cells))
    return rows


def tabulate(rows, n_cols):
    """Emit the table body as LaTeX rows."""
    lines = []
    last_lead = None
    for lead_lbl, model_lbl, cells in rows:
        this_lead = lead_lbl if lead_lbl != last_lead else ""
        if this_lead != "" and last_lead is not None:
            lines.append("\\midrule")
        lines.append(f"{this_lead} & {model_lbl} & " + " & ".join(cells) + " \\\\")
        last_lead = lead_lbl
    return "\n".join(lines)


def main():
    out_dir = Path("results")
    tex_path = Path("section5_tables.tex")

    blocks = []

    # Tamsui & Zhuoshui: RMSE, MAE, CRPS, Brier(20), POD(20), FAR(20)
    cwa_cols = ["RMSE", "MAE", "CRPS", "Brier", "POD", "FAR"]
    cwa_dec  = [3, 3, 3, 3, 3, 3]

    for tbl, leads in [("tamsui", [1, 6]), ("zhuoshui", [1, 6])]:
        d = load(out_dir, tbl)
        if tbl == "tamsui":
            models = ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]
        else:
            models = ["gru", "rmrnn_ablation", "rmrnn"]
        rows = render_table(d, models, cwa_cols, cwa_dec, leads)
        reps = d.get("reps", "?")
        blocks.append(f"% --- {tbl}: reps={reps} ---")
        # \Tab*Reps is kept for backward compat but no longer printed in tables
        blocks.append(f"\\newcommand{{\\Tab{tbl.capitalize()}Reps}}{{{reps}}}")
        blocks.append(f"\\newcommand{{\\Tab{tbl.capitalize()}Body}}{{%")
        blocks.append(tabulate(rows, len(cwa_cols)))
        blocks.append("}")
        blocks.append("")

    # CHIRPS TW & HoA: RMSE, MAE, CRPS, Brier(5), SPI-3 RMSE
    chirps_cols = ["RMSE", "MAE", "CRPS", "Brier", "SPI3_RMSE"]
    chirps_dec  = [3, 3, 3, 3, 3]

    for tbl, leads in [("chirps_tw", [1, 3]), ("chirps_hoa", [1, 3])]:
        d = load(out_dir, tbl)
        models = ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]
        rows = render_table(d, models, chirps_cols, chirps_dec, leads)
        reps = d.get("reps", "?")
        blocks.append(f"% --- {tbl}: reps={reps} ---")
        name = tbl.replace("_", "").capitalize()
        blocks.append(f"\\newcommand{{\\Tab{name}Reps}}{{{reps}}}")
        blocks.append(f"\\newcommand{{\\Tab{name}Body}}{{%")
        blocks.append(tabulate(rows, len(chirps_cols)))
        blocks.append("}")
        blocks.append("")

    # ERA5-Land: RMSE, MAE, CRPS, Brier(10), POD(10), FAR(10)
    era_cols = ["RMSE", "MAE", "CRPS", "Brier", "POD", "FAR"]
    era_dec  = [3, 3, 3, 3, 3, 3]
    d = load(out_dir, "era5land")
    models = ["persistence", "climatology", "gru", "rmrnn_ablation", "rmrnn"]
    rows = render_table(d, models, era_cols, era_dec, [1, 6])
    reps = d.get("reps", "?")
    blocks.append(f"% --- era5land: reps={reps} ---")
    blocks.append(f"\\newcommand{{\\TabEraReps}}{{{reps}}}")
    blocks.append(f"\\newcommand{{\\TabEraBody}}{{%")
    blocks.append(tabulate(rows, len(era_cols)))
    blocks.append("}")
    blocks.append("")

    tex_path.write_text("\n".join(blocks))
    print(f"wrote {tex_path} ({tex_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
