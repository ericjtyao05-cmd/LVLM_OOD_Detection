"""Turn results/<run>/summary.json into paper-ready artifacts.

Outputs (under --out):
  table_main.csv   long-form table of every metric
  table_main.tex   booktabs LaTeX table (AUROC/FPR95, ood & fake) per condition
  heatmap_<method>_fake.{png,pdf}   AUROC(fake) over the id x ood contamination grid
  heatmap_<method>_ood.{png,pdf}    AUROC(ood)  over the same grid
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

COND_RE = re.compile(r"id(\d+)_ood(\d+)")


def parse_cond(name):
    m = COND_RE.fullmatch(name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load_summary(results_dir: Path) -> dict:
    return json.load(open(results_dir / "summary.json"))


def write_csv(summary, out: Path):
    rows = []
    for res in summary["conditions"]:
        idp, oodp = parse_cond(res["condition"])
        for m, rec in res["methods"].items():
            row = {"condition": res["condition"], "id_fake_pct": idp,
                   "ood_fake_pct": oodp, "method": m, **rec}
            rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out / "table_main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def write_latex(summary, out: Path):
    methods = summary["methods"]
    lines = [r"\begin{table}[t]\centering",
             r"\caption{OOD (real novel classes) vs.\ fake (physically-impossible) "
             r"detection across fake-contamination conditions. "
             r"AUROC$\uparrow$ / FPR95$\downarrow$.}",
             r"\label{tab:main}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l" + "cccc" * len(methods) + "}",
             r"\toprule"]
    head = ["Condition"]
    for m in methods:
        head += [f"\\multicolumn{{4}}{{c}}{{{m}}}"]
    lines.append(" & ".join(head) + r" \\")
    sub = [""] + ["AUC$_{ood}$", "FPR$_{ood}$", "AUC$_{fk}$", "FPR$_{fk}$"] * len(methods)
    lines.append(" & ".join(sub) + r" \\ \midrule")
    for res in summary["conditions"]:
        cells = [res["condition"].replace("_", r"\_")]
        for m in methods:
            r = res["methods"][m]
            cells += [f"{r['auroc_ood']:.3f}", f"{r['fpr95_ood']:.3f}",
                      f"{r['auroc_fake']:.3f}", f"{r['fpr95_fake']:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "table_main.tex").write_text("\n".join(lines))


def heatmaps(summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = summary["conditions"]
    ids = sorted({parse_cond(c["condition"])[0] for c in conds})
    oods = sorted({parse_cond(c["condition"])[1] for c in conds})
    lookup = {c["condition"]: c["methods"] for c in conds}

    for m in summary["methods"]:
        for axis, key in (("fake", "auroc_fake"), ("ood", "auroc_ood")):
            grid = np.full((len(ids), len(oods)), np.nan)
            for i, a in enumerate(ids):
                for j, b in enumerate(oods):
                    name = f"id{a:02d}_ood{b:02d}"
                    if name in lookup and m in lookup[name]:
                        grid[i, j] = lookup[name][m].get(key, np.nan)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(oods))); ax.set_xticklabels([f"{b}%" for b in oods])
            ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"{a}%" for a in ids])
            ax.set_xlabel("OOD fake %"); ax.set_ylabel("ID fake %")
            ax.set_title(f"{m}: AUROC ({axis})")
            for i in range(len(ids)):
                for j in range(len(oods)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                                color="w", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"heatmap_{m}_{axis}.{ext}", dpi=150)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = load_summary(Path(args.results))
    write_csv(summary, out)
    write_latex(summary, out)
    heatmaps(summary, out)
    print(f"[paperize] wrote table_main.csv/.tex + heatmaps -> {out}/")


if __name__ == "__main__":
    main()
