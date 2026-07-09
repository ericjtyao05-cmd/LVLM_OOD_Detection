"""Turn results/<run>/summary.json into paper-ready artifacts.

The experiment is a 1-D sweep: ID-training contamination % -> real-vs-real OOD
detection (AUROC/FPR95 per detector) + clean-ID accuracy.

Outputs (under --out):
  table_main.csv    long-form table of every metric
  table_main.tex    booktabs LaTeX table
  sweep_auroc.{png,pdf}   AUROC(ood) vs contamination %, one line per detector
  sweep_fpr95.{png,pdf}   FPR95(ood) vs contamination %
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

COND_RE = re.compile(r"id(\d+)(?:_ood00)?")   # accepts legacy id00_ood00 names


def parse_pct(name: str):
    m = COND_RE.fullmatch(name)
    return int(m.group(1)) if m else None


def load_summary(results_dir: Path) -> dict:
    return json.load(open(results_dir / "summary.json"))


def _rows(summary):
    rows = []
    for res in summary["conditions"]:
        pct = parse_pct(res["condition"])
        if pct is None:
            continue
        for m, rec in res["methods"].items():
            rows.append({"condition": res["condition"], "id_fake_pct": pct,
                         "method": m, "id_acc": res.get("id_acc"), **rec})
    return sorted(rows, key=lambda r: (r["id_fake_pct"], r["method"]))


def write_csv(summary, out: Path):
    rows = _rows(summary)
    if not rows:
        return
    with open(out / "table_main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def write_latex(summary, out: Path):
    methods = summary["methods"]
    conds = [c for c in summary["conditions"] if parse_pct(c["condition"]) is not None]
    conds.sort(key=lambda c: parse_pct(c["condition"]))
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Effect of ID-training contamination with AI-generated "
             r"impossible images on \emph{real-vs-real} OOD detection. Test sets "
             r"are clean real images, identical across conditions. "
             r"AUROC$\uparrow$ / FPR95$\downarrow$; ID acc$\uparrow$.}",
             r"\label{tab:contamination}",
             r"\begin{tabular}{lc" + "cc" * len(methods) + "}",
             r"\toprule"]
    head = ["ID fake \\%", "ID acc"]
    for m in methods:
        head.append(f"\\multicolumn{{2}}{{c}}{{{m}}}")
    lines.append(" & ".join(head) + r" \\")
    sub = ["", ""] + ["AUC", "FPR95"] * len(methods)
    lines.append(" & ".join(sub) + r" \\ \midrule")
    for c in conds:
        cells = [str(parse_pct(c["condition"])),
                 f"{c.get('id_acc', float('nan')):.3f}" if c.get("id_acc") is not None else "--"]
        for m in methods:
            r = c["methods"][m]
            cells += [f"{r['auroc_ood']:.3f}", f"{r['fpr95_ood']:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "table_main.tex").write_text("\n".join(lines))


def sweeps(summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in summary["conditions"] if parse_pct(c["condition"]) is not None]
    conds.sort(key=lambda c: parse_pct(c["condition"]))
    pcts = [parse_pct(c["condition"]) for c in conds]

    for key, label, fname in (("auroc_ood", "AUROC (real ID vs real OOD)", "sweep_auroc"),
                              ("fpr95_ood", "FPR95 (real ID vs real OOD)", "sweep_fpr95")):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for m in summary["methods"]:
            ys = [c["methods"][m][key] for c in conds]
            ax.plot(pcts, ys, marker="o", label=m)
        ax.set_xlabel("ID-training fake contamination (%)")
        ax.set_ylabel(label)
        ax.set_xticks(pcts)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{fname}.{ext}", dpi=150)
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
    sweeps(summary, out)
    print(f"[paperize] wrote table_main.csv/.tex + sweep plots -> {out}/")


if __name__ == "__main__":
    main()
