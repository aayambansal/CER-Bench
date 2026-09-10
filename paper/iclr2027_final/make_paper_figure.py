"""Re-layout verified disclosure statistics for legibility; no new results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
source = ROOT / "results/readiness/qrel_disclosure/v2/summary.json"
s = json.loads(source.read_text())
assert s["replicates"] == 1000 and s["denominators"]["supported"] == 108
f = s["fractions"]
x = np.array([r["fraction"] for r in f])
highlights = {"agent": ("Agent", "#D55E00"), "bge": ("BGE", "#0072B2"),
              "splade": ("SPLADE", "#009E73"), "hybrid": ("Hybrid", "#CC79A7")}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(6.2, 4.8), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
a = fig.add_subplot(gs[0, :])
b = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])
others = [name for name in f[0]["systems"] if name not in highlights]
for i, name in enumerate(others):
    y = [r["systems"][name]["mean_R20"]["mean"] for r in f]
    lo, hi = np.array([r["systems"][name]["mean_R20"]["empirical95"] for r in f]).T
    a.plot(x, y, color="#999999", lw=0.8, alpha=0.8, label="Other six systems" if i == 0 else None)
    a.fill_between(x, lo, hi, color="#999999", alpha=0.035)
for name, (label, color) in highlights.items():
    y = [r["systems"][name]["mean_R20"]["mean"] for r in f]
    lo, hi = np.array([r["systems"][name]["mean_R20"]["empirical95"] for r in f]).T
    a.plot(x, y, color=color, lw=1.6, label=label)
    a.fill_between(x, lo, hi, color=color, alpha=0.09)
    b.plot(x, [r["systems"][name]["top1_tie_adjusted_win_rate"] for r in f], color=color, lw=1.6)
tau = [r["tau_b_vs_seed"]["mean"] for r in f]
lo, hi = np.array([r["tau_b_vs_seed"]["empirical95"] for r in f]).T
c.plot(x, tau, color="#333333", lw=1.6)
c.fill_between(x, lo, hi, color="#777777", alpha=0.2)
a.set_title("(a) Recall of identifier labels", loc="left")
a.set_ylabel("Mean R@20")
a.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
b.set_title("(b) Winner frequency", loc="left")
b.set_ylabel("Top-1 vote fraction")
b.set_ylim(-0.03, 1.03)
c.set_title("(c) Agreement with seed order", loc="left")
c.set_ylabel("Kendall tau-b")
c.set_ylim(0.3, 1.03)
for ax in (a, b, c):
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Fraction of pooled judgments revealed")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
out = HERE / "figures/qrel_disclosure_readable.pdf"
fig.savefig(out, metadata={"Title": "Conditional disclosure of historical relevance labels", "Author": "Anonymous Authors", "CreationDate": None, "ModDate": None})
fig.savefig(out.with_suffix(".png"), dpi=180)
plt.close(fig)
(HERE / "figures/readable_figure_provenance.json").write_text(json.dumps({
    "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "pdf_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "new_computation": "presentation_only_same_saved_statistics", "systems_shown": 10,
    "highlighted_systems": list(highlights), "empirical_intervals_not_population_confidence": True
}, indent=2) + "\n")
