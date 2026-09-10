#!/usr/bin/env python3
"""Create the data-rich, publication-quality robustness figure for the paper."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "paper_tables" / "submission_audit.json"
SCORES = ROOT / "results" / "baselines" / "scores_test_comprehensive.json"
OUT = ROOT / "paper" / "iclr2027_submission" / "figures" / "label_sensitivity_analysis"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GREY = "#8A8A8A"


def rank(values: dict[str, float]) -> dict[str, int]:
    return {
        method: position
        for position, (method, _) in enumerate(
            sorted(values.items(), key=lambda item: (-item[1], item[0])), start=1
        )
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    raw = json.loads(SCORES.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(7.05, 5.05), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.47, wspace=0.38)

    # A: method ranks under two qrel constructions.
    ax = fig.add_subplot(gs[0, 0])
    methods = [
        "BM25",
        "SPECTER2",
        "BGE",
        "E5-large",
        "MedCPT",
        "SPLADE",
        "Hybrid",
        "Hybrid+CE",
        "BM25+BGE-reranker",
        "Agent T=3",
    ]
    seeded = {m: audit["v1"]["scores"][m]["recall@20"] for m in methods}
    pooled = {m: audit["pooled_qrels"]["scores"][m]["recall@20"] for m in methods}
    seeded_rank, pooled_rank = rank(seeded), rank(pooled)
    colors = {m: GREY for m in methods}
    colors.update({"BM25": BLUE, "SPLADE": GREEN, "BGE": PURPLE, "Agent T=3": ORANGE})
    for method in methods:
        lw = 1.8 if method in {"BM25", "SPLADE", "BGE", "Agent T=3"} else 0.75
        alpha = 1.0 if lw > 1 else 0.55
        ax.plot(
            [0, 1],
            [seeded_rank[method], pooled_rank[method]],
            color=colors[method],
            lw=lw,
            alpha=alpha,
            marker="o",
            markersize=3.3 if lw > 1 else 2.3,
            zorder=3 if lw > 1 else 1,
        )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(10.6, 0.4)
    ax.set_xticks([0, 1], ["Seed qrels", "Pooled qrels"])
    ax.set_yticks(range(1, 11))
    ax.set_ylabel("R@20 rank (1 = best)")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for method in ("BM25", "SPLADE", "BGE", "Agent T=3"):
        ax.annotate(
            method.replace(" T=3", ""),
            (1, pooled_rank[method]),
            xytext=(4, 0),
            textcoords="offset points",
            color=colors[method],
            va="center",
            fontsize=6.4,
            fontweight="bold",
        )
    ax.text(0.0, 1.08, "A", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.08,
        1.08,
        "Ranking changes with qrel construction",
        transform=ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
    )

    # B: family-level v1 pattern for representative systems.
    ax = fig.add_subplot(gs[0, 1])
    families = [
        "constraint",
        "comparative",
        "contradiction",
        "multihop",
        "temporal",
        "aggregation",
        "negative",
    ]
    labels = [
        "Constr.",
        "Compar.",
        "Contrad.",
        "Multi-hop",
        "Temporal",
        "Aggreg.",
        "Negative",
    ]
    rows = [("BM25", "bm25"), ("Hybrid", "hybrid"), ("Agent", "agent")]
    matrix = np.array(
        [
            [raw[key]["per_family"][family]["recall@20"] for family in families]
            for _, key in rows
        ]
    )
    # pcolormesh keeps the heatmap cells as vector rectangles in the PDF.
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap="cividis",
        vmin=0.15,
        vmax=0.75,
        shading="flat",
        rasterized=False,
    )
    ax.set_xticks(
        np.arange(len(labels)), labels, rotation=36, ha="right", rotation_mode="anchor"
    )
    ax.set_yticks(np.arange(len(rows)), [name for name, _ in rows])
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.34 or matrix[i, j] > 0.66 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.9,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("R@20", labelpad=2)
    cbar.ax.tick_params(labelsize=6, length=2)
    ax.text(0.0, 1.08, "B", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.08,
        1.08,
        "No system dominates every task family",
        transform=ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
    )

    # C: iteration ablation, with non-agentic references.
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(3)
    metrics = ["recall@5", "recall@10", "recall@20"]
    for label, method, color, marker, linestyle in [
        ("Agent, 1 round", "Agent T=1", SKY, "o", "--"),
        ("Agent, 3 rounds", "Agent T=3", ORANGE, "o", "-"),
        ("BM25", "BM25", BLUE, "s", ":"),
        ("SPLADE", "SPLADE", GREEN, "^", "-."),
    ]:
        y = [audit["v1"]["scores"][method][metric] for metric in metrics]
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            lw=1.35,
            markersize=3.7,
        )
    ax.set_xticks(x, ["R@5", "R@10", "R@20"])
    ax.set_ylim(0.2, 0.58)
    ax.set_ylabel("Recall")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    ax.legend(
        frameon=False, ncol=2, loc="upper left", handlelength=2.5, columnspacing=0.8
    )
    style_axis(ax)
    ax.text(0.0, 1.08, "C", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.08,
        1.08,
        "Iteration increases recall at depth",
        transform=ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
    )

    # D: higher-density v2 proxy qrels (only saved/evaluated pair).
    ax = fig.add_subplot(gs[1, 1])
    width = 0.32
    bm25 = [audit["v2_proxy_qrels"]["scores"]["BM25"][metric] for metric in metrics]
    agent = [
        audit["v2_proxy_qrels"]["scores"]["Agent T=3"][metric] for metric in metrics
    ]
    bars1 = ax.bar(x - width / 2, bm25, width, label="BM25", color=BLUE)
    bars2 = ax.bar(
        x + width / 2,
        agent,
        width,
        label="Agent, 3 rounds",
        color=ORANGE,
        hatch="//",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x, ["R@5", "R@10", "R@20"])
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Recall")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    style_axis(ax)
    ax.text(0.0, 1.08, "D", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(
        0.08,
        1.08,
        "Cluster-proxy qrels reverse the pairwise result",
        transform=ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.095, top=0.945)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"), dpi=600)
    plt.close(fig)

    with Image.open(OUT.with_suffix(".png")) as image_png:
        image_png.convert("L").save(
            OUT.with_name(OUT.name + "_grayscale.png"), dpi=(600, 600)
        )
    print(f"Wrote {OUT.with_suffix('.pdf')}, .svg, .png, and grayscale QA preview")


if __name__ == "__main__":
    main()
