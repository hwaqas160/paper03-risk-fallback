"""
Figures for the manuscript. Every value is read from results/final/*.json or recomputed from the
stored per-scenario records with the same functions as src/evaluate.py, so the figures cannot drift
from the tables.

    python paper/make_figures.py        # writes paper/figures/*.pdf

Palette: first three categorical slots of the validated reference palette (blue, orange, aqua) plus
neutral gray for reference points. Every series also has its own marker or a direct label, so no
identity depends on color alone (IEEE print is often grayscale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "paper" / "figures"
FINAL = ROOT / "results" / "final"

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
ALPHA = 0.05
COL_W = 3.5  # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name):
    return json.loads((FINAL / name).read_text())


def target_line(ax, orient="h"):
    (ax.axhline if orient == "h" else ax.axvline)(100 * ALPHA, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)


# --------------------------------------------------------------------------- F2
def fig_operating_points():
    from evaluate import Scenario, lam_grid, load_rows, matrices
    h = load("eval_main.json")["headline"]
    groups = {  # name in json -> (label, color, marker)
        "T4 LTT (ours)": ("LTT (certified)", BLUE, "o"),
        "tuned geometric (ours, no guarantee)": ("Tuned geometric = CRC", BLUE, "s"),
        "T1 tuned confidence (most conflicting agent)": ("T1 confidence", ORANGE, "^"),
        "T2 tuned ensemble (MC dropout)": ("T2 ensemble", ORANGE, "v"),
        "T3 open-loop conformal": ("T3 open-loop conformal", ORANGE, "D"),
        "CDT (Lekeufack+ 2024)": ("CDT", AQUA, "P"),
        "ACI (Gibbs+Candes 2021)": ("ACI", AQUA, "X"),
        "oracle (hindsight, not deployable)": ("Oracle (hindsight)", GRAY, "*"),
        "never": ("Never fire", GRAY, "o"),
        "always": ("Always fire", GRAY, "o"),
    }
    test = [s for s in (Scenario(r) for r in load_rows([str(ROOT / "results/campaign/av2_test")]))
            if "conf_conflict" in s.traces]
    g = lam_grid(test, "geom")
    M, S = matrices(test, "geom", g)[:2]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot(100 * S.mean(0), 100 * M.mean(0), color=GRAY, lw=1.0, zorder=2,
            label="Geometric score, every threshold")
    target_line(ax)
    # Direct offsets for well-separated points; leader lines for the cluster below (CDT/oracle/
    # tuned/LTT sit within 5 points of x and y of each other and cannot take direct labels).
    offsets = {"T1 confidence": (-40, 7), "T2 ensemble": (6, 1),
               "ACI": (6, 2), "Never fire": (8, -13), "Always fire": (-20, 12)}
    for key, (lab, col, mk) in groups.items():
        v = h[key]
        x, y = 100 * v["unnecessary_stop"], 100 * v["miss"]
        ax.scatter(x, y, s=34 if mk != "*" else 70, marker=mk, color=col, edgecolor="white",
                  linewidth=0.8, zorder=4)
        if lab in offsets:
            ax.annotate(lab, (x, y), xytext=offsets[lab], textcoords="offset points",
                       fontsize=6.5, color=INK)
    # Two clusters where points sit too close together for direct offset labels: stack each in a
    # column with a thin leader line back to its point.
    key_of = {v[0]: k for k, v in groups.items()}

    def leader_stack(labels, tx, ty0, dty, ha="left"):
        for i, lab in enumerate(labels):
            v = h[key_of[lab]]
            x, y = 100 * v["unnecessary_stop"], 100 * v["miss"]
            ax.annotate(lab, (x, y), xytext=(tx, ty0 - dty * i), fontsize=6.5, color=INK, ha=ha,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=2, shrinkB=4))

    leader_stack(["CDT", "Oracle (hindsight)", "Tuned geometric = CRC", "LTT (certified)"], 16, 14.2, 2.5)
    leader_stack(["T3 open-loop conformal"], 56, 8.0, 0, ha="right")
    ax.text(88, 100 * ALPHA + 0.4, "5 % target", fontsize=6.5, color=INK2, ha="right")
    ax.set_xlabel("Unnecessary stops (%)")
    ax.set_ylabel("Missed interventions (%)")
    ax.set_xlim(-3, 90)
    ax.set_ylim(-1, 18.5)
    ax.grid(True, zorder=0)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, bbox_to_anchor=(1.0, 0.86))
    fig.savefig(FIG / "operating_points.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- F3
def fig_validity():
    from evaluate import Scenario, crc, load_rows, ltt, summarize, tuned
    cache = FINAL / "validity_draws.json"
    if cache.exists():
        draws = json.loads(cache.read_text())
    else:
        # Load each split separately: calibration and test are separate databases that reuse seed
        # numbers, and load_rows de-duplicates by seed across all paths passed in one call.
        scns = []
        for split in ("av2_cal", "av2_test"):
            scns += [s for s in (Scenario(r) for r in load_rows([str(ROOT / "results/campaign" / split)]))
                     if "conf_conflict" in s.traces]
        assert len(scns) == 3990, len(scns)
        rng = np.random.default_rng(0)  # identical draws to evaluate.validity(seed=0)
        n = len(scns)
        draws = {"LTT (certified)": [], "CRC": [], "Tuned (no certificate)": []}
        for _ in range(200):
            p = rng.permutation(n)
            cal = [scns[i] for i in p[: n // 2]]
            test = [scns[i] for i in p[n // 2:]]
            draws["Tuned (no certificate)"].append(summarize(test, "geom", tuned(cal, "geom", ALPHA))["miss"])
            draws["LTT (certified)"].append(summarize(test, "geom", ltt(cal, "geom", ALPHA, 0.10))["miss"])
            draws["CRC"].append(summarize(test, "geom", crc(cal, "geom", ALPHA))["miss"])
        cache.write_text(json.dumps(draws))
    order = [("LTT (certified)", BLUE), ("CRC", AQUA), ("Tuned (no certificate)", ORANGE)]
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 1.95), sharex=True)
    bins = np.arange(1.5, 8.01, 0.25)
    for ax, (lab, col) in zip(axes, order):
        x = 100 * np.asarray(draws[lab])
        viol = np.mean(x > 100 * ALPHA)
        counts, _ = np.histogram(x, bins=bins)
        ax.hist(x[x <= 100 * ALPHA], bins=bins, color=col, alpha=0.9, rwidth=0.85)
        ax.hist(x[x > 100 * ALPHA], bins=bins, color=col, alpha=0.9, rwidth=0.85, hatch="////", edgecolor="white")
        ax.axvline(100 * ALPHA, color=INK2, lw=0.8, ls=(0, (4, 2)))
        ax.set_ylim(0, counts.max() * 1.7)  # headroom so the title text clears the tallest bar
        # Left-aligned so the label never crosses the dashed target line at 5 %.
        ax.text(0.01, 0.94, f"{lab}: {100 * viol:.1f} % violate",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Test miss rate over 200 calibration/test resplits (%)")
    fig.savefig(FIG / "validity.pdf")
    plt.close(fig)
    return {k: float(np.mean(np.asarray(v) > ALPHA)) for k, v in draws.items()}


# --------------------------------------------------------------------------- F4 (shift + latency)
def fig_shift_latency():
    e = load("eval_all.json")
    main = load("eval_main.json")
    ltt = main["headline"]["T4 LTT (ours)"]
    rows = [("AV2 test\n(in domain)", ltt, None),
            ("nuScenes", e["shift"]["unweighted LTT"], e["shift"]["weighted LTT (repair)"]),
            ("Waymo", e["shift_waymo"]["unweighted LTT"], e["shift_waymo"]["weighted LTT (repair)"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.85), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(rows))
    w = 0.36
    for i, (lab, u, wt) in enumerate(rows):
        for dx, v, col, hat in ((-w / 2 if wt else 0, u, BLUE, None), (w / 2, wt, ORANGE, "////")):
            if v is None:
                continue
            m = 100 * v["miss"]
            lo, hi = 100 * v["miss_ci"][0], 100 * v["miss_ci"][1]
            a.bar(i + dx, m, width=w * 0.92, color=col, hatch=hat, edgecolor="white", linewidth=0.5, zorder=3)
            a.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.7, capsize=1.8, zorder=4)
    target_line(a)
    a.set_xticks(x)
    a.set_xticklabels([r[0] for r in rows])
    a.set_ylabel("Missed interventions (%)")
    a.set_ylim(0, 19.5)  # headroom so the legend clears the nuScenes error-bar cap (~15 %)
    a.set_yticks([0, 5, 10, 15])
    a.set_title("(a) Threshold certified on AV2", loc="left")
    # Explicit patch handles: empty a.bar() proxies rendered both swatches blue.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, label="Unweighted"),
               Patch(facecolor=ORANGE, hatch="////", edgecolor="white", label="Weighted")]
    a.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.1, fontsize=6.5,
            bbox_to_anchor=(-0.03, 1.02), labelspacing=0.3, borderaxespad=0.1)
    a.grid(True, axis="y", zorder=0)
    ab = main["ablations"]
    lat = [ab[f"latency: +{k} tick(s)"] for k in (0, 1, 2)]
    xs = [0, 0.5, 1.0]
    ys = [100 * v["miss"] for v in lat]
    b.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=4, zorder=3)
    # First point: place below in DATA coordinates (clears the y=5 target line without reaching
    # the y=0 tick labels); points-offset placement above collided with one or the other.
    b.annotate(f"{ys[0]:.1f} %\nstops {100 * lat[0]['unnecessary_stop']:.0f} %", (xs[0], 1.55),
               ha="center", va="center", fontsize=6, color=INK)
    for xx, yy, v in list(zip(xs, ys, lat))[1:]:
        b.annotate(f"{yy:.1f} %\nstops {100 * v['unnecessary_stop']:.0f} %", (xx, yy), xytext=(0, 8),
                   textcoords="offset points", ha="center", va="bottom", fontsize=6, color=INK)
    target_line(b)
    b.set_xticks(xs)
    b.set_xlabel("Actuation latency (s)")
    b.set_title("(b) Latency, AV2 test", loc="left")
    b.set_ylim(0, 12.5)
    b.set_xlim(-0.2, 1.2)
    b.grid(True, axis="y", zorder=0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "shift_latency.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_operating_points()
    v = fig_validity()
    fig_shift_latency()
    print("validity violation (should match 9.5 / 41.0 / 43.5 %):", {k: round(100 * x, 1) for k, x in v.items()})
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
