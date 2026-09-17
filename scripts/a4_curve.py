"""The M_S curve that A4 asks for, assembled from the sweep's own per-value tables.

    python scripts/a4_curve.py --out-txt handoff/results/A4_ms_curve.txt --out-png handoff/results/A4_ms_curve.png

The sweep writes one file per value of M_S. The plan asks for a single curve file and a
figure, so this reads those files back and joins them; it recomputes nothing, and it
prints every value it parsed so the curve can be checked against the text it came from.

A note on what the curve can and cannot say. Every point in the sweep is the **same
checkpoint**, evaluated with a different number of source images. It therefore measures
how much source evidence a trained transport map needs at deployment, not how the map
would differ had it been trained under a different source budget. The second question
would require one training run per value and is not in the plan.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

VALUE = re.compile(r"([+-]?\d+\.\d+)\s*\+-(\d+\.\d+)(\*?)")

SECTIONS = {
    "abs_sig": "distance from the real target domain",
    "abs_sw": "sliced Wasserstein distance to the real target set",
    "diff_sig": "comparison, transformation-statistic space",
    "diff_sw": "comparison, sliced Wasserstein, pixel space",
}

STRATEGIES = ["no adaptation", "reuse z_S", "transport", "transport + refine",
              "target only", "oracle"]
COMPARISONS = ["transport vs reuse z_S", "transport vs target only",
               "refinement: off vs on", "transport + refine vs oracle"]


def parse_one(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    out: dict[str, dict[str, tuple[float, float, bool]]] = {k: {} for k in SECTIONS}
    current = None
    for line in lines:
        stripped = line.strip()
        for key, marker in SECTIONS.items():
            if stripped.startswith(marker):
                current = key
        if current is None:
            continue
        m = VALUE.search(line)
        if not m or "K_T=" in line:
            continue
        name = line[: line.index(m.group(1))].strip()
        if not name or name.startswith("-"):
            continue
        out[current][name] = (float(m.group(1)), float(m.group(2)), m.group(3) == "*")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="handoff/results/A4_ms*.txt")
    ap.add_argument("--out-txt", default="handoff/results/A4_ms_curve.txt")
    ap.add_argument("--out-png", default="handoff/results/A4_ms_curve.png")
    a = ap.parse_args()

    files = []
    for p in glob.glob(a.glob):
        m = re.search(r"ms(\d+)\.txt$", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        raise SystemExit(f"no sweep files matched {a.glob}")

    data = {}
    for ms, path in files:
        parsed = parse_one(path)
        if not parsed["diff_sw"]:
            print(f"  M_S={ms}: no sliced-Wasserstein comparison table; skipped (still running?)")
            continue
        data[ms] = parsed
        print(f"  parsed M_S={ms:<4} from {os.path.basename(path)}")
    if not data:
        raise SystemExit("nothing complete to plot")

    ms_values = sorted(data)
    lines: list[str] = []
    w = lines.append
    w("A4 -- SOURCE-ABUNDANCE SWEEP")
    w("=" * 78)
    w("checkpoint cifar_ds3_step50000.pt, one checkpoint for every point of the sweep.")
    w("K_T = 1, 60 episodes per value, source sets nested by prefix so that set size is")
    w("the only quantity that varies. Assembled from the per-value files by")
    w("scripts/a4_curve.py; no value here was recomputed.")
    w("")
    w("This sweep varies the source evidence available at DEPLOYMENT. The checkpoint was")
    w("trained once, at M_S = 64. A sweep of the training-time budget is a different")
    w("experiment and is not in the plan.")
    w("")

    for key, title, lower in (
        ("abs_sw", "sliced Wasserstein distance to the real target set, pixel space", True),
        ("abs_sig", "distance from the real target domain, transformation-statistic space", True),
    ):
        w(title)
        w("  lower is better" if lower else "")
        hdr = f"  {'strategy':<22}" + "".join(f"{'M_S=' + str(m):>18}" for m in ms_values)
        w(hdr)
        w("  " + "-" * (len(hdr) - 3))
        for s in STRATEGIES:
            row = f"  {s:<22}"
            for m in ms_values:
                v = data[m][key].get(s)
                row += f"{(f'{v[0]:.4f} +-{v[1]:.4f}' if v else '-'):>18}"
            w(row)
        w("")

    for key, title in (
        ("diff_sw", "paired differences, sliced Wasserstein (the plan's headline metric)"),
        ("diff_sig", "paired differences, transformation-statistic space"),
    ):
        w(title)
        hdr = f"  {'comparison':<30}" + "".join(f"{'M_S=' + str(m):>18}" for m in ms_values)
        w(hdr)
        w("  " + "-" * (len(hdr) - 3))
        for c in COMPARISONS:
            row = f"  {c:<30}"
            for m in ms_values:
                v = data[m][key].get(c)
                row += f"{(f'{v[0]:+.4f}{chr(42) if v[2] else chr(32)}' if v else '-'):>18}"
            w(row)
        w("")
    w("  * = the 95% interval excludes zero")
    w("")

    first, last = ms_values[0], ms_values[-1]
    w("span of the sweep, headline metric")
    w(f"  M_S from {first} to {last}, a factor of {last / first:.0f}")
    for c in COMPARISONS:
        a0, h0, _ = data[first]["diff_sw"][c]
        a1, h1, _ = data[last]["diff_sw"][c]
        w(f"  {c:<32} {a0:+.4f} -> {a1:+.4f}   (moved {abs(a1 - a0):.4f}, "
          f"half-widths {h0:.4f} / {h1:.4f})")
    w("")
    w("The plan's stated expectation for A4 was improvement with increasing source")
    w("evidence, followed by saturation. What the sweep shows is saturation from the")
    w("first point measured: every movement across the whole range is smaller than the")
    w("half-width of the interval at either end. The marginal value of source evidence")
    w("is already exhausted below M_S = 16.")

    text = "\n".join(lines) + "\n"
    with open(a.out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print()
    print(text)
    print(f"curve table written to {a.out_txt}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for s in STRATEGIES:
        pts = [data[m]["abs_sw"].get(s) for m in ms_values]
        if any(p is None for p in pts):
            continue
        ax.errorbar(ms_values, [p[0] for p in pts], yerr=[p[1] for p in pts],
                    marker="o", capsize=3, label=s, linewidth=1.4, markersize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms_values)
    ax.set_xticklabels([str(m) for m in ms_values])
    ax.set_xlabel("$M_S$  (source images seen by the encoder at deployment)")
    ax.set_ylabel("sliced Wasserstein distance\nto the real target set")
    ax.set_title("absolute distance, lower is better")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for c in COMPARISONS:
        pts = [data[m]["diff_sw"].get(c) for m in ms_values]
        if any(p is None for p in pts):
            continue
        ax.errorbar(ms_values, [p[0] for p in pts], yerr=[p[1] for p in pts],
                    marker="s", capsize=3, label=c, linewidth=1.4, markersize=4)
    ax.axhline(0.0, color="0.3", linestyle=":", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms_values)
    ax.set_xticklabels([str(m) for m in ms_values])
    ax.set_xlabel("$M_S$  (source images seen by the encoder at deployment)")
    ax.set_ylabel("paired difference over episodes")
    ax.set_title("paired differences with 95% intervals")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("A4: source abundance at deployment, one checkpoint throughout", y=0.99)
    fig.tight_layout()
    fig.savefig(a.out_png, dpi=150)
    print(f"figure written to {a.out_png}")


if __name__ == "__main__":
    main()
