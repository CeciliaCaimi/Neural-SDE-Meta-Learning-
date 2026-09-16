"""Plot the K_T curve that A3 asks for, from a gen_strategies table.

    python scripts/plot_kt_curve.py handoff/results/A3_kt_curve.txt --out handoff/results/A3_kt_curve.png

Reads the two metric tables back out of the run's own output rather than recomputing
anything, and prints what it parsed so the plot can be checked against the text.
"""
from __future__ import annotations

import argparse
import re

VALUE = re.compile(r"(-?\d+\.\d+)\s*\+-(\d+\.\d+)")

TABLES = {
    "sig": "distance from the real target domain",
    "sw": "sliced Wasserstein distance",
}


def parse(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    k_shots: list[int] = []
    tables: dict[str, dict[str, list[tuple[float, float]]]] = {}
    current = None
    for line in lines:
        for key, marker in TABLES.items():
            if line.strip().startswith(marker):
                current = key
                tables[current] = {}
        if current and "K_T=" in line and "strategy" in line:
            k_shots = [int(m) for m in re.findall(r"K_T=(\d+)", line)]
        if current and tables.get(current) is not None:
            vals = VALUE.findall(line)
            if vals and "K_T=" not in line:
                name = line[: line.index(vals[0][0])].strip()
                if name and not name.startswith("-"):
                    tables[current][name] = [(float(m), float(h)) for m, h in vals]
        if current and line.strip().startswith("the two questions"):
            current = None
    return k_shots, tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k_shots, tables = parse(a.table)
    if not k_shots:
        raise SystemExit("no K_T header found; is this a gen_strategies table?")
    print("K_T values:", k_shots)
    for key, rows in tables.items():
        print(f"\n{key}: {len(rows)} strategies")
        for name, vals in rows.items():
            if len(vals) != len(k_shots):
                raise SystemExit(f"row '{name}' has {len(vals)} values, expected {len(k_shots)}")
            print("  ", f"{name:<20}", "  ".join(f"{m:.3f}+-{h:.3f}" for m, h in vals))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "no adaptation":     dict(color="0.6", ls=":",  marker="o"),
        "reuse z_S":         dict(color="tab:orange", ls="--", marker="s"),
        "transport":         dict(color="tab:blue", ls="-", marker="o", lw=2.2),
        "transport + refine": dict(color="tab:cyan", ls="-", marker="^"),
        "target only":       dict(color="tab:red", ls="-", marker="v"),
        "oracle":            dict(color="tab:green", ls="-.", marker="*"),
        "mean z_S":          dict(color="tab:purple", ls="--", marker="d"),
        "shuffled z_S":      dict(color="tab:brown", ls="--", marker="x"),
        "relation only":     dict(color="tab:pink", ls="--", marker="P"),
    }
    keys = [k for k in ("sig", "sw") if k in tables]
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.6))
    if len(keys) == 1:
        axes = [axes]
    titles = {"sig": "transformation-statistic distance to the real target domain",
              "sw": "sliced Wasserstein distance, pixel space"}
    for ax, key in zip(axes, keys):
        for name, vals in tables[key].items():
            mu = [v[0] for v in vals]
            err = [v[1] for v in vals]
            ax.errorbar(k_shots, mu, yerr=err, capsize=3, ms=5,
                        label=name, **style.get(name, {}))
        ax.set_xscale("log")
        ax.set_xticks(k_shots)
        ax.set_xticklabels([str(k) for k in k_shots])
        ax.set_xlabel("$K_T$  (target images available at adaptation)")
        ax.set_ylabel("distance to the real target set   (lower is better)")
        ax.set_title(titles[key], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
