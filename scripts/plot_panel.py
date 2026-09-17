"""The coordinate-loss profile, Gaussian mixtures beside CIFAR.

    python scripts/plot_panel.py --out docs/figures/profile_contrast.png

This is the figure Stage B exists to produce. The same construction is applied to both:
walk the coordinate from the shared mean (s = 0) through the task's own (s = 1) and out
to twice that displacement, and record the denoising loss along the way.

The point is not that one curve sits lower -- the two families have different loss
scales and the absolute heights are not comparable. The point is the **depth of the
well**, plotted here as a percentage of the loss at s = 0, which is comparable. On
Gaussian mixtures the coordinate is worth a 26 % excursion; on CIFAR, 1.6 %. Both are
minimised in the right place, at s = 1, so the mechanism on images is not absent or
misdirected -- it is shallow, and the reading that called it flat was overstating.

Numbers are taken from the two result files rather than recomputed, and both sources are
printed so the figure can be checked against them.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

GMM_DEFAULT = "handoff/Yu-Cao/results/B_panel.txt"
CIFAR_DEFAULT = "handoff/Jinjian/results/E12a_controls.txt"
ROW = re.compile(r"^\s{2,}(\d\.\d)\s+(\d+\.\d+)\s+(\d+\.\d+)")


def read_profile(path: str) -> tuple[list[float], list[float], list[float]]:
    """Pull the first 'toward own / toward another' profile table out of a results file."""
    s_vals, own, other = [], [], []
    inside = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "toward own" in line and "toward another" in line:
                inside = True
                continue
            if inside:
                m = ROW.match(line)
                if m:
                    s_vals.append(float(m.group(1)))
                    own.append(float(m.group(2)))
                    other.append(float(m.group(3)))
                elif s_vals:
                    break
    if not s_vals:
        raise SystemExit(f"no profile table found in {path}")
    return s_vals, own, other


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gmm", default=GMM_DEFAULT)
    ap.add_argument("--cifar", default=CIFAR_DEFAULT)
    ap.add_argument("--out", default="docs/figures/profile_contrast.png")
    a = ap.parse_args()

    panels = []
    for label, path in (("2-D Gaussian mixtures (stage 1)", a.gmm),
                        ("CIFAR-100 domain shift (stage 3)", a.cifar)):
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        s, own, other = read_profile(full)
        base = own[0]
        depth = 100.0 * (base - min(own)) / base
        print(f"{label}\n  source {path}")
        print(f"  s       {'  '.join(f'{v:>7.1f}' for v in s)}")
        print(f"  own     {'  '.join(f'{v:>7.4f}' for v in own)}")
        print(f"  another {'  '.join(f'{v:>7.4f}' for v in other)}")
        print(f"  well depth toward own: {depth:.1f}% of the loss at s=0, "
              f"minimum at s={s[own.index(min(own))]:.1f}\n")
        panels.append((label, s, own, other, depth))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (label, s, own, other, depth) in zip(axes, panels):
        base = own[0]
        ax.plot(s, [100.0 * v / base for v in own], "o-", color="tab:blue", lw=2.2,
                label="toward this task's own coordinate")
        ax.plot(s, [100.0 * v / base for v in other], "s--", color="tab:red",
                label="toward another task's")
        ax.axvline(1.0, color="0.7", ls=":", lw=1)
        ax.set_title(f"{label}\nwell depth {depth:.1f}% of the loss", fontsize=10)
        ax.set_xlabel("$s$   (0 = shared mean coordinate, 1 = this task's own)")
        ax.set_ylabel("denoising loss, % of the loss at $s=0$")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("written to", a.out)


if __name__ == "__main__":
    main()
