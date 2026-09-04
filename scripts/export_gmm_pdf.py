"""Export the report scatter figure as a vector PDF for use with LaTeX includegraphics.

Uses the same task and sampling parameters as the inline SVG version, so both agree.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from domains.gmm2d import build_task_family                      # noqa: E402

TEAL, AMBER, GREY = "#1D4E4A", "#B06A14", "#8A908E"
LIM, N_PTS = 4.2, 420


def draw(ax, pts, colour, title, sub, means=None):
    ax.scatter(pts[:, 0], pts[:, 1], s=5, c=colour, alpha=0.5, linewidths=0)
    if means is not None:
        ax.scatter(means[:, 0], means[:, 1], s=70, facecolors="none",
                   edgecolors=colour, linewidths=1.4)
    ax.axhline(0, color="#DDDAD4", lw=0.6, zorder=0)
    ax.axvline(0, color="#DDDAD4", lw=0.6, zorder=0)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(labelsize=7, colors="#69716E", length=2.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#DDDAD4"); sp.set_linewidth(0.8)
    ax.set_title(title, fontsize=9.5, color=colour, pad=7, loc="left")
    ax.set_xlabel(sub, fontsize=7, color="#69716E", labelpad=6)


def main() -> None:
    g = torch.Generator().manual_seed(3)
    fam = build_task_family(n_train=192, n_test=32, seed=12345)

    best, best_sep = None, -1.0
    for t in fam["test"]:
        if t.relation != "rotate":
            continue
        d = torch.cdist(t.source.means, t.source.means)
        sep = float(d[~torch.eye(4, dtype=torch.bool)].min())
        if sep > best_sep:
            best, best_sep = t, sep
    t = best

    src = t.source.sample(N_PTS, g).numpy()
    tgt = t.target.sample(N_PTS, g).numpy()
    noise = torch.randn(N_PTS, 2, generator=g).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    draw(axes[0], src, TEAL, "source",
         "four-component mixture; open circles are component means",
         t.source.means.numpy())
    draw(axes[1], tgt, AMBER, "target = rotate(source)",
         "cluster structure preserved, configuration rotated $60^\\circ$",
         t.target.means.numpy())
    draw(axes[2], noise, GREY, "isotropic noise",
         "control: unimodal, no cluster structure")
    fig.tight_layout(pad=1.1)

    out = os.path.join(_ROOT, "artifacts", "gmm_data.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"task {t.task_id} (relation {t.relation}, min component separation {best_sep:.2f})")
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
