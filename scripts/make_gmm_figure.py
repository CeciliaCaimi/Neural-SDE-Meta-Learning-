"""Generate the 2D Gaussian mixture figure for the report (inline SVG, theme-aware).

Three panels: source samples, target samples after the relation, and isotropic noise as control.
Every point comes from the real task family; nothing here is schematic.
"""
from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from domains.gmm2d import build_task_family                      # noqa: E402

W, H = 236, 236           # size of one panel
LIM = 4.2                 # the data range +-LIM is mapped onto the panel
N_PTS = 420


def to_px(p: torch.Tensor, ox: float) -> list[tuple[float, float]]:
    """Data coordinates -> panel pixels (the y axis is flipped)."""
    out = []
    for x, y in p.tolist():
        px = ox + (x + LIM) / (2 * LIM) * W
        py = 40 + (LIM - y) / (2 * LIM) * H
        if 0 <= px - ox <= W and 0 <= py - 40 <= H:
            out.append((round(px, 1), round(py, 1)))
    return out


def panel(ox: float, title: str, sub: str, pts, colour: str,
          means=None) -> str:
    s = [f'<rect x="{ox}" y="40" width="{W}" height="{H}" fill="var(--paper)" '
         f'stroke="var(--rule)" stroke-width="1"/>']
    # Axes
    cx, cy = ox + W / 2, 40 + H / 2
    s.append(f'<line x1="{ox+6}" y1="{cy}" x2="{ox+W-6}" y2="{cy}" '
             f'stroke="var(--rule)" stroke-width="0.8"/>')
    s.append(f'<line x1="{cx}" y1="46" x2="{cx}" y2="{40+H-6}" '
             f'stroke="var(--rule)" stroke-width="0.8"/>')
    # Shared attributes hoisted onto the group; each point keeps only coordinates, halving the file
    s.append(f'<g fill="{colour}" opacity="0.5">')
    s.append("".join(f'<circle cx="{x}" cy="{y}" r="1.7"/>' for x, y in pts))
    s.append("</g>")
    if means is not None:
        for x, y in to_px(means, ox):
            s.append(f'<circle cx="{x}" cy="{y}" r="4.2" fill="none" '
                     f'stroke="{colour}" stroke-width="1.8"/>')
    s.append(f'<text x="{ox}" y="30" font-family="var(--mono)" font-size="11.5" '
             f'font-weight="600" fill="{colour}">{title}</text>')
    s.append(f'<text x="{ox}" y="{40+H+18}" font-family="var(--mono)" font-size="9.5" '
             f'fill="var(--muted)">{sub}</text>')
    return "\n      ".join(s)


def main() -> None:
    g = torch.Generator().manual_seed(3)
    fam = build_task_family(n_train=192, n_test=32, seed=12345)

    # Pick a task whose four components are clearly separated (all tasks are built identically)
    best, best_sep = None, -1.0
    for t in fam["test"]:
        if t.relation != "rotate":
            continue
        d = torch.cdist(t.source.means, t.source.means)
        sep = d[~torch.eye(4, dtype=torch.bool)].min()
        if sep > best_sep:
            best, best_sep = t, float(sep)
    t = best
    print(f"task {t.task_id}, relation {t.relation}, min component separation {best_sep:.2f}")

    src = t.source.sample(N_PTS, g)
    tgt = t.target.sample(N_PTS, g)
    noise = torch.randn(N_PTS, 2, generator=g)

    gap = 26
    ox1, ox2, ox3 = 0, W + gap, 2 * (W + gap)
    total_w = 3 * W + 2 * gap
    svg = f'''<svg viewBox="0 0 {total_w} {H + 76}" role="img"
         aria-label="Three scatter panels. Left: 420 samples of the source distribution forming four separated clusters, with open circles marking the component means. Middle: the target distribution after a rotation relation, which preserves the cluster structure while rotating it by sixty degrees. Right: isotropic Gaussian noise as a control, unimodal and without cluster structure.">
      {panel(ox1, "source", f"four-component Gaussian mixture | {N_PTS} samples | circles are component means", to_px(src, ox1), "var(--shared)", t.source.means)}
      {panel(ox2, "target = rotate(source)", "after the relation | cluster structure preserved, rotated 60 deg", to_px(tgt, ox2), "var(--coord)", t.target.means)}
      {panel(ox3, "isotropic noise", "control | unimodal, no cluster structure", to_px(noise, ox3), "var(--muted)")}
    </svg>'''

    out = os.path.join(_ROOT, "artifacts", "gmm_figure.svg")
    open(out, "w", encoding="utf-8").write(svg)
    print(f"wrote {out}  ({len(svg)/1024:.0f} KB)")

    # Measured statistics quoted in the body of the report
    own = torch.cdist(src, t.source.means).argmin(1)
    within = torch.stack([src[own == j].std(0).mean() for j in range(4)
                          if (own == j).sum() > 1]).mean()
    dd = torch.cdist(t.source.means, t.source.means)
    between = dd[~torch.eye(4, dtype=torch.bool)].mean()
    print(f"within-component spread {within:.3f} | mean between-component distance {between:.3f} | ratio {between/within:.1f}x")
    print("weights", t.source.weights.numpy().round(3))


if __name__ == "__main__":
    main()
