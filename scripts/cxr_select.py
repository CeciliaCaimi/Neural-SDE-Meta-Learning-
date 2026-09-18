"""Select a ChestX-ray14 checkpoint on the validation tasks, and record why.

    python scripts/cxr_select.py cxr2_128 cxr2_64 --out handoff/results/C_cxr_model_selection

The rule, fixed before any C3 or C4 reading: among diagnostic reads whose step has a saved
checkpoint, take the lowest validation loss_correct. delta_task, gain_vs_shuffled, r_basis and
z_spread_rel are printed beside it and are NOT used to choose -- selecting on the quantities the
experiment asks about would be circular. Writes <out>.txt and <out>.json; the JSON is what the
queue reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lines = ["ChestX-ray14 model selection on the validation tasks",
             "rule: lowest validation loss_correct among checkpointed steps; the other columns "
             "are printed, not used", "",
             f"  {'run':<16}{'steps':>7}{'best val':>10}{'at step':>9}{'delta_task':>12}"
             f"{'gain_shuf':>11}{'r_basis':>9}{'z_spread':>10}{'val at end':>12}"]
    best = None
    for run in a.runs:
        p = os.path.join(_ROOT, "checkpoints", f"{run}_log.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"no log for {run}")
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        steps = max(r["step"] for r in rows if "L_meta" in r)
        cfg_ch = None
        dg = [d for d in rows if "diagnostics" in d
              and os.path.exists(os.path.join(_ROOT, "checkpoints", f"{run}_step{d['step']}.pt"))]
        if not dg:
            raise SystemExit(f"{run}: no diagnostic read has a checkpoint")
        b = min(dg, key=lambda d: d["diagnostics"]["loss_correct"])
        v = b["diagnostics"]
        last = [d for d in rows if "diagnostics" in d][-1]["diagnostics"]["loss_correct"]
        lines.append(f"  {run:<16}{steps:>7}{v['loss_correct']:>10.4f}{b['step']:>9}"
                     f"{v['delta_task']:>+12.5f}{v['gain_vs_shuffled']:>+11.5f}"
                     f"{v['r_basis']:>9.3f}{v['z_spread_rel']:>10.4f}{last:>12.4f}")
        if best is None or v["loss_correct"] < best["val"]:
            import torch
            ck = os.path.join(_ROOT, "checkpoints", f"{run}_step{b['step']}.pt")
            cfg_ch = torch.load(ck, map_location="cpu", weights_only=False)["config"]["model"][
                "backbone_kwargs"]["base_channels"]
            best = {"run": run, "step": b["step"], "val": v["loss_correct"],
                    "ckpt": f"checkpoints/{run}_step{b['step']}.pt", "channels": cfg_ch,
                    "steps_total": steps}
    lines += ["", f"chosen: {best['ckpt']}  ({best['channels']} channels, trained "
              f"{best['steps_total']} steps, validation loss {best['val']:.4f})"]
    out = os.path.join(_ROOT, a.out)
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
