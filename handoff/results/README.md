# Results index — Stage A and the Stage C feasibility gate

Every file here was produced by a command recorded below, on branch `jinjian`, with
`.venv/Scripts/python.exe` from the repository root. Numbers quoted in this index are
copied from the files; the files are the record.

**Standing caveats, which travel with every row.** One checkpoint
(`cifar_ds3_step50000.pt`, 50 000 steps, EMA weights). **One seed, 4321**, never varied —
so every interval is a within-run interval over episodes and none of them contains the
seed-to-seed component. The **validation** split throughout; the test split is untouched
and is spent once, at the very end, per the plan. Sixty episodes per K_T, which is the
minimum that has produced stable readings on this project.

---

## C0 — Fitzpatrick17k feasibility (Stage C gate)

`C0_feasibility.txt`

Answers whether the demographic scarcity experiment can be built. **Qualified yes.** The
annotation table is fully available; the images are not. `www.dermaamin.com`, which holds
12 631 of the 16 577 rows, fails to resolve for public resolvers as well as locally
(`EDE(22) No Reachable Authority at delegation`), so 76 % of the dataset is unreachable by
any route short of the authors' request form. `atlasdermatologico.com.br` is fully live.

Phenotype grouping **I–III vs IV–VI**, chosen from the observed counts. Atlas-only gives
16 conditions above the plan's floor, but only **8** can fund a K_T = 20 support set plus a
25-image held-out query set — a 5/2/1 task split, and one held-out test task cannot carry a
paired interval. **Atlas-only is scaffolding for C1–C3, not a headline for C4.**

---

## A3 — target-scarcity sweep

`A3_kt_curve.txt`, `A3_kt_curve.png` (the curve), `A3_grid.png` (sample grid)

```
scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 2 5 10 20 --n-classes 20 --n-samples 96 --grid-out ...
```

300 episodes. Transport beats reusing the source coordinate at **every** K_T, decaying
`+1.140 ±0.108` → `+0.229 ±0.059`, every interval clear of zero.

**Retraction recorded here.** E14 had read K_T = 20 as transport *losing* to target-only
(`-0.017 ±0.044`) and that was written up as a crossing. A3 reads `+0.008 ±0.036`: the sign
flipped and both intervals contain zero, so transport was never shown to lose. In
transformation-statistic space the advantage is confined to K_T = 1. On the headline metric
it survives to K_T = 5 — see A1.

The curve was plotted from the table by `scripts/plot_kt_curve.py`, which prints what it
parsed so the figure can be checked against the text.

---

## A1 — source-dependence panel

`A1_source_dependence.txt` (cite this one), `A1_per_episode.json`, `A1_grid.png`,
`A1_source_dependence_run1.txt` (see the provenance note inside it)

```
scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 5 20 --n-classes 20 --n-samples 96 --source-controls --json-out handoff/results/A1_per_episode.json --progress 20 --grid-out handoff/results/A1_grid.png
```

180 episodes, nine conditions. **The method beats all three source controls on the headline
metric** (sliced Wasserstein), all nine cells clear of zero and same-signed:

| transport against | K_T=1 | K_T=5 | K_T=20 |
|---|---|---|---|
| mean `z_S` | +0.0301 ±0.0072 | +0.0346 ±0.0085 | +0.0338 ±0.0080 |
| shuffled `z_S` (within relation) | +0.0416 ±0.0125 | +0.0453 ±0.0149 | +0.0436 ±0.0144 |
| relation-only | +0.0254 ±0.0078 | +0.0285 ±0.0088 | +0.0295 ±0.0087 |

**The two metrics disagree about mean `z_S`, and the disagreement is the finding.** In
transformation-statistic space the same comparison contains zero at all three K_T
(`+0.017 / -0.008 / +0.011`). That instrument is three fixed convolutions measuring which
transformation was applied, and the within-relation mean preserves the transformation
perfectly — it is blind by construction to exactly what the mean destroys. Report both,
headline the Wasserstein one, and say why they differ. An earlier reading of this panel
concluded the task-specific content was unused; that conclusion was drawn from the blind
instrument and does not hold.

**The plan's main claim is met in the low-target-data regime.** It requires beating
relation-only *and* target-only: K_T=1 gives `+0.025 ±0.008` and `+0.018 ±0.009`; K_T=5
gives `+0.029 ±0.009` and `+0.011 ±0.007`; by K_T=20 target-only has caught up
(`-0.003 ±0.007`).

**The oracle gap is real**: `+0.026 / +0.027 / +0.017` in pixel space, every interval clear
of zero. Earlier readings calling the method indistinguishable from the oracle came from
the loss-shaped instrument.

**Refinement's sign is not reportable.** Pixel space at K_T=20 gives `+0.0072 ±0.0070`
(helps); transformation-statistic space at the same cell gives `-0.0087` (hurts). Four
measurements of the K_T=1 cell span 0.023 while their half-widths are ~0.014. Per the
plan's §13 it stays out of the headline method. Its value is real when the starting
coordinate is far away — `reuse z_S` improves 1.439 → 0.521 across K_T purely through
refinement — and unmeasurable when transport has already landed close.

---

## Reproducibility notes

`A1_per_episode.json` holds every per-episode value for all nine conditions and both
metrics. It exists because the first A1 run stored only summaries, and asking a different
question of it cost a second 70-minute run. Recompute summaries from the JSON; do not
regenerate samples.

Two runs of the same command now reproduce byte for byte. Refinement previously drew its
`(t, eps)` from the unseeded global stream, so strategies compared within one episode
refined against **different** noise — an unpaired difference inside a paired comparison.
`adapt(..., generator=...)` now carries a per-episode stream that is rewound for every
strategy.

Progress for long runs goes to stderr under `--progress N`; `*.log` is gitignored, so the
progress files are not part of this archive.
