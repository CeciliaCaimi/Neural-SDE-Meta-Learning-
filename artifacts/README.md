# artifacts/

Two kinds of thing live here: the split files, which the code reads, and the stage-1
console logs, which are evidence.

## Split files — inputs, not outputs

| File | Read by | Contents |
|---|---|---|
| `cifar100_domainshift.json` | `episodes/domainshift.py` | The main experiment: the 12/4/4 superclass split and, per fine class, the source and target image pools |
| `cifar100_split.json` | `episodes/splits.py` | The sibling-class stress test |

Both hold global image indices only, about 350 kB each, and are fully reproducible:

```bash
python -m runner.build_cifar100_splits
```

They are committed anyway, so that a checkout reproduces the *exact* episodes behind
every number in the report rather than an equivalent draw. The figure in the report is
generated from `cifar100_domainshift.json` by `scripts/gen_tree_tex.py`, so the split
diagram cannot drift from the split the loader actually uses.

## Stage-1 logs — evidence

The 23 `stage1_*.log` files are the raw console output of the Gaussian-mixture runs
behind experiment E1 in the report. They are kept because the tables quote them.

**These logs are in Chinese.** They predate the conversion of this project to English
and were written by an earlier version of `runner/stage1_gmm.py`; the current runner
prints English. Reproducing any run regenerates its log in English — the numbers are
unchanged, only the labels around them differ.

| Logs | Run |
|---|---|
| `stage1_80k.log` | The first k = 16 baseline, 80k steps |
| `stage1_dec-linear_k16.log`, `stage1_dec-nonlinear_k16.log` | Linear against non-linear coordinate decoder, matched at 80k |
| `stage1_ksweep_k{2,4,8,16,32,64}.log` | The k sweep at a uniform 40k steps |
| `stage1_k{2,4,8}_s80k.log` | The low end of the k sweep, matched at 80k |
| `stage1_k{32,64}_s80k.log` | k = 32 against 64, matched at 80k |
| `stage1_k{32,64}_s160k.log` | The same two at 160k — saturation or undertraining? |
| `stage1_k16_s320k.log`, `stage1_k64_s320k.log` | Both to 320k, where the captured fraction levels off |
| `stage1_related_p{0.1,0.25,0.5,1.0}.log`, `stage1_related_unrelated.log` | The relatedness sweep |

Which script produced which is in [`../scripts/README.md`](../scripts/README.md).

## What is not here

Checkpoints (roughly 430 MB each), figures and the `.txt` / `.out` summaries are
generated rather than committed; see `.gitignore`. The CIFAR-100 numbers they contain
are reproduced in full in the root [`README.md`](../README.md) and in the report.
