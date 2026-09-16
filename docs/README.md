# docs/

The written record. If you only read one thing, read
[`meta_diffusion_combined.pdf`](meta_diffusion_combined.pdf) — 28 pages covering the
framework, the data protocol, every loss term, all eleven experiments and a
step-by-step reproduction procedure.

| File | What it is |
|---|---|
| `PROTOCOL_CARD.md` | The fixed values, split checksums and file ownership for the three parallel work packages of the next stage. Read it before starting one |
| `work_split.tex` / `.pdf` | The same allocation as a five-page hand-out, opening with a table that keys every item to a numbered section of the supervisor's 9 September note. Standalone; `build_combined.py` does not read it |
| `results_summary.tex` / `.pdf` | Every experiment run so far in one line of work, one of result and one of what it established — E1 to E11 plus what re-reading the logs added. Four pages, standalone |
| `meta_diffusion_combined.pdf` | The report as read. Built from the two sources below |
| `meta_diffusion_combined.tex` | **Generated — do not edit.** `build_combined.py` overwrites it |
| `meta_diffusion_spec.tex` | Part one: the implementation specification, written when stage 1 was the only completed work |
| `meta_diffusion_record.tex` | Part two: the experimental record, dated 09/09/2026, covering the CIFAR-100 programme |
| `figures/gmm_data.pdf` | The stage-1 mixture scatter, regenerable with `python scripts/export_gmm_pdf.py` |

## Rebuilding

```bash
python docs/build_combined.py
cd docs && pdflatex meta_diffusion_combined.tex   # three times
```

Three passes, not two: the contents list grows by a page between the first and second,
which shifts every page number after it. Two passes leave the printed contents one page
out from where the sections actually are.

Edit `meta_diffusion_spec.tex` or `meta_diffusion_record.tex` and re-run the build —
never edit the combined file, since the next build discards the changes. The build
exists because the two halves are also read separately, and keeping a hand-merged third
copy in step with them is a promise that does not survive contact with a deadline. What
it repairs on the way through — duplicated labels, a second title block, undefined
colours, two overfull boxes — is documented in the script's own header.

A clean build is **28 pages, zero errors, zero overfull boxes, zero undefined
references.** Anything else means something upstream moved.

## Related documents

- [`../README.md`](../README.md) — the project overview, including what the CIFAR-100
  experiments found and why the conclusion was reached twice in the wrong direction
  before it was reached in the right one
- [`../ALGORITHM.md`](../ALGORITHM.md) — the complete specification: every equation,
  hyperparameter and diagnostic needed to reimplement the project from scratch
- [`../scripts/README.md`](../scripts/README.md) — which script produced which table
