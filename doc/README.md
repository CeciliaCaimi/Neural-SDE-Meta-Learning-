# Paper sources

`main.tex` is the paper entry point and includes `appendix.tex`. The `figures/`
and `plots/` directories contain the available images referenced by these files.
The ICLR 2027 style, bibliography style, and `references.bib` are included.

Build from this directory with a LaTeX installation containing the packages
required by the sources:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The complete paper also requires these original GMM resources, which are not
available in this directory:

- `plots/gmm_full_grid_diagnostics.tex`
- `plots/gmm_identifiability_main.png`
- `plots/gmm_identity_chain.png`
- `plots/gmm_phase_main.png`
- `plots/gmm_shift_sweep.png`
- `results_file/fig_gain_vs_relation.png`

Restore those files at the listed paths before compiling the complete paper.
The sources retain their original GMM references; no placeholder figures or
substitute results are included.

The other report files already in this directory are separate documents.
