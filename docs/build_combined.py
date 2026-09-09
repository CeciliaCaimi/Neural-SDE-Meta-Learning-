"""Concatenate the specification report and the experimental record into one document.

    python docs/build_combined.py
    cd docs && pdflatex meta_diffusion_combined.tex   # three times, for the contents

The combined document is built by splicing the two source files rather than by keeping
a third copy in sync by hand, so neither half can drift from the version that already
compiles. Five things break under a naive concatenation and are repaired here:

  1. The record's half uses colours `shared`, `coord`, `overturn` and the macros \\KT,
     \\MS, none of which the specification's preamble defines.
  2. A second \\maketitle and \\tableofcontents in mid-document would set a title block
     inside a page and print the whole contents list a second time.
  3. Six labels are defined in both halves (eq:refine, fig:arch, fig:split, sec:data,
     sec:flow, sec:loss). LaTeX keeps the last, so every reference in the first half
     would silently point into the second. The record's copies gain a "-u" suffix.
  4. The two halves write the noise symbol differently (\\epsilon against \\varepsilon).
     A \\renewcommand at the boundary keeps each half internally consistent.
  5. The record carries its own trailing "update" page. Here that page introduces the
     record rather than following it, so the trailing copy would be a blank duplicate.

Two overfull boxes inherited from the specification are also fixed; see below.
"""
import io
import os

DOCS = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(DOCS, "meta_diffusion_spec.tex")
REC = os.path.join(DOCS, "meta_diffusion_record.tex")
OUT = os.path.join(DOCS, "meta_diffusion_combined.tex")

spec = io.open(SPEC, encoding="utf-8").read()
rec = io.open(REC, encoding="utf-8").read()

# ------------------------------------------------------------------ split both files
BEGIN, END = "\\begin{document}", "\\end{document}"
spec_pre, spec_rest = spec.split(BEGIN, 1)
spec_body = spec_rest.rsplit(END, 1)[0]

# The record's figures are all wrapped in \resizebox; the specification's are not, and
# its architecture figure runs 8.1pt into the right margin. Only that one is wrapped:
# the split figure below it is narrower than the text block, and scaling it up to the
# full width would enlarge its type past the surrounding text.
spec_body = spec_body.replace(
    "\\begin{tikzpicture}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tikzpicture}", 1)
spec_body = spec_body.replace("\\end{tikzpicture}", "\\end{tikzpicture}}", 1)

# A paragraph 4.5pt over, on an unbreakable run of long words. sloppypar stretches the
# interword space instead of overflowing, and leaves the wording untouched.
para_start = "Concretely, the coordinate of shape"
para_end = "basis in score space."
i = spec_body.index(para_start)
j = spec_body.index(para_end, i) + len(para_end)
spec_body = (spec_body[:i] + "\\begin{sloppypar}\n" + spec_body[i:j]
             + "\n\\end{sloppypar}" + spec_body[j:])

rec_body = rec.split(BEGIN, 1)[1].rsplit(END, 1)[0]
abstract = rec_body[rec_body.index("\\begin{abstract}"):
                    rec_body.index("\\end{abstract}") + len("\\end{abstract}")]
after_toc = rec_body.index("\\tableofcontents") + len("\\tableofcontents")
rec_body = rec_body[after_toc:].lstrip("\n")
if rec_body.startswith("\\bigskip"):
    rec_body = rec_body[len("\\bigskip"):].lstrip("\n")

trailing = ("\\clearpage\n\\section*{update 09/09/2026}\n"
            "\\addcontentsline{toc}{section}{update 09/09/2026}")
assert trailing in rec_body, "the record's update page is not where it was"
rec_body = rec_body.replace(trailing, "", 1)

# ------------------------------------------------------- 3: de-duplicate the labels
DUPS = ("eq:refine", "fig:arch", "fig:split", "sec:data", "sec:flow", "sec:loss")
renamed = 0
for name in DUPS:
    for cmd in ("label", "ref", "eqref", "autoref"):
        old = "\\%s{%s}" % (cmd, name)
        renamed += rec_body.count(old)
        rec_body = rec_body.replace(old, "\\%s{%s-u}" % (cmd, name))
    assert "\\label{%s-u}" % name in rec_body, "label %s vanished" % name

# --------------------------------------------- 1: colours and macros the record needs
additions = "\n".join([
    r"\usepackage{array}",
    r"",
    r"% ---- required by the 09/09/2026 update -----------------------------------",
    r"\definecolor{shared}{HTML}{1D4E4A}   % shared parameters, frozen at deployment",
    r"\definecolor{coord}{HTML}{B06A14}    % the task coordinate, optimised at deployment",
    r"\definecolor{overturn}{HTML}{8C3A32} % a conclusion later overturned",
    r"\newcommand{\KT}{K_{T}}",
    r"\newcommand{\MS}{M_{S}}",
    r"",
])
anchor = "\\definecolor{tealc}"
spec_pre = spec_pre.replace(anchor, additions + anchor, 1)

# ---------------------------------- 2 and 4: the boundary page, without a second title
bridge = "\n".join([
    r"",
    r"\clearpage",
    r"\section*{update 09/09/2026}",
    r"\addcontentsline{toc}{section}{update 09/09/2026}",
    r"",
    r"% the record half writes the noise symbol as \varepsilon throughout",
    r"\renewcommand{\epshat}{\hat{\varepsilon}}",
    r"",
    r"\begin{center}",
    r"{\Large\bfseries Meta-Diffusion: Experimental Record}",
    r"\end{center}",
    r"\medskip",
    r"",
    abstract,
    r"",
])

doc = spec_pre + BEGIN + spec_body + bridge + "\n" + rec_body + END + "\n"

# ------------------------------------------------------------------------- checks
assert doc.count("\\maketitle") == 1, doc.count("\\maketitle")
assert doc.count("\\tableofcontents") == 1
assert doc.count("\\begin{document}") == 1 and doc.count("\\end{document}") == 1
assert doc.count("\\begin{tikzpicture}") == doc.count("\\end{tikzpicture}") == 5
assert doc.count("\\resizebox{\\textwidth}{!}{%") == 4
labels = [l.split("}")[0] for l in doc.split("\\label{")[1:]]
assert len(labels) == len(set(labels)), "still duplicated: %s" % (
    sorted({l for l in labels if labels.count(l) > 1}),)
assert not [c for c in doc if ord(c) < 32 and c != "\n"], "control characters"

io.open(OUT, "w", encoding="utf-8", newline="\n").write(doc)
print("wrote %s" % os.path.basename(OUT))
print("  %d labels, all distinct; %d references rewritten across %d duplicated names"
      % (len(labels), renamed, len(DUPS)))
print("  5 tikzpictures, one title block, one contents list")
