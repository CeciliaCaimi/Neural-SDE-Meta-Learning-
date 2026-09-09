"""Emit the three-way class split as a TikZ picture, read from the split file itself.

The report shows which superclasses and fine classes fall into train, validation and
test. Typing that figure by hand would let it drift silently from what the code loads,
and a split diagram that disagrees with the split is worse than no diagram. So the
figure is generated from artifacts/cifar100_domainshift.json, the same file the
episode loader reads.

Usage:
    python scripts/gen_tree_tex.py [-o OUT.tex] [--split-file PATH]

The output is a bare tikzpicture body, meant to be pasted between \\begin{tikzpicture}
and \\end{tikzpicture} in the report. It expects the colours `shared`, `coord` to be
defined by the document preamble.
"""
import argparse
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from episodes.domainshift import load_domainshift            # noqa: E402

ROW = 0.42          # cm between superclass rows
GAP = 0.45          # cm between split groups
X_SUP = 0.75        # superclass label column
X_JOIN = 6.10       # where the connector stubs meet
X_FINE = 6.40       # fine-class list column

COLOUR = {"train": "shared", "val": "coord", "test": "black!62"}
LABEL = {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}
NOTE = {"train": "gradients", "val": "every choice", "test": "final only"}


def build(split):
    groups = []
    for which in ("train", "val", "test"):
        rows = []
        for cid in sorted(split.superclass_split[which]):
            fines = sorted(f for f, c in split.classes.items() if c.coarse_id == cid)
            rows.append((split.coarse_names[cid].replace("_", " "),
                         [split.fine_names[f].replace("_", " ") for f in fines]))
        groups.append((which, rows))

    out = []
    add = out.append
    add(r"\node[anchor=west, font=\tiny\bfseries, black!55] at (%.2f,0.45) "
        r"{SUPERCLASS};" % X_SUP)
    add(r"\node[anchor=west, font=\tiny\bfseries, black!55] at (%.2f,0.45) "
        r"{ITS FIVE FINE CLASSES --- EACH ONE IS A SEPARATE TASK};" % X_FINE)
    add(r"\draw[black!30, line width=.3pt] (-0.3,0.25) -- (17.2,0.25);")

    y = 0.0
    for which, rows in groups:
        col = COLOUR[which]
        top = y
        bot = y - (len(rows) - 1) * ROW
        add(r"\draw[%s, line width=.8pt] (0.30,%.2f) -- (0.12,%.2f) -- (0.12,%.2f) "
            r"-- (0.30,%.2f);" % (col, top + 0.16, top + 0.16, bot - 0.16, bot - 0.16))
        mid = (top + bot) / 2
        # rotated, so a long label such as VALIDATION cannot reach the superclass column
        add(r"\node[rotate=90, font=\tiny\bfseries, %s] at (-0.16,%.2f) {%s};"
            % (col, mid, LABEL[which]))
        add(r"\node[rotate=90, font=\tiny, black!55] at (0.02,%.2f) {%s};"
            % (mid, NOTE[which]))
        add(r"\draw[%s, opacity=.45, line width=.3pt] (%.2f,%.2f) -- (%.2f,%.2f);"
            % (col, X_JOIN - 0.28, top, X_JOIN - 0.28, bot))
        for name, fines in rows:
            add(r"\node[anchor=west, font=\scriptsize, %s] at (%.2f,%.2f) {%s};"
                % (col, X_SUP, y, name))
            add(r"\draw[%s, opacity=.45, line width=.3pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                % (col, X_JOIN - 0.28, y, X_JOIN, y))
            add(r"\node[anchor=west, font=\scriptsize, black!78] at (%.2f,%.2f) {%s};"
                % (X_FINE, y, r" \textperiodcentered\ ".join(fines)))
            y -= ROW
        y -= GAP
    return out, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "artifacts",
                                                        "split_tree.tex"))
    ap.add_argument("--split-file", default=os.path.join(REPO, "artifacts",
                                                         "cifar100_domainshift.json"))
    a = ap.parse_args()

    split = load_domainshift(a.split_file)
    lines, groups = build(split)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    io.open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(lines))

    n_rows = sum(len(rows) for _, rows in groups)
    print("wrote %s" % a.out)
    print("  %d superclasses over %d splits, %d lines of TikZ"
          % (n_rows, len(groups), len(lines)))


if __name__ == "__main__":
    main()
