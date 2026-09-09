# Shared preamble for every driver script in this directory. Source it, do not run it:
#
#     . "$(dirname "$0")/_common.sh"
#
# It does three things that every script here needs and that half of them used to do
# differently from the other half:
#
#   1. Moves to the repository root, so a script produces the same result no matter
#      which directory it was invoked from. The runners are all `python -m`, which
#      resolves packages against the working directory, so this is not cosmetic.
#   2. Picks an interpreter. The project's virtual environment is used when present;
#      otherwise whatever `python` is on PATH. The scripts previously hard-coded
#      `.venv/Scripts/python.exe`, which exists only on Windows.
#   3. Forces UTF-8 on stdout. Several diagnostics print box-drawing characters, and
#      the Windows console default (cp1252) raises UnicodeEncodeError on them.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe          # Windows virtual environment
elif [ -x .venv/bin/python ]; then
    PY=.venv/bin/python                  # POSIX virtual environment
else
    PY=python                            # fall back to PATH
fi
export PY
export PYTHONIOENCODING=utf-8

mkdir -p artifacts checkpoints

# Torch's CIFAR loader and NumPy both emit deprecation notices that bury the numbers.
# Scripts pipe through this to drop them without hiding anything else.
NOISE='VisibleDeprecation|pickle.load'
export NOISE
