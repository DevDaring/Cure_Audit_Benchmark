"""
File: Code/CURE/make_overleaf_zip.py
Purpose: Bundle the ICLR 2027 manuscript for Overleaf: the main .tex, the ICLR style files,
         the bibliography, every generated table/macro fragment under tables_v2/ that the
         manuscript \\input{}s (plus the two macro files), and every figure it includes.
         Nothing else (no build products, no revision reports, no .md).

Usage:  python Code/CURE/make_overleaf_zip.py    -> Submission2/iclr2027_overleaf.zip
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Submission2" / "iclr2027"
OUT = ROOT / "Submission2" / "iclr2027_overleaf.zip"
MAIN = "iclr2027_CURE_Audit_Benchmark.tex"

tex = (SRC / MAIN).read_text(encoding="utf-8")
files = [MAIN, "references.bib", "iclr2027_conference.sty", "iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty", "math_commands.tex"]
files += sorted({"%s.tex" % m for m in re.findall(r"\\input\{([^}]+)\}", tex)})
files += sorted({m for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)})
missing = [f for f in files if not (SRC / f).exists()]
assert not missing, "missing: %s" % missing
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(SRC / f, f)
print("wrote %s (%d files)" % (OUT, len(files)))
for f in files:
    print("  ", f)
