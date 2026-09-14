"""
File: Code/CURE/verify_iclr_numbers.py
Purpose: After the revision loop, confirm that every number in the manuscript prose is one
         the generator produced: a macro value in tables_v2/numbers_v2.tex, a cell or caption
         value of a generated table, or a fixed protocol constant listed below. Fails loudly
         on any other number, so a Writer-introduced figure cannot slip into the paper.

Usage:  python Code/CURE/verify_iclr_numbers.py [path/to/tex]   (default: the manuscript)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ICLR = Path(__file__).resolve().parents[2] / "Submission2" / "iclr2027"
TABLES = ICLR / "tables_v2"
# protocol constants that are stated in the text and defined by the code, not measured
CONSTANTS = {"2027", "24", "200", "20,000", "20{,}000", "1,024", "1{,}024", "1024", "9", "5", "4", "3", "2", "1", "0",
             "160", "100", "300", "2,000", "2{,}000", "48", "10", "0.05", "168", "596", "50"}


def allowed_values(tex_dir: Path) -> set:
    vals = set()
    for f in TABLES.glob("*.tex"):
        txt = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\\newcommand\{\\[A-Za-z]+\}\{([^}]*)\}", txt):
            vals.add(m.group(1).replace("{,}", ","))
        for m in re.finditer(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?![\w])", txt):
            vals.add(m.group(0).lstrip("+"))
    return vals


def prose(tex: str) -> str:
    body = tex.split("\\begin{document}")[1]
    body = re.sub(r"\\begin\{(table|figure|tikzpicture|equation)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\\(?:cite[pt]?|ref|eqref|label|input|includegraphics|url|href|hypersetup)\*?(?:\[[^\]]*\])?\{[^}]*\}", " ", body)
    body = re.sub(r"%.*", "", body)
    body = re.sub(r"\$[^$]*\$", " ", body)
    return body


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ICLR / "iclr2027_CURE_Audit_Benchmark.tex"
    tex = path.read_text(encoding="utf-8")
    ok = allowed_values(path.parent) | CONSTANTS
    bad = []
    for m in re.finditer(r"(?<![\w.\\-])\d+(?:[.,]\d+)*(?:\{,\}\d+)?(?![\w])", prose(tex)):
        raw = m.group(0).replace("{,}", ",")
        if raw in ok or raw.rstrip("0").rstrip(".") in ok:
            continue
        ctx = prose(tex)[max(0, m.start() - 60): m.end() + 40].replace("\n", " ")
        bad.append((raw, ctx))
    for raw, ctx in bad:
        print("UNSOURCED %-10s :: %s" % (raw, ctx))
    print("%d prose numbers not traceable to the generator" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
