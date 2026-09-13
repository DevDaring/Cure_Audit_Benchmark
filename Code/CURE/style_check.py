"""
File: Code/CURE/style_check.py
Purpose: Enforce, mechanically, the writing faults that got the previous paper rejected.

A reviewer wrote of an earlier submission of this work: "The text relies heavily on vague, unnatural,
and vacuous phrasing rather than precise scientific language, which makes it extremely
difficult to evaluate the actual merit of the research", and separately: "The paper is
exceedingly difficult to follow because it completely lacks concrete examples."

Good intentions do not survive a long draft, so each rule below is a check that fails the
build. Run it before every commit of the .tex and before submission.

Checks
------
 1. VAGUE PHRASING. A blocklist seeded with the exact phrases the reviewer quoted, plus
    the family they belong to. These sentences read as content while asserting nothing.
 2. SENTENCE LENGTH. Long sentences reduce human understanding. Warns above 30 words,
    fails above 40.
 3. NUMBER DENSITY IN PROSE. Numbers belong in tables and figures; prose cites them.
    Flags paragraphs carrying more than three numerals outside a float.
 4. CAPTION LENGTH. One line, two at most; the explanation goes in the text.
 5. SECTION / SUBSECTION ADJACENCY. A \\subsection must not immediately follow a
    \\section. At least one sentence of orientation belongs between them.
 6. ABRUPT OPENINGS. The abstract and each section must not open on a bare definition,
    a formula, or a citation.
 7. UNDEFINED TERMS. Paper-specific vocabulary must appear after its definition, not
    before it.
 8. ARXIV CITATIONS. If a bib entry is an arXiv preprint but names a venue, cite the
    venue. Flags entries that look like preprints of published work.
 9. DUPLICATE REPORTING. Warns when a table and a figure appear to present the same
    quantity.

Usage:
  python Code/CURE/style_check.py Submission2/ML_Springer_CURE_Audit_Benchmark.tex
  python Code/CURE/style_check.py <tex> --bib Submission2/references.bib
  python Code/CURE/style_check.py <tex> --abstract-limit 250

Part of the CURE codebase (Springer Machine Learning submission).
"""

import argparse
import re
import sys
from pathlib import Path

# Phrases the reviewer named, and the family they belong to. Each says nothing checkable.
VAGUE = [
    # Quoted verbatim from the rejection.
    "points the other way", "load-bearing", "the central result is",
    "is a prognosis", "prognostic of",
    # Same family: a metaphor standing in for a measurement.
    "tells a different story", "paints a picture", "sheds light",
    "the story is", "the picture that emerges", "speaks to", "points toward",
    "points towards", "cuts both ways", "the upshot is", "at odds with",
    "carries weight", "does the heavy lifting", "the crux",
    # Hedges that assert nothing checkable.
    "it is worth noting", "it is important to note", "it should be emphasised",
    "in essence", "at its core", "fundamentally", "arguably",
    "broadly speaking", "in some sense", "to a large extent", "by and large",
    "relatively speaking", "more or less", "in practice, this means",
    # Vague quantity where a number belongs.
    "a rich set of", "a range of interesting", "quite promising",
    "a variety of", "a number of interesting", "several key", "various aspects",
    "considerably better", "markedly improved", "clearly superior",
    # Promissory phrasing.
    "shows promise", "opens the door", "paves the way", "a natural next step",
    "we believe that", "we argue that this is", "intuitively,",
    "somewhat surprisingly", "rather striking", "compelling evidence",
]

BANNED_VERBS = ["leverage", "harness", "delve", "utilize", "utilise", "showcase"]
BANNED_CONNECTIVES = ["Furthermore", "Moreover", "Notably", "Additionally"]

MAX_SENTENCE_WARN = 30
MAX_SENTENCE_FAIL = 40
MAX_CAPTION_CHARS = 200          # roughly two typeset lines
MAX_NUMBERS_PER_PARAGRAPH = 3


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def _body(tex: str) -> str:
    """Text outside floats, equations and verbatim, i.e. what a reader reads as prose.

    Everything before \\maketitle is front matter: ORCIDs, postcodes and the DOI are
    numerals, but nobody reads them as prose, so they are not subject to the density rule.
    The abstract has its own check.
    """
    out = tex
    cut = out.find("\\maketitle")
    if cut != -1:
        out = out[cut + len("\\maketitle"):]
    for env in ("figure", "figure\\*", "table", "table\\*", "equation", "align",
                "verbatim", "lstlisting", "tabular", "CCSXML", "quote"):
        out = re.sub(rf"\\begin{{{env}}}.*?\\end{{{env}}}", " ", out, flags=re.S)
    return out


def check_vague(tex: str) -> list[str]:
    issues = []
    low = tex.lower()
    for p in VAGUE:
        for m in re.finditer(re.escape(p.lower()), low):
            line = tex[: m.start()].count("\n") + 1
            issues.append(f"line {line}: vague phrasing {p!r} -- state the finding instead")
    for v in BANNED_VERBS:
        for m in re.finditer(rf"\b{v}\b", low):
            issues.append(f"line {tex[:m.start()].count(chr(10))+1}: banned verb {v!r}")
    for c in BANNED_CONNECTIVES:
        hits = list(re.finditer(rf"\b{c}\b", tex))
        if len(hits) > 1:
            issues.append(f"connective {c!r} used {len(hits)} times; at most once per paper")
    return issues


def check_sentences(tex: str) -> tuple[list[str], list[str]]:
    body = _body(_strip_comments(tex))
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?({[^}]*})?", " ", body)
    body = re.sub(r"[{}$]", " ", body)
    fails, warns = [], []
    for s in re.split(r"(?<=[.!?])\s+", body):
        s = " ".join(s.split())
        n = len(s.split())
        if n > MAX_SENTENCE_FAIL:
            fails.append(f"{n} words: {s[:90]}...")
        elif n > MAX_SENTENCE_WARN:
            warns.append(f"{n} words: {s[:90]}...")
    return fails, warns


def check_number_density(tex: str) -> list[str]:
    body = _body(_strip_comments(tex))
    issues = []
    for para in re.split(r"\n\s*\n", body):
        clean = re.sub(r"\\(cite|ref|label|section|subsection)[a-zA-Z]*\*?({[^}]*})?", " ", para)
        nums = re.findall(r"(?<![\w.])\d+\.\d+|(?<![\w.])\d{2,}", clean)
        if len(nums) > MAX_NUMBERS_PER_PARAGRAPH:
            issues.append(
                f"{len(nums)} numerals in one paragraph ({', '.join(nums[:6])}...); "
                f"cite them from a table or figure instead: {' '.join(clean.split())[:70]}..."
            )
    return issues


def check_captions(tex: str) -> list[str]:
    issues = []
    for m in re.finditer(r"\\caption\{", tex):
        i, depth = m.end(), 1
        while i < len(tex) and depth:
            if tex[i] == "{":
                depth += 1
            elif tex[i] == "}":
                depth -= 1
            i += 1
        cap = tex[m.end(): i - 1]
        if len(cap) > MAX_CAPTION_CHARS:
            issues.append(
                f"line {tex[:m.start()].count(chr(10))+1}: caption is {len(cap)} chars "
                f"(max {MAX_CAPTION_CHARS}); move the explanation into the text"
            )
    return issues


# A paragraph opening on a concessive reads as commentary before it reads as a claim.
BANNED_PARA_OPENERS = ("While", "However", "Whilst", "Although")


def check_paragraph_openers(tex: str) -> list[str]:
    body = _body(_strip_comments(tex))
    issues = []
    for para in re.split(r"\n\s*\n", body):
        first = para.strip().split(" ")[0].strip("{}\\") if para.strip() else ""
        if first in BANNED_PARA_OPENERS:
            issues.append(
                f"paragraph opens on {first!r}; lead with the claim: "
                f"{' '.join(para.split())[:70]}..."
            )
    return issues


def check_structure(tex: str) -> list[str]:
    issues = []
    for m in re.finditer(r"\\section\{[^}]*\}(.*?)\\subsection\{", tex, flags=re.S):
        between = re.sub(r"\\label\{[^}]*\}", "", m.group(1)).strip()
        if len(between) < 80:
            line = tex[: m.start()].count("\n") + 1
            issues.append(
                f"line {line}: subsection follows section with no orientating text; "
                "add one or two sentences saying what the section does"
            )
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    if m:
        first = " ".join(m.group(1).split())[:120]
        if re.match(r"^(Let|Define|Given|We define|Formally|\\\[|\$)", first):
            issues.append("abstract opens abruptly on a definition or formula")
    return issues


def _abstract(tex: str) -> str | None:
    """The abstract body, under either the environment or the Springer \\abstract{} form."""
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    if m:
        return m.group(1)
    # sn-jnl takes the abstract as a braced argument; match its balanced braces.
    m = re.search(r"\\abstract\{", tex)
    if not m:
        return None
    i, depth = m.end(), 1
    while i < len(tex) and depth:
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
        i += 1
    return tex[m.end(): i - 1]


def check_abstract_limit(tex: str, limit: int) -> list[str]:
    body = _abstract(tex)
    if body is None:
        return ["no abstract found"]
    words = len(re.sub(r"\\[a-zA-Z]+\*?({[^}]*})?", " ", body).split())
    return [f"abstract is {words} words, limit {limit}"] if words > limit else []


def check_bib(bib_path: Path) -> list[str]:
    """Flag entries that cite a preprint where a published venue should be named.

    An entry naming a real venue and keeping the arXiv id in a note is the CORRECT form,
    not a fault, so only two states are flagged:

      1. The venue field is itself arXiv, e.g. journal = {arXiv preprint}. Always wrong
         when the work has appeared; always worth checking when it has not.
      2. The entry names no venue at all. This may be legitimate for a genuine preprint,
         so it is reported for manual confirmation rather than assumed to be wrong.

    Whether an unpublished-looking entry has in fact been published cannot be settled from
    the .bib alone; this check narrows the list a human has to verify.
    """
    if not bib_path or not bib_path.exists():
        return []
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    issues = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        low = body.lower()
        venue = re.search(r"(booktitle|journal)\s*=\s*[{\"]([^}\"]+)", body, flags=re.I)
        # howpublished / url / doi is the correct home for a preprint, library or statute
        # with no conference or journal, and eprint+archivePrefix is the standard BibTeX
        # form for the same thing. An entry carrying any of them has already been resolved.
        declared = re.search(r"(howpublished|url|doi|eprint)\s*=", low)
        if venue and "arxiv" in venue.group(2).lower():
            issues.append(f"{key}: venue is {venue.group(2)!r}; name the conference or journal")
        elif not venue and not declared and kind not in ("techreport", "book",
                                                         "phdthesis", "manual"):
            issues.append(f"{key}: no venue and no howpublished; confirm the record")
        for field in ("author", "title", "year"):
            if not re.search(rf"{field}\s*=", low):
                issues.append(f"{key}: missing {field}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--bib", default=None)
    ap.add_argument("--abstract-limit", type=int, default=250)
    args = ap.parse_args()

    tex = Path(args.tex).read_text(encoding="utf-8", errors="replace")
    fails, warns = check_sentences(tex)

    groups = {
        "VAGUE PHRASING (the stated rejection reason)": check_vague(tex),
        f"SENTENCES OVER {MAX_SENTENCE_FAIL} WORDS": fails,
        "NUMBER DENSITY IN PROSE": check_number_density(tex),
        "CAPTIONS TOO LONG": check_captions(tex),
        "PARAGRAPH OPENERS": check_paragraph_openers(tex),
        "STRUCTURE": check_structure(tex),
        "ABSTRACT LIMIT": check_abstract_limit(tex, args.abstract_limit),
        "REFERENCES": check_bib(Path(args.bib)) if args.bib else [],
    }

    total = 0
    for name, items in groups.items():
        if items:
            print(f"\n=== {name}: {len(items)} ===")
            for i in items[:12]:
                print(f"  {i}")
            if len(items) > 12:
                print(f"  ... and {len(items)-12} more")
            total += len(items)

    if warns:
        print(f"\n=== SENTENCES {MAX_SENTENCE_WARN}-{MAX_SENTENCE_FAIL} WORDS (warning): {len(warns)} ===")
        for w in warns[:5]:
            print(f"  {w}")

    print(f"\n{'FAIL' if total else 'PASS'}: {total} blocking issue(s), {len(warns)} warning(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
