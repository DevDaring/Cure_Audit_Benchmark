"""
judge_unmapped.py -- optional LLM judge (DeepSeek, OpenAI-compatible API) for the generations
the deterministic parser leaves UNMAPPED: the model answered with a paraphrase of an option
("The other woman who had an injury" for "The woman with the injury"). Used only for that
residue (1.7 to 3.6 per cent of generations on the full run), never for empty outputs, and
every judged row is labelled gen_parse_X = "judge" so the judged share is reported separately
from the deterministic mapping. Judgements are cached in results/v2/judge_cache.json so a
rerun costs nothing.

Keys: DEEPSEEK_API_KEY_1 / DEEPSEEK_API_KEY_2 (round-robin), DEEPSEEK_API_BASE_URL,
DEEPSEEK_JUDGE_MODEL_NAME from Code/CURE/.env; nothing is printed or written from them.

The judge sees only the option list and the answer string, never the passage, and must return
the option letter or NONE. Temperature 0. A reply that is not a listed letter is recorded as
"judge_none" and the row stays invalid.

Usage
  python judge_unmapped.py                       # every per-item file under results/v2
  python judge_unmapped.py --dir results/v2_qwen2.5-7b-instruct --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

import common as K
import p1_pilot as P1

log = K.setup_logging("cure.ext.judge")
FILES = ["pilot_per_item.parquet", "p2_per_item.parquet"]
SYSTEM = ("You map a free-text answer to one of the listed multiple-choice options. Reply with the "
          "option letter only (A, B, C, ...) if the answer clearly refers to exactly one option, "
          "otherwise reply NONE. Never explain.")


def read_env() -> dict:
    out = {}
    for line in (K.CURE / ".env").read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Judge:
    def __init__(self, cache_path: Path):
        e = read_env()
        self.keys = [e[k] for k in ("DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2") if e.get(k)]
        self.base = e.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.model = e.get("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.i = 0
        self.calls = 0
        if not self.keys:
            raise SystemExit("no DEEPSEEK_API_KEY_* in Code/CURE/.env")

    def _key(self) -> str:
        self.i += 1
        return self.keys[self.i % len(self.keys)]

    def ask(self, answer: str, options: list[tuple[str, str]]) -> str | None:
        opts = "\n".join("(%s) %s" % (l, t) for l, t in options)
        ck = json.dumps([answer, opts])
        if ck in self.cache:
            return self.cache[ck]
        body = {"model": self.model, "temperature": 0, "max_tokens": 4,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "Options:\n%s\n\nAnswer: %s\n\nLetter:" % (opts, answer)}]}
        out = None
        for attempt in range(4):
            req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json"); req.add_header("Authorization", "Bearer " + self._key())
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    txt = json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
                self.calls += 1
                m = re.match(r"^\(?([A-Za-z])\)?", txt)
                letter = m.group(1).upper() if m and txt.upper() != "NONE" else None
                out = letter if letter in {l for l, _ in options} else None
                break
            except urllib.error.HTTPError as ex:
                if ex.code in (429, 500, 502, 503):
                    time.sleep(2 + 3 * attempt); continue
                log.error("judge http %s", ex.code); break
            except Exception as ex:
                log.error("judge error %s", str(ex)[:120]); time.sleep(2); continue
        self.cache[ck] = out
        if self.calls % 25 == 0:
            self.save()
        return out

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=0), encoding="utf-8")


def _answer_text(raw: str) -> str | None:
    """The same candidate extraction as p1_pilot.parse_answer, for the judge's input."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("answer", "")).strip()
        except Exception:
            pass
    mm = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
    if mm:
        return mm.group(1).strip()
    return raw.strip().splitlines()[0].strip() if raw.strip() else None


def judge_file(path: Path, judge: Judge, idx: dict, dry: bool) -> dict:
    df = pd.read_parquet(path)
    if "gen_raw_A" not in df:
        return {"file": path.name, "skipped": "no generations"}
    n_unmapped = n_judged = n_mapped = 0
    for i, r in df.iterrows():
        touched = False
        for side in ("A", "B"):
            if r.get(f"gen_parse_{side}") != "unmapped":
                continue
            n_unmapped += 1
            key = (r["seed_id"], r.get(f"subvariant_{side}"))
            if key not in idx or dry:
                continue
            prompt, gold = idx[key]
            ans = _answer_text(str(r[f"gen_raw_{side}"]))
            if not ans:
                continue
            opts = P1.options_of(prompt)
            letter = judge.ask(ans, opts)
            n_judged += 1
            if letter is None:
                df.at[i, f"gen_parse_{side}"] = "judge_none"
                continue
            oi = [l for l, _ in opts].index(letter)
            gi = P1.gold_index(prompt, gold)
            df.at[i, f"gen_idx_{side}"] = oi
            df.at[i, f"gen_parse_{side}"] = "judge"
            df.at[i, f"gen_valid_{side}"] = True
            df.at[i, f"gen_correct_{side}"] = (oi == gi) if gi is not None else None
            n_mapped += 1; touched = True
        if touched:
            a, b = df.at[i, "gen_idx_A"], df.at[i, "gen_idx_B"]
            va, vb = bool(df.at[i, "gen_valid_A"]), bool(df.at[i, "gen_valid_B"])
            df.at[i, "gen_flip_AB"] = (a != b) if (va and vb) else None
    if not dry:
        for c in ("gen_idx_A", "gen_idx_B"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["gen_judged_utc"] = K.utc_now()
        df.to_parquet(path, index=False)
    return {"file": path.name, "rows": len(df), "unmapped_before": n_unmapped, "judged": n_judged, "mapped_by_judge": n_mapped}


def judge_dir(d: Path, dry: bool = False) -> list[dict]:
    pen = pd.read_parquet(K.PENTAD_CLEAN); pen = pen[pen["slot"] == "c"]
    idx = {(r["seed_id"], r["subvariant"]): (str(r["prompt_text"]), str(r.get("gold_answer", ""))) for _, r in pen.iterrows()}
    judge = Judge(K.RESULTS / "v2" / "judge_cache.json")
    out = []
    for name in FILES:
        p = d / name
        if p.exists():
            rec = judge_file(p, judge, idx, dry); log.info("%s: %s", K.rel(p), rec); out.append(rec)
    judge.save()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for rec in judge_dir(Path(a.dir).resolve() if a.dir else K.RESULTS / "v2", a.dry_run):
        print(rec)


if __name__ == "__main__":
    main()
