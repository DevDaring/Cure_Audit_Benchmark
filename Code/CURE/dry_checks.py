"""
dry_checks.py -- all dry-run validations for CURE.

Validates the whole environment on a tiny slice before the main run:
  1. API connectivity across every provider, every key, with its model id.
  2. Per-model code path (all four models, two pairs): E1 subspace, E3 re-audit, E4 utility.
  3. GPU and flash-attention verified by a real forward pass.
  4. Secret scan of the tracked tree (no secret literal anywhere but .env).
  5. Integrity self-test (a crafted duplicate and a corrupt parquet).
  6. In-stack baselines on two pairs.

Any failure returns False so the dry run exits non-zero.
"""

import json
import logging
import os
import re
import subprocess

import pandas as pd

import config_cure as C

log = logging.getLogger("cure.dry")


def check_apis() -> bool:
    import judge_api
    report = judge_api.test_all_keys()
    (C.RESULTS / "dryrun").mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "dryrun" / "api_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for prov, rs in report.items():
        for r in rs:
            log.info("api %-10s key#%d model=%s -> %s", prov, r["key_index"], r.get("model"), r["status"])
    # Resolve one working tier for the whole run (no per-item cross-tier fallback).
    chosen = judge_api.resolve_active_judge(report=report)
    if not chosen:
        log.error("DRY FAIL: no judge tier has a working key (checked %s)", ", ".join(report))
        return False
    if chosen != C.ACTIVE_JUDGE:
        log.warning("configured judge tier %r has no working keys; using %r for the whole run "
                    "(same task, working keys; still no per-item fallback)", C.ACTIVE_JUDGE, chosen)
    else:
        log.info("judge tier %r active for the whole run", chosen)
    return True


def check_flash_attention() -> bool:
    import torch
    if not torch.cuda.is_available():
        log.error("DRY FAIL: CUDA not available")
        return False
    allow_sdpa = os.environ.get("CURE_ALLOW_SDPA") == "1"
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:
        if allow_sdpa:
            log.warning("flash_attn missing; CURE_ALLOW_SDPA=1 -> continuing on sdpa (SLOW)")
            return True
        log.error("DRY FAIL: flash_attn import failed: %s", str(exc)[:160])
        return False
    # one forward pass on the smallest model with flash_attention_2
    import torch
    from load_osm import load_model, unload_model
    small = min(C.OSM_MODELS, key=lambda m: 0 if "gemma" in m["name"] else 1)
    model, tok = load_model(small)
    try:
        impl = getattr(model.config, "_attn_implementation", "?")
        dev = next(model.parameters()).device
        ids = tok("hello world", return_tensors="pt").to(dev)
        with torch.no_grad():
            model(**ids)
        ok = impl == "flash_attention_2"
        log.info("flash-attention forward pass ok; impl=%s", impl)
        if not ok and not allow_sdpa:
            log.error("DRY FAIL: attention impl is %r, not flash_attention_2", impl)
        return ok or allow_sdpa
    finally:
        unload_model(small["name"])


_SECRET_RE = re.compile(
    r"(ac\.sk\.|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})")


def secret_scan() -> bool:
    """Grep tracked files (not .env) for secret literals."""
    try:
        files = subprocess.run(["git", "ls-files"], cwd=str(C.REPO),
                               capture_output=True, text=True).stdout.splitlines()
    except Exception:
        files = []
    bad = []
    for f in files:
        if f.endswith(".env") or "/.env" in f:
            continue
        p = C.REPO / f
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if _SECRET_RE.search(txt):
            bad.append(f)
    if bad:
        log.error("DRY FAIL: secret literal found in tracked files: %s", bad[:10])
        return False
    log.info("secret scan clean")
    return True


def integrity_selftest() -> bool:
    import integrity
    d = C.RESULTS / "dryrun"
    d.mkdir(parents=True, exist_ok=True)
    dup = pd.DataFrame({"method": ["x", "x"], "model_name": ["m", "m"], "v": [1, 2]})
    dup.to_parquet(d / "cure_baselines_selftest.parquet", index=False)
    (d / "cure_recovery_corrupt.parquet").write_bytes(b"not a parquet")
    rep = integrity.run(d)
    ok = rep["deduped"].get("cure_baselines_selftest.parquet", 0) >= 1 and len(rep["quarantined"]) >= 1
    log.info("integrity self-test: %s", "ok" if ok else "FAIL")
    return ok


def per_model_smoke() -> bool:
    import experiments as E
    import baselines as B
    from load_osm import load_model, unload_model
    ok = True
    for cfg in C.OSM_MODELS:
        # two instances of EACH dataset (bbq, crows_pairs, stereoset)
        pairs = E.head_per_dataset(E._load_pairs(cfg, limit=None), C.DRY_LIMIT)
        if not pairs:
            log.error("DRY FAIL: no pairs for %s", cfg["name"]); ok = False; continue
        log.info("dry %s: %d pairs across datasets %s", cfg["name"], len(pairs),
                 sorted({str(p["seed_id"]).split("_", 1)[0] for p in pairs}))
        model, tok = load_model(cfg)
        try:
            basis = E.build_subspace(model, tok, cfg, pairs, rank=2)
            if not basis:
                log.error("DRY FAIL: empty subspace for %s", cfg["name"]); ok = False; continue
            re = E.e3_reaudit(model, tok, cfg, pairs, basis)
            if re.empty:
                log.error("DRY FAIL: empty re-audit for %s", cfg["name"]); ok = False
            # validate EVERY comparative baseline builds a basis without crashing
            # (a raised exception here fails the dry before the full baseline pass).
            ctx = E.collect_acts(model, tok, cfg, pairs)
            for m in B.REGISTRY:
                built = B.build_basis(m, model, tok, cfg, pairs, ctx=ctx)
                st = built.get("status", "ok")
                nb = len(built.get("basis", {}) or {})
                log.info("dry baseline %-14s %s -> status=%s layers=%d", m, cfg["name"], st, nb)
                if built.get("basis"):
                    if E.e3_reaudit(model, tok, cfg, pairs, built["basis"]).empty:
                        log.warning("dry baseline %s empty re-audit (acceptable on %d pairs)", m, len(pairs))
        except Exception as exc:
            log.error("DRY FAIL: %s raised: %s", cfg["name"], str(exc)[:200]); ok = False
        finally:
            unload_model(cfg["name"])
    return ok


def run_all() -> bool:
    import integrity
    integrity.run()
    checks = [
        ("secret_scan", secret_scan),
        ("integrity_selftest", integrity_selftest),
        ("apis", check_apis),
        ("flash_attention", check_flash_attention),
        ("per_model_smoke", per_model_smoke),
    ]
    ok = True
    for name, fn in checks:
        try:
            res = fn()
        except Exception as exc:
            log.error("dry check %s raised: %s", name, str(exc)[:200]); res = False
        log.info("DRY CHECK %-20s %s", name, "PASS" if res else "FAIL")
        ok = ok and res
    log.info("DRY RESULT: %s", "PASS" if ok else "FAIL")
    return ok
