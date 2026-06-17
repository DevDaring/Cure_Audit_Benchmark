"""
deploy_cure.py -- provision a minimal-safe GPU on Akash for the CURE run.

Reads Code/CURE/.env for the Akash key and the keys CURE needs, renders the SDL in
memory (secrets never written to disk), polls bids, and leases the first available GPU
in minimal-safe priority order. The AKASH key is NOT injected into the VM; only the
keys CURE actually uses are passed through.

Minimal-safe default: >= 40 GB cards first (TransformerLens loads a second model copy
on top of the HF model, so 24 GB risks an OOM mid-run), with H100/H200 as fallback.

Usage:
  python deploy_cure.py                 # provision (minimal-safe order)
  python deploy_cure.py --probe         # survey bids/prices for the list, lease nothing
  python deploy_cure.py --status
  python deploy_cure.py --close
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE / ".env"
TEMPLATE = HERE / "sdl_cure.yaml"
STATE = HERE / ".deploy_state.json"
BASE = "https://console-api.akash.network"

# Keys the VM needs (the AKASH key is deliberately excluded). Real value from .env if
# present; AWS pair is dummied because the audit config requires the names but CURE
# never calls them.
INJECT_KEYS = [
    "HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED", "CURE_JUDGE_PROVIDER",
    "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
    "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_BASE_URL", "DEEPSEEK_JUDGE_MODEL_NAME",
    "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
    "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_BASE_URL",
]
DEFAULTS = {"RANDOM_SEED": "20260101", "CURE_JUDGE_PROVIDER": "gemini"}
DUMMY = {"AWS_ACCESS_KEY": "unused-by-cure", "AWS_SECRET_KEY": "unused-by-cure"}


def read_env() -> dict:
    out = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _api(method: str, path: str, key: str, body: dict | None = None) -> tuple[int, dict]:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("x-api-key", key)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def _env_block(env: dict) -> str:
    lines = []
    for k in INJECT_KEYS:
        v = env.get(k, DEFAULTS.get(k, ""))
        if v:
            lines.append(f'      - "{k}={v}"')
    for k, v in DUMMY.items():
        lines.append(f'      - "{k}={v}"')
    return "\n".join(lines)


def render_sdl(env: dict, gpu_model: str, max_price: int) -> str:
    sdl = TEMPLATE.read_text(encoding="utf-8")
    sdl = sdl.replace("__GPU_MODEL__", gpu_model)
    sdl = sdl.replace("__MAX_PRICE__", str(max_price))
    sdl = sdl.replace("__GH_TOKEN__", env.get("Github_Classic_Token", ""))
    sdl = sdl.replace("__ENV_BLOCK__", _env_block(env))
    return sdl


def _poll_bids(key: str, dseq, wait_s: int):
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(8)
        _, bids = _api("GET", f"/v1/bids?dseq={dseq}", key)
        items = bids.get("data", []) if isinstance(bids, dict) else []
        items = [b for b in items if b.get("bid", {}).get("state", "open") == "open"]
        if items:
            items.sort(key=lambda b: float(b["bid"]["price"]["amount"]))
            return items
        print("[akash] waiting for bids ...")
    return []


def probe(env: dict, gpus: list, max_price: int, wait_s: int) -> None:
    """Create a deployment per GPU, list bids and prices, then close (lease nothing)."""
    key = env["AKASH_API_KEY"]
    for gpu in gpus:
        sdl = render_sdl(env, gpu, max_price)
        code, resp = _api("POST", "/v1/deployments", key, {"data": {"sdl": sdl, "deposit": 0.5}})
        if code not in (200, 201) or "data" not in resp:
            print(f"[probe] {gpu}: create failed ({code})"); continue
        dseq = resp["data"]["dseq"]
        items = _poll_bids(key, dseq, wait_s)
        if items:
            print(f"[probe] {gpu}: {len(items)} bid(s); cheapest "
                  f"{items[0]['bid']['price']['amount']} uakt/block "
                  f"provider={items[0]['bid']['id']['provider']}")
        else:
            print(f"[probe] {gpu}: NO bids")
        _api("DELETE", f"/v1/deployments/{dseq}", key)


def provision(env: dict, gpu_model: str, deposit: float, max_price: int, wait_s: int) -> dict | None:
    key = env["AKASH_API_KEY"]
    sdl = render_sdl(env, gpu_model, max_price)
    print(f"[akash] creating deployment for GPU={gpu_model} deposit=${deposit} ...")
    code, resp = _api("POST", "/v1/deployments", key, {"data": {"sdl": sdl, "deposit": deposit}})
    if code not in (200, 201) or "data" not in resp:
        print(f"[akash] create FAILED ({code}): {json.dumps(resp)[:400]}")
        return None
    dseq = resp["data"].get("dseq")
    manifest = resp["data"].get("manifest")
    print(f"[akash] deployment created: dseq={dseq}")
    try:
        items = _poll_bids(key, dseq, wait_s)
        if not items:
            print(f"[akash] no bids for {gpu_model}; closing dseq={dseq}")
            _api("DELETE", f"/v1/deployments/{dseq}", key)
            return None
        bid = items[0]["bid"]
        print(f"[akash] {len(items)} bid(s); cheapest price={bid['price']['amount']} "
              f"provider={bid['id']['provider']}")
        bid_id = bid["id"]
        lease_body = {"manifest": manifest, "leases": [{
            "dseq": str(dseq), "gseq": bid_id["gseq"], "oseq": bid_id["oseq"],
            "provider": bid_id["provider"]}]}
        lc, lresp = _api("POST", "/v1/leases", key, lease_body)
        if lc not in (200, 201):
            print(f"[akash] lease FAILED ({lc}): {json.dumps(lresp)[:400]}; closing")
            _api("DELETE", f"/v1/deployments/{dseq}", key)
            return None
        state = {"dseq": str(dseq), "gpu_model": gpu_model, "provider": bid_id["provider"],
                 "gseq": bid_id["gseq"], "oseq": bid_id["oseq"],
                 "price_uakt_per_block": bid["price"]["amount"]}
        STATE.write_text(json.dumps(state, indent=2))
        print(f"[akash] LEASE CREATED for {gpu_model}. dseq={dseq} provider={bid_id['provider']}")
        return state
    except Exception as exc:
        print(f"[akash] provision error for {gpu_model}: {exc}; closing dseq={dseq}")
        _api("DELETE", f"/v1/deployments/{dseq}", key)
        return None


def show_status(env: dict) -> None:
    key = env["AKASH_API_KEY"]
    if not STATE.exists():
        print("no saved deployment"); return
    st = json.loads(STATE.read_text())
    code, resp = _api("GET", f"/v1/deployments/{st['dseq']}", key)
    print(f"[akash] status ({code}) for dseq={st['dseq']} ({st['gpu_model']}):")
    print(json.dumps(resp, indent=2)[:2000])


def close(env: dict) -> None:
    key = env["AKASH_API_KEY"]
    if not STATE.exists():
        print("no saved deployment"); return
    st = json.loads(STATE.read_text())
    code, resp = _api("DELETE", f"/v1/deployments/{st['dseq']}", key)
    print(f"[akash] close dseq={st['dseq']} -> {code} {json.dumps(resp)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", type=float, default=15.0)
    ap.add_argument("--max-price", type=int, default=100000)
    ap.add_argument("--wait", type=int, default=90)
    # minimal-safe: 40-48 GB cards first (no OOM with the TL second copy), H100/H200 fallback
    ap.add_argument("--gpus", default="l40s,a6000,a40,l40,a100,h100,h200")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--close", action="store_true")
    args = ap.parse_args()
    env = read_env()
    if "AKASH_API_KEY" not in env:
        print("AKASH_API_KEY missing from Code/CURE/.env"); sys.exit(1)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.status:
        show_status(env); return
    if args.close:
        close(env); return
    if args.probe:
        probe(env, gpus, args.max_price, args.wait); return
    for gpu in gpus:
        st = provision(env, gpu, args.deposit, args.max_price, args.wait)
        if st:
            print(f"[akash] provisioned {gpu}. Results push to Cure_Audit_Benchmark "
                  f"(Code/CURE/results/). `python deploy_cure.py --status` for the lease.")
            return
        print(f"[akash] {gpu} unavailable; trying next ...")
    print("[akash] no GPU could be provisioned. Try later or raise --max-price/--deposit.")


if __name__ == "__main__":
    main()
