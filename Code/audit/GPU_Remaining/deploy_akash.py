"""
deploy_akash.py -- provision a GPU on Akash via the Console Managed-Wallet API.

Tries H200 first; if no provider bids within the wait window, closes and tries H100.
Secrets (HF token, GitHub token, Akash key) are read from GPU_Remaining/.env and
injected into the SDL at submit time only -- the rendered SDL is never written to disk.

Usage:
  python deploy_akash.py            # provision (H200 -> H100), default deposit/price
  python deploy_akash.py --deposit 15 --max-price 100000 --wait 120
  python deploy_akash.py --status   # show current deployment status
  python deploy_akash.py --close    # close the saved deployment
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
TEMPLATE = HERE / "sdl_template.yaml"
STATE = HERE / ".deploy_state.json"          # gitignored (matches *.json? no -> add to .gitignore care)
BASE = "https://console-api.akash.network"


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
    # Cloudflare in front of console-api blocks the default Python-urllib UA (error 1010);
    # send a normal browser User-Agent so POST/DELETE are not rejected.
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


def render_sdl(env: dict, gpu_model: str, max_price: int) -> str:
    sdl = TEMPLATE.read_text(encoding="utf-8")
    sdl = sdl.replace("__GPU_MODEL__", gpu_model)
    sdl = sdl.replace("__HF_TOKEN__", env.get("HUGGINGFACE_TOKEN", ""))
    sdl = sdl.replace("__GH_TOKEN__", env.get("Github_Classic_Token", ""))
    sdl = sdl.replace("__MAX_PRICE__", str(max_price))
    return sdl


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
        bid = None
        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(8)
            bc, bids = _api("GET", f"/v1/bids?dseq={dseq}", key)
            items = bids.get("data", []) if isinstance(bids, dict) else []
            items = [b for b in items if b.get("bid", {}).get("state", "open") == "open"]
            if items:
                items.sort(key=lambda b: float(b["bid"]["price"]["amount"]))  # price is a decimal string
                bid = items[0]["bid"]
                print(f"[akash] {len(items)} bid(s); cheapest price={bid['price']['amount']} "
                      f"provider={bid['id']['provider']}")
                break
            print("[akash] waiting for bids ...")

        if not bid:
            print(f"[akash] no bids for {gpu_model}; closing dseq={dseq}")
            _api("DELETE", f"/v1/deployments/{dseq}", key)
            return None

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
    ap.add_argument("--deposit", type=float, default=10.0)
    ap.add_argument("--max-price", type=int, default=100000)
    ap.add_argument("--wait", type=int, default=120)
    ap.add_argument("--gpus", default="h200,h100,a100,a40,a6000,l40s,l40")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--close", action="store_true")
    args = ap.parse_args()
    env = read_env()
    if "AKASH_API_KEY" not in env:
        print("AKASH_API_KEY missing from GPU_Remaining/.env"); sys.exit(1)
    if args.status:
        show_status(env); return
    if args.close:
        close(env); return
    for gpu in [g.strip() for g in args.gpus.split(",") if g.strip()]:
        st = provision(env, gpu, args.deposit, args.max_price, args.wait)
        if st:
            print(f"[akash] provisioned {gpu}. Monitor results on GitHub main "
                  f"(Code/audit/GPU_Remaining/results/). `python deploy_akash.py --status` for lease.")
            return
        print(f"[akash] {gpu} unavailable; trying next ...")
    print("[akash] no GPU could be provisioned (H200/H100). Try later or raise --max-price/--deposit.")


if __name__ == "__main__":
    main()
