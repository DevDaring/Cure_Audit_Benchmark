"""
File: Code/CURE/Next_Run/f5_contract.py
Purpose: Regenerate Submission2/Evidence_Contract.md, the ONLY results allow-list the CURE
         manuscript may use, from the two machine-written claim ledgers of the final cycle
         (Next_Plan.md Section 12): claims_f0.json (F0 reanalysis of the earlier pool) and
         claims.json (F4 confirmatory analysis of the fresh set). Nothing is typed by hand;
         every line carries the claim id, its status and the permitted wording.

Usage:  python Code/CURE/Next_Run/f5_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nr_common as N

OUT = N.REPO / "Submission2" / "Evidence_Contract.md"
STATUS_TEXT = {
    "supported": "SUPPORTED (may be claimed as a result)",
    "limited": "LIMITED (report with its caveat; no causal or preservation claim)",
    "unsupported": "UNSUPPORTED (report as a negative result or a measured harm; never as support)",
    "measured": "MEASURED (secondary estimate with interval; no test)",
    "corrected": "CORRECTED WORDING (use this wording, not the earlier one)",
    "not_measured": "NOT MEASURED",
}


def fmt(c: dict) -> str:
    parts = ["### %s\n" % c["id"], "- **Status.** %s" % STATUS_TEXT.get(c.get("status"), c.get("status"))]
    if c.get("model"):
        parts.append("- **Model.** %s" % c["model"])
    if c.get("endpoint"):
        parts.append("- **Endpoint.** %s" % c["endpoint"])
    if c.get("population"):
        parts.append("- **Population.** %s" % c["population"])
    if c.get("estimate") is not None and c.get("interval"):
        parts.append("- **Estimate.** %+.4f [%+.4f, %+.4f]%s" % (c["estimate"], c["interval"][0], c["interval"][1],
                                                              ("; Holm p %.4f" % c["p_holm"]) if c.get("p_holm") is not None else ""))
    if c.get("achieved_ratio") is not None:
        parts.append("- **Achieved energy ratio.** %.3f" % c["achieved_ratio"])
    if c.get("energy_test_spread") is not None:
        parts.append("- **Test-set energy spread across calibrated cells.** %.3f (matched on test: %s)" % (c["energy_test_spread"], c.get("energy_matched_on_test")))
    if c.get("reason"):
        parts.append("- **Reason.** %s" % c["reason"])
    parts.append("- **Permitted wording.** %s" % c["permitted_wording"])
    return "\n".join(parts) + "\n"


def main() -> None:
    f0 = json.loads((N.FINAL / "claims_f0.json").read_text(encoding="utf-8"))
    f4 = json.loads((N.FINAL / "claims.json").read_text(encoding="utf-8"))
    comp = (N.FINAL / "COMPLETION.txt").read_text(encoding="utf-8")
    branch = comp.split("Outcome branch (Section 10):")[1].split("Budget:")[0].strip() if "Outcome branch" in comp else ""
    lines = [
        "# Evidence contract for the CURE paper (Submission2, ICLR 2027)",
        "",
        "Generated %s by Code/CURE/Next_Run/f5_contract.py from Code/CURE/results/final_20260914/claims_f0.json "
        "(F0 reanalysis of the earlier pool) and claims.json (F4 confirmatory analysis of the fresh BBQ set). "
        "This is the ONLY results allow-list the manuscript may use. A claim not listed here, or listed with a "
        "status other than SUPPORTED or MEASURED, may not be stated as a result." % N.utc_now(),
        "",
        "## Outcome rule (Next_Plan.md Section 10)",
        "",
        f4.get("outcome_rule", ""),
        "",
        "```",
        branch,
        "```",
        "",
        "## Study 2: fresh BBQ set (F4, pre-registered; template-cluster bootstrap, Holm over 8 confirmatory and 6 non-inferiority tests)",
        "",
    ]
    lines += [fmt(c) for c in f4["claims"]]
    lines += ["## Energy flags (F4)", "", "```", json.dumps(f4.get("energy_flags", {}), indent=1), "```", "",
              "## Study 1: earlier pool (F0 reanalysis of results/v2 and reanalysis*)", ""]
    lines += [fmt(c) for c in f0["claims"]]
    lines += ["## Standing prohibitions", "",
              "- No win rate. No 'provably'. No LEACE guarantee verified here. No 'repair without cost'. No accuracy-preservation claim under the full-strength span erasure. No massive-activation mechanism. No cost forecast.",
              "- H2/H3 on Llama-3.1-8B are LIMITED (test-set energy spread above 10 per cent): no energy-controlled causal claim.",
              "- The Study 1 'energy-matched' random control removed 1.6 to 3.6 per cent of the erasure's energy: call it a rank-matched control at a development-fixed strength; perturbation size is not ruled out by Study 1.",
              "- The basis is a contrast-derived projection (centred SVD of swap differences) evaluated with patching, not a direction identified by patching.",
              "- LEACE in Study 1 was labelled by swap side, which is not one coherent concept; the coherent-label baseline is Study 2's man/woman LEACE (F3).",
              "- Never mention MIRAGE, SCOPE, Patchscopes or any companion study.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote %s (%d Study 2 claims, %d Study 1 claims)" % (OUT, len(f4["claims"]), len(f0["claims"])))


if __name__ == "__main__":
    main()
