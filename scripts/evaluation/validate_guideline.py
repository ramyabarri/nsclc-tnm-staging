"""Exhaustive validation of the Guideline Logic Agent's IASLC rule engine.

The staging table lives in ``configs/iaslc_rules.yaml`` and is loaded by the
agent. This script validates that table by enumerating EVERY (T, N, M)
combination over the valid category tokens and comparing the agent's output
against an *independently written* IASLC 8th-edition oracle (``oracle_stage``
below). Because the oracle is coded by hand from the published stage groups —
not read from the YAML — full agreement is evidence that the encoded table
faithfully implements the guideline, with no data or model involved.

This isolates and certifies the deterministic component of the pipeline
(supervisor's "formal validation of the guideline engine").

Usage:
    python -m scripts.evaluation.validate_guideline
    python -m scripts.evaluation.validate_guideline --output results/staging/guideline_validation.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from agents.vision import TFactorEvidence
from agents.clinical_context import NMFactorEvidence
from agents.guideline_logic import GuidelineLogicAgent

# Category tokens to enumerate. Mirrors configs/iaslc_rules.yaml. TX/NX/MX are
# included so we also confirm the agent leaves incomplete triples indeterminate.
T_TOKENS = ["TX", "Tis", "T1mi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]
N_TOKENS = ["NX", "N0", "N1", "N2", "N3"]
M_TOKENS = ["MX", "M0", "M1a", "M1b", "M1c"]

# Size-defined T groups (invasion tokens like T3/T4 share the N/M behaviour).
_T_ALL = ["Tis", "T1mi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]


def oracle_stage(t: str, n: str, m: str) -> str:
    """Independent IASLC 8th-edition stage-group oracle.

    Hand-encoded from the published 8th-edition stage groups (Goldstraw et al.,
    2016). Returns the AJCC stage or "indeterminate" for triples the guideline
    does not assign (any TX/NX/MX, i.e. an incomplete classification).
    """
    # Distant metastasis dominates, regardless of T and N.
    if m == "M1a" or m == "M1b":
        return "IVA"
    if m == "M1c":
        return "IVB"

    # Any incomplete/unknown category leaves the stage indeterminate.
    if m != "M0" or n not in {"N0", "N1", "N2", "N3"} or t not in _T_ALL:
        return "indeterminate"

    if t == "Tis" and n == "N0":
        return "0"
    if t == "Tis":
        # Tis with nodal disease is not a defined 8th-ed group.
        return "indeterminate"

    if n == "N0":
        return {
            "T1mi": "IA1", "T1a": "IA1", "T1b": "IA2", "T1c": "IA3",
            "T2a": "IB", "T2b": "IIA", "T3": "IIB", "T4": "IIIA",
        }[t]
    if n == "N1":
        # T1(all)/T2(all) N1 -> IIB; T3/T4 N1 -> IIIA
        return "IIIA" if t in {"T3", "T4"} else "IIB"
    if n == "N2":
        # T1/T2 N2 -> IIIA; T3/T4 N2 -> IIIB
        return "IIIB" if t in {"T3", "T4"} else "IIIA"
    if n == "N3":
        # T1/T2 N3 -> IIIB; T3/T4 N3 -> IIIC
        return "IIIC" if t in {"T3", "T4"} else "IIIB"
    return "indeterminate"


def _evidence(t: str, n: str, m: str):
    """Build the T and N/M evidence objects the agent consumes for a triple."""
    t_ev = TFactorEvidence(t_category=t, confidence=1.0)
    nm_ev = NMFactorEvidence(n_category=n, m_category=m, confidence=1.0)
    return t_ev, nm_ev


def validate() -> dict:
    """Run the exhaustive comparison. Returns a result dict with mismatches."""
    agent = GuidelineLogicAgent()
    agent.load_rules()

    rows = []
    mismatches = []
    for t, n, m in itertools.product(T_TOKENS, N_TOKENS, M_TOKENS):
        t_ev, nm_ev = _evidence(t, n, m)
        result = agent.run(t_ev, nm_ev, patient_id=None)
        got = result.overall_stage
        want = oracle_stage(t, n, m)
        ok = got == want
        rows.append({"t": t, "n": n, "m": m, "agent": got, "oracle": want, "match": ok})
        if not ok:
            mismatches.append({"t": t, "n": n, "m": m, "agent": got, "oracle": want})

    return {
        "edition": agent.edition,
        "total": len(rows),
        "passed": sum(r["match"] for r in rows),
        "failed": len(mismatches),
        "mismatches": mismatches,
        "rows": rows,
    }


def _write_report(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Guideline Logic Agent — Exhaustive Rule Validation",
        "",
        f"_Edition:_ **{res['edition']}**  ",
        f"_Combinations tested:_ **{res['total']}** "
        f"({len(T_TOKENS)} T x {len(N_TOKENS)} N x {len(M_TOKENS)} M)  ",
        f"_Agreement with independent IASLC oracle:_ "
        f"**{res['passed']}/{res['total']}** "
        f"({100 * res['passed'] / res['total']:.1f}%)  ",
        "",
    ]
    if res["failed"] == 0:
        lines.append(
            "✅ **PASS** — the encoded stage table reproduces the independently "
            "hand-coded IASLC 8th-edition oracle for every category combination."
        )
    else:
        lines.append(f"❌ **{res['failed']} mismatch(es):**")
        lines.append("")
        lines.append("| T | N | M | agent | oracle |")
        lines.append("| --- | --- | --- | --- | --- |")
        for mm in res["mismatches"]:
            lines.append(
                f"| {mm['t']} | {mm['n']} | {mm['m']} | {mm['agent']} | {mm['oracle']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the IASLC guideline engine exhaustively.")
    ap.add_argument("--output", type=Path,
                    default=Path("results/staging/guideline_validation.md"))
    args = ap.parse_args()

    res = validate()
    _write_report(res, args.output)
    # Machine-readable copy alongside the markdown.
    args.output.with_suffix(".json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "rows"}, indent=2) + "\n")

    print(f"Guideline validation ({res['edition']}):")
    print(f"  {res['passed']}/{res['total']} combinations match the IASLC oracle.")
    if res["failed"]:
        print(f"  {res['failed']} MISMATCH(es):")
        for mm in res["mismatches"]:
            print(f"    ({mm['t']},{mm['n']},{mm['m']}): agent={mm['agent']} oracle={mm['oracle']}")
    else:
        print("  PASS — encoded table matches the independent oracle exactly.")
    print(f"Report: {args.output}")
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
