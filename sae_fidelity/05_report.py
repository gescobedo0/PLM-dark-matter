#!/usr/bin/env python
"""Turn results.json into the filled-in handoff decision and a markdown report.

Decision logic (handoff "Decision rule" table):
  FVU comparable overall + markers validate      -> GO (curate ~50-200 features)
  FVU comparable + markers fail                   -> clustering-only (note discrepancy)
  FVU elevated on a subset only                   -> in-distribution subset only
  FVU elevated overall                            -> NO-GO (fall back to non-SAE blocks)
Markers "validate" if a majority of non-disorder sets fire their expected
features above the background 90th percentile, and the disorder set does NOT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sae_lib.config import load_config


def decide(results: dict, marker_pass_frac: float = 0.5) -> dict:
    verdict = results["fvu_verdict"]
    rows = {(r["group"], r["subclass"]): r for r in verdict["rows"]}
    micro_all = rows.get(("microprotein", "(all)"))
    micro_all_label = micro_all["label"] if micro_all else "unknown"

    # per-subclass split within microproteins
    sub_rows = [r for r in verdict["rows"]
                if r["group"] == "microprotein" and r["subclass"] not in ("(all)",)
                and not r["subclass"].startswith("tm:")]
    elevated_subs = [r["subclass"] for r in sub_rows if r["label"] == "elevated"]
    comparable_subs = [r["subclass"] for r in sub_rows if r["label"] == "comparable"]

    # marker validation
    mrec = results.get("marker_recovery", {})
    struct = {k: v for k, v in mrec.items()
              if v.get("concept") != "disorder" and not v.get("skipped")}
    fired = [k for k, v in struct.items() if v.get("fraction_above_bg_90pct", 0) >= 0.5]
    markers_ok = (len(struct) > 0 and len(fired) / len(struct) >= marker_pass_frac)
    # disorder NEGATIVE control: structured features must NOT fire on disordered
    # proteins (low fraction-above-bg). Absent control -> treat as clean.
    dis_neg = results.get("disorder_negative_control")
    disorder_clean = (dis_neg is None) or (dis_neg.get("fraction_above_bg_90pct", 1.0) < 0.5)
    markers_validate = markers_ok and disorder_clean

    # map to the decision table
    if micro_all_label == "comparable" and markers_validate:
        decision = "GO"
        action = "Use SAE features. Curate ~50-200 features whose descriptions map to categories of interest."
    elif micro_all_label == "comparable" and not markers_validate:
        decision = "CLUSTERING_ONLY"
        action = "Use SAE features for clustering only, not semantic labeling. Report the marker discrepancy."
    elif elevated_subs and comparable_subs:
        decision = "SUBSET_ONLY"
        action = (f"Use SAE features only on in-distribution subclasses "
                  f"({', '.join(comparable_subs)}); report coverage. "
                  f"Elevated FVU on: {', '.join(elevated_subs)}.")
    elif micro_all_label == "elevated":
        decision = "NO_GO"
        action = "Fall back to non-SAE feature blocks. Report the negative result; note SAE retraining as future work."
    else:
        decision = "INCONCLUSIVE"
        action = "FVU intermediate; inspect distributions and thresholds before deciding."

    return {"decision": decision, "action": action,
            "microprotein_fvu_label": micro_all_label,
            "markers_validate": markers_validate,
            "structured_markers_fired": fired,
            "elevated_subclasses": elevated_subs,
            "comparable_subclasses": comparable_subs}


def render_md(results: dict, dec: dict) -> str:
    L = []
    L.append("# SAE fidelity diagnostic — go/no-go report\n")
    L.append(f"**Decision: `{dec['decision']}`**\n\n{dec['action']}\n")
    L.append(f"- microprotein FVU vs background: **{dec['microprotein_fvu_label']}**")
    L.append(f"- positive-control markers validate: **{dec['markers_validate']}** "
             f"(structured sets fired: {', '.join(dec['structured_markers_fired']) or 'none'})")
    if dec["elevated_subclasses"]:
        L.append(f"- elevated-FVU subclasses: {', '.join(dec['elevated_subclasses'])}")
    L.append("\n## Section 1 — reconstruction FVU (median, ratio vs background)\n")
    L.append("| group | subclass | n | median FVU | ratio | label |")
    L.append("|---|---|--:|--:|--:|---|")
    for r in results["fvu_verdict"]["rows"]:
        L.append(f"| {r['group']} | {r['subclass']} | {r['n']} | "
                 f"{r['fvu_median']:.3f} | {r['ratio_vs_bg']:.2f} | {r['label']} |")
    L.append("\n## Section 2 — feature-usage entropy (bits)\n")
    L.append(f"max possible = {results['max_entropy_bits']:.1f} bits "
             f"(log2 of {2**round(results['max_entropy_bits'])} features)\n")
    L.append("| group | entropy | active-mag median |")
    L.append("|---|--:|--:|")
    for g, e in results["entropy_bits"].items():
        mag = results.get("magnitude", {}).get(g, {}).get("mag_median", float("nan"))
        L.append(f"| {g} | {e:.2f} | {mag:.3f} |")
    if "informative" in results:
        L.append("\n**Top-feature description informativeness (fraction):** " +
                 ", ".join(f"{g}={v['mean_informative_frac']:.2f}"
                           for g, v in results["informative"].items()))
    if "marker_recovery" in results:
        L.append("\n## Section 4 — positive-control marker recovery\n")
        L.append("| marker set | concept | median %ile vs bg | frac > bg 90%ile |")
        L.append("|---|---|--:|--:|")
        for name, v in results["marker_recovery"].items():
            if v.get("skipped"):
                L.append(f"| {name} | {v['concept']} | skipped | {v['reason']} |")
            else:
                L.append(f"| {name} | {v['concept']} | "
                         f"{v['median_percentile_vs_bg']:.2f} | "
                         f"{v['fraction_above_bg_90pct']:.2f} |")
    L.append("\n![FVU ECDF](fvu_ecdf.png)\n")
    L.append("\n---\n*Pooling: max over residues (handoff rule — not mean). "
             "FVU baseline: background-group activation mean.*\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep_dir = Path(cfg["paths"]["report_dir"])
    results = json.loads((rep_dir / "results.json").read_text())
    dec = decide(results)
    md = render_md(results, dec)
    (rep_dir / "report.md").write_text(md, encoding="utf-8")
    (rep_dir / "decision.json").write_text(json.dumps(dec, indent=2))
    print(md)
    print(f"\nwrote {rep_dir/'report.md'} and {rep_dir/'decision.json'}")


if __name__ == "__main__":
    main()
