#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_v551_gate_aware_eval import (  # noqa: E402
    LABEL_TO_ID,
    gt_label,
    source_from_row,
    unified_prompt,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_field(text: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.*)$", text or "", re.M)
    return match.group(1).strip() if match else ""


def prompt_text(row: dict[str, Any]) -> str:
    return str((row.get("messages") or [{}])[0].get("content") or "")


def hit(text: str, phrases: list[str]) -> list[str]:
    lowered = f" {text.lower()} "
    return [phrase for phrase in phrases if f" {phrase.lower()} " in lowered]


def should_move_lazy_to_deep(manifest_row: dict[str, Any], source_row: dict[str, Any], rules: dict[str, Any]) -> tuple[bool, str, list[str]]:
    meta = manifest_row.get("meta") or {}
    if meta.get("v551_route_detail") != "same_label_one_side_imperfect_support":
        return False, "", []
    obs = meta.get("observer_answer")
    prov = meta.get("evidence_provider_answer")
    if obs != prov:
        return False, "", []
    if obs == "TAMPERED":
        return True, "same_tampered_weak_scope", []
    if obs != "REAL":
        return False, "", []
    report = parse_field(prompt_text(source_row), "observer_report")
    risk_hits = hit(report, rules.get("risk_phrases") or [])
    safe_hits = hit(report, rules.get("safe_phrases") or [])
    if risk_hits and not safe_hits:
        return True, "real_agreement_risk_phrase", risk_hits
    if rules.get("use_provider_medium") and str(meta.get("provider_confidence_level")).upper() == "MEDIUM":
        return True, "provider_medium_confidence", []
    return False, "", []


def build_split(old_gate_dir: Path, split: str, out_dir: Path, rules: dict[str, Any], tag: str) -> dict[str, Any]:
    old_manifest_path = old_gate_dir / f"judge_val_{split}_v5_51_gate_aware_manifest.jsonl"
    old_deep_path = old_gate_dir / f"judge_val_{split}_v5_51_gate_aware_deep_vlm_infer.jsonl"
    old_manifest = read_jsonl(old_manifest_path)
    old_deep = read_jsonl(old_deep_path)
    source_path = Path((old_manifest[0].get("meta") or {})["source_eval_file"])
    source_rows = read_jsonl(source_path)

    new_manifest: list[dict[str, Any]] = []
    new_deep: list[dict[str, Any]] = []
    lazy_rows: list[dict[str, Any]] = []
    moved = 0
    kept_lazy = 0
    old_deep_kept = 0

    for row in old_manifest:
        row = json.loads(json.dumps(row, ensure_ascii=False))
        meta = dict(row.get("meta") or {})
        source_row = source_rows[meta["row_index"]]
        old_lazy = row.get("lazy_response") is not None
        move, reason, hits = (False, "", [])
        if old_lazy:
            move, reason, hits = should_move_lazy_to_deep(row, source_row, rules)

        if old_lazy and not move:
            kept_lazy += 1
            row["deep_index"] = None
            row["meta"] = meta
            lazy_rows.append({**row, "response": row.get("lazy_response")})
            new_manifest.append(row)
            continue

        if old_lazy and move:
            observer = source_from_row(source_row, "observer", "observer_answer", "observer_bbox")
            provider = source_from_row(source_row, "provider", "evidence_provider_answer", "evidence_provider_bbox")
            meta.update(
                {
                    "v559_phrase_gate_moved_from_lazy": True,
                    "v559_phrase_gate_reason": reason,
                    "v559_phrase_gate_hits": hits[:10],
                    "v551_route_action": "FULL_RECHECK",
                    "v551_route_detail": f"v559_{reason}",
                    "v551_deep": True,
                }
            )
            deep_index = len(new_deep)
            meta["v551_deep_index"] = deep_index
            new_deep.append(
                {
                    "messages": [{"role": "user", "content": unified_prompt("FULL_RECHECK", observer, provider)}],
                    "images": source_row.get("images") or row.get("images"),
                    "label": LABEL_TO_ID.get(gt_label(source_row), row.get("label")),
                    "meta": meta,
                }
            )
            row["deep_index"] = deep_index
            row["lazy_response"] = None
            row["meta"] = meta
            new_manifest.append(row)
            moved += 1
            continue

        # Existing deep rows also regenerate their prompt so prompt policy changes
        # apply uniformly across old-deep and newly-routed-deep examples.
        deep_index = len(new_deep)
        meta["v551_deep_index"] = deep_index
        observer = source_from_row(source_row, "observer", "observer_answer", "observer_bbox")
        provider = source_from_row(source_row, "provider", "evidence_provider_answer", "evidence_provider_bbox")
        action = meta.get("v551_route_action") or "FULL_RECHECK"
        new_deep.append(
            {
                "messages": [{"role": "user", "content": unified_prompt(action, observer, provider)}],
                "images": source_row.get("images") or row.get("images"),
                "label": LABEL_TO_ID.get(gt_label(source_row), row.get("label")),
                "meta": meta,
            }
        )
        row["deep_index"] = deep_index
        row["lazy_response"] = None
        row["meta"] = meta
        new_manifest.append(row)
        old_deep_kept += 1

    prefix = f"judge_val_{split}_{tag}"
    manifest_path = out_dir / f"{prefix}_manifest.jsonl"
    deep_path = out_dir / f"{prefix}_deep_vlm_infer.jsonl"
    lazy_path = out_dir / f"{prefix}_lazy_outputs.jsonl"
    write_jsonl(manifest_path, new_manifest)
    write_jsonl(deep_path, new_deep)
    write_jsonl(lazy_path, lazy_rows)
    return {
        "split": split,
        "source_manifest": str(old_manifest_path),
        "source_rows": len(old_manifest),
        "old_deep_kept": old_deep_kept,
        "kept_lazy": kept_lazy,
        "moved_lazy_to_deep": moved,
        "manifest": str(manifest_path),
        "deep_vlm": str(deep_path),
        "lazy_outputs": str(lazy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-gate-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["bfree", "wildrf"])
    parser.add_argument("--tag", default="v5_59_phrase_gate")
    parser.add_argument("--summary-name", default="v559_phrase_gate_build_summary_20260708.json")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules_blob = json.loads(args.rules.read_text(encoding="utf-8"))
    rules = rules_blob["manual_plus_mined_clean"]
    summary = [build_split(args.old_gate_dir, split, args.out_dir, rules, args.tag) for split in args.splits]
    summary_path = args.out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    for item in summary:
        print(item["split"], item["manifest"], item["deep_vlm"])


if __name__ == "__main__":
    main()
