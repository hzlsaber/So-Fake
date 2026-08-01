#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_TO_ID = {"REAL": 0, "FULL_SYNTHETIC": 1, "TAMPERED": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABELS = set(LABEL_TO_ID)
BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)\s*,\s*"
    r"\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)\s*<\|box_end\|>",
    re.I,
)
_OBSERVER_SUPPORT_RULES: dict[str, Any] | None = None


def observer_support_rules() -> dict[str, Any] | None:
    global _OBSERVER_SUPPORT_RULES
    if _OBSERVER_SUPPORT_RULES is not None:
        return _OBSERVER_SUPPORT_RULES
    path = os.environ.get("V558_OBSERVER_SUPPORT_RULES")
    if not path:
        _OBSERVER_SUPPORT_RULES = {}
        return None
    rules_path = Path(path)
    if not rules_path.exists():
        _OBSERVER_SUPPORT_RULES = {}
        return None
    _OBSERVER_SUPPORT_RULES = json.loads(rules_path.read_text(encoding="utf-8"))
    return _OBSERVER_SUPPORT_RULES


def phrase_hits(text: str, phrases: list[str]) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for phrase in phrases if f" {phrase.lower()} " in lowered)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in LABELS else "UNKNOWN"


def support_norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"strong", "medium", "weak", "mixed"}:
        return text
    if text in {"moderate", "middle"}:
        return "medium"
    if text == "mild":
        return "weak"
    return "unknown"


def support_good(value: Any) -> bool:
    return support_norm(value) in {"medium", "strong"}


def count_patterns(text: str, patterns: list[str]) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for pattern in patterns if pattern in lowered)


def observer_support_from_report(answer: str, report: str, explicit: Any = None) -> tuple[str, str]:
    """Infer observer support conservatively when no explicit support exists.

    This is a text-commitment score, not a correctness/confidence score. It only
    asks whether the observer report gives label-aligned, image-checkable reasons.
    In particular, REAL reports based mainly on "no visible artifact" should not
    be treated as medium/strong evidence.
    """
    explicit_support = support_norm(explicit)
    if explicit_support != "unknown":
        return explicit_support, "explicit"

    answer = norm_label(answer)
    text = str(report or "").strip().lower()
    if not text or answer == "UNKNOWN":
        return "unknown", "missing_report"

    rules = observer_support_rules()
    if rules:
        answer_rules = (rules.get("answers") or {}).get(answer) or {}
        strong_hits = phrase_hits(text, answer_rules.get("strong_phrases") or [])
        medium_hits = phrase_hits(text, answer_rules.get("medium_phrases") or [])
        weak_hits = phrase_hits(text, answer_rules.get("weak_phrases") or [])
        if strong_hits >= 2:
            return "strong", "calibrated_phrase_strong"
        if strong_hits >= 1 or medium_hits >= 2:
            return "medium", "calibrated_phrase_medium"
        if weak_hits >= 1:
            return "weak", "calibrated_phrase_weak"
        return "unknown", "calibrated_phrase_no_match"

    absence_terms = [
        "no visible sign",
        "no obvious",
        "no evidence",
        "no discernible",
        "no apparent",
        "not appear",
        "does not appear",
        "without visible",
        "without obvious",
        "lack of",
        "lacks",
    ]
    real_positive_terms = [
        "camera noise",
        "sensor noise",
        "natural noise",
        "compression",
        "depth of field",
        "motion blur",
        "lens",
        "perspective",
        "natural shadow",
        "consistent shadow",
        "lighting direction",
        "reflection",
        "material texture",
        "skin texture",
        "fine detail",
        "micro texture",
        "photographic",
        "real photograph",
    ]
    synthetic_terms = [
        "synthetic",
        "ai-generated",
        "generated",
        "unnatural",
        "distorted",
        "deformed",
        "waxy",
        "overly smooth",
        "smooth rendering",
        "inconsistent lighting",
        "inconsistent shadow",
        "geometry",
        "facial feature",
        "hand",
        "artifact",
        "global",
        "entire image",
        "whole image",
        "background",
    ]
    local_terms = [
        "local",
        "localized",
        "region",
        "area",
        "part of",
        "object",
        "boundary",
        "edge",
        "around",
        "near",
        "bbox",
        "box",
    ]
    tamper_terms = [
        "tamper",
        "edited",
        "manipulated",
        "spliced",
        "pasted",
        "copy",
        "clone",
        "seam",
        "boundary",
        "edge artifact",
        "blend",
        "mismatch",
        "inconsistent texture",
        "localized artifact",
    ]

    absence = count_patterns(text, absence_terms)
    real_pos = count_patterns(text, real_positive_terms)
    synth = count_patterns(text, synthetic_terms)
    local = count_patterns(text, local_terms)
    tamper = count_patterns(text, tamper_terms)

    if answer == "REAL":
        contradiction = synth + tamper
        if real_pos >= 4 and contradiction <= 1 and absence <= 1:
            return "strong", "report_rule_real_positive"
        if real_pos >= 2 and contradiction <= 1:
            return "medium", "report_rule_real_positive"
        if absence > 0:
            return "weak", "report_rule_real_absence_only"
        return "unknown", "report_rule_real_generic"

    if answer == "FULL_SYNTHETIC":
        has_global_scope = any(term in text for term in ["entire image", "whole image", "global", "overall", "across the image"])
        if has_global_scope and synth >= 4:
            return "strong", "report_rule_full_synthetic_global"
        if synth >= 3:
            return "medium", "report_rule_full_synthetic_cues"
        if synth >= 1:
            return "weak", "report_rule_full_synthetic_sparse"
        return "unknown", "report_rule_full_synthetic_generic"

    if answer == "TAMPERED":
        if local >= 2 and tamper >= 3:
            return "strong", "report_rule_tampered_local"
        if local >= 1 and tamper >= 2:
            return "medium", "report_rule_tampered_local"
        if tamper >= 1 or local >= 1:
            return "weak", "report_rule_tampered_sparse"
        return "unknown", "report_rule_tampered_generic"

    return "unknown", "unsupported_answer"


def answer_auth(answer: str) -> str:
    answer = norm_label(answer)
    if answer == "REAL":
        return "REAL"
    if answer in {"FULL_SYNTHETIC", "TAMPERED"}:
        return "FAKE"
    return "UNKNOWN"


def answer_scope(answer: str) -> str:
    answer = norm_label(answer)
    if answer == "REAL":
        return "N/A"
    if answer == "FULL_SYNTHETIC":
        return "FULL_SYNTHETIC"
    if answer == "TAMPERED":
        return "TAMPERED"
    return "UNKNOWN"


def parse_field(text: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.*)$", text or "", re.M)
    return match.group(1).strip() if match else ""


def parse_box(text: Any) -> str:
    if text is None:
        return "NONE"
    raw = str(text).strip()
    if not raw or raw.upper() == "NONE":
        return "NONE"
    match = BOX_RE.search(raw)
    return match.group(0) if match else "NONE"


def box_valid(text: Any, explicit: Any = None) -> bool:
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str) and explicit.strip().upper() in {"TRUE", "FALSE"}:
        return explicit.strip().upper() == "TRUE"
    return parse_box(text) != "NONE"


def prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    return str(messages[0].get("content") or "")


def source_from_row(row: dict[str, Any], prefix: str, meta_answer_key: str, meta_bbox_key: str) -> dict[str, Any]:
    text = prompt_text(row)
    meta = row.get("meta") or {}
    answer = norm_label(parse_field(text, f"{prefix}_answer") or meta.get(meta_answer_key))
    bbox = parse_box(parse_field(text, f"{prefix}_bbox") or meta.get(meta_bbox_key))
    valid_raw = parse_field(text, f"{prefix}_bbox_valid")
    support_raw = parse_field(text, f"{prefix}_support") or meta.get(f"{prefix}_support")
    report_key = "observer_text" if prefix == "observer" else "provider_text"
    fallback_key = "observer_report" if prefix == "observer" else "evidence_provider_assessment"
    report = parse_field(text, report_key) or parse_field(text, fallback_key)
    if prefix == "observer":
        support, support_source = observer_support_from_report(answer, report, support_raw)
    else:
        support = support_norm(support_raw)
        support_source = "explicit" if support != "unknown" else "missing"
    return {
        "answer": answer,
        "auth": answer_auth(answer),
        "scope": answer_scope(answer),
        "bbox": bbox,
        "bbox_valid": box_valid(bbox, valid_raw),
        "support": support,
        "support_source": support_source,
        "text": report,
    }


def final_box_for_tampered(observer: dict[str, Any], provider: dict[str, Any]) -> str:
    if provider["answer"] == "TAMPERED" and provider["bbox_valid"]:
        return provider["bbox"]
    if observer["answer"] == "TAMPERED" and observer["bbox_valid"]:
        return observer["bbox"]
    return "NONE"


def route_case(observer: dict[str, Any], provider: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return route_action, route_detail, lazy_final_answer.

    The rule is intentionally conservative: only same-label agreement with usable
    support can skip image-side VLM recheck. REAL-vs-FULL_SYNTHETIC conflicts are
    the narrow targeted route; local-edit conflicts go to full recheck.
    """
    obs_ans, prov_ans = observer["answer"], provider["answer"]
    obs_auth, prov_auth = observer["auth"], provider["auth"]
    obs_good, prov_good = support_good(observer["support"]), support_good(provider["support"])

    if obs_ans == prov_ans and obs_ans in LABELS:
        if obs_ans == "TAMPERED" and not final_box_for_tampered(observer, provider) != "NONE":
            return "FULL_RECHECK", "agreement_tampered_without_valid_bbox", None
        if obs_good and prov_good:
            return "NO_CHECK_CLEAN", "same_label_good_support", obs_ans
        if obs_good or prov_good:
            return "CAUTION_ACCEPT", "same_label_one_side_imperfect_support", obs_ans
        return "FULL_RECHECK", "same_label_both_sources_weak", None

    if {obs_auth, prov_auth} == {"REAL", "FAKE"}:
        fake_answer = prov_ans if prov_auth == "FAKE" else obs_ans
        if fake_answer == "FULL_SYNTHETIC":
            return "TARGETED_RECHECK", "real_vs_full_synthetic", None
        return "FULL_RECHECK", "real_vs_tampered_or_unknown_fake_scope", None

    if obs_auth == prov_auth == "FAKE":
        return "FULL_RECHECK", "fake_scope_conflict", None

    return "FULL_RECHECK", "unknown_or_invalid_source_state", None


def unified_prompt(route_action: str, observer: dict[str, Any], provider: dict[str, Any]) -> str:
    if route_action == "TARGETED_RECHECK":
        class_definitions = """Class definitions:
- REAL: authentic image with no AI generation or manipulation.
- FULL_SYNTHETIC: the whole image is AI-generated."""
        task_rules = """Task rules:
- This is a narrow REAL-vs-FULL_SYNTHETIC dispute.
- First decide authenticity: REAL or FAKE.
- If authenticity is REAL, set fake_scope to N/A and final_answer to REAL.
- If authenticity is FAKE, set fake_scope to FULL_SYNTHETIC and final_answer to FULL_SYNTHETIC.
- Use the image to verify which of the two report claims survives.
- Explain what the rejected claim gets wrong."""
        output_format = """Required output format:
<authenticity_decision>
authenticity: REAL|FAKE
</authenticity_decision>

<scope_decision>
fake_scope: N/A|FULL_SYNTHETIC
</scope_decision>

<routing_decision>
route_action: TARGETED_RECHECK
</routing_decision>

<final_assessment>
One paragraph. Decide between the REAL claim and the FULL_SYNTHETIC claim, explain which survives, and state what the rejected claim gets wrong.
</final_assessment>

<final_answer>REAL|FULL_SYNTHETIC</final_answer>"""
    else:
        class_definitions = """Class definitions:
- REAL: authentic image with no AI generation or manipulation.
- FULL_SYNTHETIC: the whole image is AI-generated.
- TAMPERED: part of the image is edited or manipulated."""
        task_rules = """Task rules:
- The two reports cannot be accepted directly.
- Use the image to verify both report claims.
- First decide authenticity: REAL or FAKE.
- If authenticity is REAL, set fake_scope to N/A.
- If authenticity is FAKE, decide fake_scope as FULL_SYNTHETIC or TAMPERED.
- Output a bbox only when the final answer is TAMPERED.
- You may reject both reports if both fail the image-side check.
- Explain why direct acceptance is unsafe, which claim survives or fails, and why the final label follows."""
        output_format = """Required output format:
<authenticity_decision>
authenticity: REAL|FAKE
</authenticity_decision>

<scope_decision>
fake_scope: N/A|FULL_SYNTHETIC|TAMPERED
</scope_decision>

<routing_decision>
route_action: FULL_RECHECK
</routing_decision>

<final_assessment>
One paragraph. Explain what the image-side check finds, which report claim survives or fails, and why the final label follows.
</final_assessment>

<final_answer>REAL|FULL_SYNTHETIC|TAMPERED[, <|box_start|>(x1,y1),(x2,y2)<|box_end|>]</final_answer>"""
    return f"""<image>
You are a forensic decision maker.
You are given one image and two evidence reports.
Your task is to follow the recheck mode, compare the two reports against the image, and produce the final authenticity label.
Do not perform unrelated open-ended detection.

{class_definitions}

[Recheck Mode]
route_action: {route_action}

{task_rules}

{output_format}

Here are the two evidence reports:

[Visual Observer Report]
observer_answer: {observer['answer']}
observer_bbox: {observer['bbox']}
observer_report: {observer['text']}

[Forensic Evidence Provider Report]
evidence_provider_answer: {provider['answer']}
evidence_provider_bbox: {provider['bbox']}
evidence_provider_assessment: {provider['text']}

Now produce the final decision.
"""


def lazy_response(route_action: str, detail: str, final_answer: str, observer: dict[str, Any], provider: dict[str, Any]) -> str:
    authenticity = "REAL" if final_answer == "REAL" else "FAKE"
    scope = answer_scope(final_answer)
    if final_answer == "TAMPERED":
        final_text = f"{final_answer}, {final_box_for_tampered(observer, provider)}"
    else:
        final_text = final_answer
    assessment = (
        f"The two reports give the same {final_answer} answer and the gate route is {route_action}. "
        f"The available support is sufficient for direct acceptance under {detail}, so image-side VLM recheck is not used. "
        f"The final assessment is {final_answer}."
    )
    return f"""<authenticity_decision>
authenticity: {authenticity}
</authenticity_decision>

<scope_decision>
fake_scope: {scope}
</scope_decision>

<routing_decision>
route_action: {route_action}
</routing_decision>

<final_assessment>
{assessment}
</final_assessment>

<final_answer>{final_text}</final_answer>"""


def gt_label(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    label = norm_label(meta.get("gt_label_name"))
    if label != "UNKNOWN":
        return label
    raw = row.get("label")
    if isinstance(raw, int):
        return ID_TO_LABEL.get(raw, "UNKNOWN")
    if isinstance(raw, str) and raw.isdigit():
        return ID_TO_LABEL.get(int(raw), "UNKNOWN")
    return "UNKNOWN"


def build_split(src: Path, out_dir: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(src)
    manifest: list[dict[str, Any]] = []
    deep_rows: list[dict[str, Any]] = []
    lazy_rows: list[dict[str, Any]] = []
    action_counts = Counter()
    label_counts = Counter()
    deep_label_counts = Counter()

    for idx, row in enumerate(rows):
        observer = source_from_row(row, "observer", "observer_answer", "observer_bbox")
        provider = source_from_row(row, "provider", "evidence_provider_answer", "evidence_provider_bbox")
        action, detail, lazy_final = route_case(observer, provider)
        label = gt_label(row)
        action_counts[action] += 1
        label_counts[label] += 1
        image_path = (row.get("images") or [{}])[0].get("path")
        base_meta = dict(row.get("meta") or {})
        base_meta.update(
            {
                "schema_version": "v551_gate_aware_eval_v1",
                "source_eval_file": str(src),
                "row_index": idx,
                "gt_label_name": label,
                "v551_route_action": action,
                "v551_route_detail": detail,
                "v551_deep": action in {"TARGETED_RECHECK", "FULL_RECHECK"},
                "observer_support": observer["support"],
                "provider_support": provider["support"],
                "observer_support_source": observer["support_source"],
                "provider_support_source": provider["support_source"],
                "observer_answer": observer["answer"],
                "evidence_provider_answer": provider["answer"],
                "observer_bbox": observer["bbox"],
                "evidence_provider_bbox": provider["bbox"],
            }
        )
        manifest_row = {
            "row_index": idx,
            "split": split,
            "sample_id": base_meta.get("sample_id"),
            "image": image_path,
            "label": LABEL_TO_ID.get(label, row.get("label")),
            "meta": base_meta,
        }
        if action in {"TARGETED_RECHECK", "FULL_RECHECK"}:
            deep_index = len(deep_rows)
            base_meta["v551_deep_index"] = deep_index
            deep_label_counts[label] += 1
            deep_rows.append(
                {
                    "messages": [{"role": "user", "content": unified_prompt(action, observer, provider)}],
                    "images": [{"bytes": None, "path": image_path}],
                    "label": LABEL_TO_ID.get(label, row.get("label")),
                    "meta": base_meta,
                }
            )
            manifest_row["deep_index"] = deep_index
            manifest_row["lazy_response"] = None
        else:
            response = lazy_response(action, detail, lazy_final or observer["answer"], observer, provider)
            lazy_rows.append({**manifest_row, "response": response})
            manifest_row["deep_index"] = None
            manifest_row["lazy_response"] = response
        manifest.append(manifest_row)

    prefix = f"judge_val_{split}_v5_51_gate_aware"
    manifest_path = out_dir / f"{prefix}_manifest.jsonl"
    deep_path = out_dir / f"{prefix}_deep_vlm_infer.jsonl"
    lazy_path = out_dir / f"{prefix}_lazy_outputs.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(deep_path, deep_rows)
    write_jsonl(lazy_path, lazy_rows)
    return {
        "split": split,
        "source": str(src),
        "rows": len(rows),
        "manifest": str(manifest_path),
        "deep_vlm": str(deep_path),
        "lazy_outputs": str(lazy_path),
        "action_counts": dict(action_counts),
        "label_counts": dict(label_counts),
        "deep_label_counts": dict(deep_label_counts),
    }


def render_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# v5.51 Gate-Aware Eval Build",
        "",
        "This rebuild keeps the full eval row order in a manifest, sends only routed deep rows to the VLM, and stores lazy gate outputs separately.",
        "",
    ]
    for summary in summaries:
        n = summary["rows"]
        actions = Counter(summary["action_counts"])
        deep = actions["TARGETED_RECHECK"] + actions["FULL_RECHECK"]
        lazy = actions["NO_CHECK_CLEAN"] + actions["CAUTION_ACCEPT"]
        lines += [
            f"## {summary['split']}",
            "",
            f"- source: `{summary['source']}`",
            f"- rows: {n}",
            f"- lazy rows: {lazy}",
            f"- deep VLM rows: {deep}",
            f"- manifest: `{summary['manifest']}`",
            f"- deep VLM input: `{summary['deep_vlm']}`",
            f"- lazy output: `{summary['lazy_outputs']}`",
            "",
            "| route_action | count |",
            "|---|---:|",
        ]
        for key, value in actions.most_common():
            lines.append(f"| {key} | {value} |")
        lines += ["", "| GT label | all | deep |", "|---|---:|---:|"]
        labels = sorted(set(summary["label_counts"]) | set(summary["deep_label_counts"]))
        for label in labels:
            lines.append(f"| {label} | {summary['label_counts'].get(label, 0)} | {summary['deep_label_counts'].get(label, 0)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-eval", type=Path, required=True)
    parser.add_argument("--ood-eval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_split(args.in_eval, args.out_dir, "in"),
        build_split(args.ood_eval, args.out_dir, "ood"),
    ]
    report = args.out_dir / "v5_51_gate_aware_eval_build_summary.md"
    report.write_text(render_report(summaries), encoding="utf-8")
    print(report)
    for summary in summaries:
        print(summary["split"], summary["manifest"], summary["deep_vlm"])


if __name__ == "__main__":
    main()
