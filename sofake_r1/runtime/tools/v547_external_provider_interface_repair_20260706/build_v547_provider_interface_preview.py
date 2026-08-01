#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("SOFAKE_R1_WORKSPACE_ROOT", Path.home()))
QWEN = ROOT / "ms-swift/examples/train/seq_cls/qwen2_5"
BASE = QWEN / "judge_data/locked_provider_v58_20260629"
V539_PROMPTS = BASE / "external_social_v5_39_evidence_gate_prompt_20260705"

BFREE_RAW = QWEN / "judge_data/bfree_online_sofake_r1_v1/feta_evidence.jsonl"
WILDRF_RAW_FILES = [
    QWEN / "judge_data/wildrf_evidence_only_v1/facebook/feta_evidence.jsonl",
    QWEN / "judge_data/wildrf_evidence_only_v1/reddit/feta_evidence.jsonl",
    QWEN / "judge_data/wildrf_evidence_only_v1/twitter/feta_evidence.jsonl",
]

OUT_DIR = ROOT / "So-Fake/v547_external_provider_interface_repair_20260706"
OUT_SUMMARY = OUT_DIR / "v547_provider_interface_field_inventory_20260706.json"
OUT_REPORT = OUT_DIR / "v547_provider_interface_field_inventory_20260706.md"
OUT_EXAMPLES = OUT_DIR / "v547_provider_interface_examples_20260706.md"

LABELS = {0: "REAL", 1: "FULL_SYNTHETIC", 2: "TAMPERED"}
BOX_RE = re.compile(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def path_key(path: str) -> str:
    return os.path.realpath(path)


def image_path(row: dict[str, Any]) -> str:
    images = row.get("images") or []
    if images and isinstance(images[0], dict) and images[0].get("path"):
        return str(images[0]["path"])
    return str(row.get("meta", {}).get("image_path", ""))


def extract_field(block: str, field: str, fallback: str = "") -> str:
    m = re.search(rf"^{re.escape(field)}:\s*(.*?)(?=\n[a-zA-Z_]+:\s|\Z)", block, flags=re.S | re.M)
    if not m:
        return fallback
    return re.sub(r"\s+", " ", m.group(1)).strip()


def extract_block(text: str, title: str, next_title: str | None = None) -> str:
    if next_title:
        pat = rf"\[{re.escape(title)}\]\s*(.*?)(?=\n\[{re.escape(next_title)}\])"
    else:
        pat = rf"\[{re.escape(title)}\]\s*(.*?)(?=\n\[STEP 3:|\Z)"
    m = re.search(pat, text, flags=re.S)
    return m.group(1).strip() if m else ""


def load_raw_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            out[path_key(str(row["path"]))] = row
    return out


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def desc_from_raw(row: dict[str, Any] | None) -> dict[str, float] | None:
    if not row:
        return None
    desc = row.get("desc")
    if not isinstance(desc, list) or not desc:
        return None
    valid = [r for r in desc if isinstance(r, list) and len(r) >= 5]
    if not valid:
        return None
    selected = max(valid, key=lambda r: float(r[0]))
    # Existing provider assets use [mass, entropy, concentration, peak, area].
    return {
        "mass": float(selected[0]),
        "entropy": float(selected[1]),
        "concentration": float(selected[2]),
        "peak": float(selected[3]),
        "area": float(selected[4]),
    }


def grid_bbox_to_224(row: dict[str, Any] | None) -> str:
    if not row:
        return "NONE"
    bbox = row.get("bbox")
    grid = row.get("grid")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "NONE"
    if not isinstance(grid, list) or len(grid) < 2:
        return "NONE"
    h, w = float(grid[0]), float(grid[1])
    if h <= 0 or w <= 0:
        return "NONE"
    x1, y1, x2, y2 = [float(v) for v in bbox]
    vals = [
        round(max(0, min(224, x1 / w * 224))),
        round(max(0, min(224, y1 / h * 224))),
        round(max(0, min(224, x2 / w * 224))),
        round(max(0, min(224, y2 / h * 224))),
    ]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return "NONE"
    return f"<|box_start|>({vals[0]},{vals[1]}),({vals[2]},{vals[3]})<|box_end|>"


def bbox_ratio(bbox: str) -> float | None:
    m = BOX_RE.search(bbox or "")
    if not m:
        return None
    x1, y1, x2, y2 = [int(v) for v in m.groups()]
    return max(0, x2 - x1) * max(0, y2 - y1) / float(224 * 224)


def margin_from_logits(logits: list[float] | None, pred: int) -> float | None:
    if not isinstance(logits, list) or len(logits) < 3 or pred not in {0, 1, 2}:
        return None
    ordered = sorted([float(v) for v in logits], reverse=True)
    return ordered[0] - ordered[1]


def confidence_level(margin: float | None, meta: dict[str, Any]) -> str:
    # Use raw margin only as a confidence proxy. For WildRF, keep the locked
    # provider effective_s in metadata but do not expose transform names.
    if margin is None:
        if meta.get("provider_effective_s") is not None:
            try:
                s = float(meta["provider_effective_s"])
                if s >= 3.2:
                    return "HIGH"
                if s >= 2.4:
                    return "MEDIUM"
            except Exception:
                pass
        return "UNKNOWN"
    if margin >= 8.0:
        return "HIGH"
    if margin >= 3.0:
        return "MEDIUM"
    return "LOW"


def descriptor_strength(answer: str, desc: dict[str, float] | None, bbox: str) -> str:
    if desc is None:
        return "unknown"
    mass = desc["mass"]
    area = desc["area"]
    entropy = desc["entropy"]
    peak = desc["peak"]
    concentration = desc["concentration"]
    bbox_present = bbox != "NONE"

    if answer == "REAL":
        # External descriptor scale is larger than internal v531, so the REAL
        # rule is deliberately conservative: REAL can still be mixed even when
        # the locked class is REAL.
        score = 0
        if mass <= 25.0:
            score += 1
        if area <= 0.10:
            score += 1
        if peak <= 0.95:
            score += 1
        if concentration >= 0.97 and area <= 0.18:
            score += 1
        if score >= 3:
            return "strong"
        if score >= 2:
            return "moderate"
        return "weak"

    if answer == "FULL_SYNTHETIC":
        votes = int(mass >= 70.0) + int(area >= 0.25) + int(entropy >= 4.7)
        if votes >= 3:
            return "strong"
        if votes >= 2:
            return "moderate"
        if votes == 1:
            return "mild"
        return "weak"

    if answer == "TAMPERED":
        votes = (
            int(peak >= 0.95)
            + int(concentration >= 0.75)
            + int(area >= 0.05)
            + int(area <= 0.55)
            + int(bbox_present)
        )
        if votes >= 4 and bbox_present:
            return "strong"
        if votes >= 3:
            return "moderate"
        if votes >= 2:
            return "mild"
        return "weak"

    return "unknown"


def support_level(desc_strength: str, conf_level: str) -> str:
    if desc_strength == "unknown":
        return "UNKNOWN"
    if desc_strength == "strong" and conf_level in {"HIGH", "MEDIUM"}:
        return "STRONG"
    if desc_strength == "strong":
        return "MODERATE"
    if desc_strength == "moderate" and conf_level == "HIGH":
        return "MODERATE"
    if desc_strength == "moderate":
        return "MILD"
    if desc_strength == "mild" and conf_level in {"HIGH", "MEDIUM"}:
        return "MILD"
    return "WEAK"


def evidence_class(answer: str, support: str, desc: dict[str, float] | None, bbox: str) -> str:
    if answer == "REAL":
        return "REAL_STRONG" if support == "STRONG" else "REAL_WEAK_OR_MIXED"
    if answer == "FULL_SYNTHETIC":
        if support == "STRONG":
            return "FULL_SYNTHETIC_STRONG"
        if support == "MODERATE":
            return "FULL_SYNTHETIC_MODERATE"
        return "FULL_SYNTHETIC_WEAK_OR_CONFLICTED"
    if answer == "TAMPERED":
        ratio = bbox_ratio(bbox)
        if ratio is not None and ratio >= 0.45:
            return "TAMPERED_BROAD_REGION"
        if support == "STRONG":
            return "TAMPERED_STRONG_LOCAL"
        if support == "MODERATE":
            return "TAMPERED_MODERATE_LOCAL"
        return "TAMPERED_WEAK_OR_NO_BBOX"
    return "UNKNOWN"


def descriptor_text(desc: dict[str, float] | None) -> str:
    if not desc:
        return "descriptors unavailable"
    return (
        f"mass={desc['mass']:.2f}, area={desc['area']:.4f}, "
        f"entropy={desc['entropy']:.4f}, peak={desc['peak']:.4f}, "
        f"concentration={desc['concentration']:.4f}"
    )


def assessment(answer: str, ev_class: str, bbox: str, support: str) -> str:
    if ev_class == "REAL_STRONG":
        return f"Evidence provider answer is REAL. Low anomaly extent and a strong class margin give clear REAL-side evidence. Evidence provider bbox is {bbox}."
    if ev_class == "REAL_WEAK_OR_MIXED":
        return f"Evidence provider answer is REAL. The locked class is REAL, but the topology descriptors are mixed rather than cleanly low-anomaly, so provider support is {support.lower()}. Evidence provider bbox is {bbox}."
    if ev_class == "FULL_SYNTHETIC_STRONG":
        return f"Evidence provider answer is FULL_SYNTHETIC. Broad high-activation topology and strong class support give image-level fake evidence. Evidence provider bbox is {bbox}."
    if ev_class == "FULL_SYNTHETIC_MODERATE":
        return f"Evidence provider answer is FULL_SYNTHETIC. The topology signal is image-level but not maximally strong, giving moderate full-synthesis support. Evidence provider bbox is {bbox}."
    if ev_class == "FULL_SYNTHETIC_WEAK_OR_CONFLICTED":
        return f"Evidence provider answer is FULL_SYNTHETIC. The class points to image-level fake evidence, but descriptor or margin support is weak or mixed. Evidence provider bbox is {bbox}."
    if ev_class == "TAMPERED_STRONG_LOCAL":
        return f"Evidence provider answer is TAMPERED. Localized topology support and a valid provider bbox give strong local-edit evidence. Evidence provider bbox is {bbox}."
    if ev_class == "TAMPERED_BROAD_REGION":
        return f"Evidence provider answer is TAMPERED. The provider marks a broad candidate edited region with {support.lower()} support. Evidence provider bbox is {bbox}."
    if ev_class == "TAMPERED_MODERATE_LOCAL":
        return f"Evidence provider answer is TAMPERED. A valid bbox is present, but descriptor or margin strength keeps the local-edit evidence moderate. Evidence provider bbox is {bbox}."
    if ev_class == "TAMPERED_WEAK_OR_NO_BBOX":
        return f"Evidence provider answer is TAMPERED. The class points to local editing, but localization or topology support is weak. Evidence provider bbox is {bbox}."
    return f"Evidence provider answer is {answer}. Provider support is {support}. Evidence provider bbox is {bbox}."


def normalized_provider(row: dict[str, Any], raw: dict[str, Any] | None) -> dict[str, Any]:
    meta = row.get("meta", {})
    answer = str(meta.get("evidence_provider_answer", "")).upper()
    locked_pred = int(meta.get("provider_pred3", {"REAL": 0, "FULL_SYNTHETIC": 1, "TAMPERED": 2}.get(answer, -1)))
    raw_pred = raw.get("feta_pred") if raw else None
    desc = desc_from_raw(raw)
    raw_bbox = grid_bbox_to_224(raw)
    bbox = raw_bbox if answer == "TAMPERED" else "NONE"
    margin = margin_from_logits(raw.get("logits") if raw else None, locked_pred)
    conf = confidence_level(margin, meta)
    ds = descriptor_strength(answer, desc, bbox)
    support = support_level(ds, conf)
    ev_class = evidence_class(answer, support, desc, bbox)
    return {
        "answer": answer,
        "bbox": bbox,
        "bbox_valid": bool(bbox != "NONE"),
        "support": support,
        "evidence_class": ev_class,
        "descriptor_strength": ds,
        "confidence_level": conf,
        "descriptor_values": desc,
        "descriptor_text": descriptor_text(desc),
        "raw_bbox_224": raw_bbox,
        "locked_pred3": locked_pred,
        "raw_feta_pred": raw_pred,
        "raw_locked_pred_mismatch": raw_pred is not None and int(raw_pred) != locked_pred,
        "logit_margin_proxy": None if margin is None else round(margin, 4),
        "assessment": assessment(answer, ev_class, bbox, support),
    }


def old_provider_block(row: dict[str, Any]) -> str:
    text = row["messages"][0]["content"]
    block = extract_block(text, "Forensic Evidence Provider Report")
    return block


def observer_block(row: dict[str, Any]) -> str:
    text = row["messages"][0]["content"]
    return extract_block(text, "Visual Observer Report", "Forensic Evidence Provider Report")


def collect_dataset(name: str, prompt_path: Path, raw_map: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(prompt_path)
    out_rows = []
    stats: dict[str, Any] = {
        "n": len(rows),
        "raw_hit": 0,
        "raw_missing": 0,
        "meta_provider_answer_counts": Counter(),
        "raw_pred_counts": Counter(),
        "locked_raw_pred_pairs": Counter(),
        "provider_answer_tampered": 0,
        "tampered_bbox_available": 0,
        "descriptor_available": 0,
        "evidence_class_counts": Counter(),
        "support_counts": Counter(),
        "bfree_or_wildrf_meta_fields": Counter(),
        "provider_answer_exact_accuracy": Counter(),
        "provider_answer_binary_accuracy": Counter(),
        "evidence_class_exact_accuracy": Counter(),
        "evidence_class_binary_accuracy": Counter(),
    }
    for row in rows:
        img = image_path(row)
        raw = raw_map.get(path_key(img))
        if raw:
            stats["raw_hit"] += 1
        else:
            stats["raw_missing"] += 1
        meta = row.get("meta", {})
        for key in meta:
            stats["bfree_or_wildrf_meta_fields"][key] += 1
        prov = normalized_provider(row, raw)
        stats["meta_provider_answer_counts"][prov["answer"]] += 1
        if prov["raw_feta_pred"] is not None:
            stats["raw_pred_counts"][str(prov["raw_feta_pred"])] += 1
            stats["locked_raw_pred_pairs"][f"{prov['locked_pred3']}->{prov['raw_feta_pred']}"] += 1
        if prov["answer"] == "TAMPERED":
            stats["provider_answer_tampered"] += 1
            if prov["bbox_valid"]:
                stats["tampered_bbox_available"] += 1
        if prov["descriptor_values"]:
            stats["descriptor_available"] += 1
        stats["evidence_class_counts"][prov["evidence_class"]] += 1
        stats["support_counts"][prov["support"]] += 1
        gt = str(meta.get("gt_label_name", "")).upper()
        pred = prov["answer"]
        exact_ok = pred == gt
        binary_ok = (pred != "REAL") == (gt != "REAL")
        stats["provider_answer_exact_accuracy"][pred] += int(exact_ok)
        stats["provider_answer_exact_accuracy"][f"{pred}__total"] += 1
        stats["provider_answer_binary_accuracy"][pred] += int(binary_ok)
        stats["provider_answer_binary_accuracy"][f"{pred}__total"] += 1
        cls = prov["evidence_class"]
        stats["evidence_class_exact_accuracy"][cls] += int(exact_ok)
        stats["evidence_class_exact_accuracy"][f"{cls}__total"] += 1
        stats["evidence_class_binary_accuracy"][cls] += int(binary_ok)
        stats["evidence_class_binary_accuracy"][f"{cls}__total"] += 1
        out_rows.append({"row": row, "raw": raw, "provider": prov})
    clean_stats = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            clean_stats[key] = dict(value)
        else:
            clean_stats[key] = value
    return out_rows, clean_stats


def pick_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [
        ("REAL_STRONG", None),
        ("REAL_WEAK_OR_MIXED", None),
        ("FULL_SYNTHETIC_STRONG", None),
        ("FULL_SYNTHETIC_WEAK_OR_CONFLICTED", None),
        ("TAMPERED_STRONG_LOCAL", None),
        ("TAMPERED_BROAD_REGION", None),
    ]
    picked = []
    used = set()
    for ev_class, _ in wanted:
        for item in rows:
            sid = item["row"].get("meta", {}).get("sample_id")
            if sid in used:
                continue
            if item["provider"]["evidence_class"] == ev_class:
                picked.append(item)
                used.add(sid)
                break
    if len(picked) < 6:
        for item in rows:
            sid = item["row"].get("meta", {}).get("sample_id")
            if sid not in used:
                picked.append(item)
                used.add(sid)
            if len(picked) >= 6:
                break
    return picked


def render_report(summary: dict[str, Any]) -> str:
    def fmt_acc(counter: dict[str, int]) -> str:
        parts = []
        for key in sorted(k for k in counter if not k.endswith("__total")):
            total = counter.get(f"{key}__total", 0)
            ok = counter.get(key, 0)
            if total:
                parts.append(f"{key}: {ok}/{total}={ok / total:.3f}")
        return "; ".join(parts)

    lines = [
        "# v5.47 External Provider Interface Field Inventory",
        "",
        "## Core Finding",
        "",
        "The external BFree/WildRF evaluation does have usable raw provider signals, but the v5.39 gate prompt collapses them into a generic provider sentence.",
        "For v5.47, the locked provider class should remain the class source, while raw topology descriptors and grid bbox should be used only as evidence-shape fields.",
        "",
        "## Dataset Coverage",
        "",
    ]
    for name, stats in summary["datasets"].items():
        lines += [
            f"### {name}",
            "",
            f"- rows: `{stats['n']}`",
            f"- raw provider hit: `{stats['raw_hit']}` / `{stats['n']}`",
            f"- descriptor available: `{stats['descriptor_available']}` / `{stats['n']}`",
            f"- provider_answer=TAMPERED bbox available: `{stats['tampered_bbox_available']}` / `{stats['provider_answer_tampered']}`",
            f"- locked provider answer counts: `{stats['meta_provider_answer_counts']}`",
            f"- v547 evidence class counts: `{stats['evidence_class_counts']}`",
            f"- support counts: `{stats['support_counts']}`",
            f"- locked_pred3 -> raw_feta_pred pairs: `{stats['locked_raw_pred_pairs']}`",
            f"- provider answer exact accuracy: `{fmt_acc(stats['provider_answer_exact_accuracy'])}`",
            f"- provider answer binary accuracy: `{fmt_acc(stats['provider_answer_binary_accuracy'])}`",
            f"- evidence class exact accuracy: `{fmt_acc(stats['evidence_class_exact_accuracy'])}`",
            f"- evidence class binary accuracy: `{fmt_acc(stats['evidence_class_binary_accuracy'])}`",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "- `raw_feta_pred` should not overwrite the locked provider answer because it mismatches the locked `provider_pred3` on many examples.",
        "- `desc` and `grid/bbox` are still useful: they let us rebuild provider evidence in the same style as SFT instead of using generic text.",
        "- Visible prompts should not expose `blur_only05` or `log_scaled_a2`; those are audit metadata, not human-readable evidence.",
        "- For binary social datasets, exact FULL_SYNTHETIC/TAMPERED scope is less reliable than REAL/FAKE, but TAMPERED still needs a visible bbox when the provider claims it.",
        "- On BFree/WildRF, provider=TAMPERED is mostly a fake-side signal: exact three-way accuracy is low because the external ground truth has no true TAMPERED class, but binary accuracy can still be high.",
        "- Therefore v5.47 should preserve `evidence_provider_answer` for compatibility, but expose an additional fake-side interpretation through support/evidence class so the DM does not over-penalize TAMPERED on binary social data.",
        "",
    ]
    return "\n".join(lines)


def render_examples(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# v5.47 Provider Interface Examples", ""]
    for dataset, rows in groups.items():
        lines += [f"## {dataset}", ""]
        for item in pick_examples(rows):
            row = item["row"]
            meta = row["meta"]
            prov = item["provider"]
            raw = item["raw"]
            lines += [
                f"### {meta.get('sample_id')} | GT={meta.get('gt_label_name')} | O={meta.get('observer_answer')} | P={prov['answer']}",
                "",
                f"- image: `{image_path(row)}`",
                f"- locked_pred3 / raw_feta_pred: `{prov['locked_pred3']}` / `{prov['raw_feta_pred']}`",
                f"- evidence_class: `{prov['evidence_class']}`",
                f"- support: `{prov['support']}`",
                f"- descriptor: `{prov['descriptor_text']}`",
                f"- raw_bbox_224: `{prov['raw_bbox_224']}`",
                "",
                "**Old v5.39 provider block**",
                "",
                "```text",
                old_provider_block(row),
                "```",
                "",
                "**v5.47 repaired provider block**",
                "",
                "```text",
                f"evidence_provider_answer: {prov['answer']}",
                f"evidence_provider_bbox: {prov['bbox']}",
                f"evidence_provider_assessment: {prov['assessment']}",
                f"evidence_provider_evidence_class: {prov['evidence_class']}",
                f"evidence_provider_support: {prov['support']}",
                f"evidence_provider_descriptors: {prov['descriptor_text']}",
                "```",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    bfree_raw = load_raw_map([BFREE_RAW])
    wildrf_raw = load_raw_map(WILDRF_RAW_FILES)
    datasets = {
        "bfree": (
            V539_PROMPTS / "bfree_v5_39_evidence_gate_prompt_infer.jsonl",
            bfree_raw,
        ),
        "wildrf": (
            V539_PROMPTS / "wildrf_v5_39_evidence_gate_prompt_infer.jsonl",
            wildrf_raw,
        ),
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    summary = {"datasets": {}}
    for name, (prompt_path, raw) in datasets.items():
        rows, stats = collect_dataset(name, prompt_path, raw)
        grouped[name] = rows
        summary["datasets"][name] = stats

    write_json(OUT_SUMMARY, summary)
    write_text(OUT_REPORT, render_report(summary))
    write_text(OUT_EXAMPLES, render_examples(grouped))
    print(f"WROTE {OUT_SUMMARY}")
    print(f"WROTE {OUT_REPORT}")
    print(f"WROTE {OUT_EXAMPLES}")


if __name__ == "__main__":
    main()
