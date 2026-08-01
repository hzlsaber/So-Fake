#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw


FINAL_BLOCK_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.I | re.S)
FINAL_LINE_RE = re.compile(r"<final_answer>\s*([^\n]+)", re.I)
FIELD_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$")
BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)\s*,\s*"
    r"\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)\s*<\|box_end\|>",
    re.I,
)
LABELS = {"REAL", "FULL_SYNTHETIC", "TAMPERED"}
LABEL_FROM_ID = {0: "REAL", 1: "FULL_SYNTHETIC", 2: "TAMPERED"}


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("response"), str):
        return row["response"]
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def parse_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        m = FIELD_RE.match(line)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def parse_box(value: Any) -> list[int] | None:
    if not value:
        return None
    text = str(value)
    if text.upper() == "NONE":
        return None
    m = BOX_RE.search(text)
    if not m:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if len(nums) < 4:
            return None
        vals = [float(x) for x in nums[:4]]
    else:
        vals = [float(x) for x in m.groups()]
    x1, y1, x2, y2 = [int(round(v)) for v in vals]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_text(box: list[int] | None) -> str:
    if not box:
        return "NONE"
    x1, y1, x2, y2 = box
    return f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"


def parse_final(text: str) -> tuple[str, list[int] | None]:
    m = FINAL_BLOCK_RE.search(text or "") or FINAL_LINE_RE.search(text or "")
    if not m:
        return "PARSE_FAIL", None
    payload = m.group(1).strip()
    label_match = re.search(r"\b(REAL|FULL_SYNTHETIC|TAMPERED)\b", payload, re.I)
    label = label_match.group(1).upper() if label_match else "PARSE_FAIL"
    return label, parse_box(payload)


def authenticity(scope: str) -> str:
    return "REAL" if scope == "REAL" else "FAKE"


def _fmt_float(value: Any, ndigits: int = 3) -> str | None:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return None


def _descriptor_values(row: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    values = meta.get("provider_descriptor_values")
    if isinstance(values, dict):
        return values

    raw = None
    if row is not None:
        raw = row.get("desc") or row.get("descriptors")
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], list):
            raw = max(raw, key=lambda xs: float(xs[0]) if xs else -1.0)
        if len(raw) >= 5:
            # The compatible provider export stores [mass, entropy, concentration, peak, area].
            return {
                "mass": raw[0],
                "entropy": raw[1],
                "concentration": raw[2],
                "peak": raw[3],
                "area": raw[4],
            }
    return {}


def provider_assessment(
    answer: str | None,
    confidence: Any,
    support: Any,
    descriptors: dict[str, Any],
    box: list[int] | None,
    loc: dict[str, Any] | None,
) -> str:
    answer = (answer or "UNKNOWN").upper()
    conf = _fmt_float(confidence)
    mass = _fmt_float(descriptors.get("mass"))
    area = _fmt_float(descriptors.get("area"))
    peak = _fmt_float(descriptors.get("peak"))
    concentration = _fmt_float(descriptors.get("concentration"))
    entropy = _fmt_float(descriptors.get("entropy"))

    bits = []
    if conf is not None:
        bits.append(f"confidence {conf}")
    if support:
        bits.append(f"{str(support).lower()} support")
    prefix = f"The provider predicts {answer}"
    if bits:
        prefix += " with " + " and ".join(bits)
    prefix += "."

    desc_parts = []
    if mass is not None:
        desc_parts.append(f"mass {mass}")
    if peak is not None:
        desc_parts.append(f"peak {peak}")
    if concentration is not None:
        desc_parts.append(f"concentration {concentration}")
    if area is not None:
        desc_parts.append(f"area {area}")
    if entropy is not None:
        desc_parts.append(f"entropy {entropy}")
    desc_text = " The anomaly descriptors are " + ", ".join(desc_parts) + "." if desc_parts else ""

    if answer == "REAL":
        tail = " The topology response is treated as insufficient fake-side evidence, so no tamper localization is used."
    elif answer == "FULL_SYNTHETIC":
        tail = " The provider interprets the anomaly pattern as image-level synthetic evidence rather than a localized edit."
    elif answer == "TAMPERED":
        if box:
            tail = f" The provider interprets the bounded support region {box_text(box)} as localized manipulation evidence."
        else:
            tail = " The provider predicts localized manipulation, but no valid support box is available."
    else:
        tail = " The provider state is not parseable as a supported scope prediction."

    if loc and loc.get("status") and loc.get("status") != "ok":
        tail += f" Localization status: {loc.get('status')}."
    return prefix + desc_text + tail


def gt_label(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    if isinstance(meta.get("gt_label_name"), str):
        return meta["gt_label_name"].upper()
    raw = row.get("label")
    if isinstance(raw, int):
        return LABEL_FROM_ID.get(raw, "UNKNOWN")
    if isinstance(raw, str) and raw.isdigit():
        return LABEL_FROM_ID.get(int(raw), "UNKNOWN")
    return "UNKNOWN"


def image_path(row: dict[str, Any]) -> str:
    images = row.get("images") or []
    if images and isinstance(images[0], dict) and images[0].get("path"):
        return str(images[0]["path"])
    return str(row.get("image") or (row.get("meta") or {}).get("image_path") or "")


def index_by_sample_or_image(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        keys = [
            row.get("sample_id"),
            row.get("sid"),
            meta.get("sample_id"),
            row.get("image"),
            row.get("path"),
            image_path(row),
        ]
        for key in keys:
            if key:
                out[str(key)] = row
    return out


def index_topology_maps(root: Path | None) -> dict[str, str]:
    if root is None or not root.exists():
        return {}
    out: dict[str, str] = {}
    for path in root.rglob("*_topology_max.png"):
        sid = path.name.removesuffix("_topology_max.png")
        out[sid] = str(path)
    return out


def observer_from_row(row: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    if row is None:
        return {
            "answer": meta.get("observer_answer"),
            "authenticity": authenticity(str(meta.get("observer_answer"))) if meta.get("observer_answer") in LABELS else None,
            "bbox": parse_box(meta.get("observer_bbox")),
            "bbox_text": meta.get("observer_bbox") or "NONE",
            "report": meta.get("observer_report") or meta.get("observer_text") or "",
        }
    text = response_text(row)
    fields = parse_fields(text)
    answer = (
        fields.get("observer_answer")
        or fields.get("observer_scope")
        or fields.get("answer")
        or meta.get("observer_answer")
    )
    answer = str(answer).upper() if answer else None
    box = parse_box(fields.get("observer_bbox") or fields.get("bbox") or meta.get("observer_bbox"))
    return {
        "answer": answer,
        "authenticity": fields.get("observer_authenticity") or (authenticity(answer) if answer in LABELS else None),
        "bbox": box,
        "bbox_text": box_text(box),
        "report": fields.get("observer_report") or fields.get("reasoning") or text,
        "raw_response": text,
    }


def provider_from_row(
    row: dict[str, Any] | None,
    loc: dict[str, Any] | None,
    meta: dict[str, Any],
    topology_map_path: str | None = None,
) -> dict[str, Any]:
    desc = _descriptor_values(row, meta)
    box = parse_box(meta.get("evidence_provider_bbox"))
    if box is None and row is not None:
        box = parse_box(row.get("bbox"))
    if box is None and loc is not None:
        box = parse_box(loc.get("bbox"))
    answer = None
    if row is not None:
        answer = row.get("evidence_answer") or row.get("answer")
    answer = str(answer or meta.get("evidence_provider_answer") or "").upper() or None
    confidence = (row or {}).get("evidence_confidence", meta.get("provider_confidence"))
    support = meta.get("provider_support") or meta.get("v547_provider_support")
    assessment = meta.get("evidence_provider_assessment") or ""
    if not assessment:
        assessment = provider_assessment(answer, confidence, support, desc, box, loc)
    return {
        "answer": answer,
        "authenticity": authenticity(answer) if answer in LABELS else None,
        "confidence": confidence,
        "support": support,
        "descriptors": desc or meta.get("v547_provider_descriptors"),
        "bbox": box,
        "bbox_text": box_text(box),
        "assessment": assessment,
        "topology_map_path": topology_map_path,
        "localization": loc or {},
    }


def clamp_box(box: list[int], size: tuple[int, int]) -> list[int] | None:
    w, h = size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def save_box_mask(image: str, box: list[int], path: Path) -> bool:
    try:
        with Image.open(image) as im:
            size = im.size
    except Exception:
        return False
    box = clamp_box(box, size)
    if not box:
        return False
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)
    return True


def load_sam2(device: str, config: str, ckpt: str, no_postprocess: bool):
    sam2_root = os.environ.get("SOFAKE_R1_SAM2_ROOT", "")
    if sam2_root:
        sys.path.insert(0, sam2_root)
    from sam2.build_sam import build_sam2  # noqa: PLC0415
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

    if not ckpt:
        raise RuntimeError("SAM2 checkpoint is required when --mask-mode sam2 is enabled.")
    sam = build_sam2(config, ckpt, device=device, apply_postprocessing=not no_postprocess)
    return SAM2ImagePredictor(sam)


def save_sam2_mask(predictor, image: str, box: list[int], path: Path, multimask: bool = True) -> bool:
    try:
        import numpy as np  # noqa: PLC0415

        with Image.open(image).convert("RGB") as im:
            image_np = np.asarray(im)
            box = clamp_box(box, im.size)
        if not box:
            return False
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=multimask,
            return_logits=False,
        )
        idx = int(np.argmax(scores))
        mask = (np.asarray(masks[idx]) > 0).astype(np.uint8) * 255
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(path)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Export unified So-Fake-R1 inference records and final masks.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--observer-jsonl", type=Path)
    ap.add_argument("--provider-compatible", type=Path)
    ap.add_argument("--provider-localization", type=Path)
    ap.add_argument("--topology-map-dir", type=Path)
    ap.add_argument("--mask-mode", choices=["box", "sam2"], default="box")
    ap.add_argument("--sam-device", default="cuda")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam-ckpt", default=os.environ.get("SOFAKE_R1_SAM2_CKPT", ""))
    ap.add_argument("--sam-no-postprocess", action="store_true")
    ap.add_argument("--sam-multimask", type=int, default=1)
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)
    merged = read_jsonl(args.merged)
    if len(manifest) != len(merged):
        raise SystemExit(f"row count mismatch: manifest={len(manifest)} merged={len(merged)}")

    observer_map = index_by_sample_or_image(read_jsonl(args.observer_jsonl))
    provider_map = index_by_sample_or_image(read_jsonl(args.provider_compatible))
    loc_map = index_by_sample_or_image(read_jsonl(args.provider_localization))
    topology_map = index_topology_maps(args.topology_map_dir)

    predictor = None
    if args.mask_mode == "sam2":
        predictor = load_sam2(args.sam_device, args.sam_config, args.sam_ckpt, args.sam_no_postprocess)

    out_rows: list[dict[str, Any]] = []
    mask_written = 0
    for man, pred in zip(manifest, merged, strict=True):
        meta = dict((pred.get("meta") or man.get("meta") or {}))
        sid = str(meta.get("sample_id") or man.get("sample_id") or pred.get("sample_id") or len(out_rows))
        img = image_path(pred) or str(man.get("image") or "")
        keys = [sid, img, str(man.get("image") or "")]
        obs_row = next((observer_map[k] for k in keys if k in observer_map), None)
        prov_row = next((provider_map[k] for k in keys if k in provider_map), None)
        loc_row = next((loc_map[k] for k in keys if k in loc_map), None)

        text = response_text(pred)
        final_answer, final_box = parse_final(text)
        final_auth = authenticity(final_answer) if final_answer in LABELS else "PARSE_FAIL"
        mask_path = None
        if final_answer == "TAMPERED" and final_box and img:
            mask_path = args.mask_dir / f"{sid}.png"
            ok = (
                save_sam2_mask(predictor, img, final_box, mask_path, bool(args.sam_multimask))
                if predictor is not None
                else save_box_mask(img, final_box, mask_path)
            )
            if ok:
                mask_written += 1
            else:
                mask_path = None

        out_rows.append(
            {
                "sample_id": sid,
                "image": img,
                "gt_label": gt_label(pred),
                "route_action": meta.get("v551_route_action") or meta.get("route_action"),
                "route_detail": meta.get("v551_route_detail") or meta.get("route_detail"),
                "output_source": meta.get("v551_output_source"),
                "observer": observer_from_row(obs_row, meta),
                "provider": provider_from_row(prov_row, loc_row, meta, topology_map.get(sid)),
                "decision_maker": {
                    "raw_response": text,
                    "final_answer": final_answer,
                    "final_authenticity": final_auth,
                    "final_bbox": final_box,
                    "final_bbox_text": box_text(final_box),
                },
                "final": {
                    "authenticity": final_auth,
                    "scope": final_answer,
                    "bbox": final_box,
                    "bbox_text": box_text(final_box),
                    "mask_path": str(mask_path) if mask_path else None,
                    "mask_type": args.mask_mode if mask_path else None,
                },
            }
        )

    write_jsonl(args.out, out_rows)
    summary = {
        "rows": len(out_rows),
        "mask_mode": args.mask_mode,
        "masks_written": mask_written,
        "out": str(args.out),
        "mask_dir": str(args.mask_dir),
    }
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
