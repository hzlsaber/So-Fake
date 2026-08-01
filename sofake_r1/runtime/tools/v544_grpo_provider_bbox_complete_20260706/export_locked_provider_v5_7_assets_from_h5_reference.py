#!/usr/bin/env python3
"""Export SFT assets for the locked evidence-provider Decision Maker data.

For selected samples this writes:
  - current Qwen-FETA topology maps for V5/V5.7 Image 2
  - old-compatible provider evidence JSONL (desc/logits/bbox/grid)
  - conditional localization evidence JSONL

Localization is emitted only when the locked provider predicts TAMPERED. The
reported bbox is derived from the provider-side prompt head and, when enabled,
SAM2's final mask output. No GT mask is read for prompt-side localization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", Path(__file__).resolve().parents[2]))
BUNDLE_TOOLS = RELEASE_ROOT / "tools"
BUNDLE_M0_PROBE = BUNDLE_TOOLS / "m0_probe"
for _path in (
    BUNDLE_TOOLS,
    BUNDLE_M0_PROBE,
):
    try:
        _ok = _path.exists()
    except OSError:
        _ok = False
    if _ok:
        sys.path.insert(0, str(_path))

from export_v5_topology_maps import (  # noqa: E402
    FETAv0,
    map_to_rgb,
    output_path_for_sid,
    stats_path_for_sid,
    topology_visual_stats,
)
from m4_feta_v0 import unpermute_feat_to_raster  # noqa: E402
from m44_l19_tamper_mask_branch import bbox_from_binary  # noqa: E402
from m56_l19_heatmap_bbox_prompt import (  # noqa: E402
    L19HeatmapBBoxHead,
    grid_bbox_to_image,
    norm_box_to_pixel,
    run_sam_predict,
    top_point_from_heatmap,
)


LABEL_NAMES = {0: "REAL", 1: "FULL_SYNTHETIC", 2: "TAMPERED"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_locked_evidence(path: Path) -> dict[str, dict]:
    out = {}
    for row in read_jsonl(path):
        sid = row.get("sid") or row.get("sample_id")
        if sid:
            out[str(sid)] = row
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evidence_grid_bbox(e_main: np.ndarray) -> list[float]:
    e = np.asarray(e_main, dtype=np.float32)
    if e.size == 0:
        return [-1.0, -1.0, -1.0, -1.0]
    q90 = float(np.quantile(e, 0.90))
    thr = max(0.50, min(0.75, q90))
    mask = e >= thr
    if not mask.any() and float(e.max()) > 0:
        mask = e >= max(0.25, 0.75 * float(e.max()))
    box = bbox_from_binary(mask)
    if box is None:
        y, x = np.unravel_index(int(np.argmax(e)), e.shape)
        box = (int(x), int(y), int(x) + 1, int(y) + 1)
    return [float(v) for v in box]


def load_qwen_feta(path: Path, device: str) -> FETAv0:
    model = FETAv0(in_dim=1280, hidden=128, K=2, use_spatial_conv=True, desc_set="minimal")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def run_qwen_feta(model: FETAv0, grp: h5py.Group, device: str) -> dict:
    grid_thw = grp.attrs["grid_thw"]
    T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    feat = torch.from_numpy(grp["layer_19"][:]).float()
    feat = unpermute_feat_to_raster(feat, T, H, W).unsqueeze(0).to(device)
    attn = torch.ones(1, feat.shape[1], device=device)
    out = model(
        feat,
        attn,
        torch.tensor([T], device=device),
        torch.tensor([H], device=device),
        torch.tensor([W], device=device),
    )
    e = out["E"][0, :, :H, :W].detach().cpu().numpy()
    return {
        "e_main": np.maximum(e[0], e[1]),
        "desc": out["descriptors"][0].detach().cpu().numpy().astype(float).tolist(),
        "qwen_logits": out["logits"][0].detach().cpu().numpy().astype(float).tolist(),
        "grid": [int(H), int(W)],
        "feat": feat.detach().cpu(),
        "T": T,
        "H": H,
        "W": W,
    }


def load_prompt_head(path: Path, device: str, hidden: int, depth: int) -> L19HeatmapBBoxHead:
    model = L19HeatmapBBoxHead(in_dim=1280, hidden=hidden, depth=depth)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def make_sam_mask_input(logit_grid: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    img = Image.fromarray(np.asarray(logit_grid, dtype=np.float32), mode="F").resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(alpha * arr + beta, -32.0, 32.0).astype(np.float32)
    return arr[None, :, :]


def maybe_load_sam2(args):
    if not args.run_sam2:
        return None
    sam2_root = os.environ.get("SOFAKE_R1_SAM2_ROOT", "")
    if sam2_root:
        sys.path.insert(0, sam2_root)
    from sam2.build_sam import build_sam2  # noqa: PLC0415
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

    if not args.sam_ckpt:
        raise RuntimeError("SAM2 checkpoint is required when --run_sam2 is enabled.")
    sam = build_sam2(args.sam_config, args.sam_ckpt, device=args.device, apply_postprocessing=not args.no_postprocess)
    return SAM2ImagePredictor(sam)


@torch.no_grad()
def localization_for_sample(prompt_model, predictor, feat_cpu, sample: dict, args) -> dict:
    image_path = sample["image"]
    try:
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
    except Exception as exc:
        return {"sample_id": sample["sample_id"], "status": "unavailable", "reason": f"image_error:{exc}"}

    feat = feat_cpu.to(args.device)
    grid_thw = sample["_grid"]
    T, H, W = [int(x) for x in grid_thw]
    out = prompt_model(
        feat,
        torch.tensor([T], device=args.device),
        torch.tensor([H], device=args.device),
        torch.tensor([W], device=args.device),
    )
    prob = out["prob"][0, :H, :W].detach().cpu().numpy()
    logit = out["heat_logits"][0, :H, :W].detach().cpu().numpy()
    heat_box = grid_bbox_to_image(bbox_from_binary(prob >= float(args.prompt_thr)), prob.shape, image_size, pad=args.box_pad)
    bbox_head = norm_box_to_pixel(out["bbox"][0].detach().cpu().numpy(), image_size)
    prompt_box = heat_box or bbox_head
    point = top_point_from_heatmap(prob, image_size)

    if predictor is None:
        box = prompt_box
        return {
            "sample_id": sample["sample_id"],
            "status": "available" if box else "unavailable",
            "bbox": list(box) if box else None,
            "mask_area": float((prob >= float(args.prompt_thr)).mean()),
            "mask_confidence": float(prob.max()) if prob.size else 0.0,
            "prompt_bbox": list(prompt_box) if prompt_box else None,
            "prompt_point": point.reshape(-1).astype(float).tolist(),
            "sam2_enabled": False,
        }

    try:
        mask_input = make_sam_mask_input(logit, args.mask_alpha, args.mask_beta)
        mask, score = run_sam_predict(
            predictor,
            np.asarray(image),
            box=prompt_box,
            point=point,
            mask_input=mask_input,
            multimask=bool(args.multimask),
        )
        box = bbox_from_binary(np.asarray(mask).astype(bool))
        if box is None:
            box = prompt_box
        return {
            "sample_id": sample["sample_id"],
            "status": "available" if box else "unavailable",
            "bbox": list(box) if box else None,
            "mask_area": float(np.asarray(mask).astype(bool).mean()),
            "mask_confidence": float(score),
            "prompt_bbox": list(prompt_box) if prompt_box else None,
            "prompt_point": point.reshape(-1).astype(float).tolist(),
            "prompt_thr": float(args.prompt_thr),
            "mask_alpha": float(args.mask_alpha),
            "mask_beta": float(args.mask_beta),
            "sam2_enabled": True,
        }
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"],
            "status": "unavailable",
            "reason": f"sam2_error:{type(exc).__name__}:{exc}",
            "prompt_bbox": list(prompt_box) if prompt_box else None,
            "prompt_point": point.reshape(-1).astype(float).tolist(),
            "sam2_enabled": True,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--locked_evidence", type=Path, required=True)
    ap.add_argument("--map_dir", type=Path, required=True)
    ap.add_argument("--compatible_evidence_out", type=Path, required=True)
    ap.add_argument("--localization_out", type=Path, required=True)
    ap.add_argument("--summary_out", type=Path, required=True)
    ap.add_argument("--qwen_feta_ckpt", type=Path, default=Path(os.environ.get("SOFAKE_R1_QWEN_FETA_CKPT", "")))
    ap.add_argument("--prompt_ckpt", type=Path, default=Path(os.environ.get("SOFAKE_R1_PROMPT_CKPT", "")))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run_sam2", action="store_true")
    ap.add_argument("--sam_config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam_ckpt", default=os.environ.get("SOFAKE_R1_SAM2_CKPT", ""))
    ap.add_argument("--no_postprocess", action="store_true")
    ap.add_argument("--multimask", type=int, default=1)
    ap.add_argument("--prompt_thr", type=float, default=0.10)
    ap.add_argument("--box_pad", type=int, default=0)
    ap.add_argument("--mask_alpha", type=float, default=0.25)
    ap.add_argument("--mask_beta", type=float, default=-0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--allow_missing_locked_evidence", action="store_true")
    args = ap.parse_args()

    samples = read_jsonl(args.samples)
    locked = load_locked_evidence(args.locked_evidence)
    args.map_dir.mkdir(parents=True, exist_ok=True)

    qwen_feta = load_qwen_feta(args.qwen_feta_ckpt, args.device)
    prompt_model = load_prompt_head(args.prompt_ckpt, args.device, args.hidden, args.depth)
    predictor = maybe_load_sam2(args)

    compatible_rows = []
    localization_rows = []
    counters = Counter()
    with h5py.File(args.h5, "r") as h5:
        for i, sample in enumerate(samples, 1):
            sid = sample["sample_id"]
            ev = locked.get(sid)
            if ev is None:
                counters["missing_locked_evidence"] += 1
                continue
            if sid not in h5:
                counters["missing_h5_sid"] += 1
                continue
            grp = h5[sid]
            image_path = sample["image"]
            try:
                image = Image.open(image_path).convert("RGB")
                image_size = image.size
            except Exception:
                counters["missing_image"] += 1
                continue

            q = run_qwen_feta(qwen_feta, grp, args.device)
            e_main = q["e_main"]
            map_path = output_path_for_sid(args.map_dir, sid)
            stats_path = stats_path_for_sid(args.map_dir, sid)
            if args.force or not (map_path.exists() and stats_path.exists()):
                map_path.parent.mkdir(parents=True, exist_ok=True)
                map_to_rgb(e_main, image_size, args.cmap).save(map_path)
                stats = topology_visual_stats(e_main)
                stats.update({
                    "sample_id": sid,
                    "image_path": image_path,
                    "map_path": str(map_path),
                    "h5": str(args.h5),
                    "ckpt": str(args.qwen_feta_ckpt),
                })
                stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                counters["maps_written"] += 1
            else:
                counters["maps_skipped"] += 1

            pred = int(ev["evidence_pred"])
            compatible_rows.append({
                "path": image_path,
                "label": int(sample.get("gt_cls", ev.get("label", -1))),
                "feta_pred": pred,
                "logits": ev.get("logits"),
                "desc": q["desc"],
                "bbox": evidence_grid_bbox(e_main),
                "grid": q["grid"],
                "sample_id": sid,
                "sid": sid,
                "evidence_answer": LABEL_NAMES.get(pred, ev.get("evidence_answer")),
                "evidence_confidence": ev.get("evidence_confidence"),
                "qwen_logits": q["qwen_logits"],
                "suspicion": ev.get("suspicion"),
                "effective_s": ev.get("effective_s"),
                "transform": ev.get("transform"),
            })

            if pred == 2:
                loc_sample = dict(sample)
                loc_sample["_grid"] = [q["T"], q["H"], q["W"]]
                loc = localization_for_sample(prompt_model, predictor, q["feat"], loc_sample, args)
                counters[f"loc_{loc.get('status', 'unknown')}"] += 1
            else:
                loc = {"sample_id": sid, "status": "not_applicable"}
                counters["loc_not_applicable"] += 1
            localization_rows.append(loc)

            if i % 100 == 0 or i == len(samples):
                print(
                    f"progress {i}/{len(samples)} "
                    f"maps={counters['maps_written']} loc_avail={counters['loc_available']} "
                    f"missing={counters['missing_h5_sid'] + counters['missing_locked_evidence'] + counters['missing_image']}",
                    flush=True,
                )

    write_jsonl(args.compatible_evidence_out, compatible_rows)
    write_jsonl(args.localization_out, localization_rows)
    summary = {
        "samples": str(args.samples),
        "h5": str(args.h5),
        "locked_evidence": str(args.locked_evidence),
        "map_dir": str(args.map_dir),
        "compatible_evidence_out": str(args.compatible_evidence_out),
        "localization_out": str(args.localization_out),
        "qwen_feta_ckpt": str(args.qwen_feta_ckpt),
        "prompt_ckpt": str(args.prompt_ckpt),
        "run_sam2": bool(args.run_sam2),
        "sam_config": args.sam_config,
        "sam_ckpt": args.sam_ckpt if args.run_sam2 else "",
        "prompt_thr": args.prompt_thr,
        "mask_alpha": args.mask_alpha,
        "mask_beta": args.mask_beta,
        "rows": len(samples),
        "compatible_rows": len(compatible_rows),
        "localization_rows": len(localization_rows),
        "counters": dict(counters),
        "provider_answers": dict(Counter(r["evidence_answer"] for r in compatible_rows)),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    fatal_missing = counters["missing_h5_sid"] + counters["missing_image"]
    if counters["missing_locked_evidence"] and not args.allow_missing_locked_evidence:
        fatal_missing += counters["missing_locked_evidence"]
    if args.allow_missing_locked_evidence and len(compatible_rows) != len(locked):
        raise SystemExit(
            "incomplete asset export after allowing missing locked evidence: "
            f"compatible_rows={len(compatible_rows)} locked_rows={len(locked)} counters={dict(counters)}"
        )
    if fatal_missing:
        raise SystemExit(f"incomplete asset export: {dict(counters)}")


if __name__ == "__main__":
    main()
