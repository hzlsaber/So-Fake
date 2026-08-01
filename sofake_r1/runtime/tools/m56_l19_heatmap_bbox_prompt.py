#!/usr/bin/env python3
"""L19 heatmap + bbox prompt head for SAM2 refinement.

This is a controlled localization experiment:

  frozen Qwen L19 tokens
      -> lightweight shared localization neck
      -> heatmap head, supervised by patch mask
      -> bbox head, supervised by pixel GT bbox

The pixel GT bbox is always computed in image space by resizing `mask_path` to
the corresponding `image_path` size first. This avoids using stale patch-space
boxes when source image/mask sizes do not match.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

_sam2_root = os.environ.get("SOFAKE_R1_SAM2_ROOT", "")
if _sam2_root:
    sys.path.insert(0, _sam2_root)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None

from m4_feta_v0 import FETADataset, collate
from m44_l19_tamper_mask_branch import bbox_from_binary, bbox_iou


def safe_div(a, b):
    return float(a / b) if b else 0.0


def pixel_gt_mask(mask_path, image_size):
    m = Image.open(mask_path).convert("RGB")
    if m.size != tuple(image_size):
        m = m.resize(tuple(image_size), Image.Resampling.NEAREST)
    return np.asarray(m).max(axis=-1) > 0


def pixel_gt_bbox(image_path, mask_path):
    with Image.open(image_path) as image:
        image_size = image.size
    gt = pixel_gt_mask(mask_path, image_size)
    box = bbox_from_binary(gt)
    if box is None:
        return None, image_size, gt
    return box, image_size, gt


def compute_bbox_record(task):
    sid, image_path, mask_path = task
    if not image_path or not mask_path or not Path(image_path).exists() or not Path(mask_path).exists():
        return {"sid": sid, "skip": "missing_path"}
    try:
        box, image_size, _ = pixel_gt_bbox(image_path, mask_path)
        if box is None:
            return {"sid": sid, "skip": "empty_mask"}
        return {
            "sid": sid,
            "image_path": image_path,
            "mask_path": mask_path,
            "image_w": int(image_size[0]),
            "image_h": int(image_size[1]),
            "bbox": [int(x) for x in box],
        }
    except Exception as exc:
        return {"sid": sid, "skip": f"error:{type(exc).__name__}:{exc}"}


def bbox_cache_file(h5_path, cls_filter, cache_dir):
    cls_tag = "all" if cls_filter is None else "_".join(str(int(x)) for x in sorted(cls_filter))
    return Path(cache_dir) / f"{Path(h5_path).stem}_cls{cls_tag}_pixel_bbox_v1.jsonl"


def read_bbox_cache(path):
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "skip" not in rec:
                records[str(rec["sid"])] = rec
    return records


def write_bbox_cache(path, candidates, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_sid = {str(r["sid"]): r for r in records if "skip" not in r}
    with path.open("w", encoding="utf-8") as f:
        for sid, _, _ in candidates:
            rec = by_sid.get(str(sid))
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_bbox_candidates(h5_path, cls_filter, limit_for_uncached=0):
    cls_filter = None if cls_filter is None else {int(x) for x in cls_filter}
    candidates = []
    with h5py.File(h5_path, "r") as f:
        for sid in sorted(k for k in f.keys() if k != "meta"):
            g = f[sid]
            gt_cls = int(g.attrs["gt_cls"])
            if cls_filter is not None and gt_cls not in cls_filter:
                continue
            image_path = str(g.attrs.get("image_path", ""))
            mask_path = str(g.attrs.get("mask_path", ""))
            candidates.append((str(sid), image_path, mask_path))
            if limit_for_uncached and len(candidates) >= int(limit_for_uncached):
                break
    return candidates


def build_bbox_records(h5_path, cls_filter, limit, cache_dir, cache_workers):
    cache_path = bbox_cache_file(h5_path, cls_filter, cache_dir)
    full_records = read_bbox_cache(cache_path)
    if full_records:
        candidates = collect_bbox_candidates(h5_path, cls_filter, limit_for_uncached=0)
        return candidates, full_records, {"loaded_cache": str(cache_path), "n_cached": len(full_records)}

    # For small smoke tests, avoid accidentally building the full cache.
    uncached_limit = int(limit) if limit and limit > 0 else 0
    candidates = collect_bbox_candidates(h5_path, cls_filter, limit_for_uncached=uncached_limit)
    print(
        f"[bbox cache] build {len(candidates)} records for {Path(h5_path).name} "
        f"(cache={'disabled_for_limit' if uncached_limit else cache_path})",
        flush=True,
    )
    records = []
    skips = defaultdict(int)
    if cache_workers and cache_workers > 1:
        with Pool(processes=int(cache_workers)) as pool:
            for idx, rec in enumerate(pool.imap_unordered(compute_bbox_record, candidates, chunksize=32), 1):
                if "skip" in rec:
                    skips[rec["skip"]] += 1
                records.append(rec)
                if idx % 1000 == 0:
                    print(f"  bbox processed {idx}/{len(candidates)}", flush=True)
    else:
        for idx, task in enumerate(candidates, 1):
            rec = compute_bbox_record(task)
            if "skip" in rec:
                skips[rec["skip"]] += 1
            records.append(rec)
            if idx % 1000 == 0:
                print(f"  bbox processed {idx}/{len(candidates)}", flush=True)

    if not uncached_limit:
        write_bbox_cache(cache_path, candidates, records)
        cache_status = str(cache_path)
    else:
        cache_status = "not_written_for_limited_dataset"
    good = {str(r["sid"]): r for r in records if "skip" not in r}
    return candidates, good, {
        "built_cache": cache_status,
        "n_candidates": len(candidates),
        "n_good": len(good),
        "skips": dict(skips),
    }


def box_to_norm(box, image_size):
    iw, ih = image_size
    x1, y1, x2, y2 = box
    return np.asarray([x1 / iw, y1 / ih, x2 / iw, y2 / ih], dtype=np.float32)


def norm_box_to_pixel(box, image_size):
    iw, ih = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    out = [
        int(np.floor(x1 * iw)),
        int(np.floor(y1 * ih)),
        int(np.ceil(x2 * iw)),
        int(np.ceil(y2 * ih)),
    ]
    out[0] = max(0, min(iw - 1, out[0]))
    out[1] = max(0, min(ih - 1, out[1]))
    out[2] = max(out[0] + 1, min(iw, out[2]))
    out[3] = max(out[1] + 1, min(ih, out[3]))
    return tuple(out)


def grid_bbox_to_image(bbox, grid_hw, image_size, pad=0):
    if bbox is None:
        return None
    h, w = grid_hw
    iw, ih = image_size
    x1, y1, x2, y2 = bbox
    box = [
        int(np.floor(x1 * iw / w)) - pad,
        int(np.floor(y1 * ih / h)) - pad,
        int(np.ceil(x2 * iw / w)) + pad,
        int(np.ceil(y2 * ih / h)) + pad,
    ]
    box[0] = max(0, min(iw - 1, box[0]))
    box[1] = max(0, min(ih - 1, box[1]))
    box[2] = max(box[0] + 1, min(iw, box[2]))
    box[3] = max(box[1] + 1, min(ih, box[3]))
    return tuple(box)


def binary_metrics(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    inter = tp
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / union) if union > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return iou, prec, rec, f1


class BBoxFETADataset(FETADataset):
    def __init__(self, h5_path, layer, unpermute=False, cls_filter=None, limit=0,
                 cache_dir=".cache/l19_bbox", cache_workers=8):
        super().__init__(h5_path, layer, unpermute=unpermute)
        cls_filter = None if cls_filter is None else {int(x) for x in cls_filter}
        self.bbox_norm = {}
        self.image_paths = {}
        self.mask_paths = {}
        self.image_sizes = {}
        self.gt_boxes = {}

        candidates, records, status = build_bbox_records(
            h5_path,
            cls_filter,
            limit=limit,
            cache_dir=cache_dir,
            cache_workers=cache_workers,
        )
        keep = []
        for sid, _, _ in candidates:
            rec = records.get(str(sid))
            if rec is None:
                continue
            keep.append(sid)
            image_size = (int(rec["image_w"]), int(rec["image_h"]))
            box = tuple(int(x) for x in rec["bbox"])
            self.bbox_norm[sid] = box_to_norm(box, image_size)
            self.image_paths[sid] = rec["image_path"]
            self.mask_paths[sid] = rec["mask_path"]
            self.image_sizes[sid] = image_size
            self.gt_boxes[sid] = box
            if limit and limit > 0 and len(keep) >= int(limit):
                break
        self.sids = keep
        self.drop_stats = status

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sid = item["sid"]
        item["bbox_norm"] = torch.from_numpy(self.bbox_norm[sid]).float()
        item["image_path"] = self.image_paths[sid]
        item["mask_path"] = self.mask_paths[sid]
        item["image_size"] = self.image_sizes[sid]
        item["gt_box"] = self.gt_boxes[sid]
        return item


def collate_bbox(batch):
    out = collate(batch)
    out["bbox_norm"] = torch.stack([b["bbox_norm"] for b in batch], dim=0)
    out["image_paths"] = [b["image_path"] for b in batch]
    out["mask_paths"] = [b["mask_path"] for b in batch]
    out["image_sizes"] = [b["image_size"] for b in batch]
    out["gt_boxes"] = [b["gt_box"] for b in batch]
    return out


class L19HeatmapBBoxHead(nn.Module):
    def __init__(self, in_dim=1280, hidden=128, depth=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        blocks = []
        for _ in range(depth):
            blocks.extend([
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            ])
        self.blocks = nn.Sequential(*blocks)
        self.heatmap = nn.Conv2d(hidden, 1, kernel_size=1)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, feat, Ts, Hs, Ws):
        x = self.proj(self.norm(feat))
        B, _, D = x.shape
        H_max = int(Hs.max().item())
        W_max = int(Ws.max().item())
        T_max = int(Ts.max().item())
        assert T_max == 1, f"multi-frame not supported (T={T_max})"

        grid = x.new_zeros(B, D, H_max, W_max)
        attn_grid = x.new_zeros(B, H_max, W_max)
        for i in range(B):
            h, w = int(Hs[i].item()), int(Ws[i].item())
            grid[i, :, :h, :w] = x[i, : h * w].view(h, w, D).permute(2, 0, 1)
            attn_grid[i, :h, :w] = 1.0

        z = self.blocks(grid)
        heat_logits = self.heatmap(z).squeeze(1)
        prob = torch.sigmoid(heat_logits) * attn_grid

        valid = attn_grid.unsqueeze(1)
        denom = valid.sum(dim=(-1, -2)).clamp_min(1.0)
        avg_pool = (z * valid).sum(dim=(-1, -2)) / denom
        z_masked = z.masked_fill(valid == 0, torch.finfo(z.dtype).min)
        max_pool = z_masked.amax(dim=(-1, -2))

        raw = self.bbox_mlp(torch.cat([avg_pool, max_pool], dim=-1))
        cx = torch.sigmoid(raw[:, 0])
        cy = torch.sigmoid(raw[:, 1])
        bw = torch.sigmoid(raw[:, 2])
        bh = torch.sigmoid(raw[:, 3])
        x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
        bbox = torch.stack([x1, y1, x2, y2], dim=-1)
        return {
            "heat_logits": heat_logits,
            "prob": prob,
            "attn_grid": attn_grid,
            "bbox": bbox,
        }


def build_mask_grid(batch, like_2d, device):
    B, H_max, W_max = like_2d.shape
    target = torch.zeros(B, H_max, W_max, device=device, dtype=like_2d.dtype)
    pm = batch["patch_mask"].to(device)
    for i in range(B):
        h, w = int(batch["H"][i].item()), int(batch["W"][i].item())
        target[i, :h, :w] = pm[i, : h * w].view(h, w)
    return target


def generalized_iou_loss(pred, target):
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target.unbind(dim=-1)
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp_min(0.0) * (iy2 - iy1).clamp_min(0.0)
    p_area = (px2 - px1).clamp_min(0.0) * (py2 - py1).clamp_min(0.0)
    t_area = (tx2 - tx1).clamp_min(0.0) * (ty2 - ty1).clamp_min(0.0)
    union = p_area + t_area - inter
    iou = inter / union.clamp_min(1e-6)
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c_area = (cx2 - cx1).clamp_min(0.0) * (cy2 - cy1).clamp_min(0.0)
    giou = iou - (c_area - union) / c_area.clamp_min(1e-6)
    return 1.0 - giou


def heatmap_bbox_loss(out, batch, args):
    device = out["heat_logits"].device
    valid = out["attn_grid"]
    target = build_mask_grid(batch, out["heat_logits"], device)
    heat_logits = out["heat_logits"]
    bce_all = F.binary_cross_entropy_with_logits(heat_logits, target, reduction="none")
    bce = (bce_all * valid).sum() / valid.sum().clamp_min(1.0)
    prob = torch.sigmoid(heat_logits)
    inter = (prob * target * valid).sum(dim=(-1, -2))
    denom = (prob * valid).sum(dim=(-1, -2)) + (target * valid).sum(dim=(-1, -2))
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6))
    dice = dice.mean()

    target_box = batch["bbox_norm"].to(device)
    l1 = F.l1_loss(out["bbox"], target_box)
    giou = generalized_iou_loss(out["bbox"], target_box).mean()
    loss = args.heatmap_weight * (bce + dice) + args.bbox_weight * (l1 + giou)
    return loss, {
        "bce": float(bce.item()),
        "dice": float(dice.item()),
        "bbox_l1": float(l1.item()),
        "bbox_giou": float(giou.item()),
    }


def mean_dict(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "ge_0.5": float((vals >= 0.5).mean()),
    }


@torch.no_grad()
def evaluate_prompt_rows(model, ds, args, split):
    model.eval()
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_bbox)
    rows = []
    for batch in loader:
        out = model(batch["feat"].to(args.device), batch["T"], batch["H"], batch["W"])
        probs = out["prob"].cpu().numpy()
        pred_boxes = out["bbox"].cpu().numpy()
        for i, sid in enumerate(batch["sids"]):
            image_size = batch["image_sizes"][i]
            gt_box = tuple(batch["gt_boxes"][i])
            h, w = int(batch["H"][i]), int(batch["W"][i])
            prob = probs[i, :h, :w]
            pred_box = norm_box_to_pixel(pred_boxes[i], image_size)
            row = {
                "split": split,
                "sid": str(sid),
                "source": batch["sources"][i],
                "image_path": batch["image_paths"][i],
                "mask_path": batch["mask_paths"][i],
                "image_w": int(image_size[0]),
                "image_h": int(image_size[1]),
                "gt_box": ",".join(str(x) for x in gt_box),
                "pred_bbox_head": ",".join(str(x) for x in pred_box),
                "bbox_head_iou": bbox_iou(pred_box, gt_box),
                "heat_peak": float(prob.max()) if prob.size else 0.0,
            }
            for thr in args.thresholds:
                heat_box = grid_bbox_to_image(bbox_from_binary(prob >= float(thr)), prob.shape, image_size,
                                              pad=args.box_pad)
                row[f"heat_bbox@{thr:.2f}"] = "" if heat_box is None else ",".join(str(x) for x in heat_box)
                row[f"heat_bbox_iou@{thr:.2f}"] = bbox_iou(heat_box, gt_box)
            rows.append(row)
    summary = {
        "n": len(rows),
        "bbox_head_iou": mean_dict(rows, "bbox_head_iou"),
    }
    for thr in args.thresholds:
        summary[f"heat_bbox_iou@{thr:.2f}"] = mean_dict(rows, f"heat_bbox_iou@{thr:.2f}")
    return rows, summary


def select_sam_mask(masks, scores):
    idx = int(np.argmax(scores))
    return np.asarray(masks[idx]) > 0, float(scores[idx])


def run_sam_predict(predictor, image_np, *, box=None, point=None, mask_input=None, multimask=True):
    predictor.set_image(image_np)
    point_coords = None
    point_labels = None
    if point is not None:
        point_coords = point.astype(np.float32)
        point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None if box is None else np.asarray(box, dtype=np.float32),
        mask_input=mask_input,
        multimask_output=multimask,
        return_logits=False,
    )
    return select_sam_mask(masks, scores)


def top_point_from_heatmap(prob_grid, image_size):
    h, w = prob_grid.shape
    y, x = np.unravel_index(int(np.argmax(prob_grid)), prob_grid.shape)
    iw, ih = image_size
    return np.asarray([[((x + 0.5) * iw / w), ((y + 0.5) * ih / h)]], dtype=np.float32)


def heatmap_to_sam_mask_input(prob_grid, mode="logit", scale=1.0):
    arr = np.asarray(prob_grid, dtype=np.float32)
    img = Image.fromarray(arr, mode="F").resize((256, 256), Image.Resampling.BILINEAR)
    p = np.asarray(img, dtype=np.float32)
    if mode == "prob":
        out = p
    elif mode == "centered_prob":
        out = (p - 0.5) * 2.0
    elif mode == "logit":
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        out = np.log(p / (1.0 - p))
    else:
        raise ValueError(f"unknown mask_input_mode: {mode}")
    out = np.clip(out * float(scale), -32.0, 32.0).astype(np.float32)
    return out[None, :, :]


def parse_box(s):
    if not s:
        return None
    return tuple(int(x) for x in str(s).split(","))


@torch.no_grad()
def run_sam2_eval(model, ds, rows, args, out_dir):
    if build_sam2 is None or SAM2ImagePredictor is None:
        raise RuntimeError(
            "SAM2 is required for --run_sam2. Install sam2 or set SOFAKE_R1_SAM2_ROOT."
        )
    print("[load SAM2]", flush=True)
    sam_model = build_sam2(args.sam_config, args.sam_ckpt, device=args.device,
                           apply_postprocessing=not args.no_postprocess)
    predictor = SAM2ImagePredictor(sam_model)

    # Recompute heatmaps only for points; keep bbox rows from evaluate_prompt_rows.
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_bbox)
    by_sid = {r["sid"]: r for r in rows}
    done = 0
    for batch in loader:
        out = model(batch["feat"].to(args.device), batch["T"], batch["H"], batch["W"])
        probs = out["prob"].cpu().numpy()
        for i, sid in enumerate(batch["sids"]):
            sid = str(sid)
            row = by_sid[sid]
            image = Image.open(row["image_path"]).convert("RGB")
            image_np = np.asarray(image)
            gt = pixel_gt_mask(row["mask_path"], image.size)
            gt_box = parse_box(row["gt_box"])
            bbox_head = parse_box(row["pred_bbox_head"])
            h, w = int(batch["H"][i]), int(batch["W"][i])
            prob = probs[i, :h, :w]
            point = top_point_from_heatmap(prob, image.size)
            mask_input = heatmap_to_sam_mask_input(
                prob,
                mode=args.mask_input_mode,
                scale=args.mask_input_scale,
            )

            mask_b, score_b = run_sam_predict(
                predictor, image_np, box=bbox_head, point=None, multimask=args.multimask
            )
            iou, _, _, f1 = binary_metrics(mask_b, gt)
            row["sam_bbox_head_iou"] = iou
            row["sam_bbox_head_f1"] = f1
            row["sam_bbox_head_score"] = score_b

            mask_bp, score_bp = run_sam_predict(
                predictor, image_np, box=bbox_head, point=point, multimask=args.multimask
            )
            iou, _, _, f1 = binary_metrics(mask_bp, gt)
            row["sam_bbox_head_point_iou"] = iou
            row["sam_bbox_head_point_f1"] = f1
            row["sam_bbox_head_point_score"] = score_bp

            mask_m, score_m = run_sam_predict(
                predictor,
                image_np,
                box=None,
                point=None,
                mask_input=mask_input,
                multimask=args.multimask,
            )
            iou, _, _, f1 = binary_metrics(mask_m, gt)
            row["sam_mask_input_iou"] = iou
            row["sam_mask_input_f1"] = f1
            row["sam_mask_input_score"] = score_m

            mask_bm, score_bm = run_sam_predict(
                predictor,
                image_np,
                box=bbox_head,
                point=None,
                mask_input=mask_input,
                multimask=args.multimask,
            )
            iou, _, _, f1 = binary_metrics(mask_bm, gt)
            row["sam_bbox_head_mask_iou"] = iou
            row["sam_bbox_head_mask_f1"] = f1
            row["sam_bbox_head_mask_score"] = score_bm

            mask_bmp, score_bmp = run_sam_predict(
                predictor,
                image_np,
                box=bbox_head,
                point=point,
                mask_input=mask_input,
                multimask=args.multimask,
            )
            iou, _, _, f1 = binary_metrics(mask_bmp, gt)
            row["sam_bbox_head_mask_point_iou"] = iou
            row["sam_bbox_head_mask_point_f1"] = f1
            row["sam_bbox_head_mask_point_score"] = score_bmp

            for thr in args.thresholds:
                heat_box = parse_box(row[f"heat_bbox@{thr:.2f}"])
                if heat_box is None:
                    row[f"sam_heat_bbox@{thr:.2f}_iou"] = 0.0
                    row[f"sam_heat_bbox_point@{thr:.2f}_iou"] = 0.0
                    row[f"sam_heat_bbox_mask@{thr:.2f}_iou"] = 0.0
                    row[f"sam_heat_bbox_mask_point@{thr:.2f}_iou"] = 0.0
                    continue
                mask_hb, _ = run_sam_predict(
                    predictor, image_np, box=heat_box, point=None, multimask=args.multimask
                )
                iou, _, _, _ = binary_metrics(mask_hb, gt)
                row[f"sam_heat_bbox@{thr:.2f}_iou"] = iou
                mask_hbp, _ = run_sam_predict(
                    predictor, image_np, box=heat_box, point=point, multimask=args.multimask
                )
                iou, _, _, _ = binary_metrics(mask_hbp, gt)
                row[f"sam_heat_bbox_point@{thr:.2f}_iou"] = iou

                mask_hbm, _ = run_sam_predict(
                    predictor,
                    image_np,
                    box=heat_box,
                    point=None,
                    mask_input=mask_input,
                    multimask=args.multimask,
                )
                iou, _, _, _ = binary_metrics(mask_hbm, gt)
                row[f"sam_heat_bbox_mask@{thr:.2f}_iou"] = iou

                mask_hbmp, _ = run_sam_predict(
                    predictor,
                    image_np,
                    box=heat_box,
                    point=point,
                    mask_input=mask_input,
                    multimask=args.multimask,
                )
                iou, _, _, _ = binary_metrics(mask_hbmp, gt)
                row[f"sam_heat_bbox_mask_point@{thr:.2f}_iou"] = iou

            mask_oracle, score_oracle = run_sam_predict(
                predictor, image_np, box=gt_box, point=None, multimask=args.multimask
            )
            iou, _, _, f1 = binary_metrics(mask_oracle, gt)
            row["sam_oracle_bbox_iou"] = iou
            row["sam_oracle_bbox_f1"] = f1
            row["sam_oracle_bbox_score"] = score_oracle

            done += 1
            if done % 10 == 0:
                print(f"  SAM2 processed {done}/{len(rows)}", flush=True)

    keys = [
        "sam_bbox_head_iou",
        "sam_bbox_head_point_iou",
        "sam_mask_input_iou",
        "sam_bbox_head_mask_iou",
        "sam_bbox_head_mask_point_iou",
        "sam_oracle_bbox_iou",
    ]
    for thr in args.thresholds:
        keys.extend([
            f"sam_heat_bbox@{thr:.2f}_iou",
            f"sam_heat_bbox_point@{thr:.2f}_iou",
            f"sam_heat_bbox_mask@{thr:.2f}_iou",
            f"sam_heat_bbox_mask_point@{thr:.2f}_iou",
        ])
    summary = {"n": len(rows)}
    for k in keys:
        summary[k] = mean_dict(rows, k)
    with (out_dir / "sam2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def train_model(model, train_ds, val_ds, args):
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_bbox, drop_last=False)
    best_state = None
    best_score = -1.0
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sums = defaultdict(float)
        n = 0
        for batch in loader:
            out = model(batch["feat"].to(args.device), batch["T"], batch["H"], batch["W"])
            loss, parts = heatmap_bbox_loss(out, batch, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(batch["gt_cls"].numel())
            n += bs
            sums["loss"] += float(loss.item()) * bs
            for k, v in parts.items():
                sums[k] += v * bs
        sched.step()

        _, val_summary = evaluate_prompt_rows(model, val_ds, args, "val_in")
        score = val_summary["bbox_head_iou"]["mean"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rec = {
            "epoch": ep + 1,
            "train": {k: safe_div(v, n) for k, v in sums.items()},
            "val_in_bbox_head_iou_mean": score,
            "val_in_bbox_head_iou_median": val_summary["bbox_head_iou"]["median"],
            "val_in_heat_bbox_iou@0.10_mean": val_summary["heat_bbox_iou@0.10"]["mean"],
        }
        history.append(rec)
        print(
            f"  ep {ep+1}/{args.epochs} loss={rec['train']['loss']:.4f} "
            f"bce={rec['train']['bce']:.4f} dice={rec['train']['dice']:.4f} "
            f"l1={rec['train']['bbox_l1']:.4f} giou={rec['train']['bbox_giou']:.4f} "
            f"val_bbox={score:.3f} val_heat01={rec['val_in_heat_bbox_iou@0.10_mean']:.3f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def write_rows(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== L19 heatmap + bbox prompt head ===", flush=True)
    print("GT bbox: mask_path resized to image_path size, then pixel bbox -> normalized image bbox", flush=True)
    train_ds = BBoxFETADataset(args.train_h5, args.layer, unpermute=bool(args.unpermute),
                               cls_filter=[2], limit=args.train_limit,
                               cache_dir=args.bbox_cache_dir, cache_workers=args.bbox_cache_workers)
    val_in_ds = BBoxFETADataset(args.val_in_h5, args.layer, unpermute=bool(args.unpermute),
                                cls_filter=[2], limit=args.eval_limit,
                                cache_dir=args.bbox_cache_dir, cache_workers=args.bbox_cache_workers)
    val_ood_ds = BBoxFETADataset(args.val_ood_h5, args.layer, unpermute=bool(args.unpermute),
                                 cls_filter=[2], limit=args.eval_limit,
                                 cache_dir=args.bbox_cache_dir, cache_workers=args.bbox_cache_workers)
    print(
        f"datasets: train={len(train_ds)} val_in={len(val_in_ds)} val_ood={len(val_ood_ds)} "
        f"drops train={train_ds.drop_stats} val_ood={val_ood_ds.drop_stats}",
        flush=True,
    )

    model = L19HeatmapBBoxHead(in_dim=1280, hidden=args.hidden, depth=args.depth)
    ckpt = out_dir / "l19_heatmap_bbox_head_best.pt"
    if args.load_ckpt and Path(args.load_ckpt).exists() and not args.retrain:
        model.load_state_dict(torch.load(args.load_ckpt, map_location="cpu"), strict=True)
        history = [{"loaded_from": args.load_ckpt}]
    else:
        model, history = train_model(model, train_ds, val_in_ds, args)
        torch.save(model.state_dict(), ckpt)

    model.to(args.device)
    model.eval()
    prompt_results = {}
    all_rows = {}
    for split, ds in [("val_in", val_in_ds), ("val_ood", val_ood_ds)]:
        rows, summary = evaluate_prompt_rows(model, ds, args, split)
        prompt_results[split] = summary
        all_rows[split] = rows
        write_rows(rows, out_dir / f"{split}_prompt_rows.csv")
        print(
            f"{split}: bbox_head={summary['bbox_head_iou']['mean']:.3f}/"
            f"{summary['bbox_head_iou']['median']:.3f} "
            f"heat01={summary['heat_bbox_iou@0.10']['mean']:.3f}/"
            f"{summary['heat_bbox_iou@0.10']['median']:.3f}",
            flush=True,
        )

    sam2_summary = None
    if args.run_sam2:
        sam2_summary = run_sam2_eval(model, val_ood_ds, all_rows["val_ood"], args, out_dir)
        print(
            "SAM2 val_ood: "
            f"bbox_head={sam2_summary['sam_bbox_head_iou']['mean']:.3f}/"
            f"{sam2_summary['sam_bbox_head_iou']['median']:.3f} "
            f"bbox_head+point={sam2_summary['sam_bbox_head_point_iou']['mean']:.3f}/"
            f"{sam2_summary['sam_bbox_head_point_iou']['median']:.3f} "
            f"bbox_head+mask+point={sam2_summary['sam_bbox_head_mask_point_iou']['mean']:.3f}/"
            f"{sam2_summary['sam_bbox_head_mask_point_iou']['median']:.3f} "
            f"heat01+point={sam2_summary['sam_heat_bbox_point@0.10_iou']['mean']:.3f}/"
            f"{sam2_summary['sam_heat_bbox_point@0.10_iou']['median']:.3f} "
            f"heat01+mask+point={sam2_summary['sam_heat_bbox_mask_point@0.10_iou']['mean']:.3f}/"
            f"{sam2_summary['sam_heat_bbox_mask_point@0.10_iou']['median']:.3f}",
            flush=True,
        )
        write_rows(all_rows["val_ood"], out_dir / "val_ood_sam2_rows.csv")

    summary = {
        "setting": {
            "layer": args.layer,
            "unpermute": bool(args.unpermute),
            "hidden": args.hidden,
            "depth": args.depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "heatmap_weight": args.heatmap_weight,
            "bbox_weight": args.bbox_weight,
            "gt_bbox_definition": "resize mask_path to image_path size, compute pixel bbox, normalize by image size",
        },
        "train_history": history,
        "prompt_bbox": prompt_results,
        "sam2_val_ood": sam2_summary,
        "ckpt": str(ckpt if not args.load_ckpt else args.load_ckpt),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"OUT_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", default="")
    ap.add_argument("--val_in_h5", default="")
    ap.add_argument("--val_ood_h5", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--unpermute", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heatmap_weight", type=float, default=1.0)
    ap.add_argument("--bbox_weight", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.5])
    ap.add_argument("--box_pad", type=int, default=0)
    ap.add_argument("--bbox_cache_dir", default=os.environ.get("SOFAKE_R1_BBOX_CACHE_DIR", ".cache/l19_bbox"))
    ap.add_argument("--bbox_cache_workers", type=int, default=16)
    ap.add_argument("--run_sam2", action="store_true")
    ap.add_argument("--sam_config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam_ckpt", default=os.environ.get("SOFAKE_R1_SAM2_CKPT", ""))
    ap.add_argument("--multimask", type=int, default=1)
    ap.add_argument("--no_postprocess", action="store_true")
    ap.add_argument("--mask_input_mode", choices=["logit", "prob", "centered_prob"], default="logit")
    ap.add_argument("--mask_input_scale", type=float, default=1.0)
    ap.add_argument("--load_ckpt", default="")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    args.unpermute = bool(args.unpermute)
    args.multimask = bool(args.multimask)
    main(args)
