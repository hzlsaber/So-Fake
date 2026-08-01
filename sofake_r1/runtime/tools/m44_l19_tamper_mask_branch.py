#!/usr/bin/env python3
"""L19-only tamper mask branch over frozen Qwen/FETA evidence provider.

This controlled experiment keeps the clean FETA classification path frozen and
trains only a lightweight dense mask head from cached Qwen layer-19 features.

Purpose:
  test whether the stable L19 frozen representation contains readable tamper
  localization information when explicit TAMPERED mask supervision is added.

Deliberately excluded by default:
  - REAL / FULL_SYNTHETIC zero-mask loss
  - residual/NPR cue
  - multi-layer/depth FPN
  - mask-to-classifier feedback
  - target-domain prior or Chameleon-tuned threshold

Optional controlled ablation:
  add weak REAL-only suppression, e.g. lambda * mean(M_real). FS remains ignored.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from m4_feta_v0 import FETADataset, FETAv0, collate


CLASS_NAMES = {0: "REAL", 1: "FS", 2: "TAMP"}


def safe_div(a, b):
    return float(a / b) if b else 0.0


class FilteredFETADataset(FETADataset):
    def __init__(self, h5_path, layer, unpermute=False, cls_filter=None, limit=0):
        super().__init__(h5_path, layer, unpermute=unpermute)
        if cls_filter is not None:
            cls_filter = {int(x) for x in cls_filter}
            with h5py.File(h5_path, "r") as f:
                self.sids = [sid for sid in self.sids if int(f[sid].attrs["gt_cls"]) in cls_filter]
        if limit and limit > 0:
            self.sids = self.sids[: int(limit)]


class L19TamperMaskHead(nn.Module):
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
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)

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
        logits = self.out(z).squeeze(1)
        prob = torch.sigmoid(logits) * attn_grid
        return {"logits": logits, "prob": prob, "attn_grid": attn_grid}


def build_mask_grid(batch, like_2d, device):
    B, H_max, W_max = like_2d.shape
    target = torch.zeros(B, H_max, W_max, device=device, dtype=like_2d.dtype)
    pm = batch["patch_mask"].to(device)
    for i in range(B):
        h, w = int(batch["H"][i].item()), int(batch["W"][i].item())
        target[i, :h, :w] = pm[i, : h * w].view(h, w)
    return target


def boundary_map(x, valid=None, kernel_size=3):
    if valid is not None:
        x = x * valid
    pad = kernel_size // 2
    x4 = x.unsqueeze(1)
    dil = F.max_pool2d(x4, kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)
    ero = -F.max_pool2d(-x4, kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)
    b = (dil - ero).clamp(0.0, 1.0)
    if valid is not None:
        b = b * valid
    return b


def mask_loss_controlled(out, batch, loss_kind="bce_dice", focal_gamma=2.0, focal_alpha=0.25,
                         real_neg_weight=0.0, boundary_weight=0.0, boundary_kernel=3):
    logits = out["logits"]
    valid = out["attn_grid"]
    device = logits.device
    cls = batch["gt_cls"].to(device)
    is_tamp = cls == 2
    is_real = cls == 0

    target = build_mask_grid(batch, logits, device)

    if bool(is_tamp.any()):
        logits_t = logits[is_tamp]
        target_t = target[is_tamp]
        valid_t = valid[is_tamp]

        bce_all = F.binary_cross_entropy_with_logits(logits_t, target_t, reduction="none")
        if loss_kind == "focal_dice":
            prob = torch.sigmoid(logits_t)
            pt = target_t * prob + (1.0 - target_t) * (1.0 - prob)
            alpha_t = target_t * focal_alpha + (1.0 - target_t) * (1.0 - focal_alpha)
            bce_all = alpha_t * ((1.0 - pt).clamp_min(0.0) ** focal_gamma) * bce_all
        elif loss_kind != "bce_dice":
            raise ValueError(f"unknown loss_kind: {loss_kind}")

        bce = (bce_all * valid_t).sum() / valid_t.sum().clamp_min(1.0)

        prob_t = torch.sigmoid(logits_t)
        inter = (prob_t * target_t * valid_t).sum(dim=(-1, -2))
        denom = (prob_t * valid_t).sum(dim=(-1, -2)) + (target_t * valid_t).sum(dim=(-1, -2))
        dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6))
        dice = dice.mean()
        tamp_loss = bce + dice

        if boundary_weight > 0:
            pred_boundary = boundary_map(prob_t, valid_t, kernel_size=int(boundary_kernel))
            target_boundary = boundary_map(target_t, valid_t, kernel_size=int(boundary_kernel))
            boundary_bce_all = F.binary_cross_entropy(
                pred_boundary.clamp(1e-6, 1.0 - 1e-6),
                target_boundary,
                reduction="none",
            )
            boundary_bce = (boundary_bce_all * valid_t).sum() / valid_t.sum().clamp_min(1.0)
            boundary_inter = (pred_boundary * target_boundary * valid_t).sum(dim=(-1, -2))
            boundary_denom = (
                (pred_boundary * valid_t).sum(dim=(-1, -2))
                + (target_boundary * valid_t).sum(dim=(-1, -2))
            )
            boundary_dice = 1.0 - ((2.0 * boundary_inter + 1e-6) / (boundary_denom + 1e-6))
            boundary = boundary_bce + boundary_dice.mean()
        else:
            boundary = logits.new_zeros(())
    else:
        bce = logits.new_zeros(())
        dice = logits.new_zeros(())
        boundary = logits.new_zeros(())
        tamp_loss = logits.new_zeros(())

    if real_neg_weight > 0 and bool(is_real.any()):
        prob_real = torch.sigmoid(logits[is_real])
        valid_real = valid[is_real]
        real_loss = (prob_real * valid_real).sum(dim=(-1, -2)) / valid_real.sum(dim=(-1, -2)).clamp_min(1.0)
        real_loss = real_loss.mean()
    else:
        real_loss = logits.new_zeros(())

    loss = tamp_loss + float(boundary_weight) * boundary + float(real_neg_weight) * real_loss
    return loss, {
        "bce": float(bce.item()),
        "dice": float(dice.item()),
        "boundary": float(boundary.item()),
        "real": float(real_loss.item()),
        "n_tamp": int(is_tamp.sum().item()),
        "n_real": int(is_real.sum().item()),
    }


def bbox_from_binary(mask):
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def binary_metrics(pred, y):
    cm = np.zeros((2, 2), dtype=np.int64)
    for g, p in zip(y, pred):
        cm[int(g), int(p)] += 1
    real_rec = safe_div(cm[0, 0], cm[0].sum())
    fake_rec = safe_div(cm[1, 1], cm[1].sum())
    real_prec = safe_div(cm[0, 0], cm[:, 0].sum())
    fake_prec = safe_div(cm[1, 1], cm[:, 1].sum())
    real_f1 = safe_div(2 * real_prec * real_rec, real_prec + real_rec)
    fake_f1 = safe_div(2 * fake_prec * fake_rec, fake_prec + fake_rec)
    return {
        "acc": safe_div(np.trace(cm), cm.sum()),
        "balanced_acc": (real_rec + fake_rec) / 2,
        "macro_f1": (real_f1 + fake_f1) / 2,
        "recall": [real_rec, fake_rec],
        "precision": [real_prec, fake_prec],
        "f1": [real_f1, fake_f1],
        "cm": cm.tolist(),
    }


def three_way_summary(pred, y):
    cm = np.zeros((3, 3), dtype=np.int64)
    for g, p in zip(y, pred):
        cm[int(g), int(p)] += 1
    return {
        "n": int(cm.sum()),
        "acc": safe_div(np.trace(cm), cm.sum()),
        "recall": [safe_div(cm[c, c], cm[c].sum()) for c in range(3)],
        "cm": cm.tolist(),
        "flip": {
            "TAMP->FS": safe_div(cm[2, 1], cm[2].sum()),
            "FS->TAMP": safe_div(cm[1, 2], cm[1].sum()),
            "REAL->TAMP": safe_div(cm[0, 2], cm[0].sum()),
        },
    }


@torch.no_grad()
def evaluate_cls(model, ds, device="cuda", batch_size=64, num_workers=4, chameleon=False):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    preds, gts = [], []
    for batch in loader:
        feat = batch["feat"].to(device)
        attn = batch["attn"].to(device)
        out = model(feat, attn, batch["T"], batch["H"], batch["W"])
        pred3 = out["logits"].argmax(dim=-1).cpu().numpy().astype(np.int64)
        y = batch["gt_cls"].numpy().astype(np.int64)
        if chameleon:
            preds.append((pred3 != 0).astype(np.int64))
            gts.append((y != 0).astype(np.int64))
        else:
            preds.append(pred3)
            gts.append(y)
    pred = np.concatenate(preds, axis=0)
    y = np.concatenate(gts, axis=0)
    return binary_metrics(pred, y) if chameleon else three_way_summary(pred, y)


def try_auc(scores, labels):
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(labels)) < 2:
            return {"roc_auc": None, "ap": None}
        return {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
        }
    except Exception:
        return {"roc_auc": None, "ap": None}


def class_activation_summary(values):
    out = {}
    for c, rows in values.items():
        out[CLASS_NAMES[c]] = {}
        if not rows:
            continue
        keys = rows[0].keys()
        for k in keys:
            arr = np.asarray([r[k] for r in rows], dtype=np.float64)
            out[CLASS_NAMES[c]][k] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
            }
    return out


@torch.no_grad()
def evaluate_mask(model, ds, device="cuda", batch_size=64, num_workers=4, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [float(t) for t in thresholds]
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    by_thr = {f"{t:.2f}": {"patch_iou": [], "bbox_iou": [], "pred_area": []} for t in thresholds}
    activation = defaultdict(list)
    pixel_scores, pixel_labels = [], []
    sample_max_scores, sample_has_tamp = [], []
    n_tamp = 0

    for batch in loader:
        feat = batch["feat"].to(device)
        out = model(feat, batch["T"], batch["H"], batch["W"])
        prob = out["prob"].cpu().numpy()
        cls = batch["gt_cls"].numpy().astype(np.int64)
        for i, cls_i in enumerate(cls.tolist()):
            h, w = int(batch["H"][i].item()), int(batch["W"][i].item())
            p = prob[i, :h, :w]
            g = batch["patch_mask"][i, : h * w].numpy().reshape(h, w)
            activation[int(cls_i)].append({
                "mean_prob": float(p.mean()),
                "peak_prob": float(p.max()) if p.size else 0.0,
                "area@0.5": float((p >= 0.5).mean()) if p.size else 0.0,
            })
            sample_max_scores.append(float(p.max()) if p.size else 0.0)
            sample_has_tamp.append(1 if cls_i == 2 else 0)

            if cls_i != 2:
                continue
            n_tamp += 1
            gt_bin = g >= 0.5
            gt_bbox = bbox_from_binary(gt_bin)
            pixel_scores.extend(p.flatten().astype(np.float64).tolist())
            pixel_labels.extend(gt_bin.flatten().astype(np.int64).tolist())

            for t in thresholds:
                key = f"{t:.2f}"
                pred_bin = p >= t
                inter = np.logical_and(pred_bin, gt_bin).sum()
                union = np.logical_or(pred_bin, gt_bin).sum()
                by_thr[key]["patch_iou"].append(float(inter / union) if union > 0 else 0.0)
                by_thr[key]["bbox_iou"].append(bbox_iou(bbox_from_binary(pred_bin), gt_bbox))
                by_thr[key]["pred_area"].append(float(pred_bin.mean()))

    threshold_metrics = {}
    for key, vals in by_thr.items():
        threshold_metrics[key] = {
            "patch_iou_mean": float(np.mean(vals["patch_iou"])) if vals["patch_iou"] else 0.0,
            "patch_iou_median": float(np.median(vals["patch_iou"])) if vals["patch_iou"] else 0.0,
            "bbox_iou_mean": float(np.mean(vals["bbox_iou"])) if vals["bbox_iou"] else 0.0,
            "bbox_iou_median": float(np.median(vals["bbox_iou"])) if vals["bbox_iou"] else 0.0,
            "pred_area_mean_on_tamp": float(np.mean(vals["pred_area"])) if vals["pred_area"] else 0.0,
        }
    best_thr = max(threshold_metrics.keys(), key=lambda k: threshold_metrics[k]["patch_iou_mean"])

    return {
        "n_tamp_eval": int(n_tamp),
        "best_threshold_by_patch_iou": best_thr,
        "thresholds": threshold_metrics,
        "activation_by_class": class_activation_summary(activation),
        "tamp_patch_auc": try_auc(pixel_scores, pixel_labels),
        "sample_tamp_auc_by_peak": try_auc(sample_max_scores, sample_has_tamp),
    }


def train_mask_head(model, train_ds, val_in_ds, args):
    device = args.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    best_state = None
    best_score = -1.0
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        sums = defaultdict(float)
        n_batches = 0
        n_samples = 0
        n_tamp = 0
        for batch in loader:
            feat = batch["feat"].to(device)
            out = model(feat, batch["T"], batch["H"], batch["W"])
            loss, parts = mask_loss_controlled(
                out,
                batch,
                loss_kind=args.loss_kind,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                real_neg_weight=args.real_neg_weight,
                boundary_weight=getattr(args, "boundary_weight", 0.0),
                boundary_kernel=getattr(args, "boundary_kernel", 3),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(batch["gt_cls"].numel())
            n_batches += 1
            n_samples += bs
            n_tamp += int(parts["n_tamp"])
            sums["n_real"] += int(parts["n_real"])
            sums["loss"] += float(loss.item()) * bs
            sums["bce"] += parts["bce"] * bs
            sums["dice"] += parts["dice"] * bs
            sums["boundary"] += parts["boundary"] * bs
            sums["real"] += parts["real"] * bs
        sched.step()

        val_mask = evaluate_mask(
            model,
            val_in_ds,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            thresholds=args.thresholds,
        )
        best_thr = val_mask["best_threshold_by_patch_iou"]
        score = val_mask["thresholds"][best_thr]["patch_iou_mean"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rec = {
            "epoch": ep + 1,
            "train": {
                "loss": safe_div(sums["loss"], n_samples),
                "bce": safe_div(sums["bce"], n_samples),
                "dice": safe_div(sums["dice"], n_samples),
                "boundary": safe_div(sums["boundary"], n_samples),
                "real": safe_div(sums["real"], n_samples),
                "n_samples": int(n_samples),
                "n_tamp": int(n_tamp),
                "n_real": int(sums["n_real"]),
                "n_batches": int(n_batches),
            },
            "val_in_best_thr": best_thr,
            "val_in_best_patch_iou": score,
            "val_in_best_bbox_iou": val_mask["thresholds"][best_thr]["bbox_iou_mean"],
            "val_in_thr05_patch_iou": val_mask["thresholds"].get("0.50", {}).get("patch_iou_mean", 0.0),
            "val_in_thr05_bbox_iou": val_mask["thresholds"].get("0.50", {}).get("bbox_iou_mean", 0.0),
        }
        history.append(rec)
        print(
            f"  ep {ep+1}/{args.epochs} "
            f"loss={rec['train']['loss']:.4f} bce={rec['train']['bce']:.4f} "
            f"dice={rec['train']['dice']:.4f} boundary={rec['train']['boundary']:.4f} "
            f"real={rec['train']['real']:.4f} "
            f"val_best_thr={best_thr} val_patch_iou={score:.3f} "
            f"val_bbox_iou={rec['val_in_best_bbox_iou']:.3f} "
            f"thr05_iou={rec['val_in_thr05_patch_iou']:.3f} ({time.time()-t0:.0f}s)",
            file=sys.stderr,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def load_frozen_feta(args):
    model = FETAv0(
        in_dim=1280,
        hidden=args.feta_hidden,
        K=args.feta_K,
        use_spatial_conv=bool(args.feta_spatial_conv),
        desc_set=args.feta_desc_set,
    )
    state = torch.load(args.feta_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== L19-only Tamper Mask Branch ===", file=sys.stderr)
    print(f"  trainable: mask head only", file=sys.stderr)
    print(f"  frozen FETA ckpt: {args.feta_ckpt}", file=sys.stderr)
    print(f"  unpermute(L0 window->raster) = {bool(args.unpermute)}", file=sys.stderr)
    print(f"  real_neg_weight = {args.real_neg_weight}", file=sys.stderr)

    train_classes = [2]
    if args.real_neg_weight > 0:
        train_classes = [0, 2]
    train_ds = FilteredFETADataset(
        args.train_h5,
        args.layer,
        unpermute=args.unpermute,
        cls_filter=train_classes,
        limit=args.train_limit,
    )
    val_in_ds = FilteredFETADataset(args.val_in_h5, args.layer, unpermute=args.unpermute, limit=args.eval_limit)
    val_ood_ds = FilteredFETADataset(args.val_ood_h5, args.layer, unpermute=args.unpermute, limit=args.eval_limit)
    print(
        f"  train_classes={train_classes} train={len(train_ds)} val_in={len(val_in_ds)} val_ood={len(val_ood_ds)}",
        file=sys.stderr,
    )

    feta = load_frozen_feta(args).to(device)
    mask_head = L19TamperMaskHead(in_dim=1280, hidden=args.hidden, depth=args.depth)

    cls_results = {}
    for split, ds in [("val_in", val_in_ds), ("val_ood", val_ood_ds)]:
        r = evaluate_cls(feta, ds, device=device, batch_size=args.eval_batch_size, num_workers=args.num_workers)
        cls_results[split] = r
        print(
            f"  frozen-cls {split}: acc={r['acc']:.3f} "
            f"R={r['recall'][0]:.3f} FS={r['recall'][1]:.3f} T={r['recall'][2]:.3f} "
            f"T->FS={r['flip']['TAMP->FS']:.3f}",
            file=sys.stderr,
        )

    if args.chameleon_h5:
        ch_parts = [
            FilteredFETADataset(p, args.layer, unpermute=args.unpermute, limit=args.chameleon_limit)
            for p in args.chameleon_h5
        ]
        ch_ds = ConcatDataset(ch_parts)
        r = evaluate_cls(
            feta,
            ch_ds,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            chameleon=True,
        )
        cls_results["chameleon"] = r
        print(
            f"  frozen-cls chameleon: acc={r['acc']:.3f} bal={r['balanced_acc']:.3f} "
            f"R={r['recall'][0]:.3f} F={r['recall'][1]:.3f}",
            file=sys.stderr,
        )

    mask_head, history = train_mask_head(mask_head, train_ds, val_in_ds, args)
    torch.save(mask_head.state_dict(), out_dir / "l19_tamper_mask_head_best.pt")

    mask_results = {}
    for split, ds in [("val_in", val_in_ds), ("val_ood", val_ood_ds)]:
        r = evaluate_mask(
            mask_head,
            ds,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            thresholds=args.thresholds,
        )
        mask_results[split] = r
        best_thr = r["best_threshold_by_patch_iou"]
        thr05 = r["thresholds"].get("0.50", {})
        print(
            f"  mask {split}: best_thr={best_thr} "
            f"best_iou={r['thresholds'][best_thr]['patch_iou_mean']:.3f} "
            f"best_bbox={r['thresholds'][best_thr]['bbox_iou_mean']:.3f} "
            f"thr05_iou={thr05.get('patch_iou_mean', 0.0):.3f} "
            f"thr05_bbox={thr05.get('bbox_iou_mean', 0.0):.3f}",
            file=sys.stderr,
        )
        act = r["activation_by_class"]
        for cname in ["REAL", "FS", "TAMP"]:
            if cname in act:
                print(
                    f"    act {split} {cname}: "
                    f"mean={act[cname]['mean_prob']['median']:.3f} "
                    f"peak={act[cname]['peak_prob']['median']:.3f} "
                    f"area05={act[cname]['area@0.5']['median']:.3f}",
                    file=sys.stderr,
                )

    source_thr = mask_results["val_in"]["best_threshold_by_patch_iou"]
    summary = {
        "setting": {
            "layer": args.layer,
            "unpermute": bool(args.unpermute),
            "frozen_feta_ckpt": args.feta_ckpt,
            "trainable": "mask_head_only",
            "mask_supervision": "TAMPERED_mask_plus_weak_REAL_suppression" if args.real_neg_weight > 0 else "TAMPERED_only",
            "train_classes": train_classes,
            "real_neg_weight": args.real_neg_weight,
            "boundary_weight": args.boundary_weight,
            "boundary_kernel": args.boundary_kernel,
            "loss_kind": args.loss_kind,
            "hidden": args.hidden,
            "depth": args.depth,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "classification_frozen": cls_results,
        "train_history": history,
        "mask": mask_results,
        "source_selected_threshold": source_thr,
        "ood_at_source_selected_threshold": mask_results["val_ood"]["thresholds"].get(source_thr),
    }
    out_json = out_dir / "l19_tamper_mask_branch_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_json}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", default="")
    ap.add_argument("--val_in_h5", default="")
    ap.add_argument("--val_ood_h5", default="")
    ap.add_argument("--chameleon_h5", nargs="*", default=[])
    ap.add_argument("--feta_ckpt", default=os.environ.get("SOFAKE_R1_QWEN_FETA_CKPT", ""))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--unpermute", type=int, default=1)
    ap.add_argument("--feta_hidden", type=int, default=128)
    ap.add_argument("--feta_K", type=int, default=2)
    ap.add_argument("--feta_spatial_conv", type=int, default=1)
    ap.add_argument("--feta_desc_set", default="minimal")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--loss_kind", choices=["bce_dice", "focal_dice"], default="bce_dice")
    ap.add_argument("--real_neg_weight", type=float, default=0.0,
                    help="Weak REAL-only suppression weight. 0 reproduces the "
                         "TAMPERED-only mask-head experiment; values >0 train "
                         "on REAL+TAMP and add weight * mean(sigmoid(mask_real)).")
    ap.add_argument("--boundary_weight", type=float, default=0.0,
                    help="Boundary-aware loss weight. 0 reproduces the original BCE+Dice mask head.")
    ap.add_argument("--boundary_kernel", type=int, default=3,
                    help="Kernel size for patch-grid morphological boundary target.")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.25)
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
    ap.add_argument("--chameleon_limit", type=int, default=0)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()
    args.unpermute = bool(args.unpermute)
    main(args)
