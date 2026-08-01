#!/usr/bin/env python3
"""NPR-residual FETA classification-only probe.

This is a controlled test of replacing Qwen L19 features with a fixed
NPR-like residual input:

  image -> x - nearest_down_up(x) -> small CNN -> K evidence maps
        -> 5 descriptors/head -> 3-class classifier

No NPR checkpoint is used. Chameleon is evaluation-only and is folded to
REAL-vs-FAKE from the model's 3-way prediction.
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["REAL", "FULL_SYNTHETIC", "TAMPERED"]
DESC_NAMES = ["mass", "entropy", "concentration", "peak", "area"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_div(a, b):
    return float(a / b) if b else 0.0


def read_jsonl(path, limit=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            image_path = r.get("image_path") or r.get("image")
            if not image_path:
                continue
            rows.append({
                "image_path": image_path,
                "gt_cls": int(r["gt_cls"]),
                "sid": str(r.get("sid") or r.get("sample_id") or len(rows)),
                "source": str(r.get("source", "unknown")),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def read_chameleon_roots(roots, limit=0):
    rows = []
    for root in roots:
        root = Path(root)
        for class_dir, gt in [("0_real", 0), ("1_fake", 1)]:
            d = root / class_dir
            if not d.exists():
                continue
            paths = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])
            for p in paths:
                rows.append({
                    "image_path": str(p),
                    "gt_cls": gt,
                    "sid": p.stem,
                    "source": "Chameleon",
                })
                if limit and len(rows) >= limit:
                    return rows
    return rows


class ImageJsonDataset(Dataset):
    def __init__(self, rows, image_size=224, train=False):
        self.rows = rows
        self.image_size = image_size
        tf = []
        tf.append(transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC))
        if train:
            tf.append(transforms.RandomHorizontalFlip(p=0.5))
        tf.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf = transforms.Compose(tf)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        x = self.tf(img)
        y = torch.tensor(int(row["gt_cls"]), dtype=torch.long)
        return x, y, row["sid"], row["image_path"], row["source"]


def compute_descriptors(E, eps=1e-8):
    """E: (B, K, H, W), returns (B, K, 5)."""
    B, K, H, W = E.shape
    n = H * W
    flat = E.flatten(2)
    mass = flat.sum(dim=-1)
    peak = flat.amax(dim=-1).clamp_min(0)
    area = torch.sigmoid(20.0 * (flat - 0.5)).sum(dim=-1) / n

    p = flat / mass.unsqueeze(-1).clamp_min(eps)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)

    k_top = max(1, n // 4)
    top_vals = torch.topk(flat, k_top, dim=-1).values
    concentration = top_vals.sum(dim=-1) / mass.clamp_min(eps)
    return torch.stack([mass, entropy, concentration, peak, area], dim=-1)


class NPRFETACls(nn.Module):
    def __init__(self, hidden=96, K=2, image_size=224):
        super().__init__()
        self.K = K
        self.image_size = image_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),   # 224 -> 112
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.evidence_heads = nn.ModuleList([nn.Conv2d(hidden, 1, 1) for _ in range(K)])
        self.classifier = nn.Sequential(
            nn.Linear(K * len(DESC_NAMES), 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    @staticmethod
    def npr_residual(x):
        downup = F.interpolate(
            F.interpolate(x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True),
            scale_factor=2.0,
            mode="nearest",
            recompute_scale_factor=True,
        )
        return (x - downup) * (2.0 / 3.0)

    def forward(self, x):
        z = self.stem(self.npr_residual(x))
        E = torch.cat([torch.sigmoid(head(z)) for head in self.evidence_heads], dim=1)
        desc = compute_descriptors(E)
        logits = self.classifier(desc.flatten(1))
        return {"logits": logits, "E": E, "descriptors": desc}


def summarize_three(y, pred, probs):
    cm = np.zeros((3, 3), dtype=np.int64)
    for g, p in zip(y, pred):
        if 0 <= int(g) < 3:
            cm[int(g), int(p)] += 1
    yb = (y != 0).astype(np.int64)
    score_fake = probs[:, 1] + probs[:, 2]
    pb = (score_fake >= 0.5).astype(np.int64)
    out = {
        "n": int(len(y)),
        "acc3": float(accuracy_score(y, pred)),
        "macro_f1_3": float(f1_score(y, pred, average="macro")),
        "cm3": cm.tolist(),
        "recall3": {CLASS_NAMES[c]: safe_div(cm[c, c], cm[c].sum()) for c in range(3)},
        "flip": {
            "TAMP->FS": safe_div(cm[2, 1], cm[2].sum()),
            "FS->TAMP": safe_div(cm[1, 2], cm[1].sum()),
            "REAL->TAMP": safe_div(cm[0, 2], cm[0].sum()),
        },
        "bin_acc": float(accuracy_score(yb, pb)),
        "bin_f1": float(f1_score(yb, pb)),
        "bin_ap": float(average_precision_score(yb, score_fake)) if len(np.unique(yb)) > 1 else 0.0,
        "bin_auroc": float(roc_auc_score(yb, score_fake)) if len(np.unique(yb)) > 1 else 0.0,
        "binary_recall_real": float(((pb == 0) & (yb == 0)).sum() / max(1, (yb == 0).sum())),
        "binary_recall_fake": float(((pb == 1) & (yb == 1)).sum() / max(1, (yb == 1).sum())),
    }
    return out


def summarize_chameleon(y, pred3, probs):
    yb = y.astype(np.int64)
    score_fake = probs[:, 1] + probs[:, 2]
    pb = (pred3 != 0).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for g, p in zip(yb, pb):
        cm[int(g), int(p)] += 1
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(yb, pb)),
        "balanced_acc": float(0.5 * (safe_div(cm[0, 0], cm[0].sum()) + safe_div(cm[1, 1], cm[1].sum()))),
        "macro_f1": float(f1_score(yb, pb, average="macro")),
        "fake_ap": float(average_precision_score(yb, score_fake)) if len(np.unique(yb)) > 1 else 0.0,
        "fake_auroc": float(roc_auc_score(yb, score_fake)) if len(np.unique(yb)) > 1 else 0.0,
        "recall": {
            "REAL": safe_div(cm[0, 0], cm[0].sum()),
            "FAKE": safe_div(cm[1, 1], cm[1].sum()),
        },
        "cm": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device, out_csv, chameleon=False):
    model.eval()
    rows = []
    ys, preds, probs_all = [], [], []
    desc_sum = {f"k{k}_{d}": [] for k in range(model.K) for d in DESC_NAMES}
    for x, y, sid, path, source in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        probs = torch.softmax(out["logits"].float(), dim=1).cpu().numpy()
        pred = probs.argmax(axis=1).astype(np.int64)
        desc = out["descriptors"].detach().cpu().numpy()
        y_np = y.numpy().astype(np.int64)
        ys.append(y_np)
        preds.append(pred)
        probs_all.append(probs)
        for i in range(len(y_np)):
            row = {
                "sid": sid[i],
                "image_path": path[i],
                "source": source[i],
                "gt_cls": int(y_np[i]),
                "pred_cls": int(pred[i]),
                "p_real": float(probs[i, 0]),
                "p_full": float(probs[i, 1]),
                "p_tamp": float(probs[i, 2]),
                "fake_score": float(probs[i, 1] + probs[i, 2]),
            }
            for k in range(model.K):
                for j, name in enumerate(DESC_NAMES):
                    val = float(desc[i, k, j])
                    row[f"k{k}_{name}"] = val
                    desc_sum[f"k{k}_{name}"].append(val)
            rows.append(row)
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    probs = np.concatenate(probs_all)
    summary = summarize_chameleon(y, pred, probs) if chameleon else summarize_three(y, pred, probs)
    summary["desc_mean"] = {k: float(np.mean(v)) if v else 0.0 for k, v in desc_sum.items()}
    summary["desc_std"] = {k: float(np.std(v)) if v else 0.0 for k, v in desc_sum.items()}

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_rows = read_jsonl(args.train_jsonl, args.train_limit)
    val_in_rows = read_jsonl(args.val_in_jsonl, args.val_limit)
    val_ood_rows = read_jsonl(args.val_ood_jsonl, args.val_limit)
    cham_rows = read_chameleon_roots(args.chameleon_roots, args.chameleon_limit) if args.chameleon_roots else []
    print(
        f"rows train={len(train_rows)} {Counter(r['gt_cls'] for r in train_rows)} "
        f"val_in={len(val_in_rows)} val_ood={len(val_ood_rows)} chameleon={len(cham_rows)}",
        flush=True,
    )

    train_ds = ImageJsonDataset(train_rows, image_size=args.image_size, train=True)
    val_in_ds = ImageJsonDataset(val_in_rows, image_size=args.image_size, train=False)
    val_ood_ds = ImageJsonDataset(val_ood_rows, image_size=args.image_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_in_loader = DataLoader(
        val_in_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    val_ood_loader = DataLoader(
        val_ood_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    cham_loader = None
    if cham_rows:
        cham_ds = ImageJsonDataset(cham_rows, image_size=args.image_size, train=False)
        cham_loader = DataLoader(
            cham_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True,
        )

    model = NPRFETACls(hidden=args.hidden, K=args.K, image_size=args.image_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model=NPRFETACls K={args.K} hidden={args.hidden} image={args.image_size} "
        f"params={n_params/1e6:.3f}M loss=CE smoothing={args.smoothing}",
        flush=True,
    )

    global_step = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for step, (x, y, _sid, _path, _source) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            use_amp = bool(args.amp) and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out["logits"], y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            bs = y.numel()
            total += bs
            loss_sum += float(loss.detach().cpu()) * bs
            correct += int((out["logits"].detach().argmax(dim=1) == y).sum().item())
            global_step += 1
            if step % args.log_every == 0:
                print(
                    f"ep {ep}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={loss_sum/max(1,total):.4f} acc3={correct/max(1,total):.3f} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        print(
            f"epoch {ep} train_loss={loss_sum/max(1,total):.4f} "
            f"train_acc3={correct/max(1,total):.4f}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    ckpt = out_dir / f"npr_feta_cls_{args.tag}.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    results = {"args": vars(args), "ckpt": str(ckpt)}
    results["val_in"] = evaluate(model, val_in_loader, device, out_dir / "val_in_rows.csv")
    print("[val_in]", json.dumps(results["val_in"], ensure_ascii=False), flush=True)
    results["val_ood"] = evaluate(model, val_ood_loader, device, out_dir / "val_ood_rows.csv")
    print("[val_ood]", json.dumps(results["val_ood"], ensure_ascii=False), flush=True)
    if cham_loader is not None:
        results["chameleon"] = evaluate(model, cham_loader, device, out_dir / "chameleon_rows.csv", chameleon=True)
        print("[chameleon]", json.dumps(results["chameleon"], ensure_ascii=False), flush=True)

    out_json = out_dir / f"npr_feta_cls_{args.tag}_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULT_JSON={out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="")
    ap.add_argument("--val_in_jsonl", default="")
    ap.add_argument("--val_ood_jsonl", default="")
    ap.add_argument("--chameleon_roots", nargs="*", default=[])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--chameleon_limit", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
