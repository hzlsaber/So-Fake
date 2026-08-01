#!/usr/bin/env python3
"""Internalize an NPR-like low-level branch into FETA.

This is not frozen descriptor concatenation.  The low-level branch is trained
as part of the evidence provider:

  cached Qwen feature -> frozen/optional FETA semantic evidence -> qwen logits
  image -> NPR residual -> trainable low-level evidence maps/descriptors/logits
  [qwen evidence, low-level evidence] -> residual logit correction

The first version keeps Qwen-FETA frozen and learns a residual correction from
the NPR-like branch.  This preserves the strong SoFakeOOD baseline at
initialization while allowing the low-level evidence to become part of the
trained model.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from m4_feta_v0 import FETAv0, unpermute_feat_to_raster
from m31_npr_feta_cls import NPRFETACls


ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["REAL", "FULL_SYNTHETIC", "TAMPERED"]
RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", Path(__file__).resolve().parents[1]))
DEFAULT_QWEN_CKPT = Path(os.environ.get("SOFAKE_R1_QWEN_FETA_CKPT", ""))
DEFAULT_NPR_CKPT = Path(os.environ.get("SOFAKE_R1_NPR_CKPT", ""))


def safe_div(a, b):
    return float(a / b) if b else 0.0


def binary_metrics(pred_bin, y_bin):
    cm = np.zeros((2, 2), dtype=np.int64)
    for g, p in zip(y_bin, pred_bin):
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


def three_way_metrics(pred, y):
    cm = np.zeros((3, 3), dtype=np.int64)
    for g, p in zip(y, pred):
        cm[int(g), int(p)] += 1
    return {
        "acc": safe_div(np.trace(cm), cm.sum()),
        "recall": [safe_div(cm[c, c], cm[c].sum()) for c in range(3)],
        "cm": cm.tolist(),
        "flip": {
            "TAMP->FS": safe_div(cm[2, 1], cm[2].sum()),
            "FS->TAMP": safe_div(cm[1, 2], cm[1].sum()),
            "REAL->TAMP": safe_div(cm[0, 2], cm[0].sum()),
        },
    }


def summarize_predictions(y, pred, split):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    out = {
        "n": int(len(y)),
        "binary_folded": binary_metrics((pred != 0).astype(np.int64), (y != 0).astype(np.int64)),
    }
    if split != "chameleon":
        out["three_way"] = three_way_metrics(pred, y)
    return out


class H5QwenImageDataset(Dataset):
    def __init__(self, h5_path, layer=19, image_size=224, unpermute=True, train=False, limit=0):
        self.h5_path = h5_path
        self.layer = layer
        self.unpermute = unpermute
        self.train = train
        self.blur_prob = 0.0
        self.blur_sig = (0.0, 3.0)
        self.jpg_prob = 0.0
        self.jpg_qual = (30, 100)
        with h5py.File(h5_path, "r") as f:
            keys = sorted([k for k in f.keys() if k != "meta"])
        if limit and limit > 0:
            keys = keys[:limit]
        self.sids = keys
        self._handle = None
        tf = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
        # Keep image/qwen-feature pairing deterministic in this first joint test.
        tf.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf = transforms.Compose(tf)
        self.image_size = image_size

    def set_artifact_aug(self, blur_prob=0.0, blur_sig=(0.0, 3.0), jpg_prob=0.0, jpg_qual=(30, 100)):
        self.blur_prob = float(blur_prob)
        self.blur_sig = tuple(float(x) for x in blur_sig)
        self.jpg_prob = float(jpg_prob)
        self.jpg_qual = tuple(int(x) for x in jpg_qual)

    @staticmethod
    def _sample_range(vals):
        if len(vals) == 1:
            return vals[0]
        lo, hi = vals[0], vals[-1]
        return random.random() * (hi - lo) + lo

    def _artifact_augment(self, img):
        # CNNDetection-style non-geometric robustness augmentation. We avoid
        # crop/flip here because the paired Qwen feature is cached offline.
        if self.train and self.blur_prob > 0 and random.random() < self.blur_prob:
            sigma = self._sample_range(self.blur_sig)
            img = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        if self.train and self.jpg_prob > 0 and random.random() < self.jpg_prob:
            qual = int(round(self._sample_range(self.jpg_qual)))
            qual = max(1, min(100, qual))
            out = BytesIO()
            img.save(out, format="JPEG", quality=qual)
            out.seek(0)
            img = Image.open(out).convert("RGB")
            img.load()
            out.close()
        return img

    def _open(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.sids)

    def _read_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        img = self._artifact_augment(img)
        return self.tf(img)

    def __getitem__(self, idx):
        self._open()
        sid = self.sids[idx]
        grp = self._handle[sid]
        feat = torch.from_numpy(grp[f"layer_{self.layer}"][:]).float()
        patch_mask = torch.from_numpy(grp["patch_mask"][:]).float()
        gt_cls = int(grp.attrs["gt_cls"])
        grid_thw = grp.attrs["grid_thw"]
        T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        if self.unpermute:
            feat = unpermute_feat_to_raster(feat, T, H, W)
        image_path = str(grp.attrs.get("image_path", ""))
        return {
            "feat": feat,
            "patch_mask": patch_mask,
            "gt_cls": gt_cls,
            "T": T,
            "H": H,
            "W": W,
            "image": self._read_image(image_path),
            "image_path": image_path,
            "source": str(grp.attrs.get("source", "unknown")),
            "sid": sid,
        }


def collate_joint(batch):
    max_len = max(b["feat"].shape[0] for b in batch)
    B = len(batch)
    D = batch[0]["feat"].shape[1]
    feats = torch.zeros(B, max_len, D)
    pm = torch.zeros(B, max_len)
    attn = torch.zeros(B, max_len)
    cls = torch.zeros(B, dtype=torch.long)
    Hs = torch.zeros(B, dtype=torch.long)
    Ws = torch.zeros(B, dtype=torch.long)
    Ts = torch.zeros(B, dtype=torch.long)
    images = torch.stack([b["image"] for b in batch], dim=0)
    paths, sources, sids = [], [], []
    for i, b in enumerate(batch):
        n = b["feat"].shape[0]
        feats[i, :n] = b["feat"]
        pm[i, :n] = b["patch_mask"]
        attn[i, :n] = 1.0
        cls[i] = int(b["gt_cls"])
        Hs[i] = int(b["H"])
        Ws[i] = int(b["W"])
        Ts[i] = int(b["T"])
        paths.append(b["image_path"])
        sources.append(b["source"])
        sids.append(b["sid"])
    return {
        "feat": feats,
        "patch_mask": pm,
        "attn": attn,
        "gt_cls": cls,
        "H": Hs,
        "W": Ws,
        "T": Ts,
        "image": images,
        "image_paths": paths,
        "sources": sources,
        "sids": sids,
    }


class NPRInternalizedFETA(nn.Module):
    def __init__(self, qwen_ckpt, npr_ckpt="", hidden=128, npr_hidden=96,
                 freeze_qwen=True, zero_init_delta=True,
                 fusion_mode="residual_mlp", suspicion_max=4.0):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.suspicion_max = suspicion_max
        self.qwen = FETAv0(
            in_dim=1280,
            hidden=hidden,
            K=2,
            use_spatial_conv=True,
            desc_set="minimal",
        )
        self.qwen.load_state_dict(torch.load(qwen_ckpt, map_location="cpu"), strict=True)
        self.freeze_qwen = freeze_qwen
        if freeze_qwen:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

        self.npr = NPRFETACls(hidden=npr_hidden, K=2, image_size=224)
        if npr_ckpt:
            ckpt = torch.load(npr_ckpt, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.npr.load_state_dict(state, strict=True)

        q_dim = 2 * 5
        n_dim = 2 * 5
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(q_dim + n_dim + 6),
            nn.Linear(q_dim + n_dim + 6, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        if zero_init_delta:
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)
        self.suspicion_mlp = nn.Sequential(
            nn.LayerNorm(n_dim + 3),
            nn.Linear(n_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Start near the Qwen-FETA baseline. The suspicion branch must earn its
        # influence through the binary auxiliary loss and final CE.
        nn.init.zeros_(self.suspicion_mlp[-1].weight)
        nn.init.constant_(self.suspicion_mlp[-1].bias, -6.0)

    def forward(self, feat, attn, Ts, Hs, Ws, image):
        if self.freeze_qwen:
            with torch.no_grad():
                q = self.qwen(feat, attn, Ts, Hs, Ws)
        else:
            q = self.qwen(feat, attn, Ts, Hs, Ws)
        n = self.npr(image)
        q_desc = q["descriptors"].flatten(1)
        n_desc = n["descriptors"].flatten(1)
        q_logits = q["logits"].float()
        n_logits = n["logits"].float()
        mix = torch.cat([q_desc.float(), n_desc.float(), q_logits, n_logits], dim=-1)
        suspicion_raw = self.suspicion_mlp(torch.cat([n_desc.float(), n_logits], dim=-1)).squeeze(-1)
        suspicion = torch.sigmoid(suspicion_raw) * self.suspicion_max
        if self.fusion_mode == "residual_mlp":
            delta = self.delta_mlp(mix)
            logits = q_logits + delta
        elif self.fusion_mode == "monotonic_suspicion":
            delta = torch.zeros_like(q_logits)
            delta[:, 0] = -suspicion
            delta[:, 1] = 0.5 * suspicion
            delta[:, 2] = 0.5 * suspicion
            logits = q_logits + delta
        else:
            raise ValueError(f"unknown fusion_mode: {self.fusion_mode}")
        return {
            "logits": logits,
            "delta_logits": delta,
            "suspicion_raw": suspicion_raw,
            "suspicion": suspicion,
            "qwen_logits": q_logits,
            "npr_logits": n_logits,
            "qwen_descriptors": q_desc,
            "npr_descriptors": n_desc,
        }


def binary_ce_from_logits(logits, cls):
    bin_logits = torch.stack([logits[:, 0], torch.logsumexp(logits[:, 1:], dim=-1)], dim=-1)
    target = (cls != 0).long()
    return F.cross_entropy(bin_logits, target)


def loss_fn(out, cls, w_final=1.0, w_npr_bin=0.3, w_delta_l2=0.0, w_susp_bin=0.0):
    loss_final = F.cross_entropy(out["logits"], cls)
    loss_npr_bin = binary_ce_from_logits(out["npr_logits"], cls) if w_npr_bin > 0 else out["logits"].new_zeros(())
    loss_delta = (out["delta_logits"] ** 2).mean() if w_delta_l2 > 0 else out["logits"].new_zeros(())
    if w_susp_bin > 0:
        target = (cls != 0).float()
        loss_susp = F.binary_cross_entropy_with_logits(out["suspicion_raw"], target)
    else:
        loss_susp = out["logits"].new_zeros(())
    loss = (
        w_final * loss_final
        + w_npr_bin * loss_npr_bin
        + w_delta_l2 * loss_delta
        + w_susp_bin * loss_susp
    )
    return loss, {
        "final_ce": float(loss_final.detach().cpu()),
        "npr_bin": float(loss_npr_bin.detach().cpu()),
        "delta_l2": float(loss_delta.detach().cpu()),
        "susp_bin": float(loss_susp.detach().cpu()),
    }


def make_loader(h5_paths, args, split, train=False):
    datasets = [
        H5QwenImageDataset(
            p,
            layer=args.layer,
            image_size=args.image_size,
            unpermute=bool(args.unpermute),
            train=train,
            limit=args.train_limit if train else args.eval_limit,
        )
        for p in h5_paths
    ]
    for ds in datasets:
        ds.set_artifact_aug(
            blur_prob=getattr(args, "train_blur_prob", 0.0) if train else 0.0,
            blur_sig=getattr(args, "train_blur_sig", [0.0, 3.0]),
            jpg_prob=getattr(args, "train_jpg_prob", 0.0) if train else 0.0,
            jpg_qual=getattr(args, "train_jpg_qual", [30, 100]),
        )
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        collate_fn=collate_joint,
        pin_memory=True,
        drop_last=train,
    )
    print(f"{split}: n={len(ds)} from {h5_paths}", flush=True)
    train_blur_prob = getattr(args, "train_blur_prob", 0.0)
    train_jpg_prob = getattr(args, "train_jpg_prob", 0.0)
    if train and (train_blur_prob > 0 or train_jpg_prob > 0):
        print(
            f"{split} artifact_aug blur_prob={train_blur_prob} blur_sig={getattr(args, 'train_blur_sig', [0.0, 3.0])} "
            f"jpg_prob={train_jpg_prob} jpg_qual={getattr(args, 'train_jpg_qual', [30, 100])}",
            flush=True,
        )
    return loader


@torch.no_grad()
def evaluate(model, loader, device, split, out_csv):
    model.eval()
    rows = []
    ys, pred_final, pred_qwen, pred_npr = [], [], [], []
    for batch in loader:
        feat = batch["feat"].to(device, non_blocking=True)
        attn = batch["attn"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        cls = batch["gt_cls"].numpy().astype(np.int64)
        out = model(feat, attn, batch["T"], batch["H"], batch["W"], image)
        pf = out["logits"].argmax(dim=-1).cpu().numpy().astype(np.int64)
        pq = out["qwen_logits"].argmax(dim=-1).cpu().numpy().astype(np.int64)
        pn = out["npr_logits"].argmax(dim=-1).cpu().numpy().astype(np.int64)
        delta = out["delta_logits"].detach().cpu().numpy()
        npr_desc = out["npr_descriptors"].detach().cpu().numpy()
        ys.append(cls)
        pred_final.append(pf)
        pred_qwen.append(pq)
        pred_npr.append(pn)
        for i in range(len(cls)):
            rows.append({
                "sid": batch["sids"][i],
                "image_path": batch["image_paths"][i],
                "source": batch["sources"][i],
                "gt_cls": int(cls[i]),
                "pred_final": int(pf[i]),
                "pred_qwen": int(pq[i]),
                "pred_npr": int(pn[i]),
                "delta_real": float(delta[i, 0]),
                "delta_fs": float(delta[i, 1]),
                "delta_tamp": float(delta[i, 2]),
                "suspicion": float(out["suspicion"].detach().cpu().numpy()[i]),
                "npr_desc_mean": float(npr_desc[i].mean()),
                "npr_desc_max": float(npr_desc[i].max()),
            })
    y = np.concatenate(ys)
    pf = np.concatenate(pred_final)
    pq = np.concatenate(pred_qwen)
    pn = np.concatenate(pred_npr)
    result = {
        "final": summarize_predictions(y, pf, split),
        "qwen": summarize_predictions(y, pq, split),
        "npr": summarize_predictions(y, pn, split),
    }
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_loader = make_loader([args.train_h5], args, "train", train=True)
    val_in_loader = make_loader([args.val_in_h5], args, "val_in", train=False)
    val_ood_loader = make_loader([args.val_ood_h5], args, "val_ood", train=False)
    ch_loader = make_loader(args.chameleon_h5, args, "chameleon", train=False) if args.chameleon_h5 else None

    model = NPRInternalizedFETA(
        qwen_ckpt=args.qwen_ckpt,
        npr_ckpt=args.npr_ckpt,
        hidden=args.hidden,
        npr_hidden=args.npr_hidden,
        freeze_qwen=bool(args.freeze_qwen),
        zero_init_delta=bool(args.zero_init_delta),
        fusion_mode=args.fusion_mode,
        suspicion_max=args.suspicion_max,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"model params total={total/1e6:.3f}M trainable={trainable/1e6:.3f}M", flush=True)
    print(
        f"loss weights final={args.w_final} npr_bin={args.w_npr_bin} delta_l2={args.w_delta_l2} "
        f"susp_bin={args.w_susp_bin} freeze_qwen={bool(args.freeze_qwen)} "
        f"zero_init_delta={bool(args.zero_init_delta)} fusion={args.fusion_mode}",
        flush=True,
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    t0 = time.time()
    global_step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        sums = Counter()
        correct = 0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            feat = batch["feat"].to(device, non_blocking=True)
            attn = batch["attn"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            cls = batch["gt_cls"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            use_amp = bool(args.amp) and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(feat, attn, batch["T"], batch["H"], batch["W"], image)
                loss, parts = loss_fn(out, cls, args.w_final, args.w_npr_bin,
                                      args.w_delta_l2, args.w_susp_bin)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            bs = int(cls.numel())
            seen += bs
            global_step += 1
            correct += int((out["logits"].detach().argmax(dim=-1) == cls).sum().item())
            for k, v in parts.items():
                sums[k] += v * bs
            if step % args.log_every == 0:
                msg = " ".join(f"{k}={sums[k]/seen:.4f}" for k in sorted(sums))
                print(
                    f"ep {ep}/{args.epochs} step {step}/{len(train_loader)} {msg} "
                    f"acc={correct/max(1, seen):.3f} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        msg = " ".join(f"{k}={sums[k]/seen:.4f}" for k in sorted(sums))
        print(f"epoch {ep} {msg} acc={correct/max(1, seen):.4f}", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break

    ckpt = out_dir / f"npr_internalized_feta_{args.tag}.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    results = {"args": vars(args), "ckpt": str(ckpt)}
    results["val_in"] = evaluate(model, val_in_loader, device, "val_in", out_dir / "val_in_rows.csv")
    print("[val_in]", json.dumps(results["val_in"], ensure_ascii=False), flush=True)
    results["val_ood"] = evaluate(model, val_ood_loader, device, "val_ood", out_dir / "val_ood_rows.csv")
    print("[val_ood]", json.dumps(results["val_ood"], ensure_ascii=False), flush=True)
    if ch_loader is not None:
        results["chameleon"] = evaluate(model, ch_loader, device, "chameleon", out_dir / "chameleon_rows.csv")
        print("[chameleon]", json.dumps(results["chameleon"], ensure_ascii=False), flush=True)
    out_json = out_dir / f"npr_internalized_feta_{args.tag}_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULT_JSON={out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", default="")
    ap.add_argument("--val_in_h5", default="")
    ap.add_argument("--val_ood_h5", default="")
    ap.add_argument("--chameleon_h5", nargs="*", default=[])
    ap.add_argument("--qwen_ckpt", default=str(DEFAULT_QWEN_CKPT))
    ap.add_argument("--npr_ckpt", default=str(DEFAULT_NPR_CKPT))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--npr_hidden", type=int, default=96)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--freeze_qwen", type=int, default=1)
    ap.add_argument("--zero_init_delta", type=int, default=1)
    ap.add_argument("--fusion_mode", choices=["residual_mlp", "monotonic_suspicion"], default="residual_mlp")
    ap.add_argument("--suspicion_max", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--w_final", type=float, default=1.0)
    ap.add_argument("--w_npr_bin", type=float, default=0.3)
    ap.add_argument("--w_delta_l2", type=float, default=0.0)
    ap.add_argument("--w_susp_bin", type=float, default=0.0)
    ap.add_argument("--train_blur_prob", type=float, default=0.0)
    ap.add_argument("--train_blur_sig", type=float, nargs="+", default=[0.0, 3.0])
    ap.add_argument("--train_jpg_prob", type=float, default=0.0)
    ap.add_argument("--train_jpg_qual", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--unpermute", type=int, default=1)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
