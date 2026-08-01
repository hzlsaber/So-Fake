#!/usr/bin/env python3
"""Build and evaluate external binary datasets for the evidence provider.

The external datasets are binary REAL/FAKE folders.  FAKE is encoded as class 1
so the existing three-way evidence provider can be evaluated with the standard
folded binary decision: pred_fake = argmax(logits) != REAL.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from m36_npr_internalized_feta import NPRInternalizedFETA, make_loader


RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", Path(__file__).resolve().parents[1]))
DEFAULT_QWEN_CKPT = Path(os.environ.get("SOFAKE_R1_QWEN_FETA_CKPT", ""))
DEFAULT_NPR_CKPT = Path(os.environ.get("SOFAKE_R1_NPR_CKPT", ""))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


DEFAULT_DATASETS = {
    "B-Free-online": os.environ.get("BFREE_ONLINE_ROOT", ""),
    "AIGI-Now": os.environ.get("AIGI_NOW_ROOT", ""),
    "SocialRF": os.environ.get("SOCIALRF_ROOT", ""),
    "CommunityAI": os.environ.get("COMMUNITYAI_ROOT", ""),
}


DEFAULT_CKPTS = {
    "blur_only05": os.environ.get("SOFAKE_R1_PROVIDER_CKPT", ""),
}


def clean_id(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_bfree(root):
    for path in sorted(Path(root).rglob("*")):
        if not is_image(path):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2 or rel.parts[0] not in {"REAL", "FAKE"}:
            continue
        label = rel.parts[0]
        yield {
            "image": str(path),
            "gt_cls": 0 if label == "REAL" else 1,
            "source": "B-Free-online",
            "subsource": rel.parts[1] if len(rel.parts) > 2 else "",
            "dataset": "B-Free-online",
        }


def iter_aigi_now(root):
    for path in sorted(Path(root).rglob("*")):
        if not is_image(path):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        generator, edit_type, label = rel.parts[0], rel.parts[1], rel.parts[2]
        if label not in {"0_real", "1_fake"}:
            continue
        yield {
            "image": str(path),
            "gt_cls": 0 if label == "0_real" else 1,
            "source": f"{generator}/{edit_type}",
            "subsource": generator,
            "dataset": "AIGI-Now",
        }


def iter_simple_aigibench(root, dataset_name):
    for path in sorted(Path(root).rglob("*")):
        if not is_image(path):
            continue
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if "0_real" in parts:
            gt_cls = 0
        elif "1_fake" in parts:
            gt_cls = 1
        else:
            continue
        yield {
            "image": str(path),
            "gt_cls": gt_cls,
            "source": dataset_name,
            "subsource": dataset_name,
            "dataset": dataset_name,
        }


def dataset_iter(name, root):
    if name == "B-Free-online":
        return iter_bfree(root)
    if name == "AIGI-Now":
        return iter_aigi_now(root)
    if name in {"SocialRF", "CommunityAI"}:
        return iter_simple_aigibench(root, name)
    raise ValueError(f"unknown dataset builder: {name}")


def parse_name_path(values):
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected name=path, got {item}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def build_samples(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_name_path(args.datasets) if args.datasets else DEFAULT_DATASETS
    rows = []
    for name, root in datasets.items():
        out_path = out_dir / f"{clean_id(name)}.jsonl"
        n = 0
        n_real = 0
        n_fake = 0
        with out_path.open("w", encoding="utf-8") as f:
            for rec in dataset_iter(name, root):
                rec = dict(rec)
                rec["sample_id"] = f"{clean_id(name)}_{n:07d}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                n_real += int(rec["gt_cls"] == 0)
                n_fake += int(rec["gt_cls"] != 0)
                if args.limit and n >= args.limit:
                    break
        rows.append({
            "dataset": name,
            "root": root,
            "samples_jsonl": str(out_path),
            "n": n,
            "real": n_real,
            "fake": n_fake,
        })
    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "external_samples_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(manifest.to_string(index=False))
    print(f"MANIFEST={manifest_path}")


def binary_metrics(y, pred3):
    y = np.asarray(y, dtype=np.int64)
    pred3 = np.asarray(pred3, dtype=np.int64)
    yb = (y != 0).astype(np.int64)
    pb = (pred3 != 0).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for g, p in zip(yb, pb):
        cm[int(g), int(p)] += 1
    real_r = cm[0, 0] / max(1, cm[0].sum())
    fake_r = cm[1, 1] / max(1, cm[1].sum())
    return {
        "n": int(len(y)),
        "acc": float(np.trace(cm) / max(1, cm.sum())),
        "balanced_acc": float((real_r + fake_r) / 2),
        "real_recall": float(real_r),
        "fake_recall": float(fake_r),
        "cm00_real_as_real": int(cm[0, 0]),
        "cm01_real_as_fake": int(cm[0, 1]),
        "cm10_fake_as_real": int(cm[1, 0]),
        "cm11_fake_as_fake": int(cm[1, 1]),
        "pred_real": int((pb == 0).sum()),
        "pred_fake": int((pb == 1).sum()),
        "pred_REAL": int((pred3 == 0).sum()),
        "pred_FS": int((pred3 == 1).sum()),
        "pred_TAMP": int((pred3 == 2).sum()),
    }


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    ys, sources, sids, paths, q_logits, susp = [], [], [], [], [], []
    for batch in loader:
        feat = batch["feat"].to(device, non_blocking=True)
        attn = batch["attn"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        out = model(feat, attn, batch["T"], batch["H"], batch["W"], image)
        ys.append(batch["gt_cls"].numpy().astype(np.int64))
        sources.extend(batch["sources"])
        sids.extend(batch["sids"])
        paths.extend(batch["image_paths"])
        q_logits.append(out["qwen_logits"].float().cpu().numpy())
        susp.append(out["suspicion"].float().cpu().numpy())
    return {
        "y": np.concatenate(ys),
        "source": np.asarray(sources),
        "sid": np.asarray(sids),
        "image_path": np.asarray(paths),
        "q_logits": np.concatenate(q_logits),
        "suspicion": np.concatenate(susp),
    }


def log_scaled(s, smax=4.0, beta=2.0):
    return smax * np.log1p(beta * np.clip(s, 0.0, smax)) / np.log1p(beta * smax)


def apply_transform(q_logits, suspicion, name):
    if name == "qwen_only":
        sf = np.zeros_like(suspicion, dtype=np.float32)
    elif name == "identity":
        sf = suspicion.astype(np.float32)
    elif name == "scale_1.25":
        sf = (1.25 * suspicion).astype(np.float32)
    elif name == "log_scaled_a2":
        sf = log_scaled(suspicion, 4.0, 2.0).astype(np.float32)
    else:
        raise ValueError(name)
    logits = q_logits.copy()
    logits[:, 0] -= sf
    logits[:, 1] += 0.5 * sf
    logits[:, 2] += 0.5 * sf
    return logits, sf


def load_model(args, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    local = argparse.Namespace(**vars(args))
    for k in [
        "hidden", "npr_hidden", "freeze_qwen",
        "zero_init_delta", "fusion_mode", "suspicion_max", "layer",
        "image_size", "unpermute",
    ]:
        if k in ckpt_args:
            setattr(local, k, ckpt_args[k])
    model = NPRInternalizedFETA(
        qwen_ckpt=local.qwen_ckpt,
        npr_ckpt=local.npr_ckpt,
        hidden=local.hidden,
        npr_hidden=local.npr_hidden,
        freeze_qwen=bool(local.freeze_qwen),
        zero_init_delta=bool(local.zero_init_delta),
        fusion_mode=local.fusion_mode,
        suspicion_max=local.suspicion_max,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, local


def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_name_path(args.h5)
    ckpts = parse_name_path(args.ckpt) if args.ckpt else DEFAULT_CKPTS
    device = torch.device(args.device)
    rows = []
    prediction_rows = []
    qwen_done = set()

    for weight_name, ckpt_path in ckpts.items():
        print(f"\n[load] {weight_name}: {ckpt_path}", flush=True)
        model, model_args = load_model(args, ckpt_path, device)
        for dataset_name, h5_path in datasets.items():
            print(f"[eval] weight={weight_name} dataset={dataset_name} h5={h5_path}", flush=True)
            loader = make_loader([h5_path], model_args, dataset_name, train=False)
            arr = collect(model, loader, device)
            for transform in args.transforms:
                row_weight = weight_name
                row_ckpt = ckpt_path
                if transform == "qwen_only":
                    if dataset_name in qwen_done:
                        continue
                    row_weight = "qwen_feta_only"
                    row_ckpt = getattr(model_args, "qwen_ckpt", "")
                    qwen_done.add(dataset_name)
                logits, sf = apply_transform(arr["q_logits"], arr["suspicion"], transform)
                pred = logits.argmax(axis=-1)
                groups = ["ALL"]
                if args.per_source:
                    groups.extend(sorted(set(arr["source"].tolist())))
                for group in groups:
                    mask = np.ones_like(arr["y"], dtype=bool) if group == "ALL" else (arr["source"] == group)
                    m = binary_metrics(arr["y"][mask], pred[mask])
                    m.update({
                        "dataset": dataset_name,
                        "source": group,
                        "weight": row_weight,
                        "ckpt": row_ckpt,
                        "transform": transform,
                        "suspicion_mean": float(arr["suspicion"][mask].mean()),
                        "suspicion_median": float(np.median(arr["suspicion"][mask])),
                        "effective_s_mean": float(sf[mask].mean()),
                        "effective_s_median": float(np.median(sf[mask])),
                    })
                    rows.append(m)
                if args.write_predictions:
                    for i in range(len(arr["y"])):
                        prediction_rows.append({
                            "dataset": dataset_name,
                            "weight": row_weight,
                            "transform": transform,
                            "sid": str(arr["sid"][i]),
                            "source": str(arr["source"][i]),
                            "image_path": str(arr["image_path"][i]),
                            "gt_cls": int(arr["y"][i]),
                            "pred3": int(pred[i]),
                            "pred_bin": int(pred[i] != 0),
                            "suspicion": float(arr["suspicion"][i]),
                            "effective_s": float(sf[i]),
                        })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "external_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n[summary ALL]")
    cols = [
        "dataset", "weight", "transform", "n", "acc", "balanced_acc",
        "real_recall", "fake_recall", "pred_real", "pred_fake",
        "suspicion_median", "effective_s_median",
    ]
    print(summary[summary["source"] == "ALL"][cols].to_string(index=False))
    print(f"SUMMARY_CSV={summary_path}")
    if args.write_predictions:
        pred_path = out_dir / "external_eval_predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)
        print(f"PRED_CSV={pred_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build_samples")
    b.add_argument("--out_dir", required=True)
    b.add_argument("--datasets", nargs="*", default=None,
                   help="Optional name=path list. Defaults to the four external datasets.")
    b.add_argument("--limit", type=int, default=0)

    e = sub.add_parser("eval")
    e.add_argument("--h5", nargs="+", required=True, help="name=features.h5 entries")
    e.add_argument("--ckpt", nargs="*", default=None, help="name=checkpoint.pt entries")
    e.add_argument("--transforms", nargs="+", default=["qwen_only", "identity", "scale_1.25", "log_scaled_a2"])
    e.add_argument("--out_dir", required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--eval_batch_size", type=int, default=256)
    e.add_argument("--batch_size", type=int, default=128)
    e.add_argument("--num_workers", type=int, default=4)
    e.add_argument("--eval_limit", type=int, default=0)
    e.add_argument("--train_limit", type=int, default=0)
    e.add_argument("--layer", type=int, default=19)
    e.add_argument("--image_size", type=int, default=224)
    e.add_argument("--unpermute", type=int, default=1)
    e.add_argument("--qwen_ckpt", default=str(DEFAULT_QWEN_CKPT))
    e.add_argument("--npr_ckpt", default=str(DEFAULT_NPR_CKPT))
    e.add_argument("--hidden", type=int, default=128)
    e.add_argument("--npr_hidden", type=int, default=96)
    e.add_argument("--freeze_qwen", type=int, default=1)
    e.add_argument("--zero_init_delta", type=int, default=1)
    e.add_argument("--fusion_mode", default="monotonic_suspicion")
    e.add_argument("--suspicion_max", type=float, default=4.0)
    e.add_argument("--per_source", action="store_true")
    e.add_argument("--write_predictions", action="store_true")

    args = ap.parse_args()
    if args.cmd == "build_samples":
        build_samples(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
