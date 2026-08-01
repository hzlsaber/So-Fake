#!/usr/bin/env python3
"""Export locked FETA+NPR evidence-provider predictions from cached HDF5.

This is the classifier-side evidence provider fixed on 2026-06-29:
FETA L19 + NPR-like monotonic suspicion correction, evaluated with
``log_scaled_a2`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", Path(__file__).resolve().parents[1]))
TOOLS_DIR = RELEASE_ROOT / "tools"
M0_PROBE_DIR = TOOLS_DIR / "m0_probe"
DEFAULT_QWEN_CKPT = RELEASE_ROOT / "weights/provider/feta_v0_L19_cls_only10d_L19_u1_e1.pt"
DEFAULT_NPR_CKPT = RELEASE_ROOT / "weights/provider/npr_feta_cls_npr_residual_feta10d_e1.pt"
for _path in (TOOLS_DIR, M0_PROBE_DIR):
    if _path.exists():
        sys.path.insert(0, str(_path))

from m50_external_dataset_eval import apply_transform, collect, load_model
from m36_npr_internalized_feta import CLASS_NAMES, make_loader


def load_h5_meta(h5_path: str) -> dict[str, dict]:
    meta = {}
    with h5py.File(h5_path, "r") as h5:
        for sid in sorted(k for k in h5.keys() if k != "meta"):
            g = h5[sid]
            meta[sid] = {
                "path": str(g.attrs.get("image_path", "")),
                "label": int(g.attrs.get("gt_cls", -1)),
                "source": str(g.attrs.get("source", "")),
                "mask_path": str(g.attrs.get("mask_path", "")),
            }
    return meta


def softmax(xs: np.ndarray) -> np.ndarray:
    x = xs.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-12)


def safe_float(x) -> float:
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def as_float_list(xs) -> list[float]:
    return [safe_float(x) for x in np.asarray(xs).reshape(-1)]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transform", default="log_scaled_a2",
                    choices=["qwen_only", "identity", "scale_1.25", "log_scaled_a2"])
    ap.add_argument("--dataset_name", default="locked_provider")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--unpermute", type=int, default=1)
    ap.add_argument("--qwen_ckpt", default=str(DEFAULT_QWEN_CKPT))
    ap.add_argument("--npr_ckpt", default=str(DEFAULT_NPR_CKPT))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--npr_hidden", type=int, default=96)
    ap.add_argument("--freeze_qwen", type=int, default=1)
    ap.add_argument("--zero_init_delta", type=int, default=1)
    ap.add_argument("--fusion_mode", default="monotonic_suspicion")
    ap.add_argument("--suspicion_max", type=float, default=4.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, model_args = load_model(args, args.ckpt, device)
    loader = make_loader([args.h5], model_args, args.dataset_name, train=False)
    arr = collect(model, loader, device)
    final_logits, effective_s = apply_transform(arr["q_logits"], arr["suspicion"], args.transform)
    pred = final_logits.argmax(axis=-1)

    meta = load_h5_meta(args.h5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, sid in enumerate(arr["sid"]):
            sid = str(sid)
            probs = softmax(final_logits[i])
            q_probs = softmax(arr["q_logits"][i])
            p = int(pred[i])
            m = meta.get(sid, {})
            rec = {
                "sid": sid,
                "path": str(arr["image_path"][i]),
                "label": int(arr["y"][i]),
                "source": str(arr["source"][i]),
                "sample_source": m.get("source", ""),
                "mask_path": m.get("mask_path", ""),
                "evidence_pred": p,
                "evidence_answer": CLASS_NAMES[p],
                "evidence_confidence": safe_float(probs[p]),
                "logits": as_float_list(final_logits[i]),
                "probs": as_float_list(probs),
                "qwen_logits": as_float_list(arr["q_logits"][i]),
                "qwen_probs": as_float_list(q_probs),
                "qwen_pred": int(arr["q_logits"][i].argmax()),
                "suspicion": safe_float(arr["suspicion"][i]),
                "effective_s": safe_float(effective_s[i]),
                "transform": args.transform,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"exported {n}/{len(pred)}", flush=True)
    print(f"done: {n} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
