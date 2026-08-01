#!/usr/bin/env python3
"""Export standalone FETA topology maps for V5 judge SFT.

Each exported image is a text-free, bbox-free visualization of
E_main = max(K0, K1), resized to the original image size. It is meant to be
used as Image 2 in the decision-maker prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(RELEASE_ROOT / "tools"))

from m4_feta_v0 import FETAv0, unpermute_feat_to_raster  # noqa: E402


DEFAULT_SOURCE = Path(os.environ.get("SOFAKE_R1_TOPOLOGY_SOURCE", ""))
DEFAULT_OUT_DIR = Path(os.environ.get("SOFAKE_R1_TOPOLOGY_OUT_DIR", "topology_maps"))
DEFAULT_CKPT = Path(os.environ.get("SOFAKE_R1_QWEN_FETA_CKPT", ""))
H5_BY_PREFIX = {
    "grpo5k": Path(os.environ.get("SOFAKE_R1_GRPO5K_H5", "")),
    "grpo10k": Path(os.environ.get("SOFAKE_R1_GRPO10K_H5", "")),
}


def load_source_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def collect_unique_samples(rows: list[dict]) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for row in rows:
        sid = row.get("meta", {}).get("sample_id")
        image_path = row.get("images", [{}])[0].get("path")
        if not sid or not image_path or sid in seen:
            continue
        seen.add(sid)
        out.append((sid, image_path))
    return out


def resize_map(e: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    e = np.asarray(e, dtype=np.float32)
    e = np.clip(e, 0.0, 1.0)
    pil = Image.fromarray((e * 255).astype(np.uint8), mode="L")
    return np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def map_to_rgb(e: np.ndarray, size: tuple[int, int], cmap_name: str = "viridis") -> Image.Image:
    er = resize_map(e, size)
    rgba = (cm.get_cmap(cmap_name)(er) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3], mode="RGB")


def connected_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            ys = []
            xs = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append({
                "area": len(xs),
                "bbox_grid": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                "centroid_grid": [float(np.mean(xs)), float(np.mean(ys))],
            })
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def region_name(cx: float, cy: float, W: int, H: int) -> str:
    x = cx / max(W - 1, 1)
    y = cy / max(H - 1, 1)
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "central"
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "central"
    if vertical == "central" and horizontal == "central":
        return "center"
    if vertical == "central":
        return horizontal
    if horizontal == "central":
        return vertical
    return f"{vertical}-{horizontal}"


def topology_visual_stats(e: np.ndarray) -> dict:
    """Describe the visible structure of the anomaly map without using GT.

    These stats intentionally avoid repeating provider-facing metrics such as
    mass/area/concentration verbatim. They describe how Image 2 looks: number
    of hotspots, broadness, edge tendency, and approximate dominant location.
    """
    e = np.asarray(e, dtype=np.float32)
    e = np.clip(e, 0.0, 1.0)
    H, W = e.shape
    max_v = float(e.max()) if e.size else 0.0
    mean_v = float(e.mean()) if e.size else 0.0
    q80 = float(np.quantile(e, 0.80)) if e.size else 0.0
    q90 = float(np.quantile(e, 0.90)) if e.size else 0.0

    mid_thr = 0.35
    high_thr = max(0.50, min(0.75, q90))
    mid_mask = e >= mid_thr
    high_mask = e >= high_thr
    if high_mask.sum() == 0 and max_v >= 0.25:
        high_thr = max(0.25, 0.75 * max_v)
        high_mask = e >= high_thr

    mid_frac = float(mid_mask.mean()) if e.size else 0.0
    high_frac = float(high_mask.mean()) if e.size else 0.0
    comps_high = connected_components(high_mask)
    comps_mid = connected_components(mid_mask)
    hot_count = len(comps_high)
    mid_count = len(comps_mid)
    largest_high_frac = float(comps_high[0]["area"] / max(high_mask.sum(), 1)) if comps_high else 0.0
    largest_mid_frac = float(comps_mid[0]["area"] / max(mid_mask.sum(), 1)) if comps_mid else 0.0

    border = max(1, int(round(min(H, W) * 0.14)))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    edge_share = float((high_mask & border_mask).sum() / max(high_mask.sum(), 1)) if high_mask.any() else 0.0
    edge_heavy = bool(high_mask.any() and edge_share >= 0.45)

    if high_mask.any():
        yy, xx = np.nonzero(high_mask)
        weights = e[high_mask]
        cx = float(np.average(xx, weights=weights))
        cy = float(np.average(yy, weights=weights))
    elif mid_mask.any():
        yy, xx = np.nonzero(mid_mask)
        weights = e[mid_mask]
        cx = float(np.average(xx, weights=weights))
        cy = float(np.average(yy, weights=weights))
    else:
        cx = float((W - 1) / 2)
        cy = float((H - 1) / 2)

    if max_v < 0.25 or (mean_v < 0.12 and mid_frac < 0.08):
        intensity = "weak"
    elif max_v >= 0.75:
        intensity = "strong"
    else:
        intensity = "moderate"

    if mid_frac >= 0.62:
        extent = "diffuse"
    elif mid_frac >= 0.35:
        extent = "broad"
    elif mid_frac >= 0.16:
        extent = "moderate"
    elif mid_frac > 0.0:
        extent = "sparse"
    else:
        extent = "minimal"

    if extent == "diffuse":
        structure = "diffuse_field"
    elif edge_heavy and hot_count <= 2:
        structure = "edge_or_corner_hotspot"
    elif hot_count >= 3:
        structure = "multi_hotspot"
    elif extent in {"broad", "moderate"} and largest_mid_frac >= 0.55:
        # A single high-response core can sit inside a much larger smooth lobe.
        # In that case the visual map does not look like a compact hotspot.
        structure = "broad_lobe"
    elif hot_count == 1 and high_frac <= 0.18:
        structure = "single_hotspot"
    elif mid_count >= 3:
        structure = "scattered_regions"
    elif mid_mask.any():
        structure = "mixed_regions"
    else:
        structure = "weak_residual"

    return {
        "grid_size": [int(H), int(W)],
        "intensity": intensity,
        "extent": extent,
        "structure": structure,
        "dominant_location": region_name(cx, cy, W, H),
        "edge_heavy": edge_heavy,
        "hot_region_count": int(hot_count),
        "mid_region_count": int(mid_count),
        "largest_hot_region_share": round(largest_high_frac, 4),
        "largest_mid_region_share": round(largest_mid_frac, 4),
        "edge_hot_share": round(edge_share, 4),
        "high_threshold": round(float(high_thr), 4),
        "mid_threshold": mid_thr,
        "visual_note": "computed from E_main topology map only; no GT or provider answer used",
    }


@torch.no_grad()
def run_feta_for_group(model: FETAv0, grp: h5py.Group, device: str) -> np.ndarray:
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
    E = out["E"][0, :, :H, :W].detach().cpu().numpy()
    return np.maximum(E[0], E[1])


def output_path_for_sid(out_dir: Path, sid: str) -> Path:
    prefix = sid.split("_")[0]
    return out_dir / prefix / f"{sid}_topology_max.png"


def stats_path_for_sid(out_dir: Path, sid: str) -> Path:
    prefix = sid.split("_")[0]
    return out_dir / prefix / f"{sid}_topology_stats.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--limit", type=int, default=None, help="Limit source rows before collecting unique sample ids.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    rows = load_source_rows(args.source, args.limit)
    samples = collect_unique_samples(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = FETAv0(in_dim=1280, hidden=128, K=2, use_spatial_conv=True, desc_set="minimal")
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device).eval()

    handles: dict[str, h5py.File] = {}
    written = skipped = missing = 0
    try:
        for i, (sid, image_path) in enumerate(samples, 1):
            out_path = output_path_for_sid(args.out_dir, sid)
            stats_path = stats_path_for_sid(args.out_dir, sid)
            if out_path.exists() and stats_path.exists() and not args.force:
                skipped += 1
                continue
            prefix = sid.split("_")[0]
            h5_path = H5_BY_PREFIX.get(prefix)
            if h5_path is None:
                print(f"[missing_prefix] {sid}", file=sys.stderr)
                missing += 1
                continue
            if prefix not in handles:
                handles[prefix] = h5py.File(h5_path, "r")
            h5 = handles[prefix]
            if sid not in h5:
                print(f"[missing_sid] {sid} in {h5_path}", file=sys.stderr)
                missing += 1
                continue
            if not Path(image_path).exists():
                print(f"[missing_image] {image_path}", file=sys.stderr)
                missing += 1
                continue

            img = Image.open(image_path).convert("RGB")
            e_main = run_feta_for_group(model, h5[sid], args.device)
            topo = map_to_rgb(e_main, img.size, args.cmap)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            topo.save(out_path)
            stats = topology_visual_stats(e_main)
            stats.update({
                "sample_id": sid,
                "image_path": image_path,
                "map_path": str(out_path),
            })
            stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
            written += 1

            if i % 100 == 0 or i == len(samples):
                print(f"progress {i}/{len(samples)} written={written} skipped={skipped} missing={missing}", flush=True)
    finally:
        for h in handles.values():
            h.close()

    summary = {
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "source_rows": len(rows),
        "unique_samples": len(samples),
        "written": written,
        "skipped": skipped,
        "missing": missing,
        "device": args.device,
        "cmap": args.cmap,
    }
    summary_path = args.out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
