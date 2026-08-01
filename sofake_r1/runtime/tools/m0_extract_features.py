#!/usr/bin/env python3
"""
M0: extract intermediate visual features + patch masks from frozen
Qwen2.5-VL ViT.

For each input image, captures patch-token outputs at selected ViT blocks
via forward hooks, plus a binary patch_mask aligned to the ViT patch grid.

Selected layers (M0 v2 spec):
  [7, 11, 15, 19, 23, 27, 31]
  (full-attn boundaries are at 7, 15, 23, 31; windowed in between)

patch_mask:
  - REAL / FULL_SYNTHETIC -> all zeros
  - TAMPERED -> binarize source mask, smart-resize to ViT input H×W,
    bin into (H_patch × W_patch) cells of 14x14; patch=1 if coverage
    >= threshold (default 0.2)

Visual encoder: Qwen2_5_VLVisionTransformer at `model.model.visual`
  - 32 blocks
  - block output shape: (num_patches, 1280)
  - num_patches = T * H_patch * W_patch where (T, H_patch, W_patch) = grid_thw
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

LAYERS = [7, 11, 15, 19, 23, 27, 31]
PATCH_SIZE = 14
PATCH_MASK_THRESHOLD = 0.2


def compute_patch_mask(mask_path, grid_thw, patch_size=PATCH_SIZE,
                       threshold=PATCH_MASK_THRESHOLD):
    """Return uint8 patch_mask of length H_patch*W_patch."""
    T, H_patch, W_patch = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    n_patches = T * H_patch * W_patch
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros(n_patches, dtype=np.uint8)
    new_H = H_patch * patch_size
    new_W = W_patch * patch_size
    try:
        m = Image.open(mask_path).convert('RGB')
        m_resized = m.resize((new_W, new_H), Image.NEAREST)
        # Binarize: any RGB channel non-zero -> tampered. (Mask images may have
        # alpha that's 255 everywhere; convert('RGB') already drops alpha.)
        arr = np.asarray(m_resized).max(axis=-1)  # (new_H, new_W)
        binary = (arr > 0).astype(np.float32)
    except Exception:
        return np.zeros(n_patches, dtype=np.uint8)
    # Bin (H_patch, patch_size, W_patch, patch_size) -> (H_patch, W_patch)
    coverage = binary.reshape(H_patch, patch_size, W_patch, patch_size).mean(axis=(1, 3))
    patch_mask_2d = (coverage >= threshold).astype(np.uint8)
    # Replicate over T temporal patches (T=1 for still image, but be safe)
    pm = np.tile(patch_mask_2d.flatten(), T)
    return pm


def load_model(model_id: str, device: str = 'cuda', max_pixels: int = 224*224):
    # Recent transformers versions pre-allocate a large CUDA block while
    # loading. On 32GB cards this can OOM even though the model itself fits.
    try:
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    # Force the same smart_resize cap that ms-swift training used.
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size = {'longest_edge': max_pixels,
                                       'shortest_edge': max(28*28, max_pixels // 16)}
    return model, processor


def attach_hooks(visual, store: dict, layers):
    handles = []
    for layer_idx in layers:
        def make_hook(i):
            def hook(_module, _input, output):
                # output: (num_patches, hidden_dim=1280)
                store[i] = output.detach().to(torch.float16).cpu()
            return hook
        handles.append(visual.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return handles


def detach_hooks(handles):
    for h in handles:
        h.remove()


def extract_one(model, processor, img: Image.Image, store: dict,
                device: str = 'cuda'):
    """Run frozen ViT on one image. store gets filled by hooks."""
    img_inputs = processor.image_processor(images=[img], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(device, torch.bfloat16)
    grid_thw = img_inputs['image_grid_thw'].to(device)
    store.clear()
    with torch.no_grad():
        _ = model.model.visual(pixel_values, grid_thw=grid_thw)
    return grid_thw.cpu().numpy()[0]  # (T, H_patch, W_patch)


def main(args):
    samples = [json.loads(l) for l in open(args.samples)]
    if args.limit:
        samples = samples[:args.limit]
    print(f'samples: {len(samples)}', file=sys.stderr)

    model, processor = load_model(args.model, args.device, args.max_pixels)
    print(f'loaded {args.model}', file=sys.stderr)
    print(f'image_processor max_pixels={processor.image_processor.max_pixels}',
          file=sys.stderr)

    layers = [int(x) for x in args.layers]
    print(f'layers={layers}', file=sys.stderr)

    visual = model.model.visual
    store: dict = {}
    handles = attach_hooks(visual, store, layers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    with h5py.File(out_path, 'w') as h5:
        meta = h5.create_group('meta')
        meta.attrs['model'] = args.model
        meta.attrs['layers'] = np.array(layers, dtype=np.int32)
        meta.attrs['hidden_dim'] = 1280
        meta.attrs['max_pixels'] = args.max_pixels

        for i, samp in enumerate(samples):
            img_path = samp['image']
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f'  skip {img_path}: {e}', file=sys.stderr)
                n_skip += 1
                continue
            try:
                grid_thw = extract_one(model, processor, img, store, args.device)
            except Exception as e:
                print(f'  fail {img_path}: {e}', file=sys.stderr)
                n_skip += 1
                continue

            sid = samp.get('sample_id', f's{i:06d}')
            grp = h5.create_group(sid)
            grp.attrs['image_path'] = img_path
            grp.attrs['gt_cls'] = int(samp.get('gt_cls', -1))
            grp.attrs['grid_thw'] = grid_thw.astype(np.int32)
            for k, v in samp.items():
                if k in ('image', 'gt_cls', 'sample_id'):
                    continue
                if isinstance(v, (str, int, float, bool)):
                    grp.attrs[k] = v
                elif v is None:
                    grp.attrs[k] = ''
            # patch mask
            patch_mask = compute_patch_mask(
                samp.get('mask_path') or None,
                grid_thw,
                threshold=args.patch_mask_thr,
            )
            grp.create_dataset('patch_mask', data=patch_mask, compression='lzf')
            for layer_idx, feat in store.items():
                # feat shape: (num_patches, 1280)  fp16
                grp.create_dataset(f'layer_{layer_idx}', data=feat.numpy(),
                                   compression='lzf')
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f'  processed {i+1}/{len(samples)} (ok={n_ok} skip={n_skip})',
                      file=sys.stderr)

    detach_hooks(handles)
    print(f'\nDone: ok={n_ok}, skipped={n_skip}', file=sys.stderr)
    print(f'  -> {out_path}', file=sys.stderr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True,
                    help='jsonl with one record per image: {image, gt_cls, [source, mask, ...]}')
    ap.add_argument('--out', required=True, help='output HDF5 path')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--max_pixels', type=int, default=224 * 224)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--patch_mask_thr', type=float, default=PATCH_MASK_THRESHOLD)
    ap.add_argument('--layers', type=int, nargs='+', default=LAYERS,
                    help='ViT block indices to cache; default keeps the M0 v2 layer set.')
    args = ap.parse_args()
    main(args)
