#!/usr/bin/env python3
"""
FETA v0 — Forensic Evidence Topology Adapter (minimal-prior version).

Architecture:
  Frozen ViT features at single layer L (cached)
    ↓ LN + Linear(1280 → hidden)
    ↓ reshape to grid (B, hidden, H, W)
    ↓ optional 3x3 conv for spatial smoothness
  ──────────────────────────────────────────
    ↓ K parallel Conv2d(hidden → 1) evidence heads
    ↓ sigmoid → K evidence maps E ∈ [0,1]^(B,K,H,W)
  ──────────────────────────────────────────
    ↓ compute K × 6 differentiable descriptors per sample
    ↓ flatten → (B, K*6)
    ↓ small MLP (K*6 → 64 → 3)
    ↓ class logits

Losses (v0 minimal-prior):
  L = 1.0 * L_cls
    + 0.2 * L_real_low_mass         # REAL: all evidence maps low
    + 0.5 * L_tamp_mask_align       # TAMP: best matching map aligns gt mask
    + 0.2 * L_fs_active             # FS: at least one map must be active (hinge)

NOT included (kept as DIAGNOSTIC only, not loss):
  - FS entropy/diffuse regularizer
  - TAMP > FS concentration ranking
  - K-map diversity
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import functools

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# L0 fix: undo Qwen2.5-VL ViT window-permutation
# ============================================================
# Cached features were captured by a forward hook on visual.blocks[19], i.e.
# in Qwen's *window-permuted* token order (the ViT reorders tokens by
# window_index before the blocks and only restores raster order after the
# merger). FETA's `view(h, w, D)` assumes raster order, so the spatial grid is
# scrambled relative to both the image and the (raster) patch_mask. This helper
# reverses that permutation back to raster patch order. See
# memory/feta_window_permutation_bug.md.
QWEN_WINDOW_SIZE, QWEN_PATCH_SIZE, QWEN_MERGE = 112, 14, 2


@functools.lru_cache(maxsize=256)
def get_window_index(T, H, W):
    """Faithful replica of Qwen2_5_VL get_window_index. Returns a numpy
    permutation (length T*Hu*Wu) giving the raster merged-unit id at each
    window slot."""
    vit_win = QWEN_WINDOW_SIZE // QWEN_MERGE // QWEN_PATCH_SIZE   # 4
    Hu, Wu = H // QWEN_MERGE, W // QWEN_MERGE
    index = torch.arange(T * Hu * Wu).reshape(T, Hu, Wu)
    pad_h = vit_win - Hu % vit_win
    pad_w = vit_win - Wu % vit_win
    nwh = (Hu + pad_h) // vit_win
    nww = (Wu + pad_w) // vit_win
    ip = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
    ip = ip.reshape(T, nwh, vit_win, nww, vit_win)
    ip = ip.permute(0, 1, 3, 2, 4).reshape(T, nwh * nww, vit_win, vit_win).reshape(-1)
    return ip[ip != -100].numpy().astype(np.int64)


def unpermute_feat_to_raster(feat, T, H, W):
    """feat: (T*H*W, D) in Qwen window order -> raster patch order (T*H*W, D).

    Window reorder acts on merged-units (groups of MERGE*MERGE patches); within
    a unit the sub-patches are row-major (mh, mw). We undo the unit permutation
    and then lay the 2x2 sub-patches back into the full patch grid."""
    m = QWEN_MERGE
    Hu, Wu = H // m, W // m
    n_units = T * Hu * Wu
    if feat.shape[0] != n_units * m * m:
        return feat  # unexpected shape; no-op (safety)
    D = feat.shape[-1]
    win = get_window_index(T, H, W)
    rev = torch.from_numpy(np.argsort(win)).long()
    units = feat.reshape(n_units, m * m, D)[rev]          # raster super-cell order
    g = units.reshape(T, Hu, Wu, m, m, D).permute(0, 1, 3, 2, 4, 5).contiguous()
    return g.reshape(T * H * W, D)


# ============================================================
# Dataset
# ============================================================
class FETADataset(Dataset):
    def __init__(self, h5_path, layer, unpermute=False):
        self.h5_path = h5_path
        self.layer = layer
        self.unpermute = unpermute
        with h5py.File(h5_path, 'r') as f:
            self.sids = sorted([k for k in f.keys() if k != 'meta'])
        self._handle = None

    def _open(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        self._open()
        sid = self.sids[idx]
        grp = self._handle[sid]
        feat = torch.from_numpy(grp[f'layer_{self.layer}'][:]).float()
        patch_mask = torch.from_numpy(grp['patch_mask'][:]).float()
        gt_cls = int(grp.attrs['gt_cls'])
        grid_thw = grp.attrs['grid_thw']
        T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        if self.unpermute:
            # window order (as captured at block19) -> raster, to match
            # patch_mask (raster) and FETA's view(h,w) assumption.
            feat = unpermute_feat_to_raster(feat, T, H, W)
        source = str(grp.attrs.get('source', 'unknown'))
        return {'feat': feat, 'patch_mask': patch_mask, 'gt_cls': gt_cls,
                'T': T, 'H': H, 'W': W, 'source': source, 'sid': sid}


def collate(batch):
    max_len = max(b['feat'].shape[0] for b in batch)
    B = len(batch)
    D = batch[0]['feat'].shape[1]
    feats = torch.zeros(B, max_len, D)
    pm = torch.zeros(B, max_len)
    attn = torch.zeros(B, max_len)
    cls = torch.zeros(B, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    Hs = torch.zeros(B, dtype=torch.long)
    Ws = torch.zeros(B, dtype=torch.long)
    Ts = torch.zeros(B, dtype=torch.long)
    sources, sids = [], []
    for i, b in enumerate(batch):
        n = b['feat'].shape[0]
        feats[i, :n] = b['feat']
        pm[i, :n] = b['patch_mask']
        attn[i, :n] = 1.0
        cls[i] = b['gt_cls']; lens[i] = n
        Hs[i] = b['H']; Ws[i] = b['W']; Ts[i] = b['T']
        sources.append(b['source']); sids.append(b['sid'])
    return {'feat': feats, 'patch_mask': pm, 'attn': attn, 'gt_cls': cls,
            'lens': lens, 'H': Hs, 'W': Ws, 'T': Ts,
            'sources': sources, 'sids': sids}


# ============================================================
# FETA v0 model
# ============================================================
class FETAv0(nn.Module):
    def __init__(self, in_dim=1280, hidden=128, K=2, use_spatial_conv=True,
                 desc_set='legacy6', use_binary_head=False,
                 use_feature_binary_head=False):
        super().__init__()
        self.K = K
        self.desc_set = desc_set
        self.use_binary_head = use_binary_head
        self.use_feature_binary_head = use_feature_binary_head
        if desc_set not in DESC_SETS:
            raise ValueError(f'unknown desc_set: {desc_set}; choose from {list(DESC_SETS)}')
        self.desc_names = DESC_SETS[desc_set]
        self.desc_idx = torch.tensor([DESC_INDEX[n] for n in self.desc_names], dtype=torch.long)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.use_spatial_conv = use_spatial_conv
        if use_spatial_conv:
            self.spatial_conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
            self.act = nn.GELU()
        self.evidence_heads = nn.ModuleList([
            nn.Conv2d(hidden, 1, kernel_size=1) for _ in range(K)
        ])
        self.n_descs = len(self.desc_names)
        self.classifier = nn.Sequential(
            nn.Linear(K * self.n_descs, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        if use_binary_head:
            bin_in_dim = K * self.n_descs
            if use_feature_binary_head:
                # descriptor stats + global avg/max pooled grid features
                # + one evidence-weighted pooled feature per evidence map.
                bin_in_dim += (2 + K) * hidden
            self.binary_classifier = nn.Sequential(
                nn.Linear(bin_in_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )

    def forward(self, feat, attn, Ts, Hs, Ws):
        """
        feat: (B, L_max, 1280) padded
        attn: (B, L_max) mask
        Ts/Hs/Ws: per-sample grid dims
        Returns:
          logits (B, 3),
          E (B, K, H_max, W_max),
          attn_grid (B, H_max, W_max),
          descriptors (B, K, 6)
        """
        x = self.proj(self.norm(feat))  # (B, L_max, hidden)
        B, L_max, D = x.shape
        H_max = int(Hs.max().item())
        W_max = int(Ws.max().item())
        T_max = int(Ts.max().item())
        assert T_max == 1, f'multi-frame not supported (T={T_max})'

        # Scatter to grid + attn_grid
        grid = x.new_zeros(B, D, H_max, W_max)
        attn_grid = x.new_zeros(B, H_max, W_max)
        for i in range(B):
            h, w = int(Hs[i].item()), int(Ws[i].item())
            grid[i, :, :h, :w] = x[i, :h * w].view(h, w, D).permute(2, 0, 1)
            attn_grid[i, :h, :w] = 1.0

        if self.use_spatial_conv:
            grid = self.act(self.spatial_conv(grid))

        # K evidence maps
        E_logits = torch.cat([head(grid) for head in self.evidence_heads], dim=1)
        # (B, K, H_max, W_max)
        E = torch.sigmoid(E_logits)
        # Zero out invalid positions
        E = E * attn_grid.unsqueeze(1)

        descriptors_all = compute_descriptors_torch(E, attn_grid)  # (B, K, 8)
        # Slice to selected desc set for classifier input
        idx = self.desc_idx.to(descriptors_all.device)
        descriptors_used = descriptors_all.index_select(-1, idx)  # (B, K, n_descs)
        flat_desc = descriptors_used.flatten(1)
        logits = self.classifier(flat_desc)     # (B, 3)
        binary_logits = None
        if self.use_binary_head:
            if self.use_feature_binary_head:
                valid = attn_grid.unsqueeze(1)  # (B,1,H,W)
                denom = valid.sum(dim=(-1, -2)).clamp_min(1.0)  # (B,1)
                gap = (grid * valid).sum(dim=(-1, -2)) / denom  # (B,D)

                neg_inf = torch.finfo(grid.dtype).min
                grid_masked = grid.masked_fill(valid == 0, neg_inf)
                gmp = grid_masked.amax(dim=(-1, -2))             # (B,D)

                map_pools = []
                for k in range(self.K):
                    wk = E[:, k:k + 1] * valid                   # (B,1,H,W)
                    wk_denom = wk.sum(dim=(-1, -2)).clamp_min(1e-6)
                    map_pools.append((grid * wk).sum(dim=(-1, -2)) / wk_denom)
                binary_input = torch.cat([flat_desc, gap, gmp] + map_pools, dim=-1)
            else:
                binary_input = flat_desc
            binary_logits = self.binary_classifier(binary_input)
        return {
            'logits': logits,
            'binary_logits': binary_logits,
            'E': E,
            'attn_grid': attn_grid,
            'descriptors': descriptors_used,
            'descriptors_all': descriptors_all,  # for diagnostic / loss (need mass at index 0)
        }


# ============================================================
# Differentiable descriptors (torch)
# ============================================================
# Canonical order of all descriptors. Indexed by DESC_INDEX.
DESC_NAMES = ['mass', 'mean', 'entropy', 'concentration', 'peak', 'peak_ratio', 'area', 'spread',
              'log_total_response', 'effective_area', 'gini_concentration']
DESC_INDEX = {n: i for i, n in enumerate(DESC_NAMES)}

# Standard FETA descriptor recipe ('minimal') selected from the L19 ablation
# (descriptor_ablation_2026_05_26). Each desc has a fixed forensic semantic:
#   mass          — total evidence amount
#   peak          — strongest single-patch evidence
#   entropy       — diffuseness of evidence
#   concentration — localization / dominance (top quartile share)
#   area          — spatial extent above 0.5
# Strength alone or topology alone is each ~3pp below this combination; full
# 8-desc set is 1.7pp worse (peak_ratio / mean / spread are redundant / noise).
DESC_SETS = {
    'minimal':     ['mass', 'entropy', 'concentration', 'peak', 'area'],
    # First controlled grid-normalization ablation: replace raw-sum mass with
    # size-normalized mean evidence; keep the other four descriptors unchanged.
    'minimal_mean': ['mean', 'entropy', 'concentration', 'peak', 'area'],
    'full':        ['mass', 'mean', 'entropy', 'concentration', 'peak', 'peak_ratio', 'area', 'spread'],
    'strength':    ['mass', 'peak'],
    'topology':    ['entropy', 'concentration', 'area'],
    'no_topology': ['mass', 'peak', 'mean'],
    # Selected by descriptor-discovery on frozen evidence maps: compressed
    # response strength + peak strength + effective support + inequality.
    # The discovery stage may inspect OOD/Chameleon labels, but model fitting
    # with this set must still use SoFake train only.
    'discovery_greedy4': ['log_total_response', 'gini_concentration',
                          'peak', 'effective_area'],
    'baseline_plus_greedy': ['mass', 'entropy', 'concentration', 'peak', 'area',
                             'log_total_response', 'gini_concentration',
                             'effective_area'],
    # Original 6-desc set from earlier runs (the old "mass" was actually
    # the size-normalized mean; we map it to 'mean' to reproduce that behavior).
    'legacy6':     ['mean', 'peak', 'area', 'entropy', 'peak_ratio', 'spread'],
}


def compute_gini_torch(E, attn_grid, eps=1e-6):
    """Gini inequality of valid evidence values for each map.

    This is intentionally per-sample so variable Qwen grids do not get biased
    by padded zeros. Sort is piecewise differentiable and adequate for this
    small-grid descriptor.
    """
    B, K, _, _ = E.shape
    vals_out = []
    for i in range(B):
        valid = attn_grid[i].bool()
        per_k = []
        for k in range(K):
            vals = E[i, k][valid].flatten()
            n = vals.numel()
            if n == 0:
                per_k.append(E.new_zeros(()))
                continue
            total = vals.sum()
            sorted_vals, _ = vals.sort()
            ranks = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
            gini = (2.0 * (ranks * sorted_vals).sum() / (n * total + eps)) - ((n + 1.0) / n)
            per_k.append(gini.clamp(0.0, 1.0))
        vals_out.append(torch.stack(per_k, dim=0))
    return torch.stack(vals_out, dim=0)


def compute_descriptors_torch(E, attn_grid, eps=1e-8):
    """
    E: (B, K, H, W) sigmoid evidence ∈ [0,1], already zeroed on invalid.
    attn_grid: (B, H, W) ∈ {0,1}
    Returns: (B, K, len(DESC_NAMES)) in canonical DESC_NAMES order.

    Semantics:
      mass         = sum(E) over valid              (raw integral; scales with grid)
      mean         = sum(E) / valid_count           (size-normalized strength)
      entropy      = -sum(p log p), p = E / sum(E)  (spatial distribution)
      concentration= top-quartile sum / total sum   (peakedness)
      peak         = max(E) over valid
      peak_ratio   = peak / mean
      area         = soft-threshold(>0.5) fraction
      spread       = mean of std(x,y) weighted by E (spatial extent)
    """
    B, K, H, W = E.shape
    valid_count = attn_grid.sum(dim=(-1, -2)).clamp_min(1).unsqueeze(1)  # (B, 1)
    valid_4d = attn_grid.unsqueeze(1).expand(-1, K, -1, -1)              # (B, K, H, W)

    E_sum_2d = E.sum(dim=(-1, -2))                # (B, K)
    mass = E_sum_2d                               # raw sum
    mean = E_sum_2d / valid_count                 # size-normalized

    peak = E.amax(dim=(-1, -2)).clamp_min(0)
    peak_ratio = peak / (mean + eps)

    # area: soft threshold via sigmoid sharpening
    above = torch.sigmoid((E - 0.5) * 10) * valid_4d
    area = above.sum(dim=(-1, -2)) / valid_count

    # entropy
    Ev_sum_safe = E_sum_2d.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    p = E / Ev_sum_safe
    log_p = torch.log(p + eps)
    entropy = -(p * log_p * valid_4d).sum(dim=(-1, -2))

    # concentration: top quartile sum / total sum (differentiable via topk)
    N = H * W
    k_top = max(1, N // 4)
    E_flat = E.flatten(2)                          # (B, K, N)
    top_vals = E_flat.topk(k_top, dim=-1).values   # (B, K, k_top)
    concentration = top_vals.sum(dim=-1) / E_sum_2d.clamp_min(eps)

    # spread: weighted std of normalized coords
    device = E.device
    h_idx = torch.linspace(0, 1, H, device=device).float() if H > 1 else torch.zeros(H, device=device)
    w_idx = torch.linspace(0, 1, W, device=device).float() if W > 1 else torch.zeros(W, device=device)
    yy = h_idx.view(1, 1, H, 1).expand(B, K, H, W)
    xx = w_idx.view(1, 1, 1, W).expand(B, K, H, W)
    Ev_norm_sum = E_sum_2d.clamp_min(eps)
    mu_x = (xx * E).sum(dim=(-1, -2)) / Ev_norm_sum
    mu_y = (yy * E).sum(dim=(-1, -2)) / Ev_norm_sum
    var_x = ((xx - mu_x.unsqueeze(-1).unsqueeze(-1)) ** 2 * E).sum(dim=(-1, -2)) / Ev_norm_sum
    var_y = ((yy - mu_y.unsqueeze(-1).unsqueeze(-1)) ** 2 * E).sum(dim=(-1, -2)) / Ev_norm_sum
    spread = (var_x.clamp_min(0).sqrt() + var_y.clamp_min(0).sqrt()) / 2

    log_total_response = torch.log1p(mass.clamp_min(0))
    effective_area = torch.exp(entropy.clamp(max=50.0)) / valid_count
    gini_concentration = compute_gini_torch(E, attn_grid, eps=1e-6)

    # stack in canonical order
    return torch.stack([mass, mean, entropy, concentration,
                        peak, peak_ratio, area, spread,
                        log_total_response, effective_area, gini_concentration], dim=-1)


# ============================================================
# Loss
# ============================================================
def feta_v0_loss(out, batch, tau_mass=0.05, fs_loss_kind='hinge',
                 w_cls=1.0, w_real=0.2, w_tamp=0.5, w_fs=0.2,
                 w_bin=0.0):
    """fs_loss_kind: 'hinge' (relu(tau - max_mass)), 'l2' (relu(tau - max_mass)**2),
                     'bce' (BCE on mean of E for the K with max mass toward 1)."""
    logits = out['logits']
    E = out['E']
    attn_grid = out['attn_grid']
    descriptors_all = out['descriptors_all']  # (B, K, 8) in canonical order
    cls = batch['gt_cls']
    gt_mask_flat = batch['patch_mask']  # (B, L_max)

    L_cls = F.cross_entropy(logits, cls)

    if w_bin > 0:
        if out.get('binary_logits') is not None:
            binary_logits = out['binary_logits']
        else:
            binary_logits = torch.stack(
                [logits[:, 0], torch.logsumexp(logits[:, 1:], dim=-1)],
                dim=-1,
            )
        binary_target = (cls != 0).long()
        L_bin = F.cross_entropy(binary_logits, binary_target)
    else:
        L_bin = logits.new_zeros(())

    # Size-normalized strength ("mean" in canonical order) drives REAL/FS priors;
    # this matches the semantics of the original "mass" in pre-ablation runs.
    mass = descriptors_all[..., DESC_INDEX['mean']]  # (B, K)

    is_real = (cls == 0); is_fs = (cls == 1); is_tamp = (cls == 2)

    if w_real > 0 and is_real.any():
        L_real = (mass[is_real] ** 2).mean()
    else:
        L_real = mass.new_zeros(())

    # TAMP mask alignment: need patch_mask as grid (B, H, W)
    if w_tamp > 0 and is_tamp.any():
        B, K, H, W = E.shape
        # Build gt_mask_grid from flat patch_mask
        gt_mask_grid = E.new_zeros(B, H, W)
        for i in range(B):
            if cls[i].item() == 2:
                h_i = int(attn_grid[i].sum(dim=-1).max().item() if attn_grid[i].any() else H)
                # Reconstruct H_i, W_i from attn_grid: rows where attn_grid[i] != 0
                # easier: use shape from where attn is 1
                pass
        # Actually use the existing Hs, Ws from batch (we have them in batch dict)
        Hs = batch['H']; Ws = batch['W']
        for i in range(B):
            if cls[i].item() == 2:
                h_i, w_i = int(Hs[i].item()), int(Ws[i].item())
                n_i = h_i * w_i
                gt_mask_grid[i, :h_i, :w_i] = gt_mask_flat[i, :n_i].view(h_i, w_i)

        E_t = E[is_tamp]  # (Bt, K, H, W)
        gt_t = gt_mask_grid[is_tamp]  # (Bt, H, W)
        attn_t = attn_grid[is_tamp]  # (Bt, H, W)
        per_k_loss = []
        for k in range(K):
            Ek = E_t[:, k].clamp(1e-6, 1 - 1e-6)
            bce = -(gt_t * torch.log(Ek) + (1 - gt_t) * torch.log(1 - Ek))
            bce = (bce * attn_t).sum(dim=(-1, -2)) / attn_t.sum(dim=(-1, -2)).clamp_min(1)
            inter = (Ek * gt_t * attn_t).sum(dim=(-1, -2))
            denom = (Ek * attn_t).sum(dim=(-1, -2)) + (gt_t * attn_t).sum(dim=(-1, -2))
            dice = 1 - (2 * inter / denom.clamp_min(1e-6))
            per_k_loss.append(bce + dice)
        per_k_loss = torch.stack(per_k_loss, dim=-1)  # (Bt, K)
        L_tamp = per_k_loss.min(dim=-1).values.mean()
    else:
        L_tamp = mass.new_zeros(())

    # FS: at least one map should be active. Three options of increasing strength:
    #   hinge: relu(tau - max_mass)        — zero gradient once max_mass >= tau
    #   l2   : relu(tau - max_mass)^2      — smooth, still zero above tau
    #   bce  : BCE pulling the most-active map's mean toward 1 (always-on gradient)
    if w_fs > 0 and is_fs.any():
        mass_fs = mass[is_fs]              # (Bfs, K)
        mass_fs_max, k_arg = mass_fs.max(dim=-1)  # (Bfs,), (Bfs,)
        if fs_loss_kind == 'hinge':
            L_fs = F.relu(tau_mass - mass_fs_max).mean()
        elif fs_loss_kind == 'l2':
            L_fs = (F.relu(tau_mass - mass_fs_max) ** 2).mean()
        elif fs_loss_kind == 'bce':
            # BCE on mean activation of the most-active map per FS sample,
            # pulled toward 1.0. Uses the differentiable mass values directly.
            target = torch.ones_like(mass_fs_max)
            # mass_fs_max is the actual mean of E over the chosen map → clamp + BCE
            m = mass_fs_max.clamp(1e-6, 1 - 1e-6)
            L_fs = -(target * torch.log(m) + (1 - target) * torch.log(1 - m)).mean()
        else:
            raise ValueError(f'unknown fs_loss_kind: {fs_loss_kind}')
    else:
        L_fs = mass.new_zeros(())

    L = w_cls * L_cls + w_bin * L_bin + w_real * L_real + w_tamp * L_tamp + w_fs * L_fs
    parts = {'cls': L_cls.item(), 'real': L_real.item(),
             'tamp': L_tamp.item(), 'fs': L_fs.item(), 'bin': L_bin.item()}
    return L, parts


# ============================================================
# Diagnostic — full eval
# ============================================================
def compute_descriptors_numpy(E_flat, H, W):
    """numpy version for diagnostic. Same names as torch version."""
    eps = 1e-8
    n = H * W
    e = E_flat[:n].numpy() if isinstance(E_flat, torch.Tensor) else E_flat[:n]

    e_sum = float(e.sum())
    mass = e_sum                       # raw integral
    mean_e = e_sum / max(n, 1)         # size-normalized

    if e_sum > eps:
        p = e / (e_sum + eps)
        entropy = float(-(p * np.log(p + eps)).sum())
    else:
        entropy = 0.0

    k_top = max(1, n // 4)
    top_vals = np.partition(e, -k_top)[-k_top:]
    concentration = float(top_vals.sum() / (e_sum + eps))

    peak = float(e.max())
    peak_ratio = float(peak / (mean_e + eps))
    area = float((e >= 0.5).sum() / n)

    xs = np.repeat(np.linspace(0, 1, W)[None, :], H, axis=0).flatten() if W > 1 else np.zeros(n)
    ys = np.repeat(np.linspace(0, 1, H)[:, None], W, axis=1).flatten() if H > 1 else np.zeros(n)
    if e_sum > eps:
        mu_x = (xs * e).sum() / e_sum
        mu_y = (ys * e).sum() / e_sum
        sx = np.sqrt(((xs - mu_x) ** 2 * e).sum() / e_sum)
        sy = np.sqrt(((ys - mu_y) ** 2 * e).sum() / e_sum)
        spread = float((sx + sy) / 2)
    else:
        spread = 0.0

    return {'mass': mass, 'mean': mean_e, 'peak': peak, 'area': area,
            'entropy': entropy, 'concentration': concentration,
            'peak_ratio': peak_ratio, 'spread': spread}


def bbox_from_evidence_map(e_grid, threshold=0.5):
    mask = e_grid >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    aw = max(0, ax2 - ax1); ah = max(0, ay2 - ay1)
    bw = max(0, bx2 - bx1); bh = max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def evaluate(model, ds, device='cuda', batch_size=32, num_workers=2):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    preds, gts, sources = [], [], []
    # per-class descriptor accumulators
    per_class_desc = defaultdict(lambda: defaultdict(list))
    # bbox / patch IoU on TAMP
    tamp_patch_iou = []
    tamp_bbox_iou = []

    for batch in loader:
        feat = batch['feat'].to(device)
        attn = batch['attn'].to(device)
        out = model(feat, attn, batch['T'], batch['H'], batch['W'])
        logits = out['logits']
        E = out['E']  # (B, K, H, W) on device

        pred = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(pred); gts.extend(batch['gt_cls'].tolist())
        sources.extend(batch['sources'])

        # Per-sample descriptor (numpy, with concentration)
        E_cpu = E.cpu()
        for i in range(E.size(0)):
            cls_i = int(batch['gt_cls'][i].item())
            h_i, w_i = int(batch['H'][i].item()), int(batch['W'][i].item())
            # Best k = which has highest activation (for representative descriptors)
            # Save all K descriptors for diagnostic
            for k in range(model.K):
                e_grid = E_cpu[i, k, :h_i, :w_i].numpy()
                e_flat = e_grid.flatten()
                d = compute_descriptors_numpy(torch.from_numpy(e_flat), h_i, w_i)
                for key, val in d.items():
                    per_class_desc[cls_i][f'k{k}_{key}'].append(val)

            # For TAMP, compute IoU with best-matching k
            if cls_i == 2:
                gt_mask_flat = batch['patch_mask'][i, :h_i * w_i].numpy()
                gt_grid = gt_mask_flat.reshape(h_i, w_i)
                gt_bbox = bbox_from_evidence_map(gt_grid, 0.5)
                best_iou = 0; best_bbox_iou = 0
                for k in range(model.K):
                    e_grid = E_cpu[i, k, :h_i, :w_i].numpy()
                    pred_bin = (e_grid >= 0.5).astype(np.float32)
                    inter = (pred_bin * gt_grid).sum()
                    union = ((pred_bin + gt_grid) > 0).astype(np.float32).sum()
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                    pred_bbox = bbox_from_evidence_map(e_grid, 0.5)
                    biou = bbox_iou(pred_bbox, gt_bbox)
                    if biou > best_bbox_iou:
                        best_bbox_iou = biou
                tamp_patch_iou.append(best_iou)
                tamp_bbox_iou.append(best_bbox_iou)

    cm = np.zeros((3, 3), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    acc = float(np.trace(cm) / max(1, cm.sum()))
    recall = [float(cm[c, c] / max(1, cm[c].sum())) for c in (0, 1, 2)]
    flip = {
        'TAMP->FS': float(cm[2, 1] / max(1, cm[2].sum())),
        'FS->TAMP': float(cm[1, 2] / max(1, cm[1].sum())),
        'REAL->TAMP': float(cm[0, 2] / max(1, cm[0].sum())),
    }

    # Per-class descriptor summary + Cohen's d
    class_names = {0: 'REAL', 1: 'FS', 2: 'TAMP'}
    desc_keys = sorted(per_class_desc[0].keys()) if per_class_desc else []
    desc_summary = {}
    cohen_d = {}
    for key in desc_keys:
        per_cls_vals = {c: np.array(per_class_desc[c][key]) for c in (0, 1, 2)}
        desc_summary[key] = {
            class_names[c]: {
                'mean': float(per_cls_vals[c].mean()) if len(per_cls_vals[c]) else 0,
                'median': float(np.median(per_cls_vals[c])) if len(per_cls_vals[c]) else 0,
                'q25': float(np.percentile(per_cls_vals[c], 25)) if len(per_cls_vals[c]) else 0,
                'q75': float(np.percentile(per_cls_vals[c], 75)) if len(per_cls_vals[c]) else 0,
            } for c in (0, 1, 2)
        }
        # Cohen's d between class pairs
        for c1, c2 in [(0, 1), (1, 2), (0, 2)]:
            n1, n2 = class_names[c1], class_names[c2]
            v1, v2 = per_cls_vals[c1], per_cls_vals[c2]
            if len(v1) > 1 and len(v2) > 1:
                pooled = np.sqrt((v1.var() + v2.var()) / 2 + 1e-8)
                d = float((v1.mean() - v2.mean()) / pooled)
            else:
                d = 0.0
            cohen_d[f'{key}__{n1}_vs_{n2}'] = d

    return {
        'n': len(preds),
        'acc': acc, 'recall': recall, 'flip': flip, 'cm': cm.tolist(),
        'desc_summary': desc_summary,
        'cohen_d': cohen_d,
        'tamp_patch_iou_mean': float(np.mean(tamp_patch_iou)) if tamp_patch_iou else 0,
        'tamp_bbox_iou_mean': float(np.mean(tamp_bbox_iou)) if tamp_bbox_iou else 0,
        'n_tamp_eval': len(tamp_patch_iou),
    }


# ============================================================
# Training
# ============================================================
def train_feta(model, train_ds, lr=1e-3, batch_size=32, epochs=5,
               device='cuda', num_workers=2,
               w_cls=1.0, w_real=0.2, w_tamp=0.5, w_fs=0.2,
               fs_loss_kind='hinge', tau_mass=0.05, w_bin=0.0):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate, drop_last=False)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        sums = defaultdict(float); n_tot = 0; correct = 0
        for batch in loader:
            feat = batch['feat'].to(device)
            attn = batch['attn'].to(device)
            cls = batch['gt_cls'].to(device)
            batch_dev = dict(batch)
            batch_dev['patch_mask'] = batch['patch_mask'].to(device)
            batch_dev['gt_cls'] = cls
            out = model(feat, attn, batch['T'], batch['H'], batch['W'])
            L, parts = feta_v0_loss(out, batch_dev,
                                    tau_mass=tau_mass, fs_loss_kind=fs_loss_kind,
                                    w_cls=w_cls, w_real=w_real,
                                    w_tamp=w_tamp, w_fs=w_fs,
                                    w_bin=w_bin)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                sums[k] += v * cls.size(0)
            n_tot += cls.size(0)
            correct += (out['logits'].argmax(dim=-1) == cls).sum().item()
        sched.step()
        s = '  '.join(f'{k}={sums[k]/n_tot:.4f}' for k in sums)
        print(f'  ep {ep+1}/{epochs}  {s}  train_acc={correct/n_tot:.2%}  '
              f'({time.time()-t0:.0f}s)', file=sys.stderr)
    return model


def main(args):
    device = args.device
    print(f'\n=== FETA v0  layer={args.layer}  K={args.K} ===', file=sys.stderr)

    train_ds = FETADataset(args.train_h5, args.layer, unpermute=args.unpermute)
    val_in_ds = FETADataset(args.val_in_h5, args.layer, unpermute=args.unpermute)
    val_ood_ds = FETADataset(args.val_ood_h5, args.layer, unpermute=args.unpermute)
    print(f'  unpermute(L0 window->raster) = {args.unpermute}', file=sys.stderr)
    print(f'  train={len(train_ds)} val_in={len(val_in_ds)} val_ood={len(val_ood_ds)}',
          file=sys.stderr)

    torch.manual_seed(args.seed)
    model = FETAv0(in_dim=1280, hidden=args.hidden, K=args.K,
                   use_spatial_conv=args.spatial_conv, desc_set=args.desc_set,
                   use_binary_head=args.binary_head,
                   use_feature_binary_head=args.feature_binary_head)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  desc_set={args.desc_set}  n_descs={model.n_descs}  '
          f'desc_names={model.desc_names}', file=sys.stderr)
    print(f'  FETA params: {n_params/1e6:.3f}M', file=sys.stderr)

    model = train_feta(model, train_ds, lr=args.lr, batch_size=args.batch_size,
                       epochs=args.epochs, device=device, num_workers=args.num_workers,
                       w_cls=args.w_cls, w_real=args.w_real,
                       w_tamp=args.w_tamp, w_fs=args.w_fs,
                       fs_loss_kind=args.fs_loss_kind, tau_mass=args.tau_mass,
                       w_bin=args.w_bin)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''
    torch.save(model.state_dict(), out_dir / f'feta_v0_L{args.layer}{suffix}.pt')

    results = {}
    for split, ds in [('val_in', val_in_ds), ('val_ood', val_ood_ds)]:
        r = evaluate(model, ds, device=device, num_workers=args.num_workers)
        results[split] = r
        print(f'\n  ---- {split} ----', file=sys.stderr)
        print(f'  acc={r["acc"]:.2%}  R:{r["recall"][0]:.2%} F:{r["recall"][1]:.2%} '
              f'T:{r["recall"][2]:.2%}  T→F:{r["flip"]["TAMP->FS"]:.2%}  '
              f'R→T:{r["flip"]["REAL->TAMP"]:.2%}',
              file=sys.stderr)
        print(f'  TAMP patch IoU: {r["tamp_patch_iou_mean"]:.3f}  '
              f'bbox IoU: {r["tamp_bbox_iou_mean"]:.3f}', file=sys.stderr)

        # Print descriptor table (median by class, both K; covers all 8 descriptors)
        print(f'  --- descriptor table (median across class, both K) ---', file=sys.stderr)
        for d_name in ['mass', 'mean', 'entropy', 'concentration', 'peak', 'peak_ratio', 'area', 'spread']:
            for k in range(model.K):
                key = f'k{k}_{d_name}'
                if key not in r['desc_summary']: continue
                line = f'  k{k} {d_name:14s}'
                for cls in ['REAL', 'FS', 'TAMP']:
                    s = r['desc_summary'][key][cls]
                    line += f'  {cls}={s["median"]:7.3f}'
                d_rf = r['cohen_d'].get(f'{key}__REAL_vs_FS', 0)
                d_ft = r['cohen_d'].get(f'{key}__FS_vs_TAMP', 0)
                line += f'  | d(R,F)={d_rf:+.2f}  d(F,T)={d_ft:+.2f}'
                print(line, file=sys.stderr)
        # Topology direction checks
        for k in range(model.K):
            ent_fs = r['desc_summary'][f'k{k}_entropy']['FS']['median']
            ent_t = r['desc_summary'][f'k{k}_entropy']['TAMP']['median']
            conc_fs = r['desc_summary'][f'k{k}_concentration']['FS']['median']
            conc_t = r['desc_summary'][f'k{k}_concentration']['TAMP']['median']
            print(f'  k{k} direction: FS entropy {ent_fs:.3f} {">"if ent_fs>ent_t else "<"} TAMP {ent_t:.3f}  '
                  f'| FS conc {conc_fs:.3f} {">"if conc_fs>conc_t else "<"} TAMP {conc_t:.3f}',
                  file=sys.stderr)

    out_json = out_dir / f'feta_v0_L{args.layer}{suffix}_results.json'
    with open(out_json, 'w') as f:
        json.dump({'layer': args.layer, 'K': args.K, 'tag': args.tag,
                   'hidden': args.hidden, 'epochs': args.epochs,
                   'w_cls': args.w_cls, 'w_real': args.w_real,
                   'w_tamp': args.w_tamp, 'w_fs': args.w_fs,
                   'w_bin': args.w_bin,
                   'fs_loss_kind': args.fs_loss_kind, 'tau_mass': args.tau_mass,
                   'desc_set': args.desc_set, 'desc_names': model.desc_names,
                   'binary_head': args.binary_head,
                   'feature_binary_head': args.feature_binary_head,
                   'results': results}, f, indent=2)
    print(f'  -> {out_json}', file=sys.stderr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_h5', required=True)
    ap.add_argument('--val_in_h5', required=True)
    ap.add_argument('--val_ood_h5', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--spatial_conv', type=int, default=1)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    # loss-weight knobs
    ap.add_argument('--w_cls', type=float, default=1.0)
    ap.add_argument('--w_real', type=float, default=0.2)
    ap.add_argument('--w_tamp', type=float, default=0.5)
    ap.add_argument('--w_fs', type=float, default=0.2)
    ap.add_argument('--w_bin', type=float, default=0.0,
                    help='Auxiliary REAL-vs-FAKE CE weight. Uses binary logits '
                         '[REAL, logsumexp(FULL_SYNTHETIC,TAMPERED)] and keeps '
                         'the original 3-way classifier unchanged when 0.')
    ap.add_argument('--binary_head', type=int, default=0,
                    help='Use an independent 2-way REAL-vs-FAKE classifier head '
                         'on the same descriptors. When 0, --w_bin falls back to '
                         'logsumexp over the 3-way logits.')
    ap.add_argument('--feature_binary_head', type=int, default=0,
                    help='If --binary_head is enabled, feed the binary head with '
                         'descriptors plus global avg/max pooled grid features '
                         'and evidence-map-weighted pooled grid features.')
    ap.add_argument('--fs_loss_kind', choices=['hinge', 'l2', 'bce'], default='hinge')
    ap.add_argument('--tau_mass', type=float, default=0.05,
                    help='Target mean activation for the most-active FS map. '
                         'Used by hinge and l2; ignored by bce.')
    ap.add_argument('--desc_set',
                    choices=['minimal', 'minimal_mean', 'full', 'strength', 'topology',
                             'no_topology', 'discovery_greedy4',
                             'baseline_plus_greedy', 'legacy6'],
                    default='minimal',
                    help='Which descriptor subset to feed the classifier MLP. '
                         'Default `minimal` is the recipe chosen from the L19 '
                         'descriptor ablation (see DESC_SETS docstring).')
    ap.add_argument('--tag', default='', help='Optional suffix in output filename for variants')
    ap.add_argument('--unpermute', type=int, default=0,
                    help='L0 fix: undo Qwen ViT window-permutation on cached '
                         'features (raster order) before FETA. 1=on, 0=legacy.')
    args = ap.parse_args()
    args.spatial_conv = bool(args.spatial_conv)
    args.unpermute = bool(args.unpermute)
    args.binary_head = bool(args.binary_head)
    args.feature_binary_head = bool(args.feature_binary_head)
    main(args)
