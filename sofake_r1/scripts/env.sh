#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOFAKE_R1_DIR="${SOFAKE_R1_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Runtime code is stored in this repository; model weights are downloaded or
# supplied separately.
export SOFAKE_R1_RELEASE="${SOFAKE_R1_RELEASE:-$SOFAKE_R1_DIR/runtime}"
export SOFAKE_R1_MODEL_DIR="${SOFAKE_R1_MODEL_DIR:-$SOFAKE_R1_DIR/models/So-Fake-R1}"

export SOFAKE_R1_OBSERVER_MODEL="${SOFAKE_R1_OBSERVER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export SOFAKE_R1_OBSERVER_ADAPTER="${SOFAKE_R1_OBSERVER_ADAPTER:-$SOFAKE_R1_MODEL_DIR/observer_adapter}"
export SOFAKE_R1_OBSERVER_USE_ADAPTER="${SOFAKE_R1_OBSERVER_USE_ADAPTER:-1}"

export SOFAKE_R1_PROVIDER_CKPT="${SOFAKE_R1_PROVIDER_CKPT:-$SOFAKE_R1_MODEL_DIR/provider/npr_internalized_feta_mono_blur_only05_e1.pt}"
export SOFAKE_R1_QWEN_FETA_CKPT="${SOFAKE_R1_QWEN_FETA_CKPT:-$SOFAKE_R1_MODEL_DIR/provider/feta_v0_L19_cls_only10d_L19_u1_e1.pt}"
export SOFAKE_R1_NPR_CKPT="${SOFAKE_R1_NPR_CKPT:-$SOFAKE_R1_MODEL_DIR/provider/npr_feta_cls_npr_residual_feta10d_e1.pt}"
export SOFAKE_R1_PROMPT_CKPT="${SOFAKE_R1_PROMPT_CKPT:-$SOFAKE_R1_MODEL_DIR/provider/l19_heatmap_bbox_head_best.pt}"
export SOFAKE_R1_DM_CKPT="${SOFAKE_R1_DM_CKPT:-$SOFAKE_R1_MODEL_DIR/decision_maker_adapter}"

export SOFAKE_R1_PLUGIN="${SOFAKE_R1_PLUGIN:-$SOFAKE_R1_RELEASE/plugins/judge_v564_hybrid_route_binary_scope_rm_plugin.py}"
export SOFAKE_R1_RULES="${SOFAKE_R1_RULES:-$SOFAKE_R1_RELEASE/rules/v559_external_lazy_phrase_gate_rules_20260708.json}"
export SOFAKE_R1_TOOLS="${SOFAKE_R1_TOOLS:-$SOFAKE_R1_RELEASE/tools/v551_unified_vlm_recheck_sft_2465_20260707}"
export SOFAKE_R1_BUILDER="${SOFAKE_R1_BUILDER:-$SOFAKE_R1_RELEASE/tools/v564_new_provider_external_20260711/build_v564_new_provider_external_gate.py}"
export SOFAKE_R1_FEATURE_EXTRACT="${SOFAKE_R1_FEATURE_EXTRACT:-$SOFAKE_R1_RELEASE/tools/m0_extract_features.py}"
export SOFAKE_R1_PROVIDER_EXPORT="${SOFAKE_R1_PROVIDER_EXPORT:-$SOFAKE_R1_RELEASE/tools/m60_export_locked_evidence_provider.py}"
export SOFAKE_R1_ASSET_EXPORT="${SOFAKE_R1_ASSET_EXPORT:-$SOFAKE_R1_RELEASE/tools/v544_grpo_provider_bbox_complete_20260706/export_locked_provider_v5_7_assets_from_h5_reference.py}"

# Optional SAM2 settings. They are only needed when MASK_MODE=sam2 or
# RUN_PROVIDER_SAM2=1.
export SOFAKE_R1_SAM2_ROOT="${SOFAKE_R1_SAM2_ROOT:-}"
export SOFAKE_R1_SAM2_CKPT="${SOFAKE_R1_SAM2_CKPT:-}"
