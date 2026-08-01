#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /abs/path/to/image [out_dir]" >&2
  echo "Env: CUDA_VISIBLE_DEVICES=0 MASK_MODE=box|sam2 RUN_PROVIDER_SAM2=0|1" >&2
  exit 2
fi

IMAGE_INPUT="$1"
if [[ ! -f "$IMAGE_INPUT" ]]; then
  echo "Missing image: $IMAGE_INPUT" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

PY="${PY:-python}"
if [[ -z "${INFER:-}" ]]; then
  if ! INFER="$("$PY" - <<'PY'
import inspect
import sys

try:
    from swift.cli import infer
except Exception as exc:
    print(
        "Cannot import ms-swift from the selected Python. "
        "Activate the ms-swift environment or set PY=/path/to/env/bin/python.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(inspect.getfile(infer))
PY
)"; then
    exit 1
  fi
fi
SWIFT_ROOT="${SWIFT_ROOT:-$SOFAKE_R1_DIR}"
QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

PY_PATHS=(
  "$SOFAKE_R1_RELEASE/tools"
  "$SOFAKE_R1_RELEASE/tools/v544_grpo_provider_bbox_complete_20260706"
  "$SOFAKE_R1_RELEASE/tools/v547_external_provider_interface_repair_20260706"
  "$SOFAKE_R1_RELEASE/tools/v551_unified_vlm_recheck_sft_2465_20260707"
  "$SOFAKE_R1_RELEASE/tools/v564_new_provider_external_20260711"
  "$SOFAKE_R1_RELEASE/plugins"
)
if [[ -n "$SOFAKE_R1_SAM2_ROOT" ]]; then
  PY_PATHS+=("$SOFAKE_R1_SAM2_ROOT")
fi
IFS=:
export PYTHONPATH="${PY_PATHS[*]}${PYTHONPATH:+:$PYTHONPATH}"
unset IFS

MASK_MODE="${MASK_MODE:-box}"
RUN_PROVIDER_SAM2="${RUN_PROVIDER_SAM2:-0}"
OBSERVER_MAX_NEW_TOKENS="${OBSERVER_MAX_NEW_TOKENS:-384}"
DM_MAX_NEW_TOKENS="${DM_MAX_NEW_TOKENS:-512}"
MASTER_PORT="${MASTER_PORT:-31664}"

IMAGE="$("$PY" - "$IMAGE_INPUT" <<'PY'
import os, sys
print(os.path.realpath(os.path.expanduser(sys.argv[1])))
PY
)"

if [[ $# -eq 2 ]]; then
  OUT_DIR="$2"
else
  stem="$(basename "$IMAGE")"
  stem="${stem%.*}"
  OUT_DIR="$SOFAKE_R1_RELEASE/outputs/single_image_${stem}_$(date +%Y%m%d_%H%M%S)"
fi
OUT_DIR="$("$PY" - "$OUT_DIR" <<'PY'
import os
import sys

print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"

WORK="$OUT_DIR/work"
PROVIDER_DIR="$WORK/provider"
OLD_GATE_DIR="$WORK/v5_51_gate_aware"
GATE_DIR="$WORK/v5_60_prompt_clean_phrase_gate"
MAP_DIR="$PROVIDER_DIR/topology_maps/single"
MASK_DIR="$OUT_DIR/masks"
mkdir -p "$WORK" "$PROVIDER_DIR" "$OLD_GATE_DIR" "$GATE_DIR" "$MAP_DIR" "$MASK_DIR"

SAMPLE_ID="${SAMPLE_ID:-single_000001}"
SAMPLES="$WORK/samples.jsonl"
OBSERVER_INPUT="$WORK/observer_input.jsonl"
OBSERVER_INFER="$WORK/observer_infer.jsonl"
H5="$WORK/features_L19.h5"
LOCKED_PROVIDER="$PROVIDER_DIR/rerun_locked_provider.jsonl"
PROVIDER_COMPAT="$PROVIDER_DIR/compatible_evidence.jsonl"
PROVIDER_LOC="$PROVIDER_DIR/localization.jsonl"
PROVIDER_SUMMARY="$PROVIDER_DIR/provider_export_summary.json"
MANIFEST="$GATE_DIR/judge_val_bfree_v5_60_prompt_clean_phrase_gate_manifest.jsonl"
DEEP_INPUT="$GATE_DIR/judge_val_bfree_v5_60_prompt_clean_phrase_gate_deep_vlm_infer.jsonl"
DEEP_PRED="$WORK/dm_deep_pred.jsonl"
MERGED="$WORK/merged_prediction.jsonl"
FINAL_JSONL="$OUT_DIR/sofake_r1_single_image_result.jsonl"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

line_count() {
  if [[ -s "$1" ]]; then
    wc -l < "$1" | tr -d ' '
  else
    echo 0
  fi
}

require_dir "$SOFAKE_R1_RELEASE/tools"
require_file "$SOFAKE_R1_BUILDER"
require_file "$SOFAKE_R1_FEATURE_EXTRACT"
require_file "$SOFAKE_R1_PROVIDER_EXPORT"
require_file "$SOFAKE_R1_ASSET_EXPORT"
require_file "$SOFAKE_R1_PLUGIN"
require_file "$SOFAKE_R1_RULES"
require_file "$SOFAKE_R1_OBSERVER_ADAPTER/adapter_model.safetensors"
require_file "$SOFAKE_R1_DM_CKPT/adapter_model.safetensors"
require_file "$SOFAKE_R1_PROVIDER_CKPT"
require_file "$SOFAKE_R1_QWEN_FETA_CKPT"
require_file "$SOFAKE_R1_NPR_CKPT"
require_file "$SOFAKE_R1_PROMPT_CKPT"

echo "=== So-Fake-R1 single-image inference ==="
echo "image=$IMAGE"
echo "out_dir=$OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "observer=$SOFAKE_R1_OBSERVER_MODEL"
echo "observer_adapter=$SOFAKE_R1_OBSERVER_ADAPTER"
echo "observer_use_adapter=$SOFAKE_R1_OBSERVER_USE_ADAPTER"
echo "qwen_base_model=$QWEN_BASE_MODEL"
echo "provider=$SOFAKE_R1_PROVIDER_CKPT"
echo "qwen_feta=$SOFAKE_R1_QWEN_FETA_CKPT"
echo "npr=$SOFAKE_R1_NPR_CKPT"
echo "decision_maker=$SOFAKE_R1_DM_CKPT"

cat > "$SAMPLES" <<JSON
{"sample_id":"$SAMPLE_ID","image":"$IMAGE","gt_cls":0,"label":0,"gt_label_name":"REAL","source":"single_image","subsource":"single_image","dataset":"single_image"}
JSON

echo "=== [1/9] build observer input ==="
"$PY" "$SOFAKE_R1_BUILDER" make-observer-input \
  --samples "$SAMPLES" \
  --out "$OBSERVER_INPUT" \
  --dataset single_image

echo "=== [2/9] Visual Observer inference ==="
cd "$SWIFT_ROOT"
observer_args=(--model "$SOFAKE_R1_OBSERVER_MODEL")
if [[ "$SOFAKE_R1_OBSERVER_USE_ADAPTER" == "1" ]] || {
  [[ "$SOFAKE_R1_OBSERVER_USE_ADAPTER" == "auto" ]] && [[ -f "$SOFAKE_R1_OBSERVER_ADAPTER/adapter_model.safetensors" ]]
}; then
  observer_args=(--model "$QWEN_BASE_MODEL" --adapters "$SOFAKE_R1_OBSERVER_ADAPTER")
fi
echo "observer_model_args=${observer_args[*]}"
"$PY" "$INFER" \
  "${observer_args[@]}" \
  --val_dataset "$OBSERVER_INPUT" \
  --max_batch_size 1 \
  --result_path "$OBSERVER_INFER" \
  --max_new_tokens "$OBSERVER_MAX_NEW_TOKENS" \
  --temperature 0
if [[ "$(line_count "$OBSERVER_INFER")" != "1" ]]; then
  echo "Observer inference did not produce exactly one row: $OBSERVER_INFER" >&2
  exit 3
fi

echo "=== [3/9] extract frozen Qwen L19 H5 features ==="
"$PY" "$SOFAKE_R1_FEATURE_EXTRACT" \
  --samples "$SAMPLES" \
  --out "$H5" \
  --model "$QWEN_BASE_MODEL" \
  --device cuda \
  --max_pixels $((224 * 224)) \
  --layers 19

echo "=== [4/9] Forensic Evidence Provider classification ==="
"$PY" "$SOFAKE_R1_PROVIDER_EXPORT" \
  --h5 "$H5" \
  --ckpt "$SOFAKE_R1_PROVIDER_CKPT" \
  --out "$LOCKED_PROVIDER" \
  --transform log_scaled_a2 \
  --dataset_name single_image \
  --qwen_ckpt "$SOFAKE_R1_QWEN_FETA_CKPT" \
  --npr_ckpt "$SOFAKE_R1_NPR_CKPT" \
  --device cuda \
  --eval_batch_size 1 \
  --batch_size 1 \
  --num_workers 0

echo "=== [5/9] Provider topology, descriptors, and support box ==="
asset_args=(
  --samples "$SAMPLES"
  --h5 "$H5"
  --locked_evidence "$LOCKED_PROVIDER"
  --map_dir "$MAP_DIR"
  --compatible_evidence_out "$PROVIDER_COMPAT"
  --localization_out "$PROVIDER_LOC"
  --summary_out "$PROVIDER_SUMMARY"
  --qwen_feta_ckpt "$SOFAKE_R1_QWEN_FETA_CKPT"
  --prompt_ckpt "$SOFAKE_R1_PROMPT_CKPT"
  --device cuda
)
if [[ "$RUN_PROVIDER_SAM2" == "1" ]]; then
  asset_args+=(--run_sam2)
fi
"$PY" "$SOFAKE_R1_ASSET_EXPORT" "${asset_args[@]}"

echo "=== [6/9] Evidence Gate ==="
"$PY" "$SOFAKE_R1_BUILDER" build \
  --bfree-samples "$SAMPLES" \
  --bfree-observer "$OBSERVER_INFER" \
  --bfree-provider "$PROVIDER_COMPAT" \
  --old-gate-dir "$OLD_GATE_DIR" \
  --phrase-dir "$GATE_DIR" \
  --tag v5_60_prompt_clean_phrase_gate

require_file "$MANIFEST"
require_file "$DEEP_INPUT"

deep_rows="$(line_count "$DEEP_INPUT")"
echo "deep_rows=$deep_rows"

echo "=== [7/9] Decision Maker routed inference ==="
if [[ "$deep_rows" == "0" ]]; then
  : > "$DEEP_PRED"
  echo "Gate selected direct acceptance; Decision Maker VLM call skipped."
else
  "$PY" -m torch.distributed.run --nproc_per_node 1 --master_port "$MASTER_PORT" "$INFER" \
    --model "$QWEN_BASE_MODEL" \
    --adapters "$SOFAKE_R1_DM_CKPT" \
    --external_plugins "$SOFAKE_R1_PLUGIN" \
    --val_dataset "$DEEP_INPUT" \
    --max_batch_size 1 \
    --result_path "$DEEP_PRED" \
    --max_new_tokens "$DM_MAX_NEW_TOKENS" \
    --write_batch_size 10 \
    --temperature 0 \
    --max_length 128000 \
    --max_pixels 1048576
fi

echo "=== [8/9] merge direct/deep predictions ==="
"$PY" "$SOFAKE_R1_TOOLS/merge_v551_gate_aware_infer.py" \
  --manifest "$MANIFEST" \
  --deep-pred "$DEEP_PRED" \
  --out "$MERGED"

echo "=== [9/9] export unified result and final mask ==="
export_args=(
  --manifest "$MANIFEST"
  --merged "$MERGED"
  --out "$FINAL_JSONL"
  --mask-dir "$MASK_DIR"
  --observer-jsonl "$OBSERVER_INFER"
  --provider-compatible "$PROVIDER_COMPAT"
  --provider-localization "$PROVIDER_LOC"
  --topology-map-dir "$MAP_DIR"
  --mask-mode "$MASK_MODE"
)
"$PY" "$SOFAKE_R1_RELEASE/tools/export_sofake_r1_infer_outputs.py" "${export_args[@]}"

"$PY" - "$FINAL_JSONL" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    row["gt_label"] = None
    row.setdefault("notes", {})["gt_label"] = "not used for single-image inference; internal dummy label only satisfies the batch data schema"
path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
PY

echo "=== Done ==="
echo "result_jsonl=$FINAL_JSONL"
echo "masks=$MASK_DIR"
echo "observer=$OBSERVER_INFER"
echo "provider=$PROVIDER_COMPAT"
echo "merged=$MERGED"

"$PY" - "$FINAL_JSONL" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
print(json.dumps({
    "sample_id": row.get("sample_id"),
    "route_action": row.get("route_action"),
    "output_source": row.get("output_source"),
    "observer_answer": (row.get("observer") or {}).get("answer"),
    "provider_answer": (row.get("provider") or {}).get("answer"),
    "final": row.get("final"),
}, ensure_ascii=False, indent=2))
PY
