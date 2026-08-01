# So-Fake-R1 Inference

This directory contains the inference-only release of So-Fake-R1.

It runs the three deployed components:

1. Visual Observer
2. Forensic Evidence Provider
3. Evidence Gate + Decision Maker

The release is intentionally focused on inference. Training scripts and intermediate experiment code are not included in this public runtime.

## 1. Install

Create an environment with PyTorch, Transformers, PEFT, Hugging Face Hub, and ms-swift. The exact environment used in our experiments is based on `swift==4.3.0.dev0`.

```bash
pip install -r sofake_r1/requirements.txt
```

Run the scripts from an environment where `swift.cli.infer` is importable. If your default `python` is different, set:

```bash
export PY=/path/to/ms-swift-env/bin/python
```

You also need access to the Qwen base model:

```bash
Qwen/Qwen2.5-VL-7B-Instruct
```

SAM2 is optional. It is only required when exporting SAM2-refined masks.

## 2. Download Weights

```bash
python sofake_r1/scripts/download_weights.py
```

By default, weights are downloaded to:

```text
sofake_r1/models/So-Fake-R1/
```

The expected structure is:

```text
observer_adapter/
decision_maker_adapter/
provider/
```

If the weights are already stored elsewhere, set:

```bash
export SOFAKE_R1_MODEL_DIR=/path/to/So-Fake-R1
```

## 3. Run The Example

```bash
bash sofake_r1/scripts/infer_one_image.sh \
  sofake_r1/examples/example.jpg \
  outputs/sofake_r1_example
```

The script writes:

```text
outputs/sofake_r1_example/
  sofake_r1_single_image_result.jsonl
  sofake_r1_single_image_result.jsonl.summary.json
  work/observer_infer.jsonl
  work/provider/compatible_evidence.jsonl
  work/merged_prediction.jsonl
  masks/
```

For the released example image, the expected routed result is:

```text
observer_answer: FULL_SYNTHETIC
provider_answer: REAL
route_action: TARGETED_RECHECK
final scope: REAL
```

## 4. Optional Settings

Use a local Qwen checkpoint instead of the Hugging Face model ID:

```bash
export QWEN_BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
```

Use another GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash sofake_r1/scripts/infer_one_image.sh \
  sofake_r1/examples/example.jpg \
  outputs/sofake_r1_example
```

Export box masks by default:

```bash
export MASK_MODE=box
```

Enable SAM2-refined masks:

```bash
export MASK_MODE=sam2
export SOFAKE_R1_SAM2_ROOT=/path/to/sam2
export SOFAKE_R1_SAM2_CKPT=/path/to/sam2.1_hiera_tiny.pt
```
