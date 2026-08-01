#!/usr/bin/env python3
"""Download So-Fake-R1 released weights from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "saberzl/So-Fake-R1"
DEFAULT_FILES = [
    "observer_adapter/*",
    "decision_maker_adapter/*",
    "provider/*",
    "model_manifest.json",
    "SHA256SUMS",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "models" / "So-Fake-R1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(args.out_dir),
        allow_patterns=DEFAULT_FILES,
    )
    print(path)


if __name__ == "__main__":
    main()
