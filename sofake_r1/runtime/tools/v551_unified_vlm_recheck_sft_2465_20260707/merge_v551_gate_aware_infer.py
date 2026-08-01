#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("response"), str):
        return row["response"]
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deep-pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    deep_pred = read_jsonl(args.deep_pred) if args.deep_pred.exists() else []
    out_rows: list[dict[str, Any]] = []
    missing_deep = 0

    for row in manifest:
        meta = dict(row.get("meta") or {})
        deep_index = row.get("deep_index")
        if deep_index is None:
            response = row.get("lazy_response") or ""
            source = "lazy_gate"
        else:
            if deep_index >= len(deep_pred):
                missing_deep += 1
                response = ""
            else:
                response = response_text(deep_pred[deep_index])
            source = "deep_vlm"
        meta["v551_output_source"] = source
        out_rows.append(
            {
                "response": response,
                "images": [{"bytes": None, "path": row.get("image")}],
                "label": row.get("label"),
                "meta": meta,
            }
        )

    if missing_deep:
        raise SystemExit(f"missing deep predictions: {missing_deep}")
    write_jsonl(args.out, out_rows)
    print(args.out)
    print(f"merged rows: {len(out_rows)}")
    print(f"deep predictions used: {sum(1 for r in manifest if r.get('deep_index') is not None)}")


if __name__ == "__main__":
    main()
