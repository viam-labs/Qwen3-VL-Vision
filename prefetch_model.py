#!/usr/bin/env python3
"""Download Qwen3-VL weights into the Hugging Face cache during module setup."""

import os
import sys

from huggingface_hub import snapshot_download

# Keep in sync with src/qwen3_vl.py DEFAULT_MODEL
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Thinking"


def main() -> int:
    model_id = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    print(f"Prefetching '{model_id}' into Hugging Face cache...", flush=True)
    path = snapshot_download(repo_id=model_id)
    print(f"Cached at {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
