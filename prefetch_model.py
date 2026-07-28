#!/usr/bin/env python3
"""Download default Qwen3.5 GGUF weights into the Hugging Face cache."""

import os
import sys

from huggingface_hub import hf_hub_download

# Keep in sync with src/qwen.py defaults
DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-F16.gguf"


def main() -> int:
    repo_id = os.environ.get("QWEN3_VL_MODEL_REPO", DEFAULT_MODEL_REPO)
    model_file = os.environ.get("QWEN3_VL_MODEL_FILE", DEFAULT_MODEL_FILE)
    mmproj_file = os.environ.get("QWEN3_VL_MMPROJ_FILE", DEFAULT_MMPROJ_FILE)

    print(f"Prefetching '{repo_id}/{model_file}'...", flush=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    print(f"Cached model at {model_path}", flush=True)

    print(f"Prefetching '{repo_id}/{mmproj_file}'...", flush=True)
    mmproj_path = hf_hub_download(repo_id=repo_id, filename=mmproj_file)
    print(f"Cached mmproj at {mmproj_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
