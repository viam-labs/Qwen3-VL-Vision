# Qwen modular vision service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a `viam-labs:vision:qwen` model.

It runs [Qwen3.5-0.8B GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) locally via [llama.cpp](https://github.com/ggerganov/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for fast image classification / Q&A and object detection (`bbox_2d` grounding) on mobile and edge devices.

Defaults (tuned for speed):
- LLM: `Qwen3.5-0.8B-Q4_K_M.gguf` (~530MB)
- Vision projector: `mmproj-F16.gguf` (~205MB)
- Image longest side: `512`
- Context: `2048`
- Accelerator: **Metal** on macOS, **CUDA** when `nvidia-smi` is present, otherwise CPU

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam-labs:vision:qwen` model from the [viam-labs qwen-vision module](https://app.viam.com/module/viam-labs/qwen-vision).

For local development:

```bash
chmod +x run.sh
# macOS Metal example:
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements.txt -r requirements-dev.txt
pytest
```

On first module setup (`run.sh` before `.installed` exists), dependencies are installed (with Metal/CUDA build flags when available) and the default GGUF + mmproj are prefetched into the Hugging Face cache. Delete `.installed` and restart the module to re-run setup after upgrading.

## Configure your vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Service** subtab and click **Create service**.
Select the `vision` type, then select the `viam-labs:vision:qwen` model.
Enter a name for your vision service and click **Create**.

On the new service panel, copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "camera": "<camera-name>"
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera` | string | **Required** | Default camera dependency for the service. |
| `model_repo` | string | Optional | Hugging Face GGUF repo. Default `unsloth/Qwen3.5-0.8B-GGUF`. |
| `model_file` | string | Optional | LLM GGUF filename. Default `Qwen3.5-0.8B-Q4_K_M.gguf`. |
| `mmproj_file` | string | Optional | Vision projector GGUF. Default `mmproj-F16.gguf`. |
| `model_path` | string | Optional | Local path to LLM GGUF (skips Hub download for the model). |
| `mmproj_path` | string | Optional | Local path to mmproj GGUF (skips Hub download for mmproj). |
| `n_gpu_layers` | number | Optional | Layers to offload to GPU/Metal (`-1` = all, default). Use `0` for CPU-only. |
| `n_ctx` | number | Optional | Context window (default `2048`). |
| `classification_prompt` | string | Optional | Default classification question. Asks for a 2–3 sentence scene description by default. |
| `max_new_tokens` | number | Optional | Max tokens for classification (default `256`). |
| `detection_max_new_tokens` | number | Optional | Max tokens for detection JSON (default `512`). |
| `max_image_side` | number | Optional | Longest image side in pixels before inference (default `512`). Lower is faster. |
| `auto_label` | bool | Optional | Moondream-style list-then-ground. Default `true`. Set `false` for a faster single-pass over a fixed category list. Overridable via `extra.auto_label`. |
| `do_sample` | bool | Optional | Enable sampling for classifications (default `false`). Detections are always greedy. |
| `temperature` | number | Optional | Sampling temperature when `do_sample` is true (default `0.7`). |
| `top_p` | number | Optional | Nucleus sampling when `do_sample` is true (default `0.8`). |
| `top_k` | number | Optional | Top-k sampling when `do_sample` is true (default `20`). |

### Example Configurations

Default Qwen3.5-0.8B Q4_K_M (fast / mobile):

```json
{
  "camera": "cam"
}
```

Higher-quality Qwen3.5-2B:

```json
{
  "camera": "cam",
  "model_repo": "unsloth/Qwen3.5-2B-GGUF",
  "model_file": "Qwen3.5-2B-Q4_K_M.gguf",
  "mmproj_file": "mmproj-F16.gguf",
  "max_image_side": 768,
  "n_ctx": 4096
}
```

Previous Qwen3-VL-2B Instruct:

```json
{
  "camera": "cam",
  "model_repo": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
  "model_file": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
  "mmproj_file": "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
}
```

CPU-only:

```json
{
  "camera": "cam",
  "n_gpu_layers": 0
}
```

## API

### get_classifications / get_classifications_from_camera

Override the prompt with `extra={"question": "..."}`.

### get_detections / get_detections_from_camera

Asks for JSON `{"bbox_2d": [x1,y1,x2,y2], "label": "..."}` on the 0–1000 grid, then converts to Viam detections.

By default this uses the **moondream auto-label** flow: list visible objects, then ground those categories for boxes (~2 vision passes; falls back to per-object grounding if needed). Pass `extra={"query": "people"}` to only list/ground that kind of thing, or set `"auto_label": false` for a faster single-pass over a fixed category list.

## Migration from `qwen3-vl-vision`

**Breaking:** module `viam-labs:qwen3-vl-vision` / model `viam-labs:vision:qwen3-vl` are replaced by:
- Module: `viam-labs:qwen-vision`
- Model: `viam-labs:vision:qwen`

Remove the old module from your machine config and add `viam-labs:qwen-vision`, then recreate the vision service with model `viam-labs:vision:qwen`. Attributes are unchanged.
