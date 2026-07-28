# Qwen3-VL modular vision service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a `viam-labs:vision:qwen3-vl` model.

It runs [Qwen3-VL-2B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) locally via [llama.cpp](https://github.com/ggerganov/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for image classification / Q&A and open-vocabulary object detection (`bbox_2d` grounding).

Defaults:
- LLM: `Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- Vision projector: `mmproj-Qwen3VL-2B-Instruct-F16.gguf`
- Accelerator: **Metal** on macOS, **CUDA** when `nvidia-smi` is present, otherwise CPU

This is much faster on Apple Silicon than the previous PyTorch/Transformers path.

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam-labs:vision:qwen3-vl` model from the [viam-labs qwen3-vl-vision module](https://app.viam.com/module/viam-labs/qwen3-vl-vision).

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
Select the `vision` type, then select the `viam-labs:vision:qwen3-vl` model.
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
| `model_repo` | string | Optional | Hugging Face GGUF repo. Default `Qwen/Qwen3-VL-2B-Instruct-GGUF`. |
| `model_file` | string | Optional | LLM GGUF filename. Default `Qwen3VL-2B-Instruct-Q4_K_M.gguf`. |
| `mmproj_file` | string | Optional | Vision projector GGUF. Default `mmproj-Qwen3VL-2B-Instruct-F16.gguf`. |
| `model_path` | string | Optional | Local path to LLM GGUF (skips Hub download for the model). |
| `mmproj_path` | string | Optional | Local path to mmproj GGUF (skips Hub download for mmproj). |
| `n_gpu_layers` | number | Optional | Layers to offload to GPU/Metal (`-1` = all, default). Use `0` for CPU-only. |
| `n_ctx` | number | Optional | Context window (default `4096`). |
| `classification_prompt` | string | Optional | Default classification question. Asks for a 2–3 sentence scene description by default. |
| `max_new_tokens` | number | Optional | Max tokens for classification (default `256`). |
| `detection_max_new_tokens` | number | Optional | Max tokens for detection JSON (default `512`). |
| `max_image_side` | number | Optional | Longest image side in pixels before inference (default `768`). Lower is faster. |
| `auto_label` | bool | Optional | If `true`, list objects then ground those categories (~2x slower, open-vocab). Default `false` (single pass over a common category list). Overridable via `extra.auto_label`. |
| `do_sample` | bool | Optional | Enable sampling for classifications (default `false`). Detections are always greedy. |
| `temperature` | number | Optional | Sampling temperature when `do_sample` is true (default `0.7`). |
| `top_p` | number | Optional | Nucleus sampling when `do_sample` is true (default `0.8`). |
| `top_k` | number | Optional | Top-k sampling when `do_sample` is true (default `20`). |

### Example Configurations

Default Q4_K_M on Metal/CUDA:

```json
{
  "camera": "cam"
}
```

Higher-quality Q8 quant:

```json
{
  "camera": "cam",
  "model_file": "Qwen3VL-2B-Instruct-Q8_0.gguf",
  "mmproj_file": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
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

By default this uses **one** vision pass grounding a fixed common-category list (person, chair, laptop, etc.) — the style Qwen3-VL follows reliably. Pass `extra={"query": "person, chair, laptop"}` for your own classes in one pass, or set `"auto_label": true` to list objects first then ground them (~2x slower, fully open-vocab).
