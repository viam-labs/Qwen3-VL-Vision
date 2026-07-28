# Qwen3-VL modular vision service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a `viam-labs:vision:qwen3-vl` model.

It runs [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) locally via [Hugging Face Transformers](https://huggingface.co/docs/transformers) for image classification / Q&A and open-vocabulary object detection (2D grounding with `bbox_2d` coordinates). The default weights are **Qwen3-VL-2B-Instruct** (faster than Thinking for detection / Q&A on Mac MPS).

Local inference needs enough GPU or Apple Silicon / CPU memory for the chosen checkpoint. Prefer Instruct for interactive use; set `model` to a Thinking checkpoint when you want CoT reasoning.

If your Python was built without `lzma` (common with some pyenv installs), this module installs a compatibility shim so `torchvision` / `AutoProcessor` can import. Prefer a Python with `libxz`/`lzma` support when possible (`brew install xz` before building Python).

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam-labs:vision:qwen3-vl` model from the [viam-labs qwen3-vl-vision module](https://app.viam.com/module/viam-labs/qwen3-vl-vision).

For local development:

```bash
chmod +x run.sh
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

On first module setup (`run.sh` before `.installed` exists), dependencies are installed and the default model weights are prefetched into the Hugging Face cache via `prefetch_model.py`. Set `HF_TOKEN` if needed for rate limits or gated assets. Override the prefetch checkpoint with `QWEN3_VL_MODEL`. Delete `.installed` and restart the module to re-run setup (needed after changing the default model).

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

The following attributes are available for `viam-labs:vision:qwen3-vl` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera` | string | **Required** | Default camera dependency for the service. Camera-based API methods use the `camera_name` argument; add extra cameras via `depends_on` if needed. |
| `model` | string | Optional | Hugging Face model id. Defaults to [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). Use `Qwen/Qwen3-VL-2B-Thinking` for CoT (much slower). |
| `classification_prompt` | string | Optional | Default question for classifications. Defaults to `"describe this image"`. Overridden by `extra.question` when provided. |
| `device_map` | string | Optional | Transformers `device_map` (default `"auto"`). |
| `dtype` | string | Optional | Torch dtype name (`"auto"`, `"bfloat16"`, `"float16"`, `"float32"`). On Apple Silicon defaults to `bfloat16`; otherwise `"auto"`. |
| `max_new_tokens` | number | Optional | Max new tokens for classification / Q&A (default `512`). Raise for Thinking models. |
| `detection_max_new_tokens` | number | Optional | Max new tokens for detection grounding (default `512`). |
| `do_sample` | bool | Optional | Enable sampling for classifications (default `false` / greedy). Detections always use greedy decoding for JSON stability. |
| `temperature` | number | Optional | Sampling temperature when `do_sample` is true (default `1.0`). |
| `top_p` | number | Optional | Nucleus sampling `top_p` when `do_sample` is true (default `0.95`). |
| `top_k` | number | Optional | `top_k` sampling when `do_sample` is true (default `20`). |

### Example Configurations

Default 2B Instruct (recommended on Mac):

```json
{
  "camera": "cam"
}
```

Thinking model (slower, longer CoT):

```json
{
  "camera": "cam",
  "model": "Qwen/Qwen3-VL-2B-Thinking",
  "max_new_tokens": 2048,
  "detection_max_new_tokens": 2048
}
```

Larger Instruct checkpoint:

```json
{
  "camera": "cam",
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "dtype": "bfloat16"
}
```

Custom classification prompt:

```json
{
  "camera": "cam",
  "classification_prompt": "what safety gear is visible?"
}
```

## API

The qwen3-vl resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html).

Camera-based methods use the `camera_name` argument. That camera must be available as a dependency (the required `camera` attribute, and any additional cameras listed in `depends_on`).

### get_classifications(image=*binary*, count)

### get_classifications_from_camera(camera_name=*string*, count)

By default, the model is asked the configured `classification_prompt` (or `"describe this image"` if unset).
Override per call with the extra parameter `question`:

```python
qwen.get_classifications(
    image,
    1,
    extra={"question": "what is the person wearing?"},
)
```

Thinking editions may emit CoT before the final answer; the module strips `</think>`-delimited reasoning and returns the final text as `class_name`.

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Detections use Qwen3-VL [2D grounding](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): a single generation that asks for a JSON list of `{"bbox_2d": [x1,y1,x2,y2], "label": "..."}` entries on the model’s **0–1000** coordinate grid, then converts those boxes to pixel and normalized Viam detections. Decoding is greedy for stable JSON.

By default, all visible objects are requested. Pass `extra={"query": "..."}` to limit the set (for example, only people or vehicles):

```python
qwen.get_detections(image, extra={"query": "people"})
```
