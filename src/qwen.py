from typing import ClassVar, Mapping, Optional, Any, List, cast
from typing_extensions import Self
import ast
import base64
import ctypes
import json
import re
from io import BytesIO
from pathlib import Path

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes

from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.logging import getLogger

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import llama_cpp
from llama_cpp._utils import suppress_stdout_stderr
from llama_cpp.llama_chat_format import MTMDChatHandler
from PIL import Image

LOGGER = getLogger(__name__)

# Qwen3.5-0.8B is natively multimodal and much faster on mobile / edge devices.
DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-F16.gguf"
DEFAULT_CLASSIFICATION_PROMPT = (
    "Describe this image in 2-3 sentences. Cover the overall scene, "
    "notable people or objects, and any important details or actions."
)
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_DETECTION_MAX_NEW_TOKENS = 512
DEFAULT_N_CTX = 2048
DEFAULT_N_GPU_LAYERS = -1
# Smaller frames = faster vision encode on phones / edge devices.
DEFAULT_MAX_IMAGE_SIDE = 512

# Qwen3.5 / Qwen3-VL grounding uses relative coordinates on a 0–1000 grid.
BBOX_SCALE = 1000.0

# Vague "detect everything" prompts often return prose or empty JSON on small models.
# Cookbook-style category grounding is what Qwen VL follows reliably.
# Common indoor / desk / street classes for a useful single-pass default.
DEFAULT_DETECTION_CATEGORIES = (
    "person, man, woman, child, chair, table, sofa, desk, bed, laptop, "
    "monitor, keyboard, mouse, phone, bottle, cup, mug, book, bag, backpack, "
    "plant, lamp, door, window, tv, remote, glasses, headphones, car, bicycle"
)

LIST_OBJECTS_PROMPT = (
    "List all distinct visible objects in this image. "
    "Include people, furniture, electronics, clothing, and other items. "
    "Return a simple comma-separated list of object names only, with no extra text."
)

DETECTION_PROMPT_QUERY = (
    'Locate every instance that belongs to the following categories: "{query}". '
    "Report bbox coordinates in JSON format as a list of objects with keys "
    '"bbox_2d" ([x1, y1, x2, y2] in 0-1000) and "label".'
)

_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

_LOG_QUIETED = False


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def _quiet_llama_log(level, text, user_data):  # noqa: ARG001
    # llama.cpp prints progress/info to stderr by default; Viam treats that as errors.
    return


def _quiet_llama_logs() -> None:
    global _LOG_QUIETED
    if _LOG_QUIETED:
        return
    try:
        llama_cpp.llama_log_set(_quiet_llama_log, ctypes.c_void_p(0))
    except Exception:
        LOGGER.debug("unable to install llama.cpp log callback", exc_info=True)
    try:
        if hasattr(llama_cpp, "mtmd_log_set"):
            llama_cpp.mtmd_log_set(_quiet_llama_log, ctypes.c_void_p(0))
        if hasattr(llama_cpp, "mtmd_helper_log_set"):
            llama_cpp.mtmd_helper_log_set(_quiet_llama_log, ctypes.c_void_p(0))
    except Exception:
        LOGGER.debug("unable to install mtmd log callback", exc_info=True)
    _LOG_QUIETED = True


def resolve_gguf_path(repo_id: str, filename: str, local_path: str = "") -> str:
    """Return a local GGUF path, downloading from Hugging Face when needed."""
    if local_path:
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise Exception(f"local model file not found: {path}")
        return str(path)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resize_for_inference(pil_image, max_side: int):
    """Downscale large camera frames; vision encode dominates latency."""
    if max_side <= 0:
        return pil_image.convert("RGB")
    image = pil_image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def pil_to_data_uri(pil_image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def strip_markdown_json(text: str) -> str:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        cleaned = cleaned.split("```", 1)[0]
    elif cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].strip().lower() in ("json", ""):
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned.strip()


class qwen(Vision, Reconfigurable):
    """
    Vision service backed by Qwen3.5 GGUF via llama.cpp (Metal / CUDA / CPU).
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "qwen")

    llm: Any
    DEPS: Mapping[ResourceName, ResourceBase]
    classification_prompt: str
    max_new_tokens: int
    detection_max_new_tokens: int
    max_image_side: int
    auto_label: bool
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    @classmethod
    def validate(cls, config: ComponentConfig):
        fields = config.attributes.fields
        camera = fields["camera"].string_value
        if not camera:
            raise Exception("camera is required")
        return [camera], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.DEPS = dependencies
        fields = config.attributes.fields

        camera_name = fields["camera"].string_value
        if Camera.get_resource_name(camera_name) not in dependencies:
            raise Exception(f"camera dependency '{camera_name}' not found")

        self.classification_prompt = (
            fields["classification_prompt"].string_value or DEFAULT_CLASSIFICATION_PROMPT
        )

        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        if "max_new_tokens" in fields:
            self.max_new_tokens = int(fields["max_new_tokens"].number_value)

        self.detection_max_new_tokens = DEFAULT_DETECTION_MAX_NEW_TOKENS
        if "detection_max_new_tokens" in fields:
            self.detection_max_new_tokens = int(
                fields["detection_max_new_tokens"].number_value
            )

        self.max_image_side = DEFAULT_MAX_IMAGE_SIDE
        if "max_image_side" in fields:
            self.max_image_side = int(fields["max_image_side"].number_value)

        # Two-pass list-then-ground. More recall, roughly 2x slower. Off by default.
        self.auto_label = False
        if "auto_label" in fields:
            self.auto_label = fields["auto_label"].bool_value

        self.do_sample = False
        if "do_sample" in fields:
            self.do_sample = fields["do_sample"].bool_value

        self.temperature = 0.7
        if "temperature" in fields:
            self.temperature = float(fields["temperature"].number_value)

        self.top_p = 0.8
        if "top_p" in fields:
            self.top_p = float(fields["top_p"].number_value)

        self.top_k = 20
        if "top_k" in fields:
            self.top_k = int(fields["top_k"].number_value)

        repo_id = fields["model_repo"].string_value or DEFAULT_MODEL_REPO
        model_file = fields["model_file"].string_value or DEFAULT_MODEL_FILE
        mmproj_file = fields["mmproj_file"].string_value or DEFAULT_MMPROJ_FILE
        model_path = fields["model_path"].string_value
        mmproj_path = fields["mmproj_path"].string_value

        n_ctx = DEFAULT_N_CTX
        if "n_ctx" in fields:
            n_ctx = int(fields["n_ctx"].number_value)

        n_gpu_layers = DEFAULT_N_GPU_LAYERS
        if "n_gpu_layers" in fields:
            n_gpu_layers = int(fields["n_gpu_layers"].number_value)

        resolved_model = resolve_gguf_path(repo_id, model_file, model_path)
        resolved_mmproj = resolve_gguf_path(repo_id, mmproj_file, mmproj_path)

        _quiet_llama_logs()
        LOGGER.info(
            "loading Qwen3.5 GGUF via llama.cpp "
            f"(model={resolved_model}, mmproj={resolved_mmproj}, "
            f"n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})"
        )
        with suppress_stdout_stderr(disable=False):
            chat_handler = MTMDChatHandler(
                clip_model_path=resolved_mmproj, verbose=False
            )
            self.llm = Llama(
                model_path=resolved_model,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        return

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        cam = cast(Camera, self.DEPS[Camera.get_resource_name(camera_name)])
        images, _ = await cam.get_images()
        if not images:
            raise Exception("get_images from cam returned no images")
        for img in images:
            if img.mime_type == CameraMimeType.JPEG:
                return img
        raise Exception(f"no images from cam is {CameraMimeType.JPEG}")

    def _strip_thinking(self, text: str) -> str:
        """Remove Qwen Thinking CoT; final answer follows </think> when present."""
        if not text:
            return ""
        match = _THINK_CLOSE_RE.search(text)
        if match:
            return text[match.end() :].strip()
        return text.strip()

    def _prepare_image(self, image: ViamImage):
        return resize_for_inference(
            viam_to_pil_image(image), self.max_image_side
        )

    def _generate(
        self,
        pil_image,
        prompt: str,
        *,
        max_new_tokens: int,
        greedy: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_uri(pil_image)},
                    },
                ],
            }
        ]

        gen_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        if greedy or not self.do_sample:
            gen_kwargs["temperature"] = 0.0
        else:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["top_k"] = self.top_k

        with suppress_stdout_stderr(disable=False):
            completion = self.llm.create_chat_completion(**gen_kwargs)

        content = completion["choices"][0]["message"].get("content") or ""
        return self._strip_thinking(content)

    def _loads_jsonish(self, text: str) -> Any:
        cleaned = strip_markdown_json(text)
        for candidate in (cleaned,):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                pass

        for pattern in (_JSON_ARRAY_RE, _JSON_OBJECT_RE):
            match = pattern.search(cleaned)
            if not match:
                continue
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(snippet)
            except (SyntaxError, ValueError):
                pass
        return None

    def _parse_detection_items(self, text: str) -> List[dict]:
        parsed = self._loads_jsonish(text)
        if parsed is None:
            LOGGER.warning(f"no JSON detections found in model output: {text!r}")
            return []

        if isinstance(parsed, dict):
            # Single object or {"detections":[...]} / {"objects":[...]} wrappers.
            for key in ("detections", "objects", "items", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            LOGGER.warning(f"unexpected detection JSON type: {type(parsed)}")
            return []

        items: List[dict] = []
        for item in parsed:
            if isinstance(item, dict):
                items.append(item)
        return items

    def _detections_from_response(
        self, text: str, width: int, height: int
    ) -> List[Detection]:
        detections: List[Detection] = []
        for item in self._parse_detection_items(text):
            bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue

            # Values may already be normalized [0,1] instead of [0,1000].
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, y1, x2, y2 = (
                    x1 * BBOX_SCALE,
                    y1 * BBOX_SCALE,
                    x2 * BBOX_SCALE,
                    y2 * BBOX_SCALE,
                )

            x1 = max(0.0, min(BBOX_SCALE, x1))
            y1 = max(0.0, min(BBOX_SCALE, y1))
            x2 = max(0.0, min(BBOX_SCALE, x2))
            y2 = max(0.0, min(BBOX_SCALE, y2))
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x_min_n = x1 / BBOX_SCALE
            y_min_n = y1 / BBOX_SCALE
            x_max_n = x2 / BBOX_SCALE
            y_max_n = y2 / BBOX_SCALE

            label = item.get("label") or item.get("class_name") or item.get("name") or ""
            if not isinstance(label, str):
                label = str(label)

            detections.append(
                Detection(
                    x_min=int(round(x_min_n * width)),
                    y_min=int(round(y_min_n * height)),
                    x_max=int(round(x_max_n * width)),
                    y_max=int(round(y_max_n * height)),
                    x_min_normalized=x_min_n,
                    y_min_normalized=y_min_n,
                    x_max_normalized=x_max_n,
                    y_max_normalized=y_max_n,
                    confidence=1.0,
                    class_name=label,
                )
            )
        return detections

    def _list_object_names(self, pil_image) -> List[str]:
        """Ask for a comma-separated object list, then ground those categories."""
        answer = self._generate(
            pil_image, LIST_OBJECTS_PROMPT, max_new_tokens=128, greedy=True
        )
        # Models sometimes return "a, b, and c" or bullet lines.
        cleaned = answer.replace("\n", ",")
        names: List[str] = []
        seen = set()
        for part in cleaned.split(","):
            name = part.strip().strip(".-•*").strip()
            if name.lower().startswith("and "):
                name = name[4:].strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        return names

    def _detection_prompt(self, query: Optional[str] = None) -> str:
        # Always use category grounding; open-vocab prose prompts under-fire on 2B.
        categories = (
            str(query).strip()
            if query and str(query).strip()
            else DEFAULT_DETECTION_CATEGORIES
        )
        return DETECTION_PROMPT_QUERY.format(query=categories)

    def _resolve_auto_label(self, extra: Optional[Mapping[str, Any]]) -> bool:
        if extra is not None and "auto_label" in extra:
            return bool(extra["auto_label"])
        return self.auto_label

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.get_detections(
            await self.get_cam_image(camera_name), extra=extra
        )

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # Default: one vision pass grounding a fixed common-category list
        # (Qwen cookbook style). Optional auto_label: list objects, then ground
        # those (~2x slower, open-vocab). Or extra.query for specific classes.
        pil_image = self._prepare_image(image)
        query = extra.get("query") if extra else None
        if not (query and str(query).strip()) and self._resolve_auto_label(extra):
            names = self._list_object_names(pil_image)
            if not names:
                LOGGER.warning("object listing returned no names; no detections")
                return []
            query = ", ".join(names)
            LOGGER.debug(f"auto-listed detection categories: {query}")

        prompt = self._detection_prompt(query)
        max_tokens = self.detection_max_new_tokens
        if extra is not None and extra.get("max_new_tokens") is not None:
            max_tokens = int(extra["max_new_tokens"])
        text = self._generate(
            pil_image, prompt, max_new_tokens=max_tokens, greedy=True
        )
        LOGGER.debug(f"detection raw model output: {text!r}")
        # Map boxes onto the original camera frame, not the resized inference image.
        original = viam_to_pil_image(image)
        width, height = original.size
        detections = self._detections_from_response(text, width, height)
        if not detections:
            LOGGER.warning(f"no detections produced from model output: {text!r}")
        return detections

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return await self.get_classifications(
            await self.get_cam_image(camera_name), count, extra=extra
        )

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        question = self.classification_prompt
        if extra is not None and extra.get("question") is not None:
            question = extra["question"]
        max_tokens = self.max_new_tokens
        if extra is not None and extra.get("max_new_tokens") is not None:
            max_tokens = int(extra["max_new_tokens"])
        answer = self._generate(
            self._prepare_image(image), question, max_new_tokens=max_tokens
        )
        return [Classification(class_name=answer, confidence=1.0)]

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        return []

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None
    ) -> Mapping[str, ValueTypes]:
        return {}

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        if return_classifications:
            result.classifications = await self.get_classifications(
                result.image, 1, extra=extra
            )
        if return_detections:
            result.detections = await self.get_detections(result.image, extra=extra)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )
