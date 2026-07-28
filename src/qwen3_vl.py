from typing import ClassVar, Mapping, Optional, Any, List, cast
from typing_extensions import Self
import base64
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
from llama_cpp.llama_chat_format import MTMDChatHandler

LOGGER = getLogger(__name__)

DEFAULT_MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
DEFAULT_CLASSIFICATION_PROMPT = "describe this image"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_DETECTION_MAX_NEW_TOKENS = 512
DEFAULT_N_CTX = 4096
DEFAULT_N_GPU_LAYERS = -1

# Qwen3-VL grounding uses relative coordinates on a 0–1000 grid.
BBOX_SCALE = 1000.0

DETECTION_PROMPT_ALL = (
    "Locate every object in this image. "
    'Output a JSON array only, where each item has keys "bbox_2d" and "label". '
    '"bbox_2d" must be [x1, y1, x2, y2] integers in the range [0, 1000]. '
    "Do not include explanations or code fences."
)

DETECTION_PROMPT_QUERY = (
    "Locate all {query} in this image. "
    'Output a JSON array only, where each item has keys "bbox_2d" and "label". '
    '"bbox_2d" must be [x1, y1, x2, y2] integers in the range [0, 1000]. '
    "Do not include explanations or code fences."
)

_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def resolve_gguf_path(repo_id: str, filename: str, local_path: str = "") -> str:
    """Return a local GGUF path, downloading from Hugging Face when needed."""
    if local_path:
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise Exception(f"local model file not found: {path}")
        return str(path)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def pil_to_data_uri(pil_image) -> str:
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class qwen3_vl(Vision, Reconfigurable):
    """
    Vision service backed by Qwen3-VL GGUF via llama.cpp (Metal / CUDA / CPU).
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "qwen3-vl")

    llm: Any
    DEPS: Mapping[ResourceName, ResourceBase]
    classification_prompt: str
    max_new_tokens: int
    detection_max_new_tokens: int
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

        LOGGER.info(
            "loading Qwen3-VL GGUF via llama.cpp "
            f"(model={resolved_model}, mmproj={resolved_mmproj}, "
            f"n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})"
        )
        chat_handler = MTMDChatHandler(clip_model_path=resolved_mmproj, verbose=False)
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

        if greedy or not self.do_sample:
            completion = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        else:
            completion = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        content = completion["choices"][0]["message"].get("content") or ""
        return self._strip_thinking(content)

    def _parse_json_array(self, text: str) -> List[Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            lines = cleaned.splitlines()
            if lines and lines[0].strip().lower() in ("json", ""):
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        match = _JSON_ARRAY_RE.search(cleaned)
        if not match:
            LOGGER.warning(f"no JSON array found in model output: {text!r}")
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            LOGGER.warning(f"failed to parse JSON array from model output: {text!r}")
            return []
        return parsed if isinstance(parsed, list) else []

    def _detections_from_response(
        self, text: str, width: int, height: int
    ) -> List[Detection]:
        detections: List[Detection] = []
        for item in self._parse_json_array(text):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue

            x1 = max(0.0, min(BBOX_SCALE, x1))
            y1 = max(0.0, min(BBOX_SCALE, y1))
            x2 = max(0.0, min(BBOX_SCALE, x2))
            y2 = max(0.0, min(BBOX_SCALE, y2))

            x_min_n = x1 / BBOX_SCALE
            y_min_n = y1 / BBOX_SCALE
            x_max_n = x2 / BBOX_SCALE
            y_max_n = y2 / BBOX_SCALE

            label = item.get("label", "")
            if label is None:
                label = ""
            elif not isinstance(label, str):
                label = str(label)

            detections.append(
                {
                    "x_min": int(round(x_min_n * width)),
                    "y_min": int(round(y_min_n * height)),
                    "x_max": int(round(x_max_n * width)),
                    "y_max": int(round(y_max_n * height)),
                    "x_min_normalized": x_min_n,
                    "y_min_normalized": y_min_n,
                    "x_max_normalized": x_max_n,
                    "y_max_normalized": y_max_n,
                    "confidence": 1,
                    "class_name": label,
                }
            )
        return detections

    def _detection_prompt(self, query: Optional[str] = None) -> str:
        if query and str(query).strip():
            return DETECTION_PROMPT_QUERY.format(query=str(query).strip())
        return DETECTION_PROMPT_ALL

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
        pil_image = viam_to_pil_image(image)
        query = extra.get("query") if extra else None
        prompt = self._detection_prompt(query)
        max_tokens = self.detection_max_new_tokens
        if extra is not None and extra.get("max_new_tokens") is not None:
            max_tokens = int(extra["max_new_tokens"])
        text = self._generate(
            pil_image, prompt, max_new_tokens=max_tokens, greedy=True
        )
        width, height = pil_image.size
        return self._detections_from_response(text, width, height)

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
            viam_to_pil_image(image), question, max_new_tokens=max_tokens
        )
        return [{"class_name": answer, "confidence": 1}]

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        return

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None
    ) -> Mapping[str, ValueTypes]:
        return

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
