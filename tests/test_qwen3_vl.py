from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from src.qwen3_vl import qwen3_vl as Qwen3VL
import importlib

qwen_mod = importlib.import_module("src.qwen3_vl")


def make_config(attrs: dict, name: str = "qwen3-vl") -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def make_jpeg_image(width: int = 100, height: int = 50) -> ViamImage:
    buf = BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return ViamImage(data=buf.getvalue(), mime_type=CameraMimeType.JPEG)


def make_camera(image: ViamImage | None = None) -> MagicMock:
    cam = MagicMock(spec=Camera)
    cam.get_images = AsyncMock(return_value=([image or make_jpeg_image()], None))
    return cam


@pytest.fixture
def mock_llama():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "decoded"}}]
    }
    handler = MagicMock()

    with (
        patch(
            "src.qwen3_vl.hf_hub_download",
            side_effect=lambda **kw: f"/tmp/{kw['filename']}",
        ) as download,
        patch("src.qwen3_vl.MTMDChatHandler", return_value=handler) as handler_cls,
        patch("src.qwen3_vl.Llama", return_value=llm) as llama_cls,
        patch("src.qwen3_vl._quiet_llama_logs"),
        patch("src.qwen3_vl.suppress_stdout_stderr"),
    ):
        yield download, handler_cls, llama_cls, llm


@pytest.fixture
def service(mock_llama):
    _, _, _, llm = mock_llama
    cam = make_camera()
    deps = {Camera.get_resource_name("cam"): cam}
    config = make_config({"camera": "cam"})
    instance = Qwen3VL.new(config, deps)
    instance._test_camera = cam
    instance._test_llm = llm
    return instance


class TestValidate:
    def test_requires_camera(self):
        with pytest.raises(Exception, match="camera is required"):
            Qwen3VL.validate(make_config({}))

    def test_returns_camera_dependency(self):
        assert Qwen3VL.validate(make_config({"camera": "cam"})) == (["cam"], [])


class TestReconfigure:
    def test_defaults(self, mock_llama):
        download, handler_cls, llama_cls, _ = mock_llama
        cam = make_camera()
        Qwen3VL.new(
            make_config({"camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        assert download.call_count == 2
        filenames = {c.kwargs["filename"] for c in download.call_args_list}
        assert filenames == {
            "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
        }
        assert handler_cls.called
        assert llama_cls.call_args.kwargs["n_gpu_layers"] == -1
        assert llama_cls.call_args.kwargs["n_ctx"] == 4096

    def test_custom_files(self, mock_llama):
        download, _, llama_cls, _ = mock_llama
        cam = make_camera()
        Qwen3VL.new(
            make_config(
                {
                    "camera": "cam",
                    "model_repo": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
                    "model_file": "Qwen3VL-2B-Instruct-Q8_0.gguf",
                    "mmproj_file": "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                    "n_gpu_layers": 20,
                    "n_ctx": 2048,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        filenames = {c.kwargs["filename"] for c in download.call_args_list}
        assert "Qwen3VL-2B-Instruct-Q8_0.gguf" in filenames
        assert llama_cls.call_args.kwargs["n_gpu_layers"] == 20
        assert llama_cls.call_args.kwargs["n_ctx"] == 2048

    def test_missing_camera_dependency(self, mock_llama):
        with pytest.raises(Exception, match="camera dependency"):
            Qwen3VL.new(make_config({"camera": "cam"}), {})


class TestGeneration:
    @pytest.mark.asyncio
    async def test_detections_use_greedy_decoding(self, service):
        response = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "box"}]'
        with patch.object(service, "_generate", wraps=service._generate) as gen:
            service._test_llm.create_chat_completion.return_value = {
                "choices": [{"message": {"content": response}}]
            }
            await service.get_detections(make_jpeg_image())
        assert gen.call_args.kwargs.get("greedy") is True

    @pytest.mark.asyncio
    async def test_generate_defaults_to_temperature_zero(self, service):
        service._test_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        await service.get_classifications(make_jpeg_image(), 1)
        kwargs = service._test_llm.create_chat_completion.call_args.kwargs
        assert kwargs.get("temperature") == 0.0
        assert kwargs.get("max_tokens") == 256


class TestClassifications:
    @pytest.mark.asyncio
    async def test_default_question(self, service):
        with patch.object(service, "_generate", return_value="a red square") as gen:
            result = await service.get_classifications(make_jpeg_image(), 1)
        assert result[0].class_name == "a red square"
        assert result[0].confidence == pytest.approx(1.0)
        assert "Describe this image in 2-3 sentences" in gen.call_args[0][1]

    @pytest.mark.asyncio
    async def test_config_classification_prompt(self, mock_llama):
        cam = make_camera()
        service = Qwen3VL.new(
            make_config(
                {
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        with patch.object(service, "_generate", return_value="a helmet") as gen:
            await service.get_classifications(make_jpeg_image(), 1)
        assert gen.call_args[0][1] == "what safety gear is visible?"

    @pytest.mark.asyncio
    async def test_extra_question_overrides_config_prompt(self, mock_llama):
        cam = make_camera()
        service = Qwen3VL.new(
            make_config(
                {
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        with patch.object(service, "_generate", return_value="yes") as gen:
            result = await service.get_classifications(
                make_jpeg_image(), 1, extra={"question": "is there a person?"}
            )
        assert result[0].class_name == "yes"
        assert gen.call_args[0][1] == "is there a person?"

    @pytest.mark.asyncio
    async def test_strips_thinking_content(self, service):
        service._test_llm.create_chat_completion.return_value = {
            "choices": [
                {"message": {"content": "<think>reasoning here</think>\nfinal answer"}}
            ]
        }
        result = await service.get_classifications(make_jpeg_image(), 1)
        assert result[0].class_name == "final answer"

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        with patch.object(service, "_generate", return_value="from cam"):
            result = await service.get_classifications_from_camera("cam", 1)
        assert result[0].class_name == "from cam"
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_llama):
        cam_a = make_camera()
        cam_b = make_camera()
        deps = {
            Camera.get_resource_name("cam-a"): cam_a,
            Camera.get_resource_name("cam-b"): cam_b,
        }
        service = Qwen3VL.new(make_config({"camera": "cam-a"}), deps)
        with patch.object(service, "_generate", return_value="from cam-b"):
            await service.get_classifications_from_camera("cam-b", 1)
        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestDetections:
    @pytest.mark.asyncio
    async def test_parses_bbox_json(self, service):
        response = (
            '[{"bbox_2d": [100, 200, 300, 400], "label": "person"}, '
            '{"bbox_2d": [500, 500, 900, 900], "label": "chair"}]'
        )
        with (
            patch.object(service, "_list_object_names") as list_objs,
            patch.object(service, "_generate", return_value=response) as gen,
        ):
            result = await service.get_detections(make_jpeg_image(100, 50))

        list_objs.assert_not_called()
        prompt = gen.call_args[0][1]
        assert 'categories: "person, man, woman' in prompt
        assert [d.class_name for d in result] == ["person", "chair"]
        assert result[0].x_min == 10
        assert result[0].y_min == 10
        assert result[0].x_max == 30
        assert result[0].y_max == 20
        assert result[0].x_min_normalized == pytest.approx(0.1)
        assert result[1].x_min == 50
        assert result[1].y_max == 45

    @pytest.mark.asyncio
    async def test_parses_single_object_json(self, service):
        # Official Qwen examples often return one object, not an array.
        response = '{"bbox_2d": [100, 200, 300, 400], "label": "cup"}'
        result = service._detections_from_response(response, 100, 50)
        assert len(result) == 1
        assert result[0].class_name == "cup"
        assert result[0].x_min == 10

    @pytest.mark.asyncio
    async def test_query_skips_listing(self, service):
        response = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "person"}]'
        with (
            patch.object(service, "_list_object_names") as list_objs,
            patch.object(service, "_generate", return_value=response) as gen,
        ):
            result = await service.get_detections(
                make_jpeg_image(), extra={"query": "people"}
            )
        list_objs.assert_not_called()
        prompt = gen.call_args[0][1]
        assert 'categories: "people"' in prompt
        assert result[0].class_name == "person"

    @pytest.mark.asyncio
    async def test_auto_label_lists_then_grounds(self, service):
        response = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "chair"}]'
        with (
            patch.object(
                service, "_list_object_names", return_value=["person", "chair"]
            ) as list_objs,
            patch.object(service, "_generate", return_value=response) as gen,
        ):
            result = await service.get_detections(
                make_jpeg_image(), extra={"auto_label": True}
            )
        list_objs.assert_called_once()
        assert 'categories: "person, chair"' in gen.call_args[0][1]
        assert result[0].class_name == "chair"

    @pytest.mark.asyncio
    async def test_auto_label_empty_list(self, service):
        with (
            patch.object(service, "_list_object_names", return_value=[]),
            patch.object(service, "_generate") as gen,
        ):
            result = await service.get_detections(
                make_jpeg_image(), extra={"auto_label": True}
            )
        assert result == []
        gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        response = '[{"bbox_2d": [0, 0, 500, 500], "label": "cup"}]'
        with patch.object(service, "_generate", return_value=response):
            result = await service.get_detections_from_camera("cam")
        assert len(result) == 1
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_llama):
        cam_a = make_camera()
        cam_b = make_camera()
        service = Qwen3VL.new(
            make_config({"camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )
        with patch.object(
            service,
            "_generate",
            return_value='[{"bbox_2d": [0, 0, 500, 500], "label": "cup"}]',
        ):
            await service.get_detections_from_camera("cam-b")
        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestImageResize:
    def test_resize_for_inference_downscales(self):
        img = Image.new("RGB", (1920, 1080), color="blue")
        out = qwen_mod.resize_for_inference(img, 768)
        assert max(out.size) == 768

    def test_resize_for_inference_skips_small(self):
        img = Image.new("RGB", (640, 480), color="blue")
        out = qwen_mod.resize_for_inference(img, 768)
        assert out.size == (640, 480)


class TestPropertiesAndCaptureAll:
    @pytest.mark.asyncio
    async def test_properties(self, service):
        props = await service.get_properties()
        assert props.classifications_supported is True
        assert props.detections_supported is True
        assert props.object_point_clouds_supported is False

    @pytest.mark.asyncio
    async def test_capture_all_respects_flags(self, service):
        with patch.object(
            service,
            "_generate",
            side_effect=[
                "a scene",
                '[{"bbox_2d": [0, 0, 1000, 1000], "label": "box"}]',
            ],
        ):
            result = await service.capture_all_from_camera(
                "cam",
                return_image=True,
                return_classifications=True,
                return_detections=True,
            )

        assert result.image is not None
        assert result.classifications[0].class_name == "a scene"
        assert result.detections[0].class_name == "box"

    @pytest.mark.asyncio
    async def test_capture_all_skips_unrequested(self, service):
        with patch.object(service, "_generate") as gen:
            result = await service.capture_all_from_camera("cam")
        assert result.image is not None
        assert not result.classifications
        assert not result.detections
        gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_all_uses_requested_camera(self, mock_llama):
        cam_a = make_camera()
        cam_b = make_camera()
        service = Qwen3VL.new(
            make_config({"camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )
        await service.capture_all_from_camera("cam-b")
        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()
