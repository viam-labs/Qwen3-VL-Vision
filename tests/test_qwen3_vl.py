from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from src.qwen3_vl import qwen3_vl as Qwen3VL


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
def mock_hf():
    model = MagicMock()
    model.device = torch.device("cpu")
    processor = MagicMock()

    def apply_chat_template(*_args, **_kwargs):
        inputs = MagicMock()
        inputs.input_ids = torch.tensor([[1, 2, 3]])
        inputs.to = MagicMock(return_value=inputs)
        return inputs

    processor.apply_chat_template.side_effect = apply_chat_template
    processor.batch_decode.return_value = ["decoded"]

    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    with (
        patch(
            "src.qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained",
            return_value=model,
        ) as from_pretrained,
        patch(
            "src.qwen3_vl.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as proc_from_pretrained,
    ):
        yield from_pretrained, proc_from_pretrained, model, processor


@pytest.fixture
def service(mock_hf):
    _, _, model, processor = mock_hf
    cam = make_camera()
    deps = {Camera.get_resource_name("cam"): cam}
    config = make_config({"camera": "cam"})
    instance = Qwen3VL.new(config, deps)
    instance._test_camera = cam
    instance._test_model = model
    instance._test_processor = processor
    return instance


class TestValidate:
    def test_requires_camera(self):
        with pytest.raises(Exception, match="camera is required"):
            Qwen3VL.validate(make_config({}))

    def test_returns_camera_dependency(self):
        assert Qwen3VL.validate(make_config({"camera": "cam"})) == (["cam"], [])


class TestReconfigure:
    def test_defaults(self, mock_hf):
        from_pretrained, proc_from_pretrained, _, _ = mock_hf
        cam = make_camera()
        Qwen3VL.new(
            make_config({"camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        from_pretrained.assert_called_once()
        assert from_pretrained.call_args[0][0] == "Qwen/Qwen3-VL-2B-Instruct"
        assert from_pretrained.call_args.kwargs["device_map"] == "auto"
        proc_from_pretrained.assert_called_once_with("Qwen/Qwen3-VL-2B-Instruct")

    def test_custom_model(self, mock_hf):
        from_pretrained, _, _, _ = mock_hf
        cam = make_camera()
        Qwen3VL.new(
            make_config(
                {
                    "camera": "cam",
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "device_map": "cpu",
                    "dtype": "float32",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        assert from_pretrained.call_args[0][0] == "Qwen/Qwen3-VL-4B-Instruct"
        assert from_pretrained.call_args.kwargs["device_map"] == "cpu"
        assert from_pretrained.call_args.kwargs["dtype"] is torch.float32

    def test_missing_camera_dependency(self, mock_hf):
        with pytest.raises(Exception, match="camera dependency"):
            Qwen3VL.new(make_config({"camera": "cam"}), {})


class TestGeneration:
    @pytest.mark.asyncio
    async def test_detections_use_greedy_decoding(self, service):
        response = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "box"}]'
        with patch.object(service, "_generate", wraps=service._generate) as gen:
            service._test_processor.batch_decode.return_value = [response]
            await service.get_detections(make_jpeg_image())
        assert gen.call_args.kwargs.get("greedy") is True

    @pytest.mark.asyncio
    async def test_generate_defaults_to_do_sample_false(self, service):
        service._test_processor.batch_decode.return_value = ["ok"]
        await service.get_classifications(make_jpeg_image(), 1)
        kwargs = service._test_model.generate.call_args.kwargs
        assert kwargs.get("do_sample") is False
        assert kwargs.get("max_new_tokens") == 512


class TestClassifications:
    @pytest.mark.asyncio
    async def test_default_question(self, service):
        with patch.object(service, "_generate", return_value="a red square") as gen:
            result = await service.get_classifications(make_jpeg_image(), 1)
        assert result == [{"class_name": "a red square", "confidence": 1}]
        assert gen.call_args[0][1] == "describe this image"

    @pytest.mark.asyncio
    async def test_config_classification_prompt(self, mock_hf):
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
    async def test_extra_question_overrides_config_prompt(self, mock_hf):
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
        assert result[0]["class_name"] == "yes"
        assert gen.call_args[0][1] == "is there a person?"

    @pytest.mark.asyncio
    async def test_strips_thinking_content(self, service):
        service._test_processor.batch_decode.return_value = [
            "<think>reasoning here</think>\nfinal answer"
        ]
        result = await service.get_classifications(make_jpeg_image(), 1)
        assert result[0]["class_name"] == "final answer"

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        with patch.object(service, "_generate", return_value="from cam"):
            result = await service.get_classifications_from_camera("cam", 1)
        assert result[0]["class_name"] == "from cam"
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_hf):
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
        with patch.object(service, "_generate", return_value=response) as gen:
            result = await service.get_detections(make_jpeg_image(100, 50))

        prompt = gen.call_args[0][1]
        assert "Locate every object" in prompt
        assert [d["class_name"] for d in result] == ["person", "chair"]
        assert result[0]["x_min"] == 10
        assert result[0]["y_min"] == 10
        assert result[0]["x_max"] == 30
        assert result[0]["y_max"] == 20
        assert result[0]["x_min_normalized"] == pytest.approx(0.1)
        assert result[1]["x_min"] == 50
        assert result[1]["y_max"] == 45

    @pytest.mark.asyncio
    async def test_query_limits_detection(self, service):
        response = '[{"bbox_2d": [0, 0, 1000, 1000], "label": "person"}]'
        with patch.object(service, "_generate", return_value=response) as gen:
            result = await service.get_detections(
                make_jpeg_image(), extra={"query": "people"}
            )
        prompt = gen.call_args[0][1]
        assert "Locate all people" in prompt
        assert result[0]["class_name"] == "person"

    @pytest.mark.asyncio
    async def test_strips_thinking_before_json(self, service):
        response = (
            "<think>I see a cup</think>\n"
            '[{"bbox_2d": [0, 0, 500, 500], "label": "cup"}]'
        )
        text = service._strip_thinking(response)
        result = service._detections_from_response(text, 100, 50)
        assert len(result) == 1
        assert result[0]["class_name"] == "cup"
        assert result[0]["x_max"] == 50

    @pytest.mark.asyncio
    async def test_parses_fenced_json(self, service):
        response = '```json\n[{"bbox_2d": [0, 0, 1000, 1000], "label": "box"}]\n```'
        result = service._detections_from_response(response, 100, 50)
        assert result[0]["class_name"] == "box"
        assert result[0]["x_max"] == 100

    @pytest.mark.asyncio
    async def test_empty_or_invalid_output(self, service):
        with patch.object(service, "_generate", return_value="no boxes here"):
            result = await service.get_detections(make_jpeg_image())
        assert result == []

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        response = '[{"bbox_2d": [0, 0, 500, 500], "label": "cup"}]'
        with patch.object(service, "_generate", return_value=response):
            result = await service.get_detections_from_camera("cam")
        assert len(result) == 1
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_hf):
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
        assert result.classifications[0]["class_name"] == "a scene"
        assert result.detections[0]["class_name"] == "box"

    @pytest.mark.asyncio
    async def test_capture_all_skips_unrequested(self, service):
        with patch.object(service, "_generate") as gen:
            result = await service.capture_all_from_camera("cam")
        assert result.image is not None
        assert not result.classifications
        assert not result.detections
        gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_all_uses_requested_camera(self, mock_hf):
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
