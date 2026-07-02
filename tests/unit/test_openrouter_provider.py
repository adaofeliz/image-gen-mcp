"""Unit tests for OpenRouter provider."""

import base64

import pytest

from image_gen_mcp.providers.base import ProviderConfig, ProviderError
from image_gen_mcp.providers.openrouter import (
    OpenRouterProvider,
    _build_image_config,
    _build_messages,
    _parse_base64_data_url,
    _size_to_aspect_ratio,
)

_SAMPLE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_SAMPLE_IMAGE_DATA_URL = f"data:image/png;base64,{_SAMPLE_B64}"
_SAMPLE_BYTES = base64.b64decode(_SAMPLE_B64)


def _fake_image_response(model_slug: str, image_data_url: str) -> dict:
    return {
        "id": "chatcmpl-123",
        "model": model_slug,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here is your image.",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2000,
            "total_tokens": 2010,
        },
        "created": 1700000000,
    }


@pytest.fixture
def provider():
    return OpenRouterProvider(
        ProviderConfig(api_key="sk-or-v1-test", enabled=True)
    )


class TestSizeToAspectRatio:
    def test_known_sizes(self):
        assert _size_to_aspect_ratio("1024x1024") == "1:1"
        assert _size_to_aspect_ratio("1536x1024") == "3:2"
        assert _size_to_aspect_ratio("1024x1536") == "2:3"
        assert _size_to_aspect_ratio("3840x2160") == "16:9"

    def test_normalized_whitespace(self):
        assert _size_to_aspect_ratio("  1024x1024  ") == "1:1"

    def test_unknown_size_returns_none(self):
        assert _size_to_aspect_ratio("2048x1152") is None
        assert _size_to_aspect_ratio("auto") is None
        assert _size_to_aspect_ratio("") is None


class TestParseBase64DataUrl:
    def test_valid_png(self):
        result_bytes, result_fmt = _parse_base64_data_url(_SAMPLE_IMAGE_DATA_URL)
        assert result_bytes == _SAMPLE_BYTES
        assert result_fmt == "png"

    def test_valid_jpeg(self):
        url = f"data:image/jpeg;base64,{_SAMPLE_B64}"
        result_bytes, result_fmt = _parse_base64_data_url(url)
        assert result_bytes == _SAMPLE_BYTES
        assert result_fmt == "jpeg"

    def test_invalid_format_raises(self):
        with pytest.raises(ProviderError, match="Unexpected image URL format"):
            _parse_base64_data_url("not-a-data-url")


class TestBuildImageConfig:
    def test_empty_params(self):
        assert _build_image_config({}) == {}

    def test_auto_values_skipped(self):
        assert _build_image_config(
            {"size": "auto", "quality": "auto", "background": "auto"}
        ) == {}

    def test_size_aspect_ratio(self):
        config = _build_image_config({"size": "1024x1024"})
        assert config == {"aspect_ratio": "1:1"}

    def test_size_image_size_fallback(self):
        config = _build_image_config({"size": "2048x1152"})
        assert config == {"size": "2048x1152"}

    def test_quality_and_format(self):
        config = _build_image_config(
            {"quality": "high", "output_format": "jpeg"}
        )
        assert config["quality"] == "high"
        assert config["output_format"] == "jpeg"

    def test_compression_for_lossy(self):
        config = _build_image_config(
            {"output_format": "jpeg", "compression": 80}
        )
        assert config["output_compression"] == 80

    def test_compression_100_skipped(self):
        config = _build_image_config(
            {"output_format": "jpeg", "compression": 100}
        )
        assert "output_compression" not in config


class TestBuildMessages:
    def test_text_only(self):
        messages = _build_messages("a sunset")
        assert messages == [{"role": "user", "content": "a sunset"}]

    def test_with_image_bytes(self):
        messages = _build_messages("edit this", _SAMPLE_BYTES)
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1] == {"type": "text", "text": "edit this"}

    def test_with_image_data_url(self):
        messages = _build_messages("edit this", _SAMPLE_IMAGE_DATA_URL)
        content = messages[0]["content"]
        assert content[0]["image_url"]["url"] == _SAMPLE_IMAGE_DATA_URL


class TestProviderInit:
    def test_creates_with_api_key(self):
        provider = OpenRouterProvider(
            ProviderConfig(api_key="sk-or-v1-test", enabled=True)
        )
        assert provider.is_available()

    def test_disabled_when_no_key(self):
        provider = OpenRouterProvider(
            ProviderConfig(api_key="", enabled=True)
        )
        assert not provider.is_available()

    def test_custom_headers_forwarded(self):
        provider = OpenRouterProvider(
            ProviderConfig(
                api_key="sk-or-v1-test",
                enabled=True,
                custom_headers={
                    "X-Title": "Test App",
                    "HTTP-Referer": "https://test.example",
                },
            )
        )
        assert "X-Title" in provider._client.headers
        assert "HTTP-Referer" in provider._client.headers


class TestModelRegistration:
    def test_supported_models(self):
        assert "gpt-5.4-image-2" in OpenRouterProvider.SUPPORTED_MODELS
        assert "gpt-5-image" in OpenRouterProvider.SUPPORTED_MODELS
        assert "gpt-5-image-mini" in OpenRouterProvider.SUPPORTED_MODELS

    def test_gpt_54_image_2_capabilities(self):
        cap = OpenRouterProvider.SUPPORTED_MODELS["gpt-5.4-image-2"]
        assert cap.supports_custom_sizes is True
        assert cap.supports_background is True
        assert cap.supports_compression is True
        assert cap.size_constraints is not None
        assert cap.size_constraints["max_edge"] == 3840
        assert "3840x2160" in cap.supported_sizes

    def test_gpt_5_image_capabilities(self):
        cap = OpenRouterProvider.SUPPORTED_MODELS["gpt-5-image"]
        assert cap.supports_custom_sizes is False
        assert "auto" in cap.supported_sizes

    def test_model_slug_raises_for_unknown(self, provider):
        with pytest.raises(ProviderError, match="not supported"):
            provider._model_slug("unknown-model")


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_successful_generation(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json=_fake_image_response(
                    "openai/gpt-5.4-image-2", _SAMPLE_IMAGE_DATA_URL
                ),
            )
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.generate_image(
            model="gpt-5.4-image-2",
            prompt="a sunset",
            quality="high",
            size="1024x1024",
            output_format="png",
        )
        assert result.image_data == _SAMPLE_BYTES
        assert result.metadata["model"] == "gpt-5.4-image-2"
        assert result.metadata["output_format"] == "png"
        assert result.metadata["usage"]["total_tokens"] == 2010

    @pytest.mark.asyncio
    async def test_unsupported_model_raises(self, provider: OpenRouterProvider):
        with pytest.raises(ProviderError, match="not supported"):
            await provider.generate_image(model="dall-e-3", prompt="test")

    @pytest.mark.asyncio
    async def test_no_images_in_response(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "No image.",
                                "images": [],
                            }
                        }
                    ]
                },
            )
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(ProviderError, match="No images"):
            await provider.generate_image(
                model="gpt-5-image", prompt="test"
            )

    @pytest.mark.asyncio
    async def test_http_error(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "Unauthorized"})
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        with pytest.raises(ProviderError, match="OpenRouter API error"):
            await provider.generate_image(
                model="gpt-5-image", prompt="test"
            )


class TestEditImage:
    @pytest.mark.asyncio
    async def test_successful_edit(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json=_fake_image_response(
                    "openai/gpt-5.4-image-2", _SAMPLE_IMAGE_DATA_URL
                ),
            )
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.edit_image(
            model="gpt-5.4-image-2",
            image_data=_SAMPLE_BYTES,
            prompt="add a sun",
            size="1024x1024",
        )
        assert result.image_data == _SAMPLE_BYTES
        assert result.metadata["operation"] == "edit"

    @pytest.mark.asyncio
    async def test_mask_data_rejected(self, provider: OpenRouterProvider):
        with pytest.raises(ProviderError, match="Mask-based editing"):
            await provider.edit_image(
                model="gpt-5.4-image-2",
                image_data=_SAMPLE_BYTES,
                prompt="test",
                mask_data=_SAMPLE_BYTES,
            )

    @pytest.mark.asyncio
    async def test_unsupported_model_raises(self, provider: OpenRouterProvider):
        with pytest.raises(ProviderError, match="not supported"):
            await provider.edit_image(
                model="dall-e-3",
                image_data=_SAMPLE_BYTES,
                prompt="test",
            )


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self, provider: OpenRouterProvider):
        import httpx

        def handler(req):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "openai/gpt-5.4-image-2"},
                        {"id": "openai/gpt-5-image"},
                        {"id": "openai/gpt-5-image-mini"},
                    ]
                },
            )

        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.check_health()
        assert result["status"] == "healthy"
        assert "gpt-5.4-image-2" in result["models_available"]

    @pytest.mark.asyncio
    async def test_degraded_missing_models(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"data": [{"id": "openai/gpt-5-image"}]}
            )
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.check_health()
        assert result["status"] == "healthy"
        assert "gpt-5-image" in result["models_available"]
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_unhealthy_no_models(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": []})
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.check_health()
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_unhealthy_on_error(self, provider: OpenRouterProvider):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(500, json={"error": "down"})
        )
        provider._client = httpx.AsyncClient(
            transport=transport,
            base_url=provider._base_url,
            headers=provider._client.headers,
        )

        result = await provider.check_health()
        assert result["status"] == "unhealthy"
        assert "error" in result


class TestEstimateCost:
    def test_returns_zero_with_note(self, provider):
        result = provider.estimate_cost("gpt-5-image", "test")
        assert result["estimated_cost_usd"] == 0.0
        assert "note" in result


class TestClose:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, provider: OpenRouterProvider):
        await provider.close()
        await provider.close()  # should not raise
