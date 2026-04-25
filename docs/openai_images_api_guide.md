# OpenAI Images API Guide

> **Note**: This server now supports multiple AI providers. For comprehensive documentation covering OpenAI, Gemini, and other providers, see the [Multi-Provider API Guide](multi_provider_api_guide.md).

This guide covers the OpenAI Images API integration used by the Image Gen MCP Server.

## Quick Reference

The Image Gen MCP Server supports the following OpenAI models:
- **gpt-image-2**: Current flagship — flexible sizing up to 4K (3840x2160), $30/1M image output tokens
- **gpt-image-1.5**: Previous flagship with fixed preset sizes
- **gpt-image-1**: Original gpt-image model
- **dall-e-3**: High-quality creative image generation
- **dall-e-2**: Standard image generation with editing capabilities

## Create Image

**Endpoint**: `POST https://api.openai.com/v1/images/generations`

Creates an image given a prompt. [Learn more](https://platform.openai.com/docs/guides/images).

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Text description of the desired image(s). Max 32,000 chars for `gpt-image-*`, 1,000 for `dall-e-2`, 4,000 for `dall-e-3` |
| `model` | string | No | Model to use: `gpt-image-2` (default), `gpt-image-1.5`, `gpt-image-1`, `dall-e-3`, or `dall-e-2` |
| `n` | integer | No | Number of images. `gpt-image-*` and `dall-e-3`: `n=1` only. `dall-e-2`: up to 10 |
| `size` | string | No | Image size. `gpt-image-1` / `gpt-image-1.5`: `1024x1024`, `1536x1024`, `1024x1536`, or `auto`. `gpt-image-2`: presets plus `3840x2160` and any `WxH` (multiples of 16, max edge 3840, aspect ≤3:1, 655,360–8,294,400 pixels) |
| `quality` | string | No | Image quality. `gpt-image-*`: `auto`, `high`, `medium`, `low`. `dall-e-3`: `hd` or `standard` |
| `style` | string | No | Style for `dall-e-3`: `vivid` or `natural` |
| `output_format` | string | No | Format for `gpt-image-*`: `png`, `jpeg`, or `webp` |
| `background` | string | No | Background. `gpt-image-1` / `gpt-image-1.5`: `transparent`, `opaque`, `auto`. `gpt-image-2`: accepts the same values, but `transparent` is auto-downgraded to `auto` since gpt-image-2 has no native transparency support |
| `moderation` | string | No | Moderation level for `gpt-image-*`: `auto` or `low` |

### Example Request

```bash
curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PROVIDERS__OPENAI__API_KEY" \
  -d '{
    "model": "gpt-image-2",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'
```

### Python Example

```python
import base64
from openai import OpenAI
client = OpenAI()

img = client.images.generate(
    model="gpt-image-2",
    prompt="A cute baby sea otter",
    n=1,
    size="1024x1024"
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

## Create Image Edit

**Endpoint**: `POST https://api.openai.com/v1/images/edits`

Creates an edited or extended image given source images and a prompt. Supports `gpt-image-*` (recommend `gpt-image-2`) and `dall-e-2`.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file/array | Yes | Image(s) to edit. For `gpt-image-*`: up to 16 images, PNG/WEBP/JPG < 50MB |
| `prompt` | string | Yes | Edit instructions. Max 32,000 chars for `gpt-image-*`, 1,000 for `dall-e-2` |
| `model` | string | No | Model to use: `gpt-image-2` (default), `gpt-image-1.5`, `gpt-image-1`, or `dall-e-2` |
| `mask` | file | No | PNG mask file indicating areas to edit |
| `n` | integer | No | Number of images to generate (1-10) |

### Example Request

```bash
curl -X POST "https://api.openai.com/v1/images/edits" \
  -H "Authorization: Bearer $PROVIDERS__OPENAI__API_KEY" \
  -F "model=gpt-image-2" \
  -F "image[]=@body-lotion.png" \
  -F "image[]=@bath-bomb.png" \
  -F 'prompt=Create a lovely gift basket with these items'
```

## Using with MCP Server

### Configuration

```bash
# Environment variables
PROVIDERS__OPENAI__API_KEY=sk-your-openai-api-key-here
PROVIDERS__OPENAI__BASE_URL=https://api.openai.com/v1
PROVIDERS__OPENAI__ORGANIZATION=org-your-org-id
PROVIDERS__OPENAI__ENABLED=true
```

### Generate Image via MCP

```python
# Generate with OpenAI model
result = await session.call_tool("generate_image", {
    "prompt": "A beautiful sunset over mountains",
    "model": "gpt-image-2",
    "quality": "high",
    "size": "1536x1024"
})
```

### Edit Image via MCP

```python
# Edit image using OpenAI models
result = await session.call_tool("edit_image", {
    "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "prompt": "Add a rainbow to the sky",
    "model": "gpt-image-2"
})
```

## Model Comparison

| Model | Generation | Editing | Max Prompt | Sizes | Quality Options |
|-------|------------|---------|------------|-------|----------------|
| gpt-image-2 | ✅ | ✅ | 32,000 chars | Presets + any WxH (mult-of-16, max edge 3840, aspect ≤3:1) | auto, high, medium, low |
| gpt-image-1.5 | ✅ | ✅ | 32,000 chars | 1024x1024, 1536x1024, 1024x1536 | auto, high, medium, low |
| gpt-image-1 | ✅ | ✅ | 32,000 chars | 1024x1024, 1536x1024, 1024x1536 | auto, high, medium, low |
| dall-e-3 | ✅ | ❌ | 4,000 chars | 1024x1024, 1792x1024, 1024x1792 | hd, standard |
| dall-e-2 | ✅ | ✅ | 1,000 chars | 256x256, 512x512, 1024x1024 | standard |

## Response Format

```json
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50
  }
}
```

## Error Handling

Common error codes:
- `400`: Bad request - invalid parameters
- `401`: Unauthorized - invalid API key
- `429`: Rate limit exceeded
- `500`: Internal server error

## Best Practices

1. **Model Selection**: 
   - Use `gpt-image-2` for best quality, flexible sizing (incl. 4K), and lowest per-image cost
   - Use `dall-e-3` for creative, artistic images
   - Use `dall-e-2` for simple images or editing

2. **Parameter Optimization**:
   - Adjust quality based on output requirements
   - Use appropriate sizes for your use case
   - Set moderation levels according to content policy needs

3. **Error Handling**: Implement retry logic with exponential backoff
4. **Rate Limiting**: Respect OpenAI's rate limits
5. **Cost Management**: Monitor token usage and costs

## Migration Notes

If you're migrating from the single-provider version:
- Add `model` parameter to specify which OpenAI model to use
- Use `list_available_models` tool to discover available models
- Consider trying different models for different use cases

For complete documentation covering all supported providers, see the [Multi-Provider API Guide](multi_provider_api_guide.md).