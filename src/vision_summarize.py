"""Turn a table/figure image crop into a dense text caption using a vision
model, so it becomes searchable by a text embedding index."""
from __future__ import annotations

import base64
import mimetypes

from config import config

_TABLE_PROMPT = (
    "This is a cropped table from a document. Describe what it contains: "
    "its title/purpose if inferable, the column headers, and the key "
    "numbers or trends a reader would care about. Be specific and dense — "
    "this caption is the only thing a search index will see, so include "
    "exact figures where legible. 3-6 sentences, no preamble."
)

_FIGURE_PROMPT = (
    "This is a cropped figure/image from a document (could be a diagram, "
    "chart, photo, or illustration, e.g. an assembly step or a schematic). "
    "Describe precisely what it shows: what kind of visual it is, the "
    "main elements, any labels or numbers visible, and what it communicates "
    "in context. Be specific and dense — this caption is the only thing a "
    "search index will see. 3-6 sentences, no preamble."
)


def _encode_image(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime


def _caption_anthropic(image_path: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    data, mime = _encode_image(image_path)
    resp = client.messages.create(
        model=config.VISION_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def _caption_gemini(image_path: str, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    data, mime = _encode_image(image_path)
    resp = client.models.generate_content(
        model=config.VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=base64.standard_b64decode(data), mime_type=mime),
            prompt,
        ],
    )
    return (resp.text or "").strip()


def _caption_groq(image_path: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    data, mime = _encode_image(image_path)
    resp = client.chat.completions.create(
        model=config.VISION_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def caption_image(image_path: str, element_type: str) -> str:
    """element_type is 'table' or 'figure' — used to pick the right prompt."""
    prompt = _TABLE_PROMPT if element_type == "table" else _FIGURE_PROMPT
    if config.VISION_PROVIDER == "gemini":
        return _caption_gemini(image_path, prompt)
    if config.VISION_PROVIDER == "groq":
        return _caption_groq(image_path, prompt)
    return _caption_anthropic(image_path, prompt)
