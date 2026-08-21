from __future__ import annotations

import base64
import mimetypes
import re

from config import config
from src.extract import extract_pdf, chunk_text_elements
from src.vision_summarize import caption_image
from src.embed_store import VectorStore
from src.retrieve import retrieve, select_image_items, RetrievedItem


def ingest(pdf_path: str, store: VectorStore | None = None, verbose: bool = True) -> VectorStore:
    """Extract, caption, embed, and store one PDF. Returns the store so
    callers can chain into `answer()` without re-opening it."""
    store = store or VectorStore()
    elements = extract_pdf(pdf_path)
    elements = chunk_text_elements(elements)

    ids, texts, metadatas = [], [], []
    for el in elements:
        if el.type == "text":
            searchable_text = el.text
        else:
            if verbose:
                print(f"  captioning {el.type} on page {el.page} ({el.id})...")
            searchable_text = caption_image(el.image_path, el.type)

        ids.append(el.id)
        texts.append(searchable_text)
        metadatas.append(
            {
                "doc_id": el.doc_id,
                "doc_name": el.doc_name,
                "page": el.page,
                "type": el.type,
                "image_path": el.image_path or "",
            }
        )

    store.add(ids=ids, texts=texts, metadatas=metadatas)
    if verbose:
        n_text = sum(1 for e in elements if e.type == "text")
        n_table = sum(1 for e in elements if e.type == "table")
        n_fig = sum(1 for e in elements if e.type == "figure")
        print(f"Indexed {pdf_path}: {n_text} text chunks, {n_table} tables, {n_fig} figures.")
    return store


def _build_context_blocks(items: list[RetrievedItem]) -> list[dict]:
    """Build Anthropic-style content blocks: text context + inline images
    for any retrieved table/figure."""
    blocks = []
    header = "\n".join(
        f"[{i+1}] ({it.type}, {it.doc_name} p.{it.page}) {it.text}" for i, it in enumerate(items)
    )
    blocks.append({"type": "text", "text": "Retrieved context:\n\n" + header})

    for it in select_image_items(items):
        if it.image_path:
            try:
                mime, _ = mimetypes.guess_type(it.image_path)
                with open(it.image_path, "rb") as f:
                    data = base64.standard_b64encode(f.read()).decode("utf-8")
                blocks.append(
                    {"type": "text", "text": f"Source image ({it.doc_name} p.{it.page}):"}
                )
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": mime or "image/png", "data": data}}
                )
            except FileNotFoundError:
                continue
    return blocks


_SYSTEM_PROMPT = (
    "You answer questions about a document using only the retrieved context "
    "provided below (text excerpts, table captions, and actual images of "
    "tables/figures). Look directly at any provided images before answering "
    "— they are the ground truth, the captions are only a search aid. Answer "
    "the user's exact question directly and concisely. Cite sources only as "
    "plain text such as '(page 2)'; never emit bracket citations, line-number "
    "citations, or internal reference tokens. Do not reveal your reasoning "
    "or use <think> tags. If the context doesn't contain the answer, say so "
    "plainly rather than guessing."
)


def _clean_answer(text: str) -> str:
    """Remove model-only reasoning and citation artifacts before rendering."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"【[^】]*】", "", text)
    text = re.sub(r"\[[^\]]*†[^\]]*\]", "", text)
    text = re.sub(r"(?<!\*)\s*\*\s+\*(?!\*)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def answer(question: str, store: VectorStore | None = None, top_k: int | None = None) -> str:
    store = store or VectorStore()
    items = retrieve(store, question, top_k=top_k)
    if not items:
        return "I couldn't find anything relevant in the indexed documents."

    content_blocks = _build_context_blocks(items)
    content_blocks.append({"type": "text", "text": f"\nQuestion: {question}"})

    if config.LLM_PROVIDER == "openai":
        return _answer_openai(content_blocks)
    if config.LLM_PROVIDER == "gemini":
        return _answer_gemini(content_blocks)
    if config.LLM_PROVIDER == "groq":
        return _answer_groq(content_blocks)
    return _answer_anthropic(content_blocks)


def _answer_anthropic(content_blocks: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.GROQ_API_KEY)
    resp = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content_blocks}],
    )
    return _clean_answer("".join(b.text for b in resp.content if b.type == "text"))


def _answer_openai(content_blocks: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    # translate Anthropic-style blocks -> OpenAI-style
    oa_content = []
    for b in content_blocks:
        if b["type"] == "text":
            oa_content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            mime = b["source"]["media_type"]
            data = b["source"]["data"]
            oa_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})

    resp = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": oa_content},
        ],
    )
    return _clean_answer(resp.choices[0].message.content or "")


def _answer_groq(content_blocks: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    text_parts = []
    for block in content_blocks:
        if block["type"] == "text":
            text_parts.append(block["text"])

    resp = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(text_parts)},
        ],
    )
    return _clean_answer(resp.choices[0].message.content or "")


def _answer_gemini(content_blocks: list[dict]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    parts = []
    for b in content_blocks:
        if b["type"] == "text":
            parts.append(b["text"])
        elif b["type"] == "image":
            raw = base64.standard_b64decode(b["source"]["data"])
            parts.append(types.Part.from_bytes(data=raw, mime_type=b["source"]["media_type"]))

    resp = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT, max_output_tokens=1000),
    )
    return _clean_answer(resp.text or "")
