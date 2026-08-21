"""Layout-aware PDF extraction.

Walks each page and produces three kinds of Element:
  - "text"   : a prose block
  - "table"  : a detected table, rendered to a PNG crop
  - "figure" : an embedded image, rendered to a PNG crop

Tables and figures get their bounding boxes rendered directly from the page
(rather than re-encoding the embedded image bytes) so the crop always looks
like what a human sees on the page, borders and surrounding whitespace
included.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from config import config


@dataclass
class Element:
    id: str
    doc_id: str
    doc_name: str
    page: int  # 1-indexed
    type: str  # "text" | "table" | "figure"
    bbox: tuple  # (x0, y0, x1, y1) in PDF points
    text: str = ""  # raw extracted text (prose, or table-as-text)
    image_path: Optional[str] = None  # set for table / figure crops


def _rects_overlap(a, b, threshold: float = 0.6) -> bool:
    """True if rect `a` is mostly contained inside rect `b`."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return (inter / area_a) >= threshold


def _render_crop(page: fitz.Page, bbox: fitz.Rect, out_path: str) -> None:
    zoom = config.PAGE_RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=bbox)
    pix.save(out_path)


def extract_pdf(pdf_path: str, doc_id: Optional[str] = None) -> list[Element]:
    """Extract all text/table/figure elements from a PDF.

    Returns a flat list of Element, in reading order per page.
    """
    doc_id = doc_id or str(uuid.uuid4())[:8]
    doc_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    elements: list[Element] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1
        exclude_boxes: list[tuple] = []

        # --- Tables ------------------------------------------------------
        try:
            table_finder = page.find_tables()
            tables = list(table_finder.tables)
        except Exception:
            tables = []

        for tab in tables:
            bbox = tuple(tab.bbox)
            exclude_boxes.append(bbox)
            elem_id = f"{doc_id}-p{page_no}-table-{uuid.uuid4().hex[:6]}"
            img_path = os.path.join(config.IMAGE_DIR, f"{elem_id}.png")
            try:
                _render_crop(page, fitz.Rect(bbox), img_path)
            except Exception:
                img_path = None
            try:
                raw_text = tab.to_pandas().to_csv(index=False)
            except Exception:
                raw_text = ""
            elements.append(
                Element(
                    id=elem_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_no,
                    type="table",
                    bbox=bbox,
                    text=raw_text,
                    image_path=img_path,
                )
            )

        # --- Figures (embedded images) ------------------------------------
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                if rect.width < config.MIN_IMAGE_DIM_PX or rect.height < config.MIN_IMAGE_DIM_PX:
                    continue
                bbox = tuple(rect)
                if any(_rects_overlap(bbox, t) for t in exclude_boxes):
                    continue  # already captured as part of a table
                exclude_boxes.append(bbox)
                elem_id = f"{doc_id}-p{page_no}-fig-{uuid.uuid4().hex[:6]}"
                img_path = os.path.join(config.IMAGE_DIR, f"{elem_id}.png")
                try:
                    _render_crop(page, rect, img_path)
                except Exception:
                    continue
                elements.append(
                    Element(
                        id=elem_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page_no,
                        type="figure",
                        bbox=bbox,
                        image_path=img_path,
                    )
                )

        # --- Text blocks ---------------------------------------------------
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if not text:
                continue
            bbox = (x0, y0, x1, y1)
            if any(_rects_overlap(bbox, ex) for ex in exclude_boxes):
                continue  # text that's actually inside a table/figure region
            elem_id = f"{doc_id}-p{page_no}-text-{uuid.uuid4().hex[:6]}"
            elements.append(
                Element(
                    id=elem_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_no,
                    type="text",
                    bbox=bbox,
                    text=text,
                )
            )

    doc.close()
    return elements


def chunk_text_elements(elements: list[Element]) -> list[Element]:
    """Merge/split raw text elements into ~CHUNK_SIZE chunks per page,
    preserving page number and a union bounding box. Table/figure elements
    pass through unchanged."""
    chunked: list[Element] = []
    by_page: dict[int, list[Element]] = {}
    for el in elements:
        if el.type != "text":
            chunked.append(el)
            continue
        by_page.setdefault(el.page, []).append(el)

    for page_no, els in by_page.items():
        # reading order: top-to-bottom, left-to-right
        els.sort(key=lambda e: (round(e.bbox[1]), e.bbox[0]))
        buf, buf_bbox, buf_ids = "", None, []

        def flush():
            if not buf.strip():
                return
            elem_id = f"{els[0].doc_id}-p{page_no}-chunk-{uuid.uuid4().hex[:6]}"
            chunked.append(
                Element(
                    id=elem_id,
                    doc_id=els[0].doc_id,
                    doc_name=els[0].doc_name,
                    page=page_no,
                    type="text",
                    bbox=buf_bbox or (0, 0, 0, 0),
                    text=buf.strip(),
                )
            )

        for el in els:
            candidate = (buf + "\n\n" + el.text).strip() if buf else el.text
            if len(candidate) > config.CHUNK_SIZE and buf:
                flush()
                # start new chunk with overlap from the tail of the previous one
                tail = buf[-config.CHUNK_OVERLAP:]
                buf = (tail + "\n\n" + el.text).strip()
                buf_bbox = el.bbox
            else:
                buf = candidate
                buf_bbox = (
                    min(buf_bbox[0], el.bbox[0]) if buf_bbox else el.bbox[0],
                    min(buf_bbox[1], el.bbox[1]) if buf_bbox else el.bbox[1],
                    max(buf_bbox[2], el.bbox[2]) if buf_bbox else el.bbox[2],
                    max(buf_bbox[3], el.bbox[3]) if buf_bbox else el.bbox[3],
                ) if buf_bbox else el.bbox
        flush()

    return chunked
