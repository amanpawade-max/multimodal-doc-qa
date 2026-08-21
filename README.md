# Multimodal Document QA

Ask questions over PDFs that mix prose, tables, and figures (research papers,
IKEA-style manuals, financial statements) and get answers grounded in the
actual tables/images, not just surrounding text.

## How it works

**Ingestion (once per document)**
1. `src/extract.py` walks each page of the PDF with PyMuPDF, pulling out
   three kinds of elements: text blocks, tables (via PyMuPDF's built-in
   table finder), and figures (embedded images). Tables and figures are
   rendered to PNG crops on disk.
2. `src/vision_summarize.py` sends each table/figure crop to Groq's
   `qwen/qwen3.6-27b` vision model and gets back a dense text caption.
   describing what it shows.
3. `src/embed_store.py` embeds the text blocks and the vision captions, and
   stores everything in a persistent ChromaDB collection. Each vector's
   metadata carries a pointer back to the source: page number, bounding
   box, and (for tables/figures) the path to the raw image crop.

**Query (per question)**
4. `src/retrieve.py` embeds the user's question, does a similarity search
   against the collection, and pulls back the top-k chunks *plus* the raw
   image crops for any table/figure hits, plus neighboring text on the same
   page for context.
5. `src/pipeline.py` assembles all of that into one multimodal prompt (text
   + images) and asks the generation model for an answer that's grounded in
   what's actually on the page.

```
PDF -> extract (text/table/figure) -> vision captions -> embed -> ChromaDB
                                                                      |
question -> embed -> similarity search -> text + image crops -> multimodal LLM -> answer
```

## Setup

```bash
pip install -r requirements.txt
```

Then create a `.env` file in the project root (never commit this) with Groq
credentials. Groq handles both image extraction and final answer generation.

```
# .env
VISION_PROVIDER=groq
VISION_MODEL=qwen/qwen3.6-27b
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...

# .env — or switch providers by setting LLM_PROVIDER and the matching key
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Ingest a document (extracts, captions, embeds, stores)
python ingest.py path/to/manual.pdf

# Ask questions against everything you've ingested
python query.py "What size wrench do I need for step 4?"

# Or run the interactive UI
streamlit run app.py
```

## Config

All tunables live in `config.py` and can be overridden via environment
variables: chunk size/overlap, top-k, `MAX_RETRIEVED_IMAGES` (defaults to one
best-matching table/figure crop), embedding provider (local
sentence-transformers or OpenAI), which models to use for vision captioning
vs. final generation, and where data is stored on disk (`./data` by
default — `./data/images` for crops, `./data/index` for the ChromaDB store).

## Design notes / things to tune for your documents

- **Table granularity**: right now each detected table is captioned as one
  unit. For documents with huge tables where users ask about specific cells
  ("what was Q3 APAC revenue"), consider also indexing `table.to_pandas()`
  row-by-row as a secondary text index.
- **Vision captioning cost**: this is the most expensive step at ingest
  time — it's O(number of tables + figures), not O(pages). Batch it or
  cache aggressively if you're re-ingesting.
- **Reranking**: plain vector similarity can be noisy for numeric lookups
  in tables. A hybrid BM25 + vector reranker is a natural next step if you
  see retrieval misses on specific figures.
- **Scanned/image-only PDFs**: this pipeline assumes a digital PDF with a
  text layer. For scanned documents, run OCR (e.g. Tesseract) before
  extraction, or route whole pages through the vision model as figures.
