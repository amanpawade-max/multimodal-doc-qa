"""Central configuration. Every value can be overridden via environment
variable so you can run against different providers/models without editing
code."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Providers -----------------------------------------------------
    # Groq is used for both image captioning and final answer generation.
    # "groq" is free and fast: get a no-card API key at
    # https://console.groq.com/keys and set GROQ_API_KEY. Its free-tier
    # vision model is labeled "Preview" by Groq, meaning the exact model
    # name can be deprecated with fairly short notice — check
    # https://console.groq.com/docs/vision if VISION_MODEL below 404s.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "groq")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    _DEFAULT_MODEL_BY_PROVIDER = {
        "anthropic": "claude-sonnet-4-6",
        "gemini": "gemini-2.5-flash",
        "groq": "openai/gpt-oss-120b",
    }

    VISION_MODEL: str = os.getenv(
        "VISION_MODEL",
        os.getenv("QWEN_VISION_MODEL", "qwen/qwen3.6-27b"),
    )
    GENERATION_MODEL: str = os.getenv(
        "GENERATION_MODEL",
        _DEFAULT_MODEL_BY_PROVIDER.get(os.getenv("LLM_PROVIDER", "groq"), "meta-llama/llama-4-scout-17b-16e-instruct"),
    )

    # --- Embeddings ------------------------------------------------------
    # "local" uses sentence-transformers (no API calls, runs on CPU).
    # "openai" uses OpenAI's embedding API instead.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Storage ---------------------------------------------------------
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")

    @property
    def IMAGE_DIR(self) -> str:
        path = os.path.join(self.DATA_DIR, "images")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def INDEX_DIR(self) -> str:
        path = os.path.join(self.DATA_DIR, "index")
        os.makedirs(path, exist_ok=True)
        return path

    # --- Chunking / retrieval --------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    # A vector search can return several loosely related figure captions.
    # Keep general text context broad, but return only the strongest visual
    # match unless an application explicitly requests more.
    MAX_RETRIEVED_IMAGES: int = int(os.getenv("MAX_RETRIEVED_IMAGES", "1"))
    CONTEXT_NEIGHBOR_CHARS: int = int(os.getenv("CONTEXT_NEIGHBOR_CHARS", "400"))

    # --- Extraction --------------------------------------------------------
    MIN_IMAGE_DIM_PX: int = int(os.getenv("MIN_IMAGE_DIM_PX", "80"))  # skip tiny decorative images
    PAGE_RENDER_DPI: int = int(os.getenv("PAGE_RENDER_DPI", "200"))


config = Config()
