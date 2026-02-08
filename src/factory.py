"""Component factory — registry pattern for OCR, chunking, embedding, etc."""

from __future__ import annotations

from typing import Any

# Lazy-import registry: maps (category, name) → fully-qualified class path
_REGISTRY: dict[tuple[str, str], str] = {
    # OCR
    ("ocr", "tesseract"): "src.ocr.tesseract_ocr.TesseractOCR",
    ("ocr", "easyocr"): "src.ocr.easyocr_extractor.EasyOCR",
    ("ocr", "paddleocr"): "src.ocr.paddleocr_extractor.PaddleOCRExtractor",
    # VLM-based OCR (for Loop 4 enhancement)
    ("ocr", "vlm"): "src.ocr.vlm_extractor.VLMExtractor",
    ("ocr", "qwen2vl"): "src.ocr.vlm_extractor.Qwen2VLExtractor",
    # Chunking
    ("chunking", "page"): "src.chunking.page_chunker.PageChunker",
    ("chunking", "parent_child"): "src.chunking.parent_child_chunker.ParentChildChunker",
    ("chunking", "semantic"): "src.chunking.semantic_chunker.SemanticChunker",
    # Embedding
    ("embedding", "sentence_transformer"): "src.embedding.sentence_transformer.SentenceTransformerEmbedding",
    # Vector store
    ("vectorstore", "chroma"): "src.vectorstore.chroma_store.ChromaStore",
    # Retrieval
    ("retrieval", "dense"): "src.retrieval.dense_retriever.DenseRetriever",
    ("retrieval", "parent_child"): "src.retrieval.parent_child_retriever.ParentChildRetriever",
    # Generation
    ("generation", "openai"): "src.generation.openai_generator.OpenAIGenerator",
}


def create(category: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered component by category and name."""
    key = (category, name)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown component: {category}/{name}. "
            f"Available: {[k for k in _REGISTRY if k[0] == category]}"
        )
    cls = _import_class(_REGISTRY[key])
    return cls(**kwargs)


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
