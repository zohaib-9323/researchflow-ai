"""Split extracted pages into overlapping chunks while preserving source metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def split_documents(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split page-level documents; every chunk keeps document_id/source/page.

    Chunks are produced per page, so a chunk never spans two pages and its page
    number is always exact.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return [chunk for chunk in chunks if chunk.page_content.strip()]
