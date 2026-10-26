"""Text extraction for PDF (PyMuPDF) and TXT files, producing LangChain Documents."""

from __future__ import annotations

import re

import pymupdf
from langchain_core.documents import Document

from app.core.exceptions import ExtractionError
from app.utils.file_validation import FileType, decode_text

# Fewer extractable non-whitespace characters than this means a scanned/image-only PDF.
_MIN_CHARS_PER_PAGE = 20


class OCRRequiredError(ExtractionError):
    code = "ocr_required"
    default_message = (
        "No extractable text was found. This looks like a scanned/image-only PDF. OCR is "
        "not supported in this version: run the file through an OCR tool first, then "
        "upload the searchable PDF."
    )


class EncryptedPDFError(ExtractionError):
    code = "encrypted_pdf"
    default_message = "The PDF is password-protected. Remove the password and upload it again."


def normalize_pdf_text(text: str) -> str:
    """Re-join layout line breaks so paragraphs chunk (and overlap) properly.

    Hyphenated line breaks are merged, single newlines become spaces, and blank lines
    (paragraph breaks) are kept.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]*\n[ \t]*(?!\n)(?<!\n\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _base_metadata(document_id: str, filename: str) -> dict[str, str]:
    return {"document_id": document_id, "source": filename}


def load_pdf(content: bytes, document_id: str, filename: str) -> list[Document]:
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"'{filename}' is damaged or not a readable PDF.") from exc

    with pdf:
        if pdf.needs_pass and not pdf.authenticate(""):
            raise EncryptedPDFError()
        if pdf.page_count == 0:
            raise ExtractionError(f"'{filename}' contains no pages.")
        pages: list[Document] = []
        try:
            for page in pdf.pages():
                text = normalize_pdf_text(page.get_text("text"))
                if text:
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={
                                **_base_metadata(document_id, filename),
                                "page": page.number + 1,
                            },
                        )
                    )
        except RuntimeError as exc:
            raise ExtractionError(f"Text extraction failed for '{filename}'.") from exc
        total_chars = sum(len("".join(p.page_content.split())) for p in pages)
        if total_chars < _MIN_CHARS_PER_PAGE:
            raise OCRRequiredError()
        page_count = pdf.page_count
    for page_doc in pages:
        page_doc.metadata["page_count"] = page_count
    return pages


def load_text(content: bytes, document_id: str, filename: str) -> list[Document]:
    text = decode_text(content).replace("\r\n", "\n").strip()
    if not text:
        raise ExtractionError(f"'{filename}' contains no text.")
    # Plain text has no pages; page stays None so citations never invent one.
    return [Document(page_content=text, metadata={**_base_metadata(document_id, filename)})]


def load_document(
    content: bytes, file_type: FileType, document_id: str, filename: str
) -> list[Document]:
    if file_type == "pdf":
        return load_pdf(content, document_id, filename)
    return load_text(content, document_id, filename)
