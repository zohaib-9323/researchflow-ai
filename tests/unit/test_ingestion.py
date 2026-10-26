from __future__ import annotations

import pytest

from app.core.exceptions import (
    ExtractionError,
    FileTooLargeError,
    UnsupportedFileError,
    ValidationFailedError,
)
from app.ingestion.chunking import split_documents
from app.ingestion.loaders import EncryptedPDFError, OCRRequiredError, load_document
from app.utils.file_validation import safe_filename, validate_upload
from tests.conftest import make_pdf, sample_bytes

MB = 1024 * 1024


class TestFileValidation:
    def test_accepts_pdf_and_txt_by_content(self) -> None:
        pdf = validate_upload("paper.PDF", make_pdf(["Hello world"]), MB)
        txt = validate_upload("notes.txt", b"plain text", MB)
        assert (pdf.file_type, txt.file_type) == ("pdf", "txt")
        assert len(pdf.content_hash) == 64

    def test_rejects_unknown_extension(self) -> None:
        with pytest.raises(UnsupportedFileError):
            validate_upload("malware.exe", b"MZ...", MB)

    def test_rejects_pdf_extension_with_non_pdf_content(self) -> None:
        with pytest.raises(UnsupportedFileError, match="not a PDF"):
            validate_upload("fake.pdf", b"just some text", MB)

    def test_rejects_binary_disguised_as_txt(self) -> None:
        with pytest.raises(UnsupportedFileError):
            validate_upload("image.txt", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 20, MB)

    def test_rejects_empty_and_oversized(self) -> None:
        with pytest.raises(ValidationFailedError, match="empty"):
            validate_upload("empty.txt", b"", MB)
        with pytest.raises(FileTooLargeError):
            validate_upload("big.txt", b"a" * (MB + 1), MB)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd.txt", "passwd.txt"),
            ("..\\..\\windows\\evil.pdf", "evil.pdf"),
            ("/abs/path/report.pdf", "report.pdf"),
            ("we!rd<name>.txt", "we_rd_name_.txt"),
            ("", "document"),
            ("...", "document"),
        ],
    )
    def test_safe_filename_strips_paths(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected


class TestLoaders:
    def test_pdf_extraction_preserves_page_numbers(self) -> None:
        content = make_pdf(["First page about solar panels.", "Second page about wells."])
        pages = load_document(content, "pdf", "doc-1", "report.pdf")
        assert [p.metadata["page"] for p in pages] == [1, 2]
        assert "solar" in pages[0].page_content
        assert pages[1].metadata["document_id"] == "doc-1"
        assert pages[1].metadata["source"] == "report.pdf"
        assert pages[0].metadata["page_count"] == 2

    def test_pdf_skips_blank_pages_but_keeps_numbering(self) -> None:
        pages = load_document(
            make_pdf(["Intro text here.", "", "Third page text."]), "pdf", "d", "x.pdf"
        )
        assert [p.metadata["page"] for p in pages] == [1, 3]

    def test_scanned_pdf_reports_ocr_required(self) -> None:
        with pytest.raises(OCRRequiredError, match="OCR is not supported"):
            load_document(make_pdf(["", ""]), "pdf", "d", "scan.pdf")

    def test_encrypted_pdf(self) -> None:
        with pytest.raises(EncryptedPDFError):
            load_document(make_pdf(["secret"], password="pw"), "pdf", "d", "locked.pdf")

    def test_malformed_pdf(self) -> None:
        with pytest.raises(ExtractionError):
            load_document(b"%PDF-1.7\n garbage that is not a pdf", "pdf", "d", "bad.pdf")

    def test_txt_has_no_page_number(self) -> None:
        pages = load_document(sample_bytes("larkspur_water_quality.txt"), "txt", "d", "w.txt")
        assert len(pages) == 1
        assert "page" not in pages[0].metadata
        assert "64 private wells" in pages[0].page_content

    def test_whitespace_only_txt_fails(self) -> None:
        with pytest.raises(ExtractionError):
            load_document(b"   \n\n  ", "txt", "d", "blank.txt")


class TestChunking:
    def test_chunks_overlap_and_keep_metadata(self) -> None:
        text = " ".join(f"Sentence number {i} discusses topic {i % 7}." for i in range(200))
        pages = load_document(make_pdf([text[:2500], text[2500:5000]]), "pdf", "doc", "a.pdf")
        chunks = split_documents(pages, chunk_size=300, chunk_overlap=60)
        assert len(chunks) > 4
        assert all(len(c.page_content) <= 300 for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert {c.metadata["page"] for c in chunks} == {1, 2}
        assert all(c.metadata["document_id"] == "doc" for c in chunks)
        # consecutive chunks on the same page share overlapping text
        first, second = chunks[0], chunks[1]
        assert first.metadata["page"] == second.metadata["page"]
        first_end = first.metadata["start_index"] + len(first.page_content)
        assert second.metadata["start_index"] < first_end
        overlap = first.page_content[
            second.metadata["start_index"] - first.metadata["start_index"] :
        ]
        assert overlap.strip()
        assert second.page_content.startswith(overlap.strip())


def test_pdf_layout_line_breaks_are_joined() -> None:
    from app.ingestion.loaders import normalize_pdf_text

    raw = "The pilot re-\nduced use by\n18 percent.\n\nSecond paragraph\nhere.\n"
    assert (
        normalize_pdf_text(raw) == "The pilot reduced use by 18 percent.\n\nSecond paragraph here."
    )
