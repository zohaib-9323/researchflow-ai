"""Upload validation: content sniffing, size limits and safe filenames."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from app.core.exceptions import FileTooLargeError, UnsupportedFileError, ValidationFailedError

FileType = Literal["pdf", "txt"]

_PDF_MAGIC = b"%PDF-"
_ALLOWED_EXTENSIONS: dict[str, FileType] = {".pdf": "pdf", ".txt": "txt"}
_TEXT_ENCODINGS = ("utf-8-sig", "utf-16")


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    file_type: FileType
    content: bytes
    size_bytes: int
    content_hash: str


def safe_filename(raw_name: str | None, max_length: int = 150) -> str:
    """Return a display-safe base name with no directory components or control chars."""
    name = PurePath((raw_name or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" ._")
    if not name:
        name = "document"
    if len(name) > max_length:
        stem, dot, suffix = name.rpartition(".")
        name = (stem[: max_length - len(suffix) - 1] + dot + suffix) if dot else name[:max_length]
    return name


def _looks_like_text(content: bytes) -> bool:
    if b"\x00" in content[:8192] and not content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False
    for encoding in _TEXT_ENCODINGS:
        try:
            decoded = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        sample = decoded[:8192]
        if not sample:
            return True
        printable = sum(ch.isprintable() or ch in "\r\n\t\f" for ch in sample)
        return printable / len(sample) > 0.95
    return False


def decode_text(content: bytes) -> str:
    for encoding in _TEXT_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedFileError("The text file is not valid UTF-8 or UTF-16.")


def validate_upload(raw_name: str | None, content: bytes, max_bytes: int) -> ValidatedUpload:
    """Validate extension *and* content; never trust the extension alone."""
    filename = safe_filename(raw_name)
    extension = PurePath(filename).suffix.lower()
    file_type = _ALLOWED_EXTENSIONS.get(extension)
    if file_type is None:
        raise UnsupportedFileError(
            f"'{filename}' is not supported. Upload PDF (.pdf) or plain-text (.txt) files."
        )
    size = len(content)
    if size == 0:
        raise ValidationFailedError(f"'{filename}' is empty.")
    if size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise FileTooLargeError(f"'{filename}' exceeds the {limit_mb:g} MB upload limit.")
    if file_type == "pdf" and not content.lstrip()[:1024].startswith(_PDF_MAGIC):
        raise UnsupportedFileError(f"'{filename}' has a .pdf extension but is not a PDF file.")
    if file_type == "txt" and not _looks_like_text(content):
        raise UnsupportedFileError(f"'{filename}' does not appear to be a plain-text file.")
    return ValidatedUpload(
        filename=filename,
        file_type=file_type,
        content=content,
        size_bytes=size,
        content_hash=hashlib.sha256(content).hexdigest(),
    )
