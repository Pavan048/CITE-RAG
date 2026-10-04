"""PRD Section 2 Stage 1 (fast validation) and Stage 2 (Docling parse + BYOK figure captioning).

Stage 1's page-count check is deliberately a lightweight structural read (pypdf's `PdfReader` walks
the xref/page-tree without decoding page content streams) so it can run synchronously in the
upload request path, before anything is queued — the full Docling parse in Stage 2 is what runs
asynchronously in the worker.
"""

from __future__ import annotations

import io

from docling_core.types.doc.document import PictureItem, SectionHeaderItem, TableItem, TextItem
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.extensibility.prompt_provider import prompt_provider
from app.models.mongo_models import ContentType
from app.services.docling_service import docling_service
from app.services.llm_client import llm_client


class InvalidPdfError(ValueError):
    pass


class FileTooLargeError(ValueError):
    pass


class TooManyPagesError(ValueError):
    pass


def validate_upload(file_path: str, file_size_bytes: int) -> int:
    """Section 2 Stage 1: "reject before queuing". Returns the page count on success."""
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise FileTooLargeError(f"File is {file_size_bytes} bytes, exceeds the {max_bytes}-byte limit")

    with open(file_path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        raise InvalidPdfError("File does not start with a PDF header")

    try:
        page_count = len(PdfReader(file_path).pages)
    except PdfReadError as e:
        raise InvalidPdfError(f"Could not read PDF structure: {e}") from e

    if page_count > settings.max_page_count:
        raise TooManyPagesError(f"Document has {page_count} pages, exceeds the {settings.max_page_count}-page limit")

    return page_count


class ParsedElement(BaseModel):
    page_number: int
    section_title: str | None
    content_type: ContentType
    text: str


class ParsedDocument(BaseModel):
    elements: list[ParsedElement]
    page_count: int
    pages_failed: list[int]
    full_markdown: str  # source for the Stage 3 document summary


def _caption_figure(image, filename: str, page_number: int, llm_api_key: str, llm_model: str) -> str:
    prompt = prompt_provider.get("ingestion.figure_caption", filename=filename, page_number=page_number)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    # `llm_api_key` (the caller's X-LLM-Key) is used for exactly this one call and not retained
    # anywhere beyond this function's local scope (Section 8).
    return llm_client.caption_image(prompt=prompt, image_bytes=buf.getvalue(), api_key=llm_api_key, model=llm_model)


def parse_document(file_path: str, filename: str, llm_api_key: str, llm_model: str) -> ParsedDocument:
    result = docling_service.convert(file_path)
    doc = result.document

    elements: list[ParsedElement] = []
    current_section: str | None = None
    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else 1

        if isinstance(item, SectionHeaderItem):
            # Not emitted as its own retrievable chunk (a bare heading isn't useful on its own) —
            # it only updates the section context attached to what follows, and marks a new
            # parent-block boundary (Section 2 Stage 3: "store parent block (full section)").
            current_section = item.text
            continue

        if isinstance(item, TableItem):
            elements.append(
                ParsedElement(
                    page_number=page_no, section_title=current_section,
                    content_type=ContentType.TABLE, text=item.export_to_markdown(doc),
                )
            )
        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None:
                continue
            caption = _caption_figure(image, filename, page_no, llm_api_key, llm_model)
            elements.append(
                ParsedElement(
                    page_number=page_no, section_title=current_section,
                    content_type=ContentType.IMAGE_CAPTION, text=caption,
                )
            )
        elif isinstance(item, TextItem):
            elements.append(
                ParsedElement(
                    page_number=page_no, section_title=current_section,
                    content_type=ContentType.TEXT, text=item.text,
                )
            )

    return ParsedDocument(
        elements=elements,
        page_count=result.page_count,
        pages_failed=result.failed_page_numbers,
        full_markdown=doc.export_to_markdown(),
    )
