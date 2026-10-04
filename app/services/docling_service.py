"""Thin wrapper around the Docling library (PRD Section 1, Section 7).

Confirmed against the installed docling==2.15.1 (see build notes): `DocumentConverter.convert(...,
raises_on_error=False)` already isolates per-page failures internally — a page whose backend fails
to load is recorded in `ConversionResult.pages` with an invalid backend and the overall status
degrades to `PARTIAL_SUCCESS`, rather than the whole conversion raising. That is exactly the "a
single corrupt page must not fail the whole job" requirement (Section 2 Stage 2, Section 8), so
this wrapper leans on Docling's own status/pages rather than re-implementing per-page isolation by
hand (e.g. re-converting page-by-page) — Docling already paid that cost internally.

`Page.page_no` (the per-page pipeline object) is 0-indexed; `ProvenanceItem.page_no` (attached to
each content item via `.prov`) is 1-indexed. Both are used below — get the offset wrong and every
`pages_failed` entry and every citation's page number is off by one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from app.config import settings


class DoclingConversionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@dataclass
class DoclingConvertResult:
    document: DoclingDocument
    status: DoclingConversionStatus
    page_count: int
    failed_page_numbers: list[int]  # 1-indexed, for direct use as `pages_failed`


class DoclingParseError(Exception):
    """Raised only for a whole-document (not single-page) failure."""


class DoclingService:
    def __init__(self) -> None:
        options = PdfPipelineOptions()
        options.do_ocr = settings.docling_ocr_enabled
        options.do_table_structure = True
        # Needed so PictureItem.get_image() below returns an actual crop instead of None — without
        # this, figure/chart captioning (Section 2 Stage 2) has nothing to send the BYOK vision key.
        options.generate_picture_images = True
        options.images_scale = settings.docling_picture_images_scale
        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def convert(self, file_path: str) -> DoclingConvertResult:
        # Never buffer the whole PDF into a Python `bytes` object ourselves (Section 8: "never
        # load a full PDF into memory") — hand Docling a path and let its own backend stream pages.
        result: ConversionResult = self._converter.convert(file_path, raises_on_error=False)

        from docling.datamodel.base_models import ConversionStatus

        if result.status == ConversionStatus.FAILURE:
            raise DoclingParseError(f"Docling failed to parse {file_path}: {[e.error_message for e in result.errors]}")

        status = (
            DoclingConversionStatus.PARTIAL_SUCCESS
            if result.status == ConversionStatus.PARTIAL_SUCCESS
            else DoclingConversionStatus.SUCCESS
        )
        failed_page_numbers = [
            page.page_no + 1
            for page in result.pages
            if page._backend is None or not page._backend.is_valid()
        ]
        return DoclingConvertResult(
            document=result.document,
            status=status,
            page_count=len(result.pages),
            failed_page_numbers=failed_page_numbers,
        )


docling_service = DoclingService()
