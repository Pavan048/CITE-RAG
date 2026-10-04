"""Smoke test against the real Docling library on a small synthetic PDF (text + table + figure)."""

from __future__ import annotations

from pathlib import Path

from docling_core.types.doc.document import PictureItem, TableItem, TextItem

from app.services.docling_service import DoclingConversionStatus, DoclingService

TEST_PDF = Path(__file__).resolve().parent.parent / "fixtures" / "test.pdf"


def test_docling_parses_text_table_and_figure() -> None:
    svc = DoclingService()
    result = svc.convert(str(TEST_PDF))

    assert result.status == DoclingConversionStatus.SUCCESS
    assert result.page_count == 2
    assert result.failed_page_numbers == []

    items = list(result.document.iterate_items())
    assert any(isinstance(item, TextItem) for item, _ in items)
    assert any(isinstance(item, TableItem) for item, _ in items)
    assert any(isinstance(item, PictureItem) for item, _ in items)

    table_item = next(item for item, _ in items if isinstance(item, TableItem))
    markdown = table_item.export_to_markdown(result.document)
    assert "Revenue" in markdown

    picture_item = next(item for item, _ in items if isinstance(item, PictureItem))
    image = picture_item.get_image(result.document)
    assert image is not None and image.size[0] > 0
