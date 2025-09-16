from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import smolvlm_picture_description

# !/usr/bin/env python3

from pathlib import Path


def pdf_to_md(pdf_path: str, output_path: str, use_document_pipeline=False):
    """Convert PDF to Markdown file."""

    # This is code is not needed, but it could improve the data if we also use an LLM to help the conversion
    if use_document_pipeline:
        print("Using document pipeline with SmolVLM model")

        # Source: https://docling-project.github.io/docling/usage/enrichments/#smolvlm-model
        pipeline_options = PdfPipelineOptions()
        pipeline_options.picture_description_options = smolvlm_picture_description
        pipeline_options.do_picture_description = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        converter = DocumentConverter()

    doc = converter.convert(pdf_path).document

    text = doc.export_to_markdown()

    # Write to markdown file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {Path(pdf_path).stem}\n\n")
        f.write(text)

    print(f"Converted {pdf_path} to {output_path}")


if __name__ == "__main__":
    pdf_to_md("../data/raw/AIACT.pdf", "data/processed/AIACT.md")
