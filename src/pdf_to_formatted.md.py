# pdf_to_formatted.md.py

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat


def pdf_to_md(pdf_path: str, output_path: str, use_document_pipeline=False):
    """Convert PDF to Markdown file with Granite-Docling (MLX) on MPS."""

    if use_document_pipeline:
        print("Using VLM pipeline (Granite-Docling MLX) on MPS")
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel import vlm_model_specs

        pipeline_options = VlmPipelineOptions(
            # Krever docling-ibm-models
            vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.MPS,  # tving MPS
                num_threads=8  # juster ved behov
            ),
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
    else:
        converter = DocumentConverter()

    doc = converter.convert(pdf_path).document
    text = doc.export_to_markdown()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(f"# {Path(pdf_path).stem}\n\n{text}", encoding="utf-8")
    print(f"Converted {pdf_path} to {output_path}")


if __name__ == "__main__":
    pdf_to_md("../data/raw/AIACT.pdf", "../data/processed/AIACT.md")
