#!/usr/bin/env python3

from pypdf import PdfReader
from pathlib import Path


def pdf_to_md(pdf_path: str, output_path: str):
    """Convert PDF to Markdown file."""
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    # Write to markdown file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {Path(pdf_path).stem}\n\n")
        f.write(text)

    print(f"Converted {pdf_path} to {output_path}")


if __name__ == "__main__":
    pdf_to_md("../data/raw/AIACT.pdf", "data/processed/AIACT.md")
