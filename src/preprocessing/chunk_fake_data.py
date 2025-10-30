import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_fake_data():
    """
    Chunk the fake data from data/raw/fakedata.md using RecursiveCharacterTextSplitter
    and save it to data/processed/fake-chunks.json in the specified format.
    """
    # Read the fake data
    input_path = Path(__file__).parent.parent.parent / "data" / "raw" / "fakedata.md"
    with open(input_path, 'r', encoding='utf-8') as file:
        document = file.read()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_text(document)

    # Create chunks in the specified format
    chunks = []
    for i, text in enumerate(texts):
        chunk = {
            "id": f"fake-chunk-{i+1}",
            "type": None,
            "paragraph_number": None,
            "page_range": None,
            "chapter_number": None,
            "chapter_name": None,
            "section_number": None,
            "section_name": None,
            "article_number": None,
            "article_name": None,
            "annex_number": None,
            "annex_name": None,
            "text": text.strip()
        }
        chunks.append(chunk)

    # Ensure output directory exists
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "fake-chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save chunks to JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Successfully created {len(chunks)} chunks and saved to {output_path}")


if __name__ == "__main__":
    chunk_fake_data()