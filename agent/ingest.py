from __future__ import annotations

import argparse
from pathlib import Path

from agent.chunking import chunk_text
from agent.config import load_settings
from agent.nvidia_client import build_client, embed_texts
from agent.pdf_utils import extract_pdf_text
from agent.retriever import save_index


def ingest_pdf(pdf_path: str | Path) -> int:
    settings = load_settings()
    client = build_client(settings)

    path = Path(pdf_path)
    text = extract_pdf_text(path)
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError(
            "No text was extracted from the PDF. This may be a scanned document that needs OCR."
        )

    embeddings = embed_texts(client, settings.embed_model, chunks)
    save_index(embeddings, chunks, path.name)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a government document PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF to ingest")
    args = parser.parse_args()

    count = ingest_pdf(args.pdf_path)
    print(f"Ingested {count} chunks.")


if __name__ == "__main__":
    main()

