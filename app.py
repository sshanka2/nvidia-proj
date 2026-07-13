from pathlib import Path

import gradio as gr

from agent.agent import ask
from agent.ingest import ingest_pdf


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def ingest_uploaded_file(file_obj) -> str:
    if file_obj is None:
        return "Upload a PDF first."

    source = Path(file_obj.name)
    destination = UPLOAD_DIR / source.name
    destination.write_bytes(source.read_bytes())

    count = ingest_pdf(destination)
    return f"Ready. Ingested {count} chunks from {source.name}."


def answer_question(question: str) -> str:
    if not question.strip():
        return "Ask a question about the uploaded document."

    answer, sources = ask(question)
    source_lines = [
        f"{idx}. {source.source} | score={source.score:.3f}"
        for idx, source in enumerate(sources, start=1)
    ]
    return answer + "\n\nSources:\n" + "\n".join(source_lines)


with gr.Blocks(title="Government Document Agent") as demo:
    gr.Markdown("# Government Document Agent")
    gr.Markdown("Upload a Tamil or Hindi government PDF, then ask questions about it.")

    with gr.Row():
        pdf_file = gr.File(label="Government PDF", file_types=[".pdf"])
        ingest_button = gr.Button("Ingest document", variant="primary")

    ingest_status = gr.Textbox(label="Status", interactive=False)
    question = gr.Textbox(label="Question", placeholder="What is this document about?")
    answer = gr.Textbox(label="Answer", lines=12)

    ingest_button.click(ingest_uploaded_file, inputs=pdf_file, outputs=ingest_status)
    question.submit(answer_question, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()

