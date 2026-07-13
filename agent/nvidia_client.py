from openai import OpenAI

from agent.config import Settings


def build_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    input_type: str = "passage",
) -> list[list[float]]:
    response = client.embeddings.create(
        model=model,
        input=texts,
        extra_body={"input_type": input_type},
    )
    return [item.embedding for item in response.data]


def answer_with_context(
    client: OpenAI,
    model: str,
    question: str,
    contexts: list[str],
    source_language_hint: str = "Tamil or Hindi",
) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a government document assistant for Indian regional-language PDFs. "
                    "Answer only from the provided context. If the answer is not in the context, "
                    "say that the document does not provide enough information. Preserve names, "
                    "dates, amounts, and scheme terms exactly. If useful, answer in English and "
                    f"include key terms from the source language ({source_language_hint})."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion:\n{question}",
            },
        ],
    )
    return response.choices[0].message.content or ""
