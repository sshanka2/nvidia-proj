from __future__ import annotations

import argparse

from agent.config import load_settings
from agent.nvidia_client import answer_with_context, build_client, embed_texts
from agent.retriever import SearchResult, search


def ask(question: str, top_k: int = 4) -> tuple[str, list[SearchResult]]:
    settings = load_settings()
    client = build_client(settings)

    query_embedding = embed_texts(
        client,
        settings.embed_model,
        [question],
        input_type="query",
    )[0]
    results = search(query_embedding, top_k=top_k)
    answer = answer_with_context(
        client=client,
        model=settings.llm_model,
        question=question,
        contexts=[result.text for result in results],
    )
    return answer, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the ingested PDF.")
    parser.add_argument("question", help="Question to answer from the document")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    answer, results = ask(args.question, top_k=args.top_k)
    print("\nAnswer:\n")
    print(answer)
    print("\nSources:\n")
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result.source} | score={result.score:.3f}")


if __name__ == "__main__":
    main()
