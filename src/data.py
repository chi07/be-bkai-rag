from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


@dataclass(frozen=True)
class RagRecord:
    doc_id: int
    context_hash: str
    context: str
    question: str
    answer: str
    source_row_id: int
    source_context_index: int


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalize_context_chunks(value: object) -> list[str]:
    if isinstance(value, list):
        return [text for text in (normalize_text(item) for item in value) if text]
    text = normalize_text(value)
    return [text] if text else []


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bkai_dataset(dataset_name: str, dataset_config: str, split: str | None = None) -> Dataset:
    loaded = load_dataset(dataset_name, dataset_config)

    if split:
        return loaded[split]

    if isinstance(loaded, Dataset):
        return loaded

    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Unexpected dataset type: {type(loaded)!r}")

    if "train" in loaded:
        return loaded["train"]

    return concatenate_datasets([loaded[name] for name in loaded.keys()])


def iter_clean_records(dataset: Dataset, limit: int | None = None) -> Iterable[RagRecord]:
    seen_contexts: dict[str, int] = {}
    emitted = 0

    for row_id, row in enumerate(dataset):
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        contexts = normalize_context_chunks(row.get("context"))

        if not question or not answer or not contexts:
            continue

        for context_index, context in enumerate(contexts):
            context_hash = stable_hash(context)
            if context_hash in seen_contexts:
                continue

            doc_id = len(seen_contexts)
            seen_contexts[context_hash] = doc_id

            yield RagRecord(
                doc_id=doc_id,
                context_hash=context_hash,
                context=context,
                question=question,
                answer=answer,
                source_row_id=row_id,
                source_context_index=context_index,
            )

            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_eval_rows(dataset: Dataset, limit: int | None = None) -> Iterable[dict[str, str | int | list[str]]]:
    emitted = 0
    for row_id, row in enumerate(dataset):
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        contexts = normalize_context_chunks(row.get("context"))

        if not question or not answer or not contexts:
            continue

        yield {
            "row_id": row_id,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "context_hashes": [stable_hash(context) for context in contexts],
        }

        emitted += 1
        if limit is not None and emitted >= limit:
            return
