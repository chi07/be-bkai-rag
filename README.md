# Simple Vietnamese RAG with Qdrant

Pipeline này index cột `context` từ dataset `sailor2/Vietnamese_RAG` config `BKAI_RAG` vào Qdrant. Khi có câu hỏi mới, hệ thống mới embed query, search Qdrant, rồi trả về top-k context liên quan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export Qdrant config:

```bash
export QDRANT_URL="http://217.216.35.94:6333"
export QDRANT_API_KEY="your-api-key"
```

Optional config:

```bash
export RAG_COLLECTION="vietnamese_rag_bkai"
export RAG_EMBEDDING_MODEL="BAAI/bge-m3"
```

## Build Qdrant Index

```bash
python -m simple_rag.ingest --recreate
```

Để test nhanh với ít dòng:

```bash
python -m simple_rag.ingest --recreate --limit 1000
```

## Ask

```bash
python -m simple_rag.ask "Câu hỏi của bạn ở đây" --top-k 5
```

## Evaluate Retrieval

```bash
python -m simple_rag.evaluate --limit 200 --top-k 5
```

Evaluation dùng `question` làm query và xem `context` gốc của cùng dòng có nằm trong top-k hay không.
