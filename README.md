# Simple Vietnamese RAG with Qdrant

Pipeline này index cột `context` từ dataset `sailor2/Vietnamese_RAG` config `BKAI_RAG` vào Qdrant. Khi có câu hỏi mới, hệ thống mới embed query, search Qdrant, rồi trả về top-k context liên quan.

Dataset có schema:

- `question`: query dùng để test
- `answer`: đáp án chuẩn
- `context`: list các context chunk đã được làm sạch/chunk sẵn

Code sẽ lưu mỗi item trong `context` thành một point riêng trong Qdrant.

## Setup

```bash
make setup
```

Tạo file `.env` ở project root:

```bash
QDRANT_URL=http://217.216.35.94:6333
QDRANT_API_KEY=your-api-key
RAG_COLLECTION=vietnamese_rag_bkai
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_BATCH_SIZE=4
RAG_MAX_SEQ_LENGTH=512
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANK_ENABLED=true
RAG_GENERATOR_PROVIDER=extractive
```

`.env` đã được ignore để không commit API key.

## Make Commands

```bash
make help
make check
```

## Build Qdrant Index

```bash
make ingest
```

`BAAI/bge-m3` là model khá nặng. Nếu gặp lỗi memory kiểu `Invalid buffer size`, giảm batch size:

```bash
make ingest RAG_BATCH_SIZE=1 RAG_MAX_SEQ_LENGTH=512
```

Nếu muốn model nhẹ hơn để chạy nhanh trên CPU:

```bash
make ingest RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Để test nhanh với ít dòng:

```bash
make ingest-test LIMIT=100
```

`make ingest-test` dùng collection `${RAG_COLLECTION}_test`.

## Ask

Retrieve context từ Qdrant:

```bash
make ask QUESTION="Câu hỏi của bạn ở đây" TOP_K=5
```

## Answer Full Flow

Chạy đủ flow:

```text
question -> embed query -> Qdrant retrieve -> rerank -> generate
```

```bash
make answer QUESTION="Lý Hải có bao giờ hết thời trong sự nghiệp ca hát không?"
```

Mặc định `RAG_GENERATOR_PROVIDER=extractive`, nên command sẽ chạy đủ retrieve/rerank nhưng chưa gọi LLM thật. Nó trả về context liên quan nhất để bạn kiểm tra pipeline.

Nếu dùng Ollama local:

```bash
RAG_GENERATOR_PROVIDER=ollama
RAG_LLM_BASE_URL=http://localhost:11434
RAG_LLM_MODEL=llama3.1
```

Sau đó chạy:

```bash
make answer QUESTION="Lý Hải có bao giờ hết thời trong sự nghiệp ca hát không?"
```

Nếu dùng một API dạng OpenAI-compatible:

```bash
RAG_GENERATOR_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=https://your-provider.example/v1
RAG_LLM_API_KEY=your-api-key
RAG_LLM_MODEL=your-model
```

Nếu dùng OpenAI API trực tiếp, bạn có thể dùng `OPENAI_API_KEY`; code sẽ tự đọc biến này nếu `RAG_LLM_API_KEY` không được set:

```bash
OPENAI_API_KEY=sk-...
RAG_GENERATOR_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=gpt-4o-mini
```

Hoặc chạy nhanh:

```bash
make answer-openai QUESTION="Lý Hải có bao giờ hết thời trong sự nghiệp ca hát không?"
```

Đổi model OpenAI bằng:

```bash
make answer-openai QUESTION="..." OPENAI_MODEL=gpt-4o-mini
```

## API Server

CLI sẽ load model lại mỗi lần chạy, nên phù hợp để test. Frontend nên gọi API server để model được load một lần lúc startup.

Chạy API với OpenAI generator:

```bash
make api-openai
```

Mặc định server chạy ở:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Chỉ gọi API sau khi log server có dòng:

```text
Application startup complete.
```

Reranker được lazy-load: server startup không bị chặn bởi reranker, nhưng request đầu tiên có `RAG_RERANK_ENABLED=true` vẫn có thể chậm vì phải tải model rerank. Để test nhanh, tắt rerank:

```bash
make api-openai RAG_RERANK_ENABLED=false
```

Gọi answer:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Làng Mẹo ở đâu và được mệnh danh là gì?",
    "retrieve_k": 20,
    "rerank_k": 5,
    "include_contexts": false
  }'
```

Chỉ retrieve context:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "Làng Mẹo ở đâu?", "top_k": 5}'
```

Nếu muốn server nhanh hơn và chấp nhận bỏ rerank:

```bash
make api-openai RAG_RERANK_ENABLED=false
```

Đổi port:

```bash
make api-openai API_PORT=8080
```

## Docker

Build image:

```bash
make docker-build
```

Docker build lần đầu có thể lâu vì project dùng `sentence-transformers`, kéo theo `torch`. Dockerfile đã cài `torch` CPU-only trước để tránh tải các package CUDA/NVIDIA rất lớn.

Nếu build trên Apple Silicon nhưng target deploy là Linux x86_64:

```bash
make docker-build DOCKER_PLATFORM=linux/amd64
```

Run API với OpenAI generator:

```bash
make docker-run-openai
```

Script này mặc định `RAG_RERANK_ENABLED=false` để API production tránh tải reranker lớn trong request đầu tiên. Bật lại rerank khi máy đủ RAM:


Container đọc config từ `.env`, expose API ở:

```text
http://localhost:8000
```

Test:

```bash
curl http://localhost:8000/health
```

Đổi port host:

```bash
make docker-run-openai DOCKER_PORT=8080
```

Docker run thủ công tương đương:

```bash
docker build -t bkai-rag:latest .
docker run --rm --env-file .env \
  -e RAG_GENERATOR_PROVIDER=openai-compatible \
  -e RAG_LLM_BASE_URL=https://api.openai.com/v1 \
  -e RAG_LLM_MODEL=gpt-4o-mini \
  -p 8000:8000 \
  -v "$HOME/.cache/bkai-rag-hf:/cache/huggingface" \
  bkai-rag:latest
```

Image dùng `python:3.12-slim`, chỉ copy `requirements.txt` và `src/`, không copy `.venv`, `.env`, git metadata hay cache local. Volume cache giúp model Hugging Face không phải tải lại sau mỗi lần chạy container.

Nếu reranker quá nặng hoặc muốn debug nhanh:

```bash
make answer QUESTION="..." RAG_RERANK_ENABLED=false
```

Có thể chỉnh số lượng retrieve/rerank:

```bash
make answer QUESTION="..." RETRIEVE_K=20 RERANK_K=5
```

## Evaluate Retrieval

```bash
make evaluate LIMIT=200 TOP_K=5
```

Evaluation dùng `question` làm query. Một sample được tính là hit nếu kết quả retrieve chứa bất kỳ context chunk nào thuộc row gốc.

## Manual Commands

Nếu không dùng `make`:

```bash
source .venv/bin/activate
export QDRANT_URL="http://your-qdrant-url:6333"
export QDRANT_API_KEY="your-api-key"

python -m src.ingest --recreate
python -m src.ingest --recreate --batch-size 1 --max-seq-length 512
python -m src.ask "Câu hỏi của bạn ở đây" --top-k 5
python -m src.answer "Câu hỏi của bạn ở đây" --retrieve-k 20 --rerank-k 5
python -m src.evaluate --limit 200 --top-k 5
```
