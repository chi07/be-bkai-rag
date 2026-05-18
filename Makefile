PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
.SILENT:

QDRANT_URL ?= http://217.216.35.94:6333
RAG_COLLECTION ?= vietnamese_rag_bkai
RAG_EMBEDDING_MODEL ?= BAAI/bge-m3
RAG_BATCH_SIZE ?= 4
RAG_MAX_SEQ_LENGTH ?= 512
RAG_RERANKER_MODEL ?= BAAI/bge-reranker-v2-m3
RAG_RERANK_ENABLED ?= true
RAG_RERANK_MAX_LENGTH ?= 512
RAG_GENERATOR_PROVIDER ?= extractive
RAG_LLM_BASE_URL ?= http://localhost:11434
RAG_LLM_MODEL ?= llama3.1
OPENAI_MODEL ?= gpt-4o-mini
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
DOCKER_IMAGE ?= bkai-rag
DOCKER_TAG ?= latest
DOCKER_PLATFORM ?=
DOCKER_PORT ?= 8000
DOCKER_CACHE_DIR ?= $(HOME)/.cache/bkai-rag-hf
TOP_K ?= 5
RETRIEVE_K ?= 20
RERANK_K ?= 5
LIMIT ?=
QUESTION ?=

-include .env

.PHONY: help setup install check ingest ingest-test ask answer answer-openai api api-openai docker-build docker-run docker-run-openai evaluate clean

help:
	@echo "Simple RAG commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup                         Create .venv and install dependencies"
	@echo "  make install                       Install dependencies into existing .venv"
	@echo "  make check                         Compile Python files"
	@echo ""
	@echo "Qdrant index:"
	@echo "  make ingest                        Build full Qdrant index"
	@echo "  make ingest-test LIMIT=100         Recreate/index a small sample"
	@echo "  make ingest RAG_BATCH_SIZE=1       Lower memory usage for large models"
	@echo ""
	@echo "Runtime:"
	@echo "  make ask QUESTION=\"...\"            Embed query and retrieve contexts"
	@echo "  make answer QUESTION=\"...\"         Retrieve, rerank, generate answer"
	@echo "  make answer-openai QUESTION=\"...\"  Generate using OPENAI_API_KEY"
	@echo "  make api                            Start FastAPI server"
	@echo "  make api-openai                     Start API with OpenAI generator"
	@echo "  make evaluate LIMIT=200 TOP_K=5    Evaluate retrieval"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build                  Build Docker image"
	@echo "  make docker-run-openai             Run API container with OpenAI"
	@echo ""
	@echo "Config can be set in .env or inline, e.g.:"
	@echo "  QDRANT_API_KEY=... make ingest"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

check:
	$(PY) -m py_compile src/*.py

ingest:
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_URL="$(QDRANT_URL)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_BATCH_SIZE="$(RAG_BATCH_SIZE)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	$(PY) -m src.ingest --recreate --batch-size $(RAG_BATCH_SIZE) --max-seq-length $(RAG_MAX_SEQ_LENGTH) $(if $(LIMIT),--limit $(LIMIT),)

ingest-test:
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_URL="$(QDRANT_URL)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)_test" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_BATCH_SIZE="$(RAG_BATCH_SIZE)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	$(PY) -m src.ingest --recreate --batch-size $(RAG_BATCH_SIZE) --max-seq-length $(RAG_MAX_SEQ_LENGTH) --limit $(if $(LIMIT),$(LIMIT),100)

ask:
	@test -n "$(QUESTION)" || (echo 'Missing QUESTION. Usage: make ask QUESTION="..."'; exit 1)
	QDRANT_URL="$(QDRANT_URL)" \
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	$(PY) -m src.ask "$(QUESTION)" --top-k $(TOP_K)

answer:
	@test -n "$(QUESTION)" || (echo 'Missing QUESTION. Usage: make answer QUESTION="..."'; exit 1)
	QDRANT_URL="$(QDRANT_URL)" \
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	RAG_RERANKER_MODEL="$(RAG_RERANKER_MODEL)" \
	RAG_RERANK_ENABLED="$(RAG_RERANK_ENABLED)" \
	RAG_RERANK_MAX_LENGTH="$(RAG_RERANK_MAX_LENGTH)" \
	RAG_GENERATOR_PROVIDER="$(RAG_GENERATOR_PROVIDER)" \
	RAG_LLM_BASE_URL="$(RAG_LLM_BASE_URL)" \
	RAG_LLM_API_KEY="$(RAG_LLM_API_KEY)" \
	OPENAI_API_KEY="$(OPENAI_API_KEY)" \
	RAG_LLM_MODEL="$(RAG_LLM_MODEL)" \
	$(PY) -m src.answer "$(QUESTION)" --retrieve-k $(RETRIEVE_K) --rerank-k $(RERANK_K)

answer-openai:
	$(MAKE) answer \
		QUESTION="$(QUESTION)" \
		RAG_GENERATOR_PROVIDER=openai-compatible \
		RAG_LLM_BASE_URL=https://api.openai.com/v1 \
		RAG_LLM_MODEL="$(OPENAI_MODEL)"

api:
	QDRANT_URL="$(QDRANT_URL)" \
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	RAG_RERANKER_MODEL="$(RAG_RERANKER_MODEL)" \
	RAG_RERANK_ENABLED="$(RAG_RERANK_ENABLED)" \
	RAG_RERANK_MAX_LENGTH="$(RAG_RERANK_MAX_LENGTH)" \
	RAG_GENERATOR_PROVIDER="$(RAG_GENERATOR_PROVIDER)" \
	RAG_LLM_BASE_URL="$(RAG_LLM_BASE_URL)" \
	RAG_LLM_API_KEY="$(RAG_LLM_API_KEY)" \
	OPENAI_API_KEY="$(OPENAI_API_KEY)" \
	RAG_LLM_MODEL="$(RAG_LLM_MODEL)" \
	$(PY) -m uvicorn src.api:app --host $(API_HOST) --port $(API_PORT)

api-openai:
	$(MAKE) api \
		RAG_GENERATOR_PROVIDER=openai-compatible \
		RAG_LLM_BASE_URL=https://api.openai.com/v1 \
		RAG_LLM_MODEL="$(OPENAI_MODEL)"

docker-build:
	docker build $(if $(DOCKER_PLATFORM),--platform $(DOCKER_PLATFORM),) -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	mkdir -p "$(DOCKER_CACHE_DIR)"
	docker run --rm \
		--env-file .env \
		-e RAG_GENERATOR_PROVIDER="$(RAG_GENERATOR_PROVIDER)" \
		-e RAG_LLM_BASE_URL="$(RAG_LLM_BASE_URL)" \
		-e RAG_LLM_MODEL="$(RAG_LLM_MODEL)" \
		-p $(DOCKER_PORT):8000 \
		-v "$(DOCKER_CACHE_DIR):/cache/huggingface" \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-openai:
	$(MAKE) docker-run \
		RAG_GENERATOR_PROVIDER=openai-compatible \
		RAG_LLM_BASE_URL=https://api.openai.com/v1 \
		RAG_LLM_MODEL="$(OPENAI_MODEL)"

evaluate:
	QDRANT_URL="$(QDRANT_URL)" \
	HF_TOKEN="$(HF_TOKEN)" \
	QDRANT_API_KEY="$(QDRANT_API_KEY)" \
	RAG_COLLECTION="$(RAG_COLLECTION)" \
	RAG_EMBEDDING_MODEL="$(RAG_EMBEDDING_MODEL)" \
	RAG_MAX_SEQ_LENGTH="$(RAG_MAX_SEQ_LENGTH)" \
	$(PY) -m src.evaluate --limit $(if $(LIMIT),$(LIMIT),200) --top-k $(TOP_K)

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
