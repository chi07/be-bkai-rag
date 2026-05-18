- Cleaned dataset: https://huggingface.co/datasets/sailor2/Vietnamese_RAG/viewer/BKAI_RAG
- Mỗi dòng có 3 trường là question-answer và context, thì cột context ta sẽ retrieve lại và khi test sẽ dùng dòng question-answer để test.
- data này đã khá sạch và được chunk sẵn rồi chỉ cần preprocessing xong là retrieve được ạ, sau đó xây module embedding, rerank và generate


Vì data đã sạch và chunk sẵn -> Chọn model embedding tiếng Việt (ví dụ: bkai-foundation-models/vietnamese-bi-encoder) -> embed toàn bộ context -> lưu vector index (QDRANT) -> xong khi người dùng đặt câu hỏi thì mới bắt đầu embed query -> Rerank -> Generate