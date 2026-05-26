from __future__ import annotations

import requests


SYSTEM_PROMPT = (
    "Bạn là một trợ lý trò chuyện tiếng Việt thân thiện, tự nhiên và gần gũi. "
    "Hãy trả lời như đang nói chuyện với một người quen: mềm mại, dễ hiểu, có nhịp điệu hội thoại, "
    "xưng hô 'mình' và 'bạn' khi phù hợp. "
    "Dù giọng văn thân mật, bạn vẫn phải bám sát ngữ cảnh được cung cấp và không bịa thêm thông tin. "
    "Nếu câu hỏi hiện tại dùng đại từ hoặc cách nói tiếp nối như 'anh ta', 'người đó', "
    "'vậy còn', hãy dùng lịch sử hội thoại để hiểu người dùng đang nhắc tới ai. "
    "Nếu ngữ cảnh không đủ thông tin, hãy nói nhẹ nhàng rằng mình chưa thấy đủ dữ kiện để khẳng định."
)


def build_history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def build_user_prompt(
    question: str,
    contexts: list[str],
    history: list[dict[str, str]] | None = None,
) -> str:
    context_block = "\n\n".join(
        f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
    )
    history_block = build_history_block(history)
    history_section = f"Lịch sử hội thoại:\n{history_block}\n\n" if history_block else ""
    return (
        f"{history_section}"
        f"Ngữ cảnh:\n{context_block}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Hãy trả lời bằng tiếng Việt theo phong cách thân mật, tự nhiên. "
        "Không cần mở đầu máy móc như 'Dựa trên ngữ cảnh'. "
        "Trả lời vừa đủ, đúng trọng tâm; nếu có thể, nối câu cho mượt như đang trò chuyện."
    )


class Generator:
    def __init__(
        self,
        provider: str = "extractive",
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        model: str = "llama3.1",
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        question: str,
        contexts: list[str],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.provider == "extractive":
            return self._extractive_answer(contexts)
        if self.provider == "ollama":
            return self._ollama_answer(question, contexts, history=history)
        if self.provider == "openai-compatible":
            return self._openai_compatible_answer(question, contexts, history=history)
        raise ValueError(
            "Unsupported RAG_GENERATOR_PROVIDER. Use: extractive, ollama, openai-compatible."
        )

    def _extractive_answer(self, contexts: list[str]) -> str:
        if not contexts:
            return "Không đủ thông tin trong ngữ cảnh đã retrieve."
        return (
            "Generator provider đang là 'extractive', nên chưa gọi LLM. "
            "Context liên quan nhất:\n"
            f"{contexts[0]}"
        )

    def _ollama_answer(
        self,
        question: str,
        contexts: list[str],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts, history)},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def _openai_compatible_answer(
        self,
        question: str,
        contexts: list[str],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts, history)},
                ],
                "temperature": 0.2,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
