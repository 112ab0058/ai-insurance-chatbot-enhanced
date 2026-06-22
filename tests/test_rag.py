from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.rag import (
    RAGError,
    LocalVectorStore,
    answer_question,
    build_prompt,
    compute_file_hash,
    format_context,
    score_to_relevance,
    split_documents,
    validate_pdf,
)


class FakeVectorStore:
    def similarity_search_with_score(self, question, k):
        assert question == "美容手術可以理賠嗎？"
        assert k == 3
        return [
            (
                Document(
                    page_content="美容手術不負給付責任，但重建基本功能者不在此限。",
                    metadata={"page": 1},
                ),
                0.25,
            )
        ]


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(content="原則上不理賠，但重建基本功能屬例外。（第 2 頁）")


def test_hash_is_stable_and_content_sensitive():
    assert compute_file_hash(b"abc") == compute_file_hash(b"abc")
    assert compute_file_hash(b"abc") != compute_file_hash(b"abd")


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (b"", "empty.pdf", "空"),
        (b"hello", "wrong.pdf", "有效"),
        (b"%PDF-test", "wrong.txt", "PDF 格式"),
    ],
)
def test_pdf_validation_errors(data, filename, message):
    with pytest.raises(RAGError, match=message):
        validate_pdf(data, filename)


def test_split_documents_keeps_page_metadata():
    docs = [Document(page_content="條款內容。" * 100, metadata={"page": 3})]
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)


def test_context_uses_human_page_numbers_and_sources():
    context, sources = format_context(
        [(Document(page_content="住院條款", metadata={"page": 0}), 0.5)]
    )
    assert "第 1 頁" in context
    assert sources[0].page == 1
    assert sources[0].relevance == 0.5
    assert sources[0].distance == 0.5


def test_local_search_finds_relevant_chinese_clause_without_api():
    store = LocalVectorStore(
        [
            Document(page_content="住院醫療費用與病房差額。", metadata={"page": 0}),
            Document(
                page_content="美容手術不負給付責任，但重建基本功能者不在此限。",
                metadata={"page": 1},
            ),
        ]
    )
    results = store.similarity_search_with_score("美容手術可以理賠嗎？", k=1)
    assert results[0][0].metadata["page"] == 1


def test_relevance_is_bounded():
    assert score_to_relevance(-2) == 1
    assert 0 <= score_to_relevance(999) <= 1
    assert score_to_relevance(float("inf")) == 0


def test_prompt_requires_grounded_answer():
    prompt = build_prompt("問題", "條款", [])
    assert "只能依據" in prompt
    assert "檢索到的保單內容：\n條款" in prompt
    assert "問題" in prompt
    assert "📌 **結論**" in prompt
    assert "📄 **條文依據**" in prompt
    assert "⚠️ **注意事項**" in prompt


def test_answer_question_with_mock_model_has_sources():
    llm = FakeLLM()
    result = answer_question(
        FakeVectorStore(),
        "美容手術可以理賠嗎？",
        "test-key",
        llm=llm,
    )
    assert "第 2 頁" in result.answer
    assert result.sources[0].page == 2
    assert "美容手術" in llm.prompt


def test_empty_question_does_not_call_model():
    with pytest.raises(RAGError, match="空白"):
        answer_question(FakeVectorStore(), "  ", "test-key", llm=FakeLLM())
