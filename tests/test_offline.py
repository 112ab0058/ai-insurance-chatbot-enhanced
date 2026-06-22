from src.offline import answer_offline


def test_cosmetic_surgery_answer_is_grounded() -> None:
    result = answer_offline("美容手術可以理賠嗎？")
    assert "重建基本功能" in result.answer
    assert result.sources[0].page == 2


def test_claim_documents_answer_has_page_four_source() -> None:
    result = answer_offline("申請文件需要準備哪些？")
    assert "醫療費用收據正本" in result.answer
    assert {source.page for source in result.sources} == {4}


def test_unknown_question_has_safe_demo_fallback() -> None:
    result = answer_offline("這份保單可以拿來貸款嗎？")
    assert "離線展示模式" in result.answer
    assert result.sources == []
