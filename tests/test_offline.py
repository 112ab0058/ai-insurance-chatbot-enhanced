from src.offline import answer_offline


def test_cosmetic_surgery_answer_is_grounded() -> None:
    result = answer_offline("美容手術可以申請理賠嗎？有沒有例外？")
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


def test_each_preset_question_routes_to_the_expected_answer() -> None:
    cases = [
        ("住院醫療費用有哪些項目可以申請理賠？", "65%"),
        ("哪些情況不在理賠範圍內？", "犯罪行為"),
        ("申請住院醫療理賠需要準備哪些文件？", "保險金申請書"),
        ("美容手術可以申請理賠嗎？有沒有例外？", "重建基本功能"),
    ]
    for question, expected in cases:
        assert expected in answer_offline(question).answer
