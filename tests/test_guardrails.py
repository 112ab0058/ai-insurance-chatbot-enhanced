from src.guardrails import (
    is_personal_policy_question,
    personal_policy_boundary_answer,
)


def test_personal_policy_questions_are_blocked() -> None:
    blocked_questions = [
        "我的保單範圍是什麼？",
        "調出我的保單",
        "我每月的費用是多少？",
        "我的保單號碼是多少？",
        "我要查繳費紀錄",
    ]

    for question in blocked_questions:
        assert is_personal_policy_question(question)


def test_clause_questions_are_not_blocked() -> None:
    allowed_questions = [
        "因意外受傷住院後，申請理賠通常要準備哪些文件？",
        "哪些情況不在理賠範圍內？",
        "美容手術可以理賠嗎？",
        "如果文件不齊全會怎麼辦？",
    ]

    for question in allowed_questions:
        assert not is_personal_policy_question(question)


def test_boundary_answer_explains_scope_without_model_claims() -> None:
    answer = personal_policy_boundary_answer("user")

    assert "無法查詢您的個人資料" in answer
    assert "保障範圍" in answer
    assert "除外責任" in answer
    assert "保戶服務系統" in answer
