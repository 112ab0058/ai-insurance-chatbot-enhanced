from __future__ import annotations

import re


PERSONAL_POLICY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"我的保單",
        r"我的保障",
        r"保單範圍",
        r"調出.*保單",
        r"查.*我的.*保單",
        r"保單號碼",
        r"保單編號",
        r"每月.*(費用|保費|繳)",
        r"(月費|保費|費用).*(多少|金額)",
        r"繳.*多少",
        r"繳費紀錄",
        r"扣款",
        r"信用卡",
        r"銀行帳戶",
        r"我的資料",
        r"個人資料",
        r"登入",
        r"帳戶",
        r"帳號",
    )
)


POLICY_CLAUSE_HINTS = (
    "可以改問：「這份保單的住院保障有哪些？」",
    "可以改問：「理賠需要準備哪些文件？」",
    "可以改問：「哪些情況不在理賠範圍內？」",
)


def is_personal_policy_question(question: str) -> bool:
    """Return True when the question requires personal account/policy data."""
    normalized = re.sub(r"\s+", "", question.strip())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in PERSONAL_POLICY_PATTERNS)


def personal_policy_boundary_answer(role: str = "user") -> str:
    """Explain the product boundary without calling the model."""
    if role == "staff":
        return (
            "📌 **結論**\n"
            "這題需要個人保單或帳務資料，本系統目前不查詢客戶個資，因此不會送出模型判斷。\n\n"
            "📄 **可查範圍**\n"
            "目前系統只根據上傳的保單條款、契約範本與保險 Q&A 回答保障範圍、除外責任、"
            "理賠文件與流程問題。\n\n"
            "⚠️ **處理建議**\n"
            "若客戶要查保費、保單號碼、繳費紀錄或個人保障明細，請改由公司內部保單系統或客服人員查核。\n\n"
            "💬 **可改問方向**\n"
            + "\n".join(f"- {hint}" for hint in POLICY_CLAUSE_HINTS)
        )
    return (
        "📌 **結論**\n"
        "這個問題需要個人保單或帳務資料，目前系統無法查詢您的個人資料。\n\n"
        "📄 **可查範圍**\n"
        "我可以協助查詢上傳文件中的保單條款，例如保障範圍、除外責任、理賠文件與申請流程。\n\n"
        "⚠️ **注意事項**\n"
        "若要查每月保費、保單號碼、繳費紀錄或個人保障明細，請直接洽保險公司或登入正式保戶服務系統確認。\n\n"
        "💬 **您可以改問**\n"
        + "\n".join(f"- {hint}" for hint in POLICY_CLAUSE_HINTS)
    )
