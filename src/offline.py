from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.rag import RAGAnswer, Source


@dataclass(frozen=True)
class DemoEntry:
    keywords: tuple[str, ...]
    answer: str
    sources: tuple[Source, ...]


DEMO_ENTRIES = (
    DemoEntry(
        keywords=("美容", "整型", "整形", "醫美"),
        answer=(
            "一般美容手術原則上不在理賠範圍內。保單的除外責任明定，因美容手術或外科整型而住院，"
            "保險公司不負給付責任；但如果是為了重建基本功能所做的必要整型，則不受這項除外限制"
            "（第 2 頁）。\n\n"
            "因此，判斷重點不是手術名稱，而是手術是否具有重建基本功能的醫療必要性。實際申請時，"
            "建議備妥診斷證明與醫師對治療目的的說明，最終仍由保險公司依個案審核。"
        ),
        sources=(
            Source(
                page=2,
                excerpt="美容手術、外科整型。但為重建其基本功能所作之必要整型，不在此限。",
                relevance=1.0,
            ),
        ),
    ),
    DemoEntry(
        keywords=("申請文件", "準備哪些", "診斷書", "收據", "申領", "理賠文件"),
        answer=(
            "申請住院醫療費用保險金時，需準備以下文件（第 4 頁）：\n\n"
            "- 保險金申請書\n"
            "- 保險單或其謄本\n"
            "- 醫療診斷書或住院證明\n"
            "- 醫療費用收據正本；電子文件必要時可能被要求補紙本\n"
            "- 受益人的身分證明\n\n"
            "保單另約定，知悉保險事故後應於十日內通知保險公司，並在通知後儘速檢具文件申請"
            "（第 4 頁）。送件前仍建議向保險公司確認最新格式與正副本要求。"
        ),
        sources=(
            Source(
                page=4,
                excerpt="受益人申領本契約第六條保險金時，應檢具保險金申請書、保險單或其謄本、醫療診斷書或住院證明、醫療費用收據正本及受益人的身分證明。",
                relevance=1.0,
            ),
            Source(
                page=4,
                excerpt="應於知悉本公司應負保險責任之事故後十日內通知本公司，並於通知後儘速檢具所需文件申請給付。",
                relevance=0.96,
            ),
        ),
    ),
    DemoEntry(
        keywords=("住院醫療", "住院保障", "哪些項目", "醫療費用", "可以申請"),
        answer=(
            "被保險人在契約有效期間內，因約定的疾病或傷害以健保身分住院診療時，可申請住院期間"
            "自行負擔、或不屬健保給付範圍的約定費用；同一次住院的給付總額以保險金額為上限"
            "（第 1 頁）。\n\n"
            "約定項目包括醫師指示用藥、特定血液費用、掛號及證明文件、救護車費、病房費差額、"
            "部分膳食與護理費，以及超過健保給付的住院醫療費用（第 1 至 2 頁）。若未以健保身分，"
            "或在不具健保資格的醫院住院，保單約定按實際費用的 65% 給付，仍受保險金額限制"
            "（第 2 頁）。"
        ),
        sources=(
            Source(
                page=1,
                excerpt="住院期間內所發生，且依全民健康保險規定其保險對象應自行負擔及不屬全民健康保險給付範圍之各項費用核付；同一次住院給付總額不得超過保險金額。",
                relevance=1.0,
            ),
            Source(
                page=2,
                excerpt="不以全民健康保險之保險對象身分住院，或前往不具有全民健康保險之醫院住院者，依實際支付各項費用之 65% 給付，仍以保險金額為限。",
                relevance=0.94,
            ),
        ),
    ),
    DemoEntry(
        keywords=("除外責任", "不理賠", "不在理賠", "哪些情況", "不給付"),
        answer=(
            "保單列出的主要不給付情況包括：故意行為（含自殺及未遂）、犯罪行為、非法施用毒品，"
            "以及美容整型、部分先天畸形、非為當次住院治療目的的牙科手術、部分輔具裝設、非直接"
            "診治目的的健康檢查或療養等（第 2 頁）。\n\n"
            "懷孕、流產或分娩原則上也屬除外，但保單另列懷孕相關疾病、醫療必要流產及符合條件的"
            "剖腹產等例外（第 2 至 3 頁）。此外，已由健保或其他實支實付保險給付的部分，也可能受"
            "給付限制（第 2 頁）。個案仍須對照事故原因與例外條件。"
        ),
        sources=(
            Source(
                page=2,
                excerpt="被保險人因故意行為、犯罪行為或非法施用毒品所致疾病或傷害而住院診療者，本公司不負給付責任。",
                relevance=1.0,
            ),
            Source(
                page=2,
                excerpt="美容手術、外科整型、外觀可見之天生畸形及非因當次住院事故治療目的所進行之牙科手術等，列為除外事故。",
                relevance=0.97,
            ),
            Source(
                page=3,
                excerpt="懷孕、流產或分娩設有懷孕相關疾病、醫療行為必要之流產及符合條件之剖腹產等例外。",
                relevance=0.91,
            ),
        ),
    ),
)


def _match_score(question: str, keywords: Iterable[str]) -> int:
    compact = "".join(question.lower().split())
    return sum(len(keyword) for keyword in keywords if keyword.lower() in compact)


def answer_offline(question: str) -> RAGAnswer:
    """Return a curated, document-grounded answer for classroom demo topics."""
    # Entries are intentionally ordered from specific to broad. This prevents a
    # cosmetic-surgery question containing generic words such as「可以申請」
    # from being routed to the general hospitalization answer.
    for entry in DEMO_ENTRIES:
        if _match_score(question, entry.keywords) > 0:
            return RAGAnswer(entry.answer, list(entry.sources))
    return RAGAnswer(
        "離線展示模式目前可回答住院保障、除外責任、申請文件與美容手術四類問題。"
        "請改用上方範例問題，或設定 API Key 後查詢其他條款。",
        [],
    )
