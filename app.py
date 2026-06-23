from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import APP_NAME, DEFAULT_PDF
from src.offline import answer_offline
from src.rag import (
    DocumentInfo,
    RAGAnswer,
    RAGError,
    answer_question,
    build_knowledge_base,
    compute_file_hash,
    friendly_error,
)
from src.ui import apply_styles, render_claims_guide, render_knowledge_overview


load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")
apply_styles()


WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "您好，我是安心保 AI 保險助理。您可以詢問住院保障、除外責任、"
        "申請文件等問題；我的回答會附上保單條款頁碼，方便您核對原文。"
    ),
    "sources": [],
    "low_confidence": False,
}


def get_github_token() -> str:
    """Read the 4o-mini service token from the environment or Streamlit Secrets."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("GITHUB_TOKEN", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_resource(show_spinner=False)
def cached_knowledge_base(
    document_hash: str, pdf_bytes: bytes, filename: str, _github_token: str
):
    """Cache local PDF parsing and search index by document hash."""
    return build_knowledge_base(pdf_bytes, filename)


def reset_messages() -> None:
    st.session_state.messages = [WELCOME_MESSAGE.copy()]


def build_handoff_summary(messages: list[dict]) -> str:
    """Build a copyable handoff summary locally without calling an LLM."""
    pairs: list[tuple[str, str]] = []
    low_confidence_questions: list[str] = []
    pending_question = ""

    for message in messages:
        if message.get("role") == "user":
            pending_question = str(message.get("content", "")).strip()
        elif message.get("role") == "assistant" and pending_question:
            answer = " ".join(str(message.get("content", "")).split())
            answer_brief = answer if len(answer) <= 180 else f"{answer[:180].rstrip()}…"
            pairs.append((pending_question, answer_brief))
            if message.get("low_confidence"):
                low_confidence_questions.append(pending_question)
            pending_question = ""

    lines = ["客戶詢問摘要："]
    if pairs:
        lines.extend(
            f"{index}. {question} → {answer}"
            for index, (question, answer) in enumerate(pairs, start=1)
        )
    else:
        lines.append("（目前尚無客戶提問）")

    flagged = "、".join(low_confidence_questions) or "無"
    lines.append(f"建議承辦人員確認事項：{flagged}")
    return "\n".join(lines)


def activate_document(pdf_bytes: bytes, filename: str, github_token: str) -> None:
    """Build and activate a document only after the new index succeeds."""
    document_hash = compute_file_hash(pdf_bytes)
    if (
        document_hash == st.session_state.get("document_hash")
        and "vectorstore" in st.session_state
    ):
        return

    vectorstore, document_info = cached_knowledge_base(
        document_hash, pdf_bytes, filename, github_token
    )
    st.session_state.vectorstore = vectorstore
    st.session_state.document_info = document_info
    st.session_state.document_hash = document_hash
    st.session_state.answer_cache = {}
    reset_messages()


def activate_offline_document(pdf_bytes: bytes, filename: str) -> None:
    document_hash = compute_file_hash(pdf_bytes)
    if (
        document_hash == st.session_state.get("document_hash")
        and "vectorstore" in st.session_state
    ):
        return
    vectorstore, document_info = cached_knowledge_base(
        document_hash, pdf_bytes, filename, ""
    )
    st.session_state.vectorstore = vectorstore
    st.session_state.document_info = document_info
    st.session_state.document_hash = document_hash
    st.session_state.answer_cache = {}
    reset_messages()


# [修改3] AI 回答會依使用者角色套用對應的結構化提示詞。
def ask(question: str, github_token: str, role: str = "一般用戶") -> None:
    question = question.strip()
    if not question:
        st.warning("請先輸入問題，再送出查詢。")
        return
    if "vectorstore" not in st.session_state:
        st.error("知識庫尚未就緒，請稍後重試或重新上傳 PDF。")
        return

    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    try:
        prior_messages = st.session_state.messages[:-1]
        cache_key = f"{st.session_state.get('document_hash', '')}:{role}:{question}"
        cached_result = st.session_state.setdefault("answer_cache", {}).get(cache_key)
        if cached_result:
            answer, sources, low_confidence = cached_result
            result = RAGAnswer(
                answer=answer,
                sources=sources,
                low_confidence=low_confidence,
            )
        elif github_token:
            result = answer_question(
                st.session_state.vectorstore,
                question,
                github_token,
                chat_history=prior_messages[-4:],
                role=role,
            )
            st.session_state.answer_cache[cache_key] = (
                result.answer,
                result.sources,
                result.low_confidence,
            )
        elif st.session_state.document_info.filename == DEFAULT_PDF:
            result = answer_offline(question)
        else:
            raise RAGError("自訂 PDF 需要設定 API Key；離線展示僅支援預設保單。")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": [source.to_dict() for source in result.sources],
                "low_confidence": result.low_confidence,
            }
        )
    except Exception as exc:  # The UI must remain usable if an external API fails.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": friendly_error(exc),
                "sources": [],
                "low_confidence": False,
            }
        )


if "messages" not in st.session_state:
    reset_messages()

# [修改1] pending_question 讓常見問題按鈕在 rerun 後自動送出。
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if st.session_state.get("user_role") not in {"一般用戶", "保險業務/客服"}:
    st.session_state["user_role"] = "一般用戶"

github_token = get_github_token()
default_path = Path(DEFAULT_PDF)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-icon">安</div>
            <div><strong>安心保</strong><span>AI Insurance Copilot</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio(
        "使用角色",
        options=["一般用戶", "保險業務/客服"],
        key="user_role",
        horizontal=True,
        help="業務/客服模式會使用較完整的條款說明與客戶話術建議。",
    )
    st.markdown("### 文件中心")
    uploaded_file = st.file_uploader(
        "替換知識文件",
        type=["pdf"],
        help="僅在本次瀏覽工作階段使用，不會保存到伺服器。",
    )
    if uploaded_file is not None:
        candidate_bytes = uploaded_file.getvalue()
        candidate_name = uploaded_file.name
    else:
        candidate_bytes = default_path.read_bytes() if default_path.exists() else b""
        candidate_name = default_path.name

    if candidate_bytes:
        try:
            with st.spinner("正在解析保單文件…"):
                if github_token:
                    activate_document(candidate_bytes, candidate_name, github_token)
                else:
                    activate_offline_document(candidate_bytes, candidate_name)
        except Exception as exc:
            st.error(f"新文件無法建立索引：{friendly_error(exc)}")
    elif not candidate_bytes:
        st.error("找不到預設 insurance.pdf。")

    info: DocumentInfo | None = st.session_state.get("document_info")
    if info:
        safe_filename = escape(info.filename)
        st.markdown(
            f"""
            <div class="document-card">
              <div class="status-row"><span class="status-dot"></span>知識庫已就緒</div>
              <strong>{safe_filename}</strong>
              <small>{info.page_count} 頁 · {info.chunk_count} 個文字區塊</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not github_token:
        st.warning("離線展示模式僅支援預設保單。")

    if info and not github_token:
        st.info("目前為免 API 離線展示模式，可使用四個範例問題。")

    # [修改 10] 將隱私說明移至側邊欄知識庫狀態下方。
    st.caption("隱私說明：上傳的 PDF 僅存於當前工作階段記憶體，關閉頁面後自動清除。")
    if st.button("清除對話", use_container_width=True, icon="🗑️"):
        reset_messages()
        st.rerun()

    # [修改 9] 顯示提問統計與建議問法。
    question_count = sum(
        1 for message in st.session_state.messages if message.get("role") == "user"
    )
    st.caption(f"本次對話已提問：{question_count} 題")
    st.divider()
    with st.expander("💡 建議問法"):
        st.markdown(
            "- 癌症住院每天可以領多少？\n"
            "- 哪些手術不在保障範圍？\n"
            "- 申請理賠需要準備哪些文件？"
        )


st.markdown(
    """
    <section class="hero">
      <div>
        <span class="eyebrow">INSURANCE INTELLIGENCE</span>
        <h1>把複雜條款，<br>變成清楚答案。</h1>
        <p>運用 RAG 搜尋保單原文，提供可追溯頁碼的保險條款問答。</p>
      </div>
      <div class="hero-shield">✓</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# [修改 10] 精簡頂部狀態列，移除工作階段標籤。
status_text = "4o-mini 已連線" if info and github_token else "離線展示模式" if info else "等待設定"
st.markdown(
    f"""
    <div class="trust-row">
      <span>● {status_text}</span><span>回答附原文頁碼</span>
      <span>不取代專業理賠審核</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_tab, claims_tab, knowledge_tab = st.tabs(
    ["💬 AI 條款問答", "📋 理賠指南", "📚 知識庫資訊"]
)

with chat_tab:
    # [修改6] 沒有向量知識庫時顯示 Empty State，並停止渲染聊天元件。
    if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None:
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px; color: #888;">
                <div style="font-size: 48px;">⬆️</div>
                <h3 style="color: #555;">請先上傳保單 PDF</h3>
                <p>從左側「文件中心」拖曳或選擇保單 PDF<br>系統將自動建立知識庫，約需 10–30 秒</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown("#### 常見問題")
    # [修改1] 四個常見問題按鈕一鍵填入 pending_question 並立即送出。
    quick_questions = {
        "住院保障": "骨折住院可以申請理賠嗎？",
        "除外責任": "哪些情況不在理賠範圍內？",
        "申請文件": "理賠需要準備哪些文件？",
        "美容手術": "美容手術可以理賠嗎？",
    }
    cols = st.columns(4)
    for i, (label, question) in enumerate(quick_questions.items()):
        with cols[i]:
            if st.button(label, use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)
    for message_index, message in enumerate(st.session_state.messages):
        avatar = "🛡️" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            # [修改2] 從 messages 內持久化的 content/page/score 重建引用條文。
            if message["role"] == "assistant" and sources:
                with st.expander("📄 引用條文（點擊展開查看依據）"):
                    for source_index, source in enumerate(sources):
                        score = float(source.get("score", 1.0))
                        confidence = max(0, round(100 - score * 100))
                        page = source.get("page", "未知")
                        content = source.get("content", "")[:300]
                        st.markdown(
                            f"""
**【來源 {source_index + 1}】** 頁碼：第 {page} 頁　｜　相似度：**{confidence} / 100**

> {content}

---
"""
                        )
            if message["role"] == "assistant":
                if github_token:
                    st.caption("本回答由 4o-mini 依文件內容生成，僅供參考；實際理賠以保險公司審核為準。")
                else:
                    st.caption("離線展示答案已預先依保單核對，僅供參考；實際理賠以保險公司審核為準。")
                # [修改5] 以原生 Streamlit 按鈕顯示可手動複製的純文字框。
                col1, col2 = st.columns([6, 1])
                with col2:
                    if st.button("📋 複製", key=f"copy_{message_index}"):
                        st.session_state[f"show_copy_{message_index}"] = True
                if st.session_state.get(f"show_copy_{message_index}"):
                    st.text_area(
                        "複製以下內容：",
                        value=message["content"],
                        height=100,
                        key=f"textarea_{message_index}",
                    )

    if st.button("轉接真人客服", use_container_width=True, icon="☎️"):
        st.session_state["show_handoff_summary"] = True
    if st.session_state.get("show_handoff_summary"):
        st.text_area(
            "客服轉接摘要（可複製貼給承辦人員）",
            value=build_handoff_summary(st.session_state.messages),
            height=220,
            key="handoff_summary_text",
        )

    has_knowledge_base = bool(info and st.session_state.get("vectorstore"))

    # [修改1] 優先取出快速問題，直接走入與手動輸入相同的問答流程。
    user_input = ""
    if "pending_question" in st.session_state and st.session_state["pending_question"]:
        user_input = st.session_state.pop("pending_question")
    typed_question = st.chat_input(
        "輸入保險條款問題，例如：美容手術可以理賠嗎？",
        disabled=not has_knowledge_base,
    )
    if typed_question:
        user_input = typed_question
    if user_input:
        # [修改4] 呼叫 4o-mini 前在 assistant 對話泡泡中顯示 Typing Indicator。
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("安心保正在查詢保單條款中..."):
                ask(user_input, github_token, st.session_state["user_role"])
        st.rerun()

with claims_tab:
    render_claims_guide()

with knowledge_tab:
    # [修改 7] 傳入所有 chunks，供知識庫資訊 Tab 預覽。
    chunks = getattr(st.session_state.get("vectorstore"), "documents", [])
    render_knowledge_overview(
        info,
        github_token_configured=bool(github_token),
        offline_mode=bool(info and not github_token),
        chunks=chunks,
    )

st.markdown(
    '<footer>2026 人工智慧跨域專題實作 · 安心保 AI 保險助理</footer>',
    unsafe_allow_html=True,
)
