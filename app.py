from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import APP_NAME, DEFAULT_PDF, SAMPLE_QUESTIONS
from src.rag import (
    DocumentInfo,
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
}


def get_api_key() -> str:
    """Read the API key from the environment or Streamlit Secrets."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_resource(show_spinner=False)
def cached_knowledge_base(
    document_hash: str, pdf_bytes: bytes, filename: str, _api_key: str
):
    """Cache parsing, embeddings, and the FAISS index by document hash."""
    return build_knowledge_base(pdf_bytes, filename, _api_key)


def reset_messages() -> None:
    st.session_state.messages = [WELCOME_MESSAGE.copy()]


def activate_document(pdf_bytes: bytes, filename: str, api_key: str) -> None:
    """Build and activate a document only after the new index succeeds."""
    document_hash = compute_file_hash(pdf_bytes)
    if document_hash == st.session_state.get("document_hash"):
        return

    vectorstore, document_info = cached_knowledge_base(
        document_hash, pdf_bytes, filename, api_key
    )
    st.session_state.vectorstore = vectorstore
    st.session_state.document_info = document_info
    st.session_state.document_hash = document_hash
    reset_messages()


def ask(question: str, api_key: str) -> None:
    question = question.strip()
    if not question:
        st.warning("請先輸入問題，再送出查詢。")
        return
    if not api_key:
        st.error("尚未設定 OPENAI_API_KEY，請先完成環境設定。")
        return
    if "vectorstore" not in st.session_state:
        st.error("知識庫尚未就緒，請稍後重試或重新上傳 PDF。")
        return

    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    try:
        prior_messages = st.session_state.messages[:-1]
        result = answer_question(
            st.session_state.vectorstore,
            question,
            api_key,
            chat_history=prior_messages[-6:],
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": [source.to_dict() for source in result.sources],
            }
        )
    except Exception as exc:  # The UI must remain usable if an external API fails.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": friendly_error(exc),
                "sources": [],
            }
        )


if "messages" not in st.session_state:
    reset_messages()

api_key = get_api_key()
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

    if api_key and candidate_bytes:
        try:
            with st.spinner("正在建立文件索引…"):
                activate_document(candidate_bytes, candidate_name, api_key)
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
    elif not api_key:
        st.warning("需要設定 OPENAI_API_KEY 才能建立知識庫。")

    st.caption("上傳文件僅保留於記憶體，重新整理工作階段後即清除。")
    if st.button("清除對話", use_container_width=True, icon="🗑️"):
        reset_messages()
        st.rerun()


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

status_text = "系統就緒" if info and api_key else "等待設定"
st.markdown(
    f"""
    <div class="trust-row">
      <span>● {status_text}</span><span>回答附原文頁碼</span>
      <span>PDF 僅限工作階段</span><span>不取代專業理賠審核</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_tab, claims_tab, knowledge_tab = st.tabs(
    ["💬 AI 條款問答", "📋 理賠指南", "📚 知識庫資訊"]
)

with chat_tab:
    st.markdown("#### 常見問題")
    question_columns = st.columns(len(SAMPLE_QUESTIONS))
    selected_question = ""
    for column, item in zip(question_columns, SAMPLE_QUESTIONS):
        with column:
            if st.button(
                item["label"],
                key=f"sample_{item['label']}",
                help=item["question"],
                use_container_width=True,
            ):
                selected_question = item["question"]

    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = "🛡️" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"查看引用依據（{len(sources)} 則）"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**來源 {index}｜第 {source['page']} 頁**  "
                            f"相關度 {source['relevance']:.0%}"
                        )
                        st.caption(source["excerpt"])
            if message["role"] == "assistant":
                st.caption("本回答由 AI 依文件內容生成，僅供參考；實際理賠以保險公司審核為準。")

    typed_question = st.chat_input(
        "輸入保險條款問題，例如：美容手術可以理賠嗎？",
        disabled=not bool(info and api_key),
    )
    incoming_question = selected_question or typed_question
    if incoming_question:
        with st.spinner("正在比對保單條款…"):
            ask(incoming_question, api_key)
        st.rerun()

with claims_tab:
    render_claims_guide()

with knowledge_tab:
    render_knowledge_overview(info, api_key_configured=bool(api_key))

st.markdown(
    '<footer>2026 人工智慧跨域專題實作 · 安心保 AI 保險助理</footer>',
    unsafe_allow_html=True,
)
