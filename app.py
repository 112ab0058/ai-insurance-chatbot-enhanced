from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.config import APP_NAME, DEFAULT_PDF, SAMPLE_QUESTIONS
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


# [修改 6] 使用瀏覽器原生 Clipboard API，失敗時顯示手動複製文字框。
def render_copy_button(answer: str, message_index: int) -> None:
    payload = json.dumps(answer, ensure_ascii=False)
    element_id = f"copy_answer_{message_index}"
    components.html(
        f"""
        <div style="text-align:right;font-family:system-ui,sans-serif">
          <button id="{element_id}" onclick="copyAnswer()"
            style="border:1px solid #d8e4e9;background:#fff;border-radius:8px;
                   padding:6px 10px;color:#486273;cursor:pointer;font-size:13px">
            📋 複製回答
          </button>
          <span id="{element_id}_status" style="margin-left:6px;color:#168b7e;font-size:12px"></span>
          <textarea id="{element_id}_fallback" readonly
            style="display:none;width:100%;height:80px;margin-top:6px"></textarea>
        </div>
        <script>
        async function copyAnswer() {{
          const text = {payload};
          const status = document.getElementById('{element_id}_status');
          try {{
            await navigator.clipboard.writeText(text);
            status.textContent = '已複製';
          }} catch (error) {{
            const fallback = document.getElementById('{element_id}_fallback');
            fallback.value = text;
            fallback.style.display = 'block';
            fallback.select();
            status.textContent = '請從下方手動複製';
          }}
        }}
        </script>
        """,
        height=105,
    )


# [修改 3] AI 回答會套用 src.rag.build_prompt 定義的三段式結構。
def ask(question: str, github_token: str) -> None:
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
        cache_key = f"{st.session_state.get('document_hash', '')}:{question}"
        cached_result = st.session_state.setdefault("answer_cache", {}).get(cache_key)
        if cached_result:
            answer, sources = cached_result
            result = RAGAnswer(answer=answer, sources=sources)
        elif github_token:
            result = answer_question(
                st.session_state.vectorstore,
                question,
                github_token,
                chat_history=prior_messages[-4:],
            )
            st.session_state.answer_cache[cache_key] = (result.answer, result.sources)
        elif st.session_state.document_info.filename == DEFAULT_PDF:
            result = answer_offline(question)
        else:
            raise RAGError("自訂 PDF 需要設定 API Key；離線展示僅支援預設保單。")
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

# [修改 1] pending_question 讓常見問題按鈕在 rerun 後自動送出。
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

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
    st.markdown("#### 常見問題")
    question_columns = st.columns(len(SAMPLE_QUESTIONS))
    # [修改 1] 點擊後寫入完整問題並立即 rerun，下一輪自動執行。
    for column, item in zip(question_columns, SAMPLE_QUESTIONS):
        with column:
            if st.button(
                item["label"],
                key=f"sample_{item['label']}",
                help=item["question"],
                use_container_width=True,
            ):
                st.session_state.pending_question = item["question"]
                st.rerun()

    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)
    for message_index, message in enumerate(st.session_state.messages):
        avatar = "🛡️" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            # [修改 2 + 修改 5] 顯示 top-3 原始條文、distance 與 0–100 信心分數。
            if sources:
                with st.expander("📄 引用條文（點擊展開）"):
                    for index, source in enumerate(sources, start=1):
                        confidence = round(float(source.get("relevance", 0)) * 100)
                        raw_distance = source.get("distance")
                        distance = (
                            1 - confidence / 100
                            if raw_distance is None
                            else float(raw_distance)
                        )
                        st.markdown(
                            f"---\n**【來源 {index}】頁碼：第 {source['page']} 頁**  \n"
                            f"距離分數：{distance:.2f}｜相似度：{confidence / 100:.2f}｜"
                            f"**相似度 {confidence} / 100**"
                        )
                        st.caption(source["excerpt"])
            if message["role"] == "assistant":
                if github_token:
                    st.caption("本回答由 4o-mini 依文件內容生成，僅供參考；實際理賠以保險公司審核為準。")
                else:
                    st.caption("離線展示答案已預先依保單核對，僅供參考；實際理賠以保險公司審核為準。")
                if message_index > 0:
                    render_copy_button(message["content"], message_index)

    has_knowledge_base = bool(info and st.session_state.get("vectorstore"))

    # [修改 8] 尚未建立知識庫時顯示上傳引導卡片。
    if not has_knowledge_base:
        st.markdown(
            """
            <div class="guide-card" style="text-align:center;padding:2.2rem">
              <div style="font-size:2.6rem">⬆️</div>
              <h3>請先上傳保單 PDF</h3>
              <p>從左側文件中心拖曳或選擇保單 PDF，系統將自動建立知識庫，約需 10–30 秒。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    typed_question = st.chat_input(
        "輸入保險條款問題，例如：美容手術可以理賠嗎？",
        disabled=not has_knowledge_base,
    )
    incoming_question = st.session_state.pending_question or typed_question
    if incoming_question:
        st.session_state.pending_question = ""
        # [修改 4] AI 回答出現前顯示 Typing Indicator。
        with st.spinner("安心保正在查詢保單條款中..."):
            ask(incoming_question, github_token)
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
