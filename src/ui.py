from __future__ import annotations

from html import escape

import streamlit as st

from src.rag import DocumentInfo


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
        :root { --navy:#0b2341; --blue:#123d67; --teal:#16a394; --mint:#e8f7f3; --ink:#172b3f; --muted:#66788a; }
        html, body, [class*="css"] { font-family:'Noto Sans TC',sans-serif; color:var(--ink); }
        [data-testid="stAppViewContainer"] { background:#f4f7fa; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#071d35 0%,#0d3154 100%); }
        [data-testid="stSidebar"] * { color:#f5fbff; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.08); border-color:rgba(255,255,255,.2); }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { color:#0b2341!important; }
        [data-testid="stFileUploaderDropzoneInstructions"] { font-size:0; }
        [data-testid="stFileUploaderDropzoneInstructions"] > div { display:none; }
        [data-testid="stFileUploaderDropzoneInstructions"]:before { content:"拖曳 PDF 到這裡";font-size:.9rem;font-weight:600; }
        [data-testid="stFileUploaderDropzoneInstructions"] small { display:none; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span { font-size:0; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span:before { content:"選擇檔案";font-size:.875rem; }
        [data-testid="stSidebar"] .stButton button { border:1px solid rgba(255,255,255,.3); background:rgba(255,255,255,.08); color:white; }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] { background:#f7fbff;border:1px solid rgba(99,221,199,.45);border-radius:10px; }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * { color:#0b2341!important; }
        [data-testid="stSidebar"] .stSelectbox svg { fill:#0b2341!important; }
        [role="listbox"] [role="option"] { color:#172b3f!important;background:#fff!important; }
        [role="listbox"] [role="option"]:hover { background:#e7f6f3!important;color:#0d746d!important; }
        .block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:2rem; }
        .brand-lockup { display:flex; align-items:center; gap:.8rem; margin:.4rem 0 2rem; }
        .brand-icon { width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#38d3bd,#128d83);display:grid;place-items:center;font-weight:800;font-size:1.15rem;box-shadow:0 8px 22px rgba(22,163,148,.3); }
        .brand-lockup strong { display:block;font-size:1.25rem;letter-spacing:.08em; }
        .brand-lockup span { display:block;font-size:.66rem;opacity:.62;letter-spacing:.1em;margin-top:.1rem; }
        .document-card { padding:1rem;border-radius:14px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.13);margin:1rem 0; }
        .document-card strong,.document-card small { display:block;margin-top:.5rem;overflow-wrap:anywhere; }
        .document-card small { opacity:.65; }
        .status-row { color:#82eadb!important;font-size:.8rem;font-weight:600; }
        .status-dot { display:inline-block;width:8px;height:8px;background:#4de0b9;border-radius:50%;margin-right:.45rem;box-shadow:0 0 0 4px rgba(77,224,185,.12); }
        .hero { position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:center;background:linear-gradient(118deg,#09233f 0%,#124a6b 68%,#147c78 100%);border-radius:24px;padding:2.8rem 3.1rem;color:white;box-shadow:0 20px 50px rgba(12,38,66,.18); }
        .hero:after { content:"";position:absolute;width:330px;height:330px;border-radius:50%;right:-110px;top:-160px;border:1px solid rgba(255,255,255,.16);box-shadow:0 0 0 45px rgba(255,255,255,.025),0 0 0 90px rgba(255,255,255,.02); }
        .hero h1 { font-size:clamp(2rem,4vw,3.15rem);line-height:1.2;margin:.65rem 0 .8rem;letter-spacing:-.04em; }
        .hero p { color:#c9dee9;margin:0;max-width:650px;font-size:1.02rem; }
        .eyebrow { color:#63ddc7;font-size:.72rem;font-weight:800;letter-spacing:.18em; }
        .hero-shield { position:relative;z-index:1;min-width:96px;height:112px;margin-left:2rem;clip-path:polygon(50% 0,95% 18%,87% 70%,50% 100%,13% 70%,5% 18%);background:linear-gradient(145deg,#37d5bd,#0b8e86);display:grid;place-items:center;font-size:2.4rem;font-weight:800;box-shadow:0 20px 35px rgba(0,0,0,.18); }
        .trust-row { display:flex;gap:1.2rem;flex-wrap:wrap;padding:1rem 1.3rem;margin:1rem 0 1.6rem;background:white;border:1px solid #e3ebf0;border-radius:14px;color:#5b6e80;font-size:.82rem;box-shadow:0 5px 18px rgba(12,38,66,.04); }
        .trust-row span:first-child { color:#138b7f;font-weight:700; }
        .trust-row.staff-mode { background:#f6f9fc;border-color:#d5e1ea;border-left:4px solid #5f7f9d;box-shadow:0 6px 20px rgba(64,91,118,.08); }
        .trust-row.staff-mode span:first-child { color:#355f7f;font-weight:800; }
        .confidence-badge { display:inline-flex;align-items:center;gap:.35rem;border-radius:999px;padding:.35rem .75rem;font-weight:800;font-size:.86rem;margin:.35rem 0 .55rem; }
        .confidence-badge.high { background:#d4edda;color:#155724;border:1px solid #b7dfc0; }
        .confidence-badge.medium { background:#fff3cd;color:#856404;border:1px solid #f1df9a; }
        .confidence-note { color:#5f6f7d;font-size:.82rem;margin-left:.5rem; }
        [data-baseweb="tab-list"] { gap:.4rem;background:white;padding:.45rem;border-radius:15px;border:1px solid #e4ebf0; }
        [data-baseweb="tab"] { border-radius:11px;padding:.65rem 1.1rem; }
        [aria-selected="true"] { background:#e7f6f3!important;color:#0d746d!important; }
        .stButton button { border-radius:12px;border:1px solid #dce7ec;min-height:44px;font-weight:600;transition:.2s; }
        .stButton button:hover { border-color:#1ba695;color:#0b756b;box-shadow:0 7px 18px rgba(22,163,148,.1); }
        .chat-divider { height:1px;background:#e0e8ed;margin:1rem 0 1.3rem; }
        [data-testid="stChatMessage"] { background:white;border:1px solid #e3eaef;border-radius:16px;padding:.45rem .8rem;box-shadow:0 5px 15px rgba(12,38,66,.035); }
        [data-testid="stChatInput"] { border-color:#b8d5d0; }
        .guide-card { background:white;border:1px solid #e3eaef;border-radius:16px;padding:1.25rem;height:100%;box-shadow:0 5px 16px rgba(12,38,66,.035); }
        .guide-number { width:32px;height:32px;border-radius:10px;background:#e3f7f3;color:#0d7d72;display:grid;place-items:center;font-weight:800;margin-bottom:.75rem; }
        .guide-card h4 { margin:.2rem 0 .45rem; }
        .guide-card p { color:#687b8b;font-size:.88rem;line-height:1.65; }
        .info-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0; }
        .info-card { background:white;border:1px solid #e2eaf0;border-radius:16px;padding:1.25rem; }
        .info-card small { color:#708191;display:block;margin-bottom:.4rem; }
        .info-card strong { font-size:1.25rem;overflow-wrap:anywhere; }
        footer { text-align:center;color:#81909d;font-size:.76rem;padding:2.4rem 0 .5rem; }
        @media(max-width:700px){
          .block-container{padding:1rem .8rem 1.5rem}.hero{padding:2rem 1.4rem;border-radius:18px}.hero-shield{display:none}.hero h1{font-size:2rem}.trust-row{gap:.65rem}.trust-row span{width:100%}.info-grid{grid-template-columns:1fr}[data-baseweb="tab"]{padding:.55rem .5rem;font-size:.82rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_claims_guide() -> None:
    st.markdown("### 理賠申請四步驟")
    st.caption("以下為一般準備流程，實際文件與結果仍以保險公司審核為準。")
    steps = [
        ("確認事故", "確認事故日期、住院原因與治療期間，並先核對保單是否仍在有效期間。"),
        ("核對條款", "查看保障項目、除外責任、等待期與給付限制，保留相關條款頁碼。"),
        ("備齊文件", "通常包含理賠申請書、診斷證明、醫療費用收據及必要的身分證明。"),
        ("送件追蹤", "向保險公司送件並保存收件證明；若被要求補件，確認期限與缺少項目。"),
    ]
    columns = st.columns(4)
    for index, (column, (title, description)) in enumerate(zip(columns, steps), start=1):
        with column:
            st.markdown(
                f'<div class="guide-card"><div class="guide-number">{index}</div>'
                f"<h4>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### 送件前檢查")
    check_columns = st.columns(2)
    items = [
        "理賠申請書已填寫並簽名",
        "診斷證明書載明住院原因與日期",
        "醫療費用明細及收據齊全",
        "確認收據正本／副本要求",
        "保留送件紀錄與聯絡窗口",
        "個案特殊文件已向保險公司確認",
    ]
    for index, item in enumerate(items):
        with check_columns[index % 2]:
            st.checkbox(item, key=f"claim_check_{index}")

    st.info("若對保障範圍不確定，可回到「AI 條款問答」，詢問文件中的申請條件與除外責任。")


def render_knowledge_overview(
    info: DocumentInfo | None,
    github_token_configured: bool,
    offline_mode: bool = False,
    chunks: list | None = None,
) -> None:
    st.markdown("### 目前知識庫")
    if info:
        safe_filename = escape(info.filename)
        st.markdown(
            f"""
            <div class="info-grid">
              <div class="info-card"><small>文件名稱</small><strong>{safe_filename}</strong></div>
              <div class="info-card"><small>文件頁數</small><strong>{info.page_count} 頁</strong></div>
              <div class="info-card"><small>檢索區塊</small><strong>{info.chunk_count} 個</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🔍 查看所有條文區塊"):
            if chunks:
                rows = []
                for index, chunk in enumerate(chunks, start=1):
                    compact = " ".join(chunk.page_content.split())
                    rows.append(
                        {
                            "編號": f"Chunk #{index}",
                            "頁碼": f"第 {int(chunk.metadata.get('page', 0)) + 1} 頁",
                            "內容預覽": compact[:150] + ("…" if len(compact) > 150 else ""),
                            "字元數": len(chunk.page_content),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("目前沒有可預覽的條文區塊。")
    else:
        st.warning("目前尚未建立知識庫。")

    st.markdown("### 系統如何回答")
    st.markdown(
        """
        1. 解析 PDF 並保留每段文字的原始頁碼。
        2. 在伺服器本機搜尋最相關的條款，不另外呼叫 Embedding API。
        3. 連線模式由 4o-mini 整理繁體中文答案；離線查詢則使用預先核對的回答。
        4. 顯示引用頁碼與原文節錄，讓使用者自行核對。
        """
    )
    key_status = "已安全載入" if github_token_configured else "尚未設定"
    st.markdown(f"**4o-mini 連線：** {key_status}  ")
    if offline_mode:
        st.info("目前使用離線查詢模式：答案已預先依預設保單核對，不會向外部 AI 服務傳送資料。")
    st.markdown("**隱私設計：** 上傳的 PDF 僅存在目前伺服器工作階段的記憶體中。")
    st.warning("此系統是條款閱讀輔助工具，不構成保險、法律或醫療建議，也不代表最終理賠結果。")
