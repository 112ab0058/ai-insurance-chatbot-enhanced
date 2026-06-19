import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="AI Insurance Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==================================
# CUSTOM CSS
# ==================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #f5f9ff 0%,
        #eef5ff 100%
    );
}

.user-msg {
    background-color: #DCF8C6;
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
    margin-bottom: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

.bot-msg {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    margin-top: 10px;
    margin-bottom: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD ENV
# ==================================

load_dotenv()

# ==================================
# SIDEBAR
# ==================================

with st.sidebar:

    st.title("📋 系統資訊")

    st.success("✅ 知識庫已連接")

    st.markdown("""
### 使用技術

- OpenAI GPT-4o-mini
- LangChain
- OpenAI Embeddings
- FAISS
- RAG

### 專題名稱

AI保險客服與知識管理 Agent

### 功能

- 保單查詢
- 理賠問答
- 智慧客服
- 知識管理
""")

# ==================================
# HEADER
# ==================================

st.title("🤖 AI保險客服與知識管理 Agent")

st.markdown("""
### 基於 RAG 與 OpenAI GPT-4o-mini 的智慧保險問答系統
""")

st.success("✅ 系統運作正常")

# ==================================
# METRICS
# ==================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📄 知識文件",
        "1"
    )

with col2:
    st.metric(
        "🤖 AI模型",
        "GPT-4o-mini"
    )

with col3:
    st.metric(
        "🗄️ 向量資料庫",
        "FAISS"
    )

st.markdown("---")

# ==================================
# SAMPLE QUESTIONS
# ==================================

with st.expander("💡 範例問題"):

    st.write("• 骨折住院可以申請理賠嗎？")
    st.write("• 哪些情況不在理賠範圍內？")
    st.write("• 美容手術可以理賠嗎？")
    st.write("• 理賠需要準備哪些文件？")
    st.write("• 癌症住院的保障內容是什麼？")
# ==================================
# LOAD PDF
# ==================================

loader = PyPDFLoader("insurance.pdf")
documents = loader.load()

# ==================================
# SPLIT DOCS
# ==================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# ==================================
# EMBEDDINGS
# ==================================

embeddings = OpenAIEmbeddings()

# ==================================
# FAISS
# ==================================

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# ==================================
# GPT MODEL
# ==================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==================================
# CHAT HISTORY
# ==================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================
# INPUT
# ==================================

question = st.text_input(
    "請輸入您的問題",
    placeholder="例如：骨折可以申請理賠嗎？"
)

if st.button("🚀 送出問題"):

    with st.spinner("🤖 AI 正在分析保單內容..."):

        docs_found = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs_found]
        )

        with st.expander("Debug Context"):
            st.write(context)

        prompt = f"""
你是一位專業保險客服。

請根據提供的保單內容回答問題。

規則：
1. 優先根據保單內容回答。
2. 可以根據保單條文進行合理推論。
3. 不要編造不存在的保障內容。
4. 若真的找不到相關資訊，再回答：
   「保單中未提及相關資訊」。
5. 回答請使用繁體中文。

保單內容：
{context}

問題：
{question}
"""

        response = llm.invoke(prompt)

        st.session_state.messages.append(
            ("user", question)
        )

        st.session_state.messages.append(
            ("bot", response.content)
        )
# ==================================
# DISPLAY CHAT
# ==================================
if st.button("🗑️ 清除聊天紀錄"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")

for role, msg in st.session_state.messages:

    if role == "user":

        st.markdown(
            f"""
<div class="user-msg">
🧑‍💼 <b>您：</b><br><br>
{msg}
</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="bot-msg">
🤖 <b>AI客服：</b><br><br>
{msg}
</div>
""",
            unsafe_allow_html=True
        )

# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.caption(
    "2026 人工智慧跨域專題實作｜AI保險客服與知識管理 Agent"
)