APP_NAME = "安心保｜AI 保險助理"
DEFAULT_PDF = "insurance.pdf"
KNOWLEDGE_DOCS = [
    {
        "label": "課堂示範保單",
        "description": "原始 insurance.pdf，適合快速展示保單問答。",
        "files": ["insurance.pdf"],
    },
    {
        "label": "完整展示知識庫",
        "description": "保單 + 人壽契約範本 + 人身保險 Q&A，適合課堂展示與客服查核。",
        "files": [
            "insurance.pdf",
            "knowledge_docs/uploaddowndoc.pdf",
            "knowledge_docs/websitedowndoc.pdf",
        ],
    },
    {
        "label": "人壽契約範本",
        "description": "傳統型個人人壽保險定型化契約條款範本。",
        "files": ["knowledge_docs/uploaddowndoc.pdf"],
    },
    {
        "label": "保險商品 Q&A",
        "description": "人身保險商品 Q&A 問答集，含住院、實支實付、保單權益等題型。",
        "files": ["knowledge_docs/websitedowndoc.pdf"],
    },
]
MODEL_NAME = "openai/gpt-4o-mini"
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 140
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

SAMPLE_QUESTIONS = [
    {"label": "住院保障", "question": "骨折住院可以申請理賠嗎？"},
    {"label": "除外責任", "question": "哪些情況不在理賠範圍內？"},
    {"label": "申請文件", "question": "理賠需要準備哪些文件？"},
    {"label": "美容手術", "question": "美容手術可以理賠嗎？"},
]
