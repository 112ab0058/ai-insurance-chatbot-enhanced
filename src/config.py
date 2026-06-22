APP_NAME = "安心保｜AI 保險助理"
DEFAULT_PDF = "insurance.pdf"
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
