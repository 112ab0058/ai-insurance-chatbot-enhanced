# 安心保 AI 保險助理

以 Streamlit、OpenAI、LangChain 與 FAISS 建立的繁體中文保險條款問答系統。系統會從 PDF 找出相關條款，回答時附上頁碼與原文節錄，方便核對內容。

## 功能

- 原生聊天介面、範例問題與對話紀錄
- PDF 條款檢索與頁碼引用
- 單份 PDF 工作階段上傳，不寫入伺服器磁碟
- 理賠流程與送件檢查表
- 文件解析、Embedding 與 FAISS 索引快取
- API、PDF 與查無資料的友善錯誤處理

## 本機執行

1. 建立並啟用 Python 3.11 虛擬環境。
2. 安裝依賴：`pip install -r requirements-dev.txt`
3. 複製 `.env.example` 為 `.env`，填入 `OPENAI_API_KEY`。
4. 執行：`streamlit run app.py`

## Streamlit Cloud 部署

將儲存庫連接至 Streamlit Community Cloud，入口檔案選擇 `app.py`，並在 App Settings → Secrets 加入：

```toml
OPENAI_API_KEY = "sk-..."
```

不要把 `.env` 或 `secrets.toml` 提交至 GitHub。

## 測試

```bash
pytest -q
```

測試使用模擬模型，不會呼叫 OpenAI API 或產生費用。

## 展示建議

依序示範「住院保障」、「美容手術」、「申請文件」，再上傳另一份 PDF 展示知識庫替換。回答屬條款閱讀輔助，實際保障與理賠結果仍以保險公司審核為準。
