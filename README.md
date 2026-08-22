# HelloAgents DeepResearch Knowledge RAG V1

这是一个 TODO 驱动的深度研究助手。V1 在原有 Planner、Tavily Web Search、Summarizer、Report Writer、SSE 和 Vue 前端基础上，加入 PDF/TXT/Markdown 本地知识库、确定性 Embedding、FAISS 持久化检索，以及 Knowledge + Web 双路 Evidence。

## 架构

```text
User Query
    ↓
Planner → 3~5 TODO
    ↓
┌──────────────────────┬──────────────────────┐
│ Local Knowledge RAG  │ Real-time Web Search │
│ Loader → Chunk       │ Tavily / configured  │
│ Embedding → FAISS    │ search backend       │
└──────────┬───────────┴──────────┬───────────┘
           └──── Unified Evidence ┘
                      ↓
             Per-task Summarizer
                      ↓
                Report Writer
                      ↓
             SSE → Vue Frontend
```

任一路失败不会拖垮另一条路径：Knowledge 不可用时退化到 Web；Web 不可用但本地 Evidence 有效时继续总结。

## 准备环境

要求 Python 3.10+、Node.js 18+。后端使用已有 `.venv` 时，在 `backend` 目录运行：

```powershell
uv sync --all-groups --python ..\.venv\Scripts\python.exe
Copy-Item .env.example .env
```

在 `.env` 填写 LLM 与 Web Search 配置。密钥文件不会进入 Git。Windows 控制台若无法显示搜索组件的 emoji，可先设置：

```powershell
$env:PYTHONUTF8='1'
```

真实论文的下载与校验信息见 [Knowledge Base 说明](backend/knowledge_base/README.md)。

## 构建本地知识索引

```powershell
cd backend
..\.venv\Scripts\python.exe scripts\build_knowledge_index.py
..\.venv\Scripts\python.exe scripts\debug_retrieval.py --top-k 1
```

默认 Top-K 为 5。索引写入 `backend/vector_store/`，不会提交到 Git。

## 启动

后端：

```powershell
cd backend\src
$env:PYTHONUTF8='1'
..\..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。前端来源区会分别显示 `Knowledge` 和 `Web` 标签。

## 测试

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -q

cd ..\frontend
npm run build
```

测试覆盖：文档加载与切块、Embedding 确定性、FAISS 持久化、四个工程 Query、三篇论文单文档/跨文档检索、Knowledge-only、Web-only、双路 Evidence 与不可用降级。

## 推荐 Demo

```text
从传统 RAG 到 Agentic RAG：对比 RAG、ReAct 与 Self-RAG 的核心思想，并结合当前互联网资料分析智能体检索增强技术的发展趋势。
```

理想结果中，经典理论主要引用 `[Knowledge] rag_2020.pdf`、`react.pdf`、`self_rag.pdf`，当前趋势引用 `[Web] https://...`。

## V1 范围

本版有意停止在 Knowledge RAG + Web Search + Unified Evidence。BM25、RRF、Reranker、Reflection Agent 和自动 Evaluation 留待 V2 决策。
