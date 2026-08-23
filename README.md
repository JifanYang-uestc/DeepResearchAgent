# DeepResearchAgent V0.3

DeepResearchAgent 是一个 TODO 驱动的深度研究助手。V0.3 在稳定的
`v0.2-knowledge-rag` 基础上增量加入 HelloAgents Semantic Embedding、
Knowledge / Web / Hybrid 自适应路由和 Knowledge Relevance Gate。

## V0.3 解决的问题

V0.2 的每个 TODO 都会执行本地 Top-K 检索和 Web Search。Top-K 只能保证
“最靠前”，不能保证“足够相关”，因此机器人趋势等域外问题也可能把
`rag_2020.pdf`、`react.pdf` 或 `self_rag.pdf` 写入最终报告。

V0.3 使用：

```text
Route First
    +
Semantic Retrieve
    +
Gate Evidence
```

先决定该查 Knowledge、Web 还是两者，再过滤相关性不足的本地候选。

## 架构

```text
User
 ↓
Planner
 ↓
TODO
 ↓
Adaptive Retrieval Router
 ↓
┌───────────┬───────────┬────────────┐
│ Knowledge │ Web       │ Hybrid     │
└─────┬─────┴─────┬─────┴─────┬──────┘
      ↓           ↓           ↓
Semantic RAG    Tavily      Both
      ↓                       ↓
Knowledge Relevance Gate ←────┘
             ↓
      Unified Evidence
             ↓
        Summarizer
             ↓
        Report Writer
             ↓
          SSE / Vue
```

`KnowledgeService` 仍是稳定边界。默认 `helloagents` 后端复用项目原有的
PDF/TXT/Markdown Loader、页码元数据、Chunker 和 FAISS 持久化，只将 V0.2
的 HashingEmbedding 替换为 HelloAgents 官方 `LocalTransformerEmbedding`。
V0.2 `HashingEmbedding + FAISS` 保留为 `legacy_faiss` 回退后端。

## 环境准备

要求 Python 3.10+、Node.js 18+。首次安装和模型下载需要网络，之后模型与
索引均可从本地缓存加载。

```powershell
cd backend
uv sync --all-groups
Copy-Item .env.example .env
```

在 `.env` 中填写自己的 LLM 和搜索配置。`.env`、API Key、模型缓存、PDF
和向量索引都不会提交到 Git。

默认本地语义模型经过实际 CPU 加载与检索验证：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## V0.3 配置

```env
# Knowledge Backend
KNOWLEDGE_BACKEND=helloagents

# Semantic Embedding
EMBEDDING_PROVIDER=local_transformer
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Adaptive Retrieval
ENABLE_RETRIEVAL_ROUTER=True

# Relevance Gate
KNOWLEDGE_PROBE_TOP_K=3
KNOWLEDGE_RELEVANCE_THRESHOLD=0.55

# V0.3 不启用 MQE / HyDE
ENABLE_ADVANCED_RAG_SEARCH=False
```

切换回 V0.2 后端：

```env
KNOWLEDGE_BACKEND=legacy_faiss
```

0.55 来自当前 RAG/ReAct/Self-RAG 真实语料校准：域内 Top-1 为
0.6821–0.7398，域外 Top-1 为 0.3577–0.4198。不同模型或语料应重新运行
调试脚本校准，不要照搬阈值。

## 构建与调试知识库

真实论文下载说明和校验值见
[Knowledge Base 说明](backend/knowledge_base/README.md)。PDF 保存在本地但不进入 Git。

```powershell
cd backend
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\build_knowledge_index.py
.\.venv\Scripts\python.exe scripts\debug_semantic_retrieval.py
```

`build_knowledge_index.py` 是显式重建命令，不是健康检查：每次执行都会从当前
`knowledge_base/` 重新加载文件、分块、生成 Embedding、替换 FAISS 与 metadata，
并刷新当前 Backend 的缓存 Retriever。添加、替换或删除 Knowledge 文件后必须再次
执行该命令。输出包含 Backend、Documents、Pages、Chunks 和最终 Index 路径。

调试输出会显示 Backend、Gate 决策、Accept/Reject、Score、Document、Page、
Chunk ID 和 Content。Semantic 索引位于
`backend/vector_store/helloagents-semantic/`，重启后直接加载。

Windows 上如果 Transformers/PyTorch 在加载本地模型时出现原生线程访问冲突，可在
测试或构建前临时使用顺序权重加载：

```powershell
$env:HF_DEACTIVATE_ASYNC_LOAD='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

## 三类路由

- Knowledge：任务明确涉及 Catalog 中的文档或稳定理论，且不要求最新信息；只检索本地知识。
- Web：任务具有时效性且 Catalog 没有直接匹配资料；只调用 Web Search。
- Hybrid：本地文档可提供理论背景，同时问题需要当前互联网信息；执行两路并过滤本地候选。

Router 输入包含完整研究主题、任务标题、意图、查询、当前日期和
Knowledge Catalog。结构化 LLM Router 超时、异常或返回无效 JSON 时，会执行
确定性回退。Web-only 不执行 Knowledge Retrieval；Knowledge-only 在有效本地
证据存在时不调用 Web。

## 启动

后端：

```powershell
cd backend\src
$env:PYTHONUTF8='1'
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。每个任务会显示检索策略、原因、置信度、系统
notice，以及 Knowledge / Web 来源。SSE 保留 V0.2 事件并新增：

```text
retrieval_route
knowledge_rejected
```

任务数据同时记录 Router、Knowledge Retrieval、Web Search 和 Total Task
实际耗时，便于诊断而不编造 Benchmark。

## 测试

快速、确定性测试不访问 Tavily、真实 LLM、DashScope 或公网 Qdrant：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check .
```

显式运行本地模型与真实论文语料的三案例验收：

```powershell
$env:RUN_SEMANTIC_LIVE='1'
.\.venv\Scripts\python.exe -m pytest tests\live\test_v03_semantic_cases.py -q
```

前端：

```powershell
cd ..\frontend
npm run build
```

## Demo A — Knowledge

```text
ReAct 是如何结合 reasoning 与 acting 的？
```

预期：`Route: Knowledge`，主要引用 `react.pdf`，保留 Page 和 Chunk ID。

## Demo B — Web

```text
机器人领域当前的发展趋势是什么？
```

预期：`Route: Web`。当前 Knowledge Catalog 只有 RAG / ReAct / Self-RAG，
因此最终 Evidence 不应出现这些论文的 `[Knowledge]` 引用。

## Demo C — Hybrid

```text
从传统 RAG 到 Agentic RAG：对比 RAG、ReAct、Self-RAG，
并结合 2026 年互联网资料分析最新发展。
```

预期：`Route: Hybrid`，Evidence 同时包含 `[Knowledge]` 和 `[Web]`。

## Compatibility 与范围

HelloAgents 0.2.9 RAG API、Qdrant 版本、结构化返回和 metadata 调查见
[V0.3 Compatibility Spike](docs/v0.3-rag-compatibility.md)。
完整开发结果、回归、验收与已知限制见
[V0.3 Development Summary](V0.3_DEVELOPMENT_SUMMARY.md)。
审查问题的修复结果见
[V0.3 Review Fix Summary](V0.3_REVIEW_FIX_SUMMARY.md)。

V0.3 到此停止，不包含 BM25、RRF、Reranker、Reflection Agent、自动研究循环、
RAGAS/DeepEval、前端文件上传、认证或数据库。
