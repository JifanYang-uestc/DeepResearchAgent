# DeepResearchAgent — User-Controlled Research Scope

DeepResearchAgent 是一个 TODO 驱动的深度研究助手。当前版本在 V0.3 Semantic RAG
基础上把“允许使用哪些信息源”交给用户，并为每次上传创建隔离的 Session RAG。

## 思路转变

V0.3 使用自动 TODO 级 Router 决定 Knowledge / Web / Hybrid。实际审查表明，freshness、
年份、文件名和全局主题会让 Router 同时承担“用户权限”和“证据相关性”两种职责。

当前版本明确拆分为三层：

```text
ResearchMode   → 用户允许哪些信息源
Relevance Gate → 上传文档中的哪些 Chunk 足够相关
Agent          → Planner → TODO → Evidence → Summary → Report
```

LLM Router 不再能够扩大用户允许的信息源。它保留为未来可选的 Hybrid 成本/延迟优化器。

## 架构

```text
                    User
                      ↓
               Research Topic
                      +
             Uploaded Documents
                      +
                Research Mode
                      ↓
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
      Web          Document        Hybrid
       ↓              ↓              ↓
   Web Search     Session RAG    Session RAG + Web
                      ↓              ↓
                Relevance Gate ←─────┘
                      ↓
                   Planner
                      ↓
                    TODO
                      ↓
               Unified Evidence
                      ↓
                 Summarizer
                      ↓
                Report Writer
                      ↓
                  SSE / Vue
```

实际执行中 Planner 先把主题拆成 TODO，然后每个 TODO 在 ResearchMode 允许的范围内收集
证据。上传后的解析、Embedding 和索引构建在研究开始前完成。

## 用户模式

| 用户状态 | Mode | Document | Web |
|---|---|---:|---:|
| 未上传文件 | Web | No | Yes |
| 上传文件 + 文档与互联网 | Hybrid | Yes | Yes |
| 上传文件 + 仅文档 | Document | Yes | No |

### Web Only

不创建 KnowledgeService，不加载 Semantic Embedding，不打开 FAISS，也不读取 Catalog。

### Document Only

只检索当前 `document_set_id` 的独立索引，候选必须通过 Relevance Gate。即使查询包含
“2026”“最新”等词也永不调用 Web。证据不足时返回：

```text
当前上传文档未提供足够证据支持该结论。
```

### Hybrid

第一版对每个 TODO 同时允许 Document 和 Web。Document 候选仍必须通过 Gate；弱相关
Chunk 被拒绝后，Web 可以继续。来源在用户界面中分别显示为 Document 和 Web。

## 上传与隔离

支持：`.pdf`、`.txt`、`.md`、`.markdown`。

默认限制：最多 10 个文件、单文件 20 MiB、文档集总计 50 MiB。服务端会清除目录语义、
拒绝不支持的扩展，并逐文件解析；一个坏 PDF 不会隐藏其它有效文件。如果所有文件都为空
或不可解析，则不会构建空索引。

```text
backend/runtime/document_sets/
└── <uuid>/
    ├── document_set.json
    ├── files/
    │   └── uploaded.md
    └── index/
        └── helloagents-semantic/
            ├── knowledge.faiss
            └── metadata.json
```

每个 Document Set 使用 UUID 和独立 FAISS。请求中不存在全局
`CURRENT_DOCUMENT_SET`；Retriever 缓存以 `document_set_id` 为键、线程安全且有容量上限。
服务重启后直接加载持久化索引，不自动重新生成 Embedding。

Runtime、上传文件和索引都被 `.gitignore` 排除。

## API

创建文档集：

```http
POST /knowledge/document-sets
```

上传并自动构建索引：

```http
POST /knowledge/document-sets/{document_set_id}/files
Content-Type: multipart/form-data
```

查询状态：

```http
GET /knowledge/document-sets/{document_set_id}
```

成功响应示例：

```json
{
  "document_set_id": "a UUID",
  "status": "ready",
  "documents": 3,
  "pages": 82,
  "chunks": 417,
  "files": [],
  "notices": []
}
```

研究请求：

```json
{
  "topic": "分析这些论文并结合最新资料研究 Agentic RAG",
  "research_mode": "hybrid",
  "document_set_id": "a ready UUID",
  "search_api": "tavily"
}
```

`document` 和 `hybrid` 必须绑定 ready 文档集，否则返回 4xx。`web` 会忽略
`document_set_id`，并保持 Knowledge 完全懒加载。同步 `/research` 与 SSE
`/research/stream` 使用相同的任务、证据和报告语义。SSE 新增：

```json
{"type":"research_mode","mode":"hybrid","document_set_id":"..."}
```

## 数据边界

> Local Semantic Embedding 不等于 Fully Local Research。

- 文档解析、Embedding 和 FAISS 检索在本地执行。
- Document Only 保证不调用 Web Search。
- 如果 Summarizer / Report Writer 配置为外部 LLM，被 Gate 接受的 Document Chunk 会发送
  给该 LLM Provider 进行总结。
- Document Only 的“No Web”不表示“No external LLM processing”。若要求完全离线，应同时
  配置 Ollama/LMStudio 等本地 LLM。
- Hybrid 会把 Web 查询发送给配置的搜索服务，并把被接受的 Document/Web Evidence 提供
  给 Summarizer。

不要上传 `.env`、API Key 或其它秘密作为参考资料。

## 环境准备

需要 Python 3.10+、Node.js 18+：

```powershell
cd backend
uv sync --all-groups
Copy-Item .env.example .env
```

核心配置：

```env
EMBEDDING_PROVIDER=local_transformer
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
KNOWLEDGE_PROBE_TOP_K=3
KNOWLEDGE_RELEVANCE_THRESHOLD=0.55
DOCUMENT_SETS_ROOT=./runtime/document_sets
MAX_UPLOAD_FILES=10
MAX_UPLOAD_FILE_SIZE=20971520
MAX_UPLOAD_TOTAL_SIZE=52428800
DOCUMENT_INDEX_CACHE_SIZE=8
CORS_ORIGINS=http://localhost:5173
```

`build_knowledge_index.py` 保留给手工兼容模式。其优先级为：显式 CLI > `.env` > 默认值。
默认 Semantic 后端不再自动回退到可能陈旧的 Legacy 索引；Legacy 只能显式选择：

```env
KNOWLEDGE_BACKEND=legacy_faiss
```

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

打开 `http://localhost:5173`，填写研究主题；不上传文件时自动联网，上传文件后默认
“文档 + 互联网”，也可以选择“仅使用上传文档”。

## 测试

默认测试不调用真实 LLM、Tavily，也不下载模型：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check .

cd ..\frontend
npm run build
```

本地 Semantic Live（需要模型已在本机可用）：

```powershell
$env:RUN_SEMANTIC_LIVE='1'
$env:HF_DEACTIVATE_ASYNC_LOAD='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest tests\live -q
```

## 强制 Demo

- Web：无文件，研究“2026 年机器人领域的发展趋势”——只有 Web。
- Document：上传 `react.pdf`、`self_rag.pdf`，选择仅文档——没有 Web。
- Hybrid：上传 RAG/ReAct/Self-RAG 论文并研究 2026 趋势——Document + Web。
- Document Insufficient：只上传 `react.pdf`，查询全球机器人融资——不联网、不引用
  `react.pdf`、明确资料不足。
- Isolation：Set A 为 `react.pdf`，Set B 为 `robotics_report.pdf`——Research B 不出现 A。

## 安全与已知限制

服务端使用 `CORS_ORIGINS`，用户可见异常保持固定安全文本，日志会脱敏 Bearer、API Key、
authorization、secret-token 和 URL 内嵌凭据。启动日志只记录 `api_key=configured|unset`。

当前没有 BM25、RRF、Reranker、Reflection、RAGAS、Langfuse、Qdrant、Redis、Celery、
多用户认证或数据库；不支持 OCR、DOCX、PPTX、XLSX、ZIP、URL 导入。Document Set 暂无
自动过期清理和删除 UI。Embedding 构建当前在上传请求中同步完成，大文件会等待较久。

开发记录：

- [V0.3 Round-Two Fix Summary](V0.3_ROUND2_FIX_SUMMARY.md)
- [User-Controlled Research Scope Development Summary](USER_CONTROLLED_RESEARCH_SCOPE_DEVELOPMENT_SUMMARY.md)
- [V0.3 Compatibility](docs/v0.3-rag-compatibility.md)
