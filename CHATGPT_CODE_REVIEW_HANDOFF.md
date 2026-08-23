# DeepResearchAgent V0.3 — ChatGPT 代码审查交接文档

## 1. 使用目的

本文件用于把 DeepResearchAgent V0.3 交给 ChatGPT 做只读代码检查、架构检查、
测试覆盖检查和安全检查。

本轮只要求 ChatGPT：

1. 阅读仓库和文档。
2. 核对实现是否满足 V0.3 目标。
3. 找出真实缺陷、回归风险、遗漏测试和安全问题。
4. 给出带文件路径、行号和优先级的审查报告。

除非用户后续明确要求，否则不要让 ChatGPT 自动修改代码、提交 Commit、创建 PR、
运行需要真实 API Key 的命令，或向第三方服务发送项目数据和测试查询。

## 2. 仓库信息

- GitHub：<https://github.com/JifanYang-uestc/DeepResearchAgent>
- 审查分支：`fix/v03-review-findings`
- V0.3 Pull Request：`#1`
- V0.3 合并提交：`bc5a51a`
- V0.2 基线 Tag：`v0.2-knowledge-rag`
- V0.3 原开发分支：`feature/semantic-rag-router`

本轮重点审查 Review Fix；建议比较：

```text
main...fix/v03-review-findings
```

先阅读 `V0.3_REVIEW_FIX_SUMMARY.md`，再根据本文件核对 V0.3 原始架构与回归。

## 3. 项目目标

DeepResearchAgent 是一个 TODO 驱动的深度研究助手。V0.3 在 V0.2 的基础上完成：

1. HelloAgents 本地 Semantic Embedding 集成。
2. Knowledge / Web / Hybrid 自适应检索路由。
3. Knowledge Relevance Gate。
4. 路由决策、拒绝原因和性能指标的可观测性。
5. 后端回归测试、真实本地语义测试和 SSE E2E 测试。
6. Vue 前端轻量路由展示。

核心问题是避免域外问题被无关本地知识污染。例如本地知识库只有
RAG、ReAct、Self-RAG 论文时，查询“机器人领域的发展趋势”不应引用这些论文。

## 4. V0.3 架构

```text
User
  -> Planner
  -> TODO
  -> Adaptive Retrieval Router
       |-> Knowledge -> Semantic RAG -> Relevance Gate
       |-> Web       -> Web Search
       `-> Hybrid    -> Semantic RAG + Web Search -> Relevance Gate
  -> Unified Evidence
  -> Summarizer
  -> Report Writer
  -> SSE
  -> Vue Frontend
```

稳定边界：

```text
DeepResearchAgent
  -> KnowledgeService
       |-> HelloAgents Semantic Backend（默认）
       `-> Legacy HashingEmbedding + FAISS（回退）
```

## 5. 关键配置

配置样例位于 `backend/.env.example`：

```env
KNOWLEDGE_BACKEND=helloagents
EMBEDDING_PROVIDER=local_transformer
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENABLE_RETRIEVAL_ROUTER=True
KNOWLEDGE_PROBE_TOP_K=3
KNOWLEDGE_RELEVANCE_THRESHOLD=0.55
ENABLE_ADVANCED_RAG_SEARCH=False
```

真实 `.env`、API Key、PDF、模型缓存和向量索引不应进入 Git。

## 6. 优先阅读顺序

### 项目与兼容性

1. `README.md`
2. `V0.3_DEVELOPMENT_SUMMARY.md`
3. `V0.3_REVIEW_FIX_SUMMARY.md`
4. `docs/v0.3-rag-compatibility.md`
5. `backend/.env.example`

### 核心实现

1. `backend/src/services/retrieval_router.py`
2. `backend/src/services/relevance_gate.py`
3. `backend/src/services/research.py`
4. `backend/src/services/knowledge.py`
5. `backend/src/rag/helloagents_backend.py`
6. `backend/src/rag/legacy_faiss_backend.py`
7. `backend/src/rag/base.py`
8. `backend/src/rag/catalog.py`
9. `backend/src/agent.py`
10. `backend/src/models.py`
11. `backend/src/main.py`
12. `frontend/src/App.vue`

### 核心测试

1. `backend/tests/test_retrieval_router.py`
2. `backend/tests/test_router_out_of_domain.py`
3. `backend/tests/test_relevance_gate.py`
4. `backend/tests/test_adaptive_research_evidence.py`
5. `backend/tests/test_helloagents_backend.py`
6. `backend/tests/test_routing_observability.py`
7. `backend/tests/test_v03_end_to_end.py`
8. `backend/tests/live/test_v03_semantic_cases.py`

## 7. 必须检查的行为

### Knowledge-only

输入：

```text
ReAct 如何结合 reasoning 和 acting？
```

期望：

- Route 为 `knowledge`。
- 主要命中 `react.pdf`。
- 保留 document、page、chunk_id 和 score。
- 有有效 Knowledge Evidence 时不调用 Web。

### Web-only

输入：

```text
机器人领域目前的发展趋势是什么？
```

本地 Catalog 只有 RAG/ReAct/Self-RAG 时，期望：

- Route 为 `web`。
- 不执行 Knowledge Retrieval。
- 最终 Knowledge Evidence 为空。
- 最终报告不得包含对上述论文的 `[Knowledge]` 引用。

### Hybrid

输入：

```text
基于 RAG、ReAct、Self-RAG，并结合 2026 年最新资料，分析 Agentic RAG 发展趋势。
```

期望：

- Route 为 `hybrid`。
- Knowledge Evidence 至少 1 条。
- Web Evidence 至少 1 条。
- Knowledge 候选必须先通过 Relevance Gate。

## 8. 必须检查的异常路径

请逐项检查实现和测试是否一致：

- Router 返回 invalid JSON。
- Router timeout 或抛出异常。
- Semantic Embedding 初始化失败。
- Vector Backend 不可用。
- Knowledge Backend 失败后回退 Web。
- Web 失败但 Knowledge 有有效 Evidence。
- Knowledge 和 Web 都不可用。
- Hybrid 中弱相关 Knowledge 被拒绝，但 Web Evidence 保留。
- 空 Evidence 时不得让模型编造总结。
- 所有降级是否有日志或用户可见 notice。

## 9. 安全审查要求

重点检查：

- 是否提交 `.env`、Token、API Key 或硬编码凭据。
- 日志、SSE、异常信息是否可能泄露密钥。
- 用户查询、文档内容何时会发送给 LLM、Tavily 或其它第三方。
- 文件路径和上传相关逻辑是否存在路径穿越风险。
- 外部响应是否在进入 Summarizer 前经过基本结构校验。
- 依赖新增是否最小化，是否存在明显版本不兼容。

不要在审查报告中复述任何真实密钥；如果发现疑似凭据，只报告文件位置、类型和处理建议。

## 10. 已完成的验证

开发交付时的结果：

| 检查 | 结果 |
|---|---|
| 后端快速测试 | `69 passed, 1 skipped` |
| 真实本地模型和论文 live test | `1 passed` |
| Ruff | `All checks passed!` |
| Vue + TypeScript production build | 通过，14 modules transformed |
| 机器人域外污染回归 | 通过，Route=Web，Knowledge sources=[] |
| 本地 SSE E2E | 通过，最后事件为 `done` |
| Web 外部响应信任边界 | 非 HTTP(S) URL 与非法结构在 Summarizer 前被拒绝 |
| Catalog 通用词跨域污染 | 制药行业查询不会命中机器人行业报告 |
| 常见 API Key 格式审计 | 当前文件 0、Git 历史 0 |

已知 warning 来自 HelloAgents 0.2.9 内部的 Pydantic V2 旧式 `Config`，以及 FastAPI
旧式 `on_event` 生命周期接口。

## 11. 本地复现命令

以下命令不应调用真实 Tavily、LLM、DashScope 或公网 Qdrant：

```powershell
cd backend
uv sync --all-groups
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check .
```

本地模型与论文验收：

```powershell
$env:RUN_SEMANTIC_LIVE='1'
.\.venv\Scripts\python.exe -m pytest tests\live\test_v03_semantic_cases.py -q
```

前端构建：

```powershell
cd ..\frontend
npm install
npm run build
```

不要执行 `backend/scripts/run_v03_acceptance.py`，除非用户明确授权将查询发送给
真实 LLM 和 Web Search 服务。

## 12. 已知限制与非目标

V0.3 明确没有实现：

- BM25、RRF、Reranker。
- Reflection Agent 和自动研究循环。
- RAGAS、DeepEval、Langfuse。
- Redis、Celery、Database、Authentication。
- 前端文件上传。
- Qdrant 默认后端。

这些不应直接作为 V0.3 缺陷，除非当前代码声称已经实现或缺失导致既有功能错误。

## 13. ChatGPT 审查输出格式

请要求 ChatGPT 按以下顺序输出：

1. **结论**：可接受 / 需修改 / 存在阻断问题。
2. **Findings**：按 P0、P1、P2、P3 排序。
3. 每条 Finding 必须包含文件路径、行号、触发条件、实际影响和最小修复建议。
4. **Acceptance Matrix**：Knowledge/Web/Hybrid、Gate、故障降级、SSE、前端分别 PASS/FAIL/NOT VERIFIED。
5. **Test Gaps**：只列能捕获真实风险的缺失测试。
6. **Security & Secrets**：单独结论。
7. **Known Limitations**：区分范围外能力与真实缺陷。
8. **最终建议**：是否适合作为 V0.3 基线。

不要只做代码摘要；没有发现问题时，应明确写“未发现可复现缺陷”，并列出没有验证的部分。

## 14. 可直接复制给 ChatGPT 的请求

```text
请连接并只读审查 GitHub 仓库：
https://github.com/JifanYang-uestc/DeepResearchAgent

审查分支：fix/v03-review-findings
比较范围：main...fix/v03-review-findings
V0.3 原合并提交：bc5a51a

首先完整阅读仓库根目录的 CHATGPT_CODE_REVIEW_HANDOFF.md，随后阅读 README.md、
V0.3_REVIEW_FIX_SUMMARY.md、V0.3_DEVELOPMENT_SUMMARY.md 和
docs/v0.3-rag-compatibility.md，再检查其中列出的核心实现与测试文件。

这是一次只读审查。不要修改代码、提交 Commit、创建 PR，也不要运行需要真实 API Key
或向第三方发送数据的命令。

重点查找：
1. 可复现的功能错误或 V0.2 回归；
2. Knowledge/Web/Hybrid 路由边界错误；
3. 弱相关 Knowledge 绕过 Relevance Gate 的路径；
4. 故障降级、空 Evidence 和 SSE 契约问题；
5. API Key 泄露、外部数据发送和依赖风险；
6. 会掩盖上述问题的缺失测试。
7. 显式 rebuild 是否始终反映当前 corpus，且不会复用旧缓存 Retriever；
8. Global freshness 是否仍可能覆盖理论 TODO；
9. Catalog title、debug index 隔离和用户可见错误净化是否存在绕过路径；
10. Web 外部响应是否可能通过非法结构、危险 URL 或旁路 direct answer 进入 Summarizer。

请按交接文档第 13 节的格式报告。先列 Findings，并为每一项提供文件路径、行号、
触发条件、影响和最小修复建议。不要把明确列为 V0.3 范围外的功能当作缺陷。
```

## 15. ChatGPT 接入提示

在 ChatGPT 中连接 GitHub App，并只授权
`JifanYang-uestc/DeepResearchAgent` 所需的读取权限。完成授权后新建对话，选择 GitHub
作为来源，然后粘贴第 14 节的请求。

如果仓库暂时搜索不到，可先确认 GitHub App 的仓库授权范围和 `main` 是否已经同步，
再重新选择仓库。不要把 `.env` 或 API Key 作为附件上传。
