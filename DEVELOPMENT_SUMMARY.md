# Knowledge RAG V1 开发总结

## 已完成

- 建立 `v0.1-baseline`，并在 `feature/knowledge-rag` 分阶段开发。
- 支持 PDF、TXT、Markdown 加载、页码元数据、确定性重叠切块。
- 实现本地多语言 Hashing Embedding、FAISS 索引持久化和 Top-K Retriever。
- Knowledge Retrieval 与 Web Search 双路执行、统一 Evidence、引用与独立降级。
- SSE 与前端显示 Knowledge/Web 来源；本地来源含页码、Chunk ID、Score。
- 建立工程事实语料和 RAG、ReAct、Self-RAG 三篇真实论文语料。

## 验证记录

- Baseline：后端健康检查 200；Tavily 返回 5 条；Planner 生成 5 个 TODO；5 个来源事件、完整总结流、5,878 字符最终报告及 `done` 事件；前端生产构建通过。
- 工程 Retrieval：四个指定 Query 均 Top-1 命中预期事实。
- 自动测试：20 passed；唯一 warning 来自 `hello-agents` 对 Pydantic V2 旧式 Config 的弃用提示。
- 真实语料：RAG、ReAct、Self-RAG 三个定向 Query 均 Top-1 命中对应 PDF；跨文档检索覆盖三篇论文。
- 前端：TypeScript 检查与 Vite production build 通过。

## V1 明确未实现

BM25、RRF、Reranker、Reflection Agent、自动 Evaluation。
