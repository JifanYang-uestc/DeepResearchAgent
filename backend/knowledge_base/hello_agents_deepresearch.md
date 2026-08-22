# HelloAgents DeepResearch 工作流摘要

本文档根据 HelloAgents 第十四章“自动化深度研究智能体”整理，用作本地知识库中的项目架构资料。它聚焦 Planner、Search、Summarizer、NoteTool、Writer 和端到端 DeepResearch Workflow。

## TODO 驱动的研究范式

DeepResearch 先把复杂问题拆成有限、互补且可执行的 TODO，再逐项收集证据和总结。该方式使长研究任务具有清晰状态、独立检索查询和可追踪的中间产物。

三阶段主流程是：

1. 规划阶段：Planner 根据用户主题生成结构化 TODO；
2. 执行阶段：每个 TODO 独立检索、收集来源并由 Summarizer 归纳；
3. 汇总阶段：Writer 将全部任务总结与来源整合为最终报告。

## Planner

Planner 负责理解研究主题、划分任务边界，并为每个任务输出标题、意图和检索 Query。好的任务应互补、避免重复，并共同覆盖用户问题。若模型无法给出有效计划，系统可创建一个基础背景任务作为降级路径。

## Search 与 Knowledge Retrieval

原始工作流使用 SearchTool 调度 Tavily、DuckDuckGo 等 Web Search。Knowledge RAG 扩展后，每个 TODO 同时走两条证据路径：

- Knowledge Retrieval 从 PDF、TXT 和 Markdown 构建的 FAISS 索引召回本地 Chunk；
- Web Search 获取实时互联网资料。

两路相互独立。Knowledge RAG 不可用时继续 Web Search；Web Search 不可用但本地 Evidence 有效时仍继续 Summarizer。

## Summarizer

Summarizer 面向单个 TODO 工作。它接收任务目标、检索 Query、Knowledge Evidence、Web Evidence 和笔记协作要求，提炼关键发现并保留可追溯引用。任务间彼此独立，因此流式模式可以并行执行多个 TODO。

## NoteTool

NoteTool 负责持久化协作状态。Planner 可创建任务笔记；Summarizer 读取并更新任务笔记；Writer 在生成报告前读取各任务笔记，也可以创建结论笔记。工具调用会转化为 SSE 事件，供前端展示执行过程。

## Writer

Writer 接收所有任务的标题、意图、Query、执行状态、总结和来源概览，生成结构化 Markdown 报告。报告应包含背景概览、核心洞见、证据与数据、风险与挑战、参考来源，并区分 `[Knowledge]` 与 `[Web]` 引用。

## SSE 与前端

后端通过 Server-Sent Events 依次发送状态、TODO 列表、任务状态、来源、总结增量、工具调用、最终报告和 done 事件。前端实时更新任务列表和来源卡片；本地来源展示文档、页码、Chunk ID 与 Score，Web 来源展示可访问 URL。

## 完整工作流

```text
User Query
    ↓
Planner TODO
    ↓
Knowledge Retrieval + Web Search
    ↓
Unified Evidence and Citations
    ↓
Per-task Summarizer + NoteTool
    ↓
Report Writer
    ↓
SSE Streaming
    ↓
Frontend Final Report
```

来源：HelloAgents 中文教程第十四章《自动化深度研究智能体》，本地参考文件 `hello-agents-source/docs/chapter14/第十四章 自动化深度研究智能体.md`。
