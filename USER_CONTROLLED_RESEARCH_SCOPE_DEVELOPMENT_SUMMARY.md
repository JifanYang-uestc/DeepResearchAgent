# User-Controlled Research Scope 开发总结

## 1. V0.3 稳定性问题与修复

本轮功能开发前先完成了 V0.3 第二轮稳定性修复：

- 同步研究接口不再错误消费异步/SSE 生成器，改为共用证据收集逻辑并直接执行同步任务。
- 任务异常会写回内部 `failed` 状态，避免接口状态与真实执行结果不一致。
- 结构化 Web 路由决定不再被本地知识目录匹配覆盖，并补齐年份、时效性关键词等路由场景。
- CLI 参数只在用户显式传入时覆盖环境变量。
- 语义知识库构建失败时不再自动回退到旧关键词后端。
- 损坏 PDF 按文件隔离，不影响同批其他有效文档。
- 日志中的凭据只显示 `configured`/`unset`，并加入通用敏感字段脱敏。
- CORS 来源改为由 `CORS_ORIGINS` 配置。

对应提交：`b90a6c4 fix: stabilize synchronous research and retrieval boundaries`。

## 2. 改造前架构

改造前只有一个由配置指定的全局知识目录。研究请求不能表达“仅联网”“仅文档”或“混合研究”，上传文档也没有会话级隔离能力。

```text
用户问题
  -> ResearchAgent
     -> 全局 KnowledgeService
     -> Web Search
  -> Reporter
```

主要限制：

- 所有请求共享同一个知识目录与索引生命周期。
- 用户不能明确控制研究范围。
- 无上传、文档集状态查询或会话级索引接口。
- 文档证据与 Web 证据的展示边界不够清晰。

## 3. 改造后架构

```text
浏览器
  -> 创建 Document Set
  -> 上传 PDF/TXT/MD 并构建隔离索引
  -> 选择 WEB / DOCUMENT / HYBRID
  -> Research API / SSE
       -> ResearchAgent（模式为最高优先级）
          -> Session KnowledgeService（DOCUMENT/HYBRID）
          -> Web Search（WEB/HYBRID）
       -> 按来源分区的 Reporter
```

关键设计：

- `ResearchMode` 是显式、可序列化的领域类型，支持 `WEB`、`DOCUMENT`、`HYBRID`。
- 每个 Document Set 使用 UUID 和独立目录，索引不会跨集合复用。
- Web 模式不创建知识库后端；Document 模式永不回退联网；Hybrid 模式同时允许两类证据。
- 同步与 SSE 入口共用相同的模式绑定、证据收集和安全错误语义。

## 4. API 变更

### Document Set

- `POST /knowledge/document-sets`：创建空文档集，返回 `document_set_id`。
- `POST /knowledge/document-sets/{document_set_id}/files`：以 multipart 上传文件并自动构建索引。
- `GET /knowledge/document-sets/{document_set_id}`：读取状态、文件数、文档数、页数、分块数和安全错误信息。

文档集状态：`uploaded`、`indexing`、`ready`、`failed`。

### Research

研究请求增加：

```json
{
  "topic": "研究主题",
  "research_mode": "HYBRID",
  "document_set_id": "uuid",
  "search_api": "tavily"
}
```

规则：

- `WEB` 不要求 `document_set_id`。
- `DOCUMENT` 和 `HYBRID` 必须绑定处于 `ready` 状态的文档集，否则返回明确的 4xx 响应。
- 响应和 SSE 都回显最终绑定的 `research_mode`；SSE 跳过任务时仍返回摘要与来源字段。

## 5. 前端变更

- 支持选择并移除 PDF、TXT、MD、MARKDOWN 文件。
- 无文件时固定使用 Web；选中文件后默认使用 Hybrid，可切换为 Document。
- 开始研究前自动创建文档集、上传文件并等待后端完成本地索引。
- 展示上传、索引构建、就绪及失败状态。
- 展示文件数、文档数、页数和分块数。
- 研究侧栏展示当前模式，Document 模式禁用无效的搜索引擎选择。
- 证据标签明确区分 `Document` 与 `Web`。

## 6. 存储结构

默认运行时目录已加入 `.gitignore`：

```text
backend/runtime/document_sets/
  <document-set-uuid>/
    document_set.json
    files/
      uploaded-file.md
    index/
      semantic-index-artifacts
```

元数据使用临时文件替换方式落盘。服务重启后可从 `document_set.json` 和既有索引恢复，不强制重建索引。

相关配置：

- `DOCUMENT_SETS_ROOT`
- `MAX_UPLOAD_FILES`
- `MAX_UPLOAD_FILE_SIZE`
- `MAX_UPLOAD_TOTAL_SIZE`
- `DOCUMENT_INDEX_CACHE_SIZE`

## 7. 文档集隔离

- 文档集标识必须是规范化 UUID。
- 所有文件与索引路径都从服务端存储根目录推导，并验证解析后的路径仍位于对应文档集目录内。
- 文件名会移除路径语义、危险字符和 Windows 保留名称，并对同名文件生成唯一名称。
- 每个文档集拥有独立 `files/` 和 `index/`。
- 内存中的 KnowledgeService 使用 `document_set_id` 作为缓存键，并使用线程锁保护；缓存为有界 LRU 风格结构。
- 并发构建 A/B 文档集的测试验证了查询结果不会串库。

## 8. Session RAG 与证据门控

模式权限矩阵：

| 模式 | 文档检索 | Web 搜索 | 文档不足时联网 |
|---|---:|---:|---:|
| WEB | 否 | 是 | 不适用 |
| DOCUMENT | 是 | 否 | 否 |
| HYBRID | 是 | 是 | 已允许 |

用户显式选择的模式优先于旧路由器决定，内部路由不能扩大权限。Document 模式证据不足时返回确定性的中文提示，不生成无来源结论；Reporter 在零来源时输出“证据不足”报告。Hybrid 报告将 Document Evidence 与 Web Evidence 分区展示。

## 9. 测试结果

最终本地验证：

| 检查 | 结果 |
|---|---|
| 后端完整测试 | PASS：99 passed，2 skipped |
| Ruff 静态检查 | PASS |
| Vue/TypeScript 生产构建 | PASS |
| 本地真实语义索引生命周期测试 | PASS：2 passed |
| 真实 LLM API | NOT RUN |
| 真实 Tavily API | NOT RUN |

覆盖重点：

- 三种模式的权限与不回退约束。
- Document 模式证据不足语义。
- 文档集上传、扩展名/空文件校验、损坏 PDF 隔离。
- 路径穿越与 Windows 保留文件名处理。
- A/B 并发隔离、重启恢复且不重复构建。
- API 创建、上传、状态查询与未就绪拒绝。
- 同步/SSE 模式一致性。
- CORS 配置与既有 V0.3 回归场景。

两个默认跳过项均为显式 opt-in 的本地真实语义模型测试，避免普通测试下载模型或访问外部服务。

## 10. 安全措施

- API Key 不写入代码、响应、日志或示例配置。
- `.env` 和运行时上传/索引目录不进入 Git。
- 上传仅允许 PDF、TXT、MD、MARKDOWN，限制单文件数、单文件大小和整批总大小。
- 文件名净化防止目录穿越和 Windows 特殊设备名。
- UUID、规范路径和根目录边界三重验证。
- 损坏文件只暴露安全的业务错误，不向前端返回内部堆栈或本地绝对路径。
- 语义后端失败不会静默降级到不等价的旧检索实现。
- CORS 来源由环境变量显式配置。

## 11. 演示与验收结果

已通过自动化场景验证：

1. 创建 Document Set，上传 Markdown 后状态变为 `ready`，统计字段可用。
2. Document 模式只返回该文档集中的证据，不调用 Web。
3. Document 模式无足够证据时返回确定性提示且不联网。
4. Web 模式不初始化 Session RAG。
5. Hybrid 模式允许同时收集 Document 与 Web 证据。
6. 两个并发文档集可分别查询自身内容，结果不交叉。
7. 服务重启后加载持久化索引，不触发重复 rebuild。
8. 前端生产构建成功，上传和研究范围参数通过 TypeScript 检查。

未调用真实 LLM 和 Tavily，因此没有产生外部费用，也没有把本机凭据暴露给测试。

## 12. Git 历史

本轮在 `feature/user-controlled-research-scope` 分支完成，关键提交为：

- `b90a6c4` — `fix: stabilize synchronous research and retrieval boundaries`
- `9081de5` — `feat: add isolated document sets and research modes`
- `ec2796d` — `feat: add document upload and research scope controls`
- 文档提交 — README 与本开发总结

未执行 push、merge 或 tag，便于先由审查方检查提交差异。

## 13. 已知限制与后续建议

- 当前索引构建在上传请求内同步完成，大文档生产化时宜迁移到后台任务队列。
- 暂无文档集删除、过期清理、配额和用户身份认证；多用户部署前必须补齐访问控制。
- 暂不支持 DOCX、网页抓取文档、图片 OCR 和扫描 PDF OCR。
- 尚未实现 BM25/向量混合检索、RRF、重排器和反思式检索。
- 文档集元数据使用本地 JSON，而非事务数据库；适用于单机开发与验收环境。
- KnowledgeService 缓存是单进程内缓存，多进程部署需要共享状态或保持纯磁盘加载语义。
- 前端目前以 TypeScript 编译和生产构建为主，尚未引入独立的组件/E2E 测试框架。
- 真实 LLM/Tavily 端到端测试需要审查者在自己的安全环境中显式配置凭据后执行。
