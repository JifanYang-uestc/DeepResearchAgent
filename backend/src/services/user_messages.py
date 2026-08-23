"""Stable user-safe messages for failures whose details belong in server logs."""

KNOWLEDGE_UNAVAILABLE = (
    "Knowledge RAG 不可用，已退化到 Web Search 或其它可用检索路径。"
)
KNOWLEDGE_FALLBACK = "Knowledge 主后端暂时不可用，已使用本地回退后端。"
CATALOG_UNAVAILABLE = "Knowledge Catalog 暂时不可用，路由器将使用安全回退策略。"
CATALOG_FALLBACK = "Knowledge Catalog 主后端暂时不可用，已使用本地回退目录。"
WEB_UNAVAILABLE = "Web Search 不可用；系统将继续使用已经获得的 Evidence。"
WEB_PROVIDER_NOTICE = "Web Search 返回服务状态提示，详细原因已记录在服务端日志。"
TASK_EXECUTION_FAILED = "任务执行失败，详细原因已记录在服务端日志。"
STREAMING_RESEARCH_FAILED = "研究流程暂时不可用，详细原因已记录在服务端日志。"
INVALID_RESEARCH_REQUEST = "研究请求或服务配置无效，请检查输入和服务端日志。"
RESEARCH_FAILED = "研究流程执行失败，详细原因已记录在服务端日志。"
