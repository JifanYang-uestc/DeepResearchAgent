import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    enable_knowledge_rag: bool = Field(
        default=True,
        title="Enable Knowledge RAG",
        description="Retrieve evidence from the local knowledge base",
    )
    knowledge_backend: str = Field(
        default="helloagents",
        title="Knowledge Backend",
        description="Knowledge backend: helloagents or legacy_faiss",
    )
    embedding_provider: str = Field(
        default="local_transformer",
        title="Embedding Provider",
        description="Semantic embedding provider used by the HelloAgents backend",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        title="Embedding Model",
        description="Verified multilingual local SentenceTransformer model",
    )
    enable_advanced_rag_search: bool = Field(
        default=False,
        title="Enable Advanced RAG Search",
        description="Reserved switch for HelloAgents MQE/HyDE; disabled in V0.3",
    )
    enable_retrieval_router: bool = Field(
        default=True,
        title="Enable Retrieval Router",
        description="Route each TODO across Knowledge, Web, or Hybrid retrieval",
    )
    knowledge_base_path: str = Field(
        default="./knowledge_base",
        title="Knowledge Base Path",
        description="Directory containing PDF, text, and Markdown knowledge documents",
    )
    knowledge_index_path: str = Field(
        default="./vector_store",
        title="Knowledge Index Path",
        description="Directory used for persisted FAISS index data",
    )
    knowledge_top_k: int = Field(
        default=5,
        ge=1,
        title="Knowledge Top K",
        description="Number of local chunks retrieved for each research task",
    )
    knowledge_minimum_score: float = Field(
        default=0.0,
        title="Knowledge Minimum Score",
        description="Minimum cosine score accepted as local evidence",
    )
    knowledge_chunk_size: int = Field(
        default=800,
        ge=100,
        title="Knowledge Chunk Size",
        description="Maximum characters per local knowledge chunk",
    )
    knowledge_chunk_overlap: int = Field(
        default=120,
        ge=0,
        title="Knowledge Chunk Overlap",
        description="Overlapping characters between adjacent chunks",
    )
    knowledge_auto_build: bool = Field(
        default=True,
        title="Knowledge Auto Build",
        description="Build the FAISS index on first use when it is missing",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
            "enable_knowledge_rag": os.getenv("ENABLE_KNOWLEDGE_RAG"),
            "knowledge_backend": os.getenv("KNOWLEDGE_BACKEND"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "enable_advanced_rag_search": os.getenv("ENABLE_ADVANCED_RAG_SEARCH"),
            "enable_retrieval_router": os.getenv("ENABLE_RETRIEVAL_ROUTER"),
            "knowledge_base_path": os.getenv("KNOWLEDGE_BASE_PATH"),
            "knowledge_index_path": os.getenv("KNOWLEDGE_INDEX_PATH"),
            "knowledge_top_k": os.getenv("KNOWLEDGE_TOP_K"),
            "knowledge_minimum_score": os.getenv("KNOWLEDGE_MINIMUM_SCORE"),
            "knowledge_chunk_size": os.getenv("KNOWLEDGE_CHUNK_SIZE"),
            "knowledge_chunk_overlap": os.getenv("KNOWLEDGE_CHUNK_OVERLAP"),
            "knowledge_auto_build": os.getenv("KNOWLEDGE_AUTO_BUILD"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm

