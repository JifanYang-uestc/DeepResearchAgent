# Knowledge Base Test Corpus

此目录包含可提交的工程测试资料，以及本地下载但不提交 Git 的公开论文。PDF 会被 `.gitignore` 排除，FAISS 索引保存在 `backend/vector_store/`，同样不进入 Git。

## 可提交资料

- `test_facts.txt`：人工控制的四个精确检索验收事实。
- `hello_agents_deepresearch.md`：HelloAgents 第十四章工作流摘要。

## 必需论文

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  
   文件：`rag_2020.pdf`  
   下载：[arXiv 2005.11401](https://arxiv.org/pdf/2005.11401)  
   本次校验：19 页，SHA-256 `23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233`

2. ReAct: Synergizing Reasoning and Acting in Language Models  
   文件：`react.pdf`  
   下载：[arXiv 2210.03629](https://arxiv.org/pdf/2210.03629)  
   本次校验：33 页，SHA-256 `f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69`

3. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection  
   文件：`self_rag.pdf`  
   下载：[arXiv 2310.11511](https://arxiv.org/pdf/2310.11511)  
   本次校验：30 页，SHA-256 `d9eaa1398abac0df67a9d0933a5bf8b6d9d83d2e72da2a486073cd842dd52978`

可选：Toolformer，保存为 `toolformer.pdf`。

## 恢复索引

在 `backend` 目录运行：

```powershell
..\.venv\Scripts\python.exe scripts\build_knowledge_index.py
```

运行四个工程检索 Query 并查看 Rank、Score、Document、Page、Chunk ID、Content：

```powershell
..\.venv\Scripts\python.exe scripts\debug_retrieval.py --top-k 1
```
