# 项目记忆：自我进化智能助手（渐进式 AI Agent 学习项目）

## 项目简介
一个个人 AI 学习项目，采用渐进式（分阶段）方式，从零构建"自我进化智能助手"系统。展示 AI Agent 技术的完整演进路线。

## 演进阶段
| 阶段 | 文件 | 功能 |
|------|------|------|
| 阶段零 | `use_tool.py` | 最小 demo：Ollama 模型调用单个工具（获取时间） |
| 阶段一 | `agent.py` | 多工具 Agent（ReAct 模式）：查时间、搜文件、计算器、读笔记 |
| 阶段二 | `rag.py` | RAG：文档加载→切分→向量化→ChromaDB→检索问答 |
| 阶段三 | `smart_agent.py` | Agent + RAG 融合：Agent 工具集新增知识库语义检索 |
| 阶段四 | `evolving_assistant.py` + `memory_system.py` | 自我进化：记忆系统 + 用户反馈循环 |
| 最终产品 | `web_app.py` | 全功能 Gradio Web 应用（整合一切 + 大量扩展） |

## 核心模块职责
- `rag.py`: RAG 核心（加载 docs/ 下 txt/md/pdf → 切分 → nomic-embed-text 向量化 → ChromaDB → 余弦检索）
- `memory_system.py`: 记忆管理（独立 ChromaDB collection，存储用户偏好/经验/纠正，LLM 提取对话经验）
- `web_app.py`: 最终 Web 产品（Gradio 6.x），13 个工具 + 流式输出 + 反馈学习 + 管理面板
- `image_editor.py`: 基于 InstructPix2Pix 的图片编辑（自然语言指令修改图片，Mac MPS 加速）
- `extract_doc.py` / `mhtml_to_html.py` / `chrome_pdf_export.py`: 辅助工具（MHTML 需求文档 → HTML/PDF）

## web_app.py 的完整工具集（13个）
get_current_time, format_timestamp, search_in_files, calculator, run_python(Docker沙箱),
read_note, search_knowledge_base, search_my_memories, web_search(DDGS),
get_weather(和风天气+Open-Meteo), get_stock_price/search_stock_code(新浪财经), image_edit

## 技术栈
- LLM: Ollama（qwen3:8b 对话 / nomic-embed-text 嵌入 / qwen2.5vl:7b 视觉）
- 向量数据库: ChromaDB（本地持久化，knowledge_db/ 目录）
- Web UI: Gradio 6.x
- 图片编辑: diffusers (InstructPix2Pix) + Mac MPS
- 代码沙箱: Docker 容器隔离（python:3.11-slim，无网络/只读/资源限制）
- PDF: PyMuPDF (fitz)
- 架构: 本地 Ollama（视觉+嵌入）+ 远程 Ollama（文本对话，10.209.42.118:11434）

## 目录结构
- `docs/`: RAG 知识库文档源（当前有 xuqiu_output.pdf）
- `knowledge_db/`: ChromaDB 持久化数据（knowledge_base + user_memories 两个 collection）
- `edited_images/`: 图片编辑输出目录

## 常用命令
- 启动 Web 应用: `python web_app.py`（端口 7860，启动时自动检查 Docker）
- 独立 RAG 问答: `python rag.py`（支持 rebuild 命令重建知识库）
- 独立 Agent: `python agent.py` 或 `python smart_agent.py`
- 进化助手（终端版）: `python evolving_assistant.py`

## 注意事项
- web_app.py 依赖远程 Ollama 服务（10.209.42.118:11434），本地需运行 Ollama 用于视觉/嵌入模型
- 和风天气 API Key 硬编码在 web_app.py 中
- image_editor.py 依赖本地 HuggingFace 缓存的 InstructPix2Pix 模型
- 对话历史滑动窗口：最多保留 10 轮
