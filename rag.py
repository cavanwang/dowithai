"""
阶段二：RAG（检索增强生成）核心实现
- 文档加载与切分：读取 ./docs/ 下的 txt/md 文件
- 向量化：使用嵌入模型生成 embedding（nomic-embed-text）
- 存储：ChromaDB 本地持久化
- 检索问答：检索相关片段 -> 拼入 prompt -> 模型生成回答
- 支持 Ollama / OpenAI 双后端（通过环境变量 LLM_BACKEND 切换）
"""

import os
import fitz  # PyMuPDF
import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llm_client import llm_chat, llm_embed, get_backend_info

console = Console()

# ==================== 配置 ====================

MODEL = 'qwen3:8b'
EMBED_MODEL = 'nomic-embed-text'
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_db')
COLLECTION_NAME = 'knowledge_base'
CHUNK_SIZE = 200       # 每个文本块的最大字符数
CHUNK_OVERLAP = 30     # 块之间的重叠字符数

# ==================== 文档加载与切分 ====================

def load_documents() -> list[dict]:
    """读取 docs/ 目录下的所有 txt、md、pdf 文件，返回文档列表"""
    documents = []
    if not os.path.exists(DOCS_DIR):
        console.print(f"[yellow]文档目录不存在: {DOCS_DIR}[/yellow]")
        return documents

    for fname in os.listdir(DOCS_DIR):
        fpath = os.path.join(DOCS_DIR, fname)

        if fname.endswith(('.txt', '.md')):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({'source': fname, 'content': content})
            console.print(f"  [dim]加载文档: {fname} ({len(content)} 字符)[/dim]")

        elif fname.lower().endswith('.pdf'):
            try:
                pdf = fitz.open(fpath)
                page_count = len(pdf)
                pages_text = []
                for page_num, page in enumerate(pdf, 1):
                    text = page.get_text()
                    if text.strip():
                        pages_text.append(f"[第{page_num}页]\n{text.strip()}")
                pdf.close()
                content = '\n\n'.join(pages_text)
                documents.append({'source': fname, 'content': content})
                console.print(f"  [dim]加载PDF: {fname} ({page_count} 页, {len(content)} 字符)[/dim]")
            except Exception as e:
                console.print(f"  [red]PDF加载失败: {fname} - {e}[/red]")

    return documents


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将文本按段落切分为块，支持重叠"""
    # 先按段落分割
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前段落本身就超过 chunk_size，按句子再切分
        if len(para) > chunk_size:
            sentences = para.replace('。', '。\n').replace('.', '.\n').split('\n')
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) <= chunk_size:
                    current_chunk += sent + ' '
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + ' '
        else:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 添加重叠（简化处理：每个块开头包含前一个块的尾部）
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1][-overlap:]
            overlapped.append(prev_tail + ' ' + chunks[i])
        chunks = overlapped

    return chunks


# ==================== 向量化与存储 ====================

def get_embedding(text: str) -> list[float]:
    """获取文本的 embedding 向量"""
    return llm_embed(text, model=EMBED_MODEL, backend='local')[0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量获取 embedding"""
    return llm_embed(texts, model=EMBED_MODEL, backend='local')


def init_chroma_collection():
    """初始化 ChromaDB 集合"""
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    return client, collection


def build_knowledge_base():
    """构建知识库：加载文档 -> 切分 -> 向量化 -> 存入 ChromaDB"""
    console.print(Panel("[bold cyan]构建知识库[/bold cyan]", expand=False))

    # 加载文档
    documents = load_documents()
    if not documents:
        console.print("[red]没有找到任何文档！[/red]")
        return

    # 切分文本
    all_chunks = []
    all_sources = []
    for doc in documents:
        chunks = split_text(doc['content'])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(doc['source'])

    console.print(f"  共切分为 [bold]{len(all_chunks)}[/bold] 个文本块")

    # 向量化（分批处理，每批 10 个）
    console.print("  正在生成向量嵌入...")
    batch_size = 10
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        embeddings = get_embeddings_batch(batch)
        all_embeddings.extend(embeddings)
        console.print(f"  [dim]已处理 {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} 块[/dim]")

    # 存入 ChromaDB
    _, collection = init_chroma_collection()

    # 生成唯一 ID
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=[{"source": src} for src in all_sources]
    )

    console.print(f"  [green]知识库构建完成！共 {len(all_chunks)} 条记录[/green]\n")


# ==================== 检索与增强生成 ====================

def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """根据查询检索最相关的知识片段"""
    _, collection = init_chroma_collection()

    if collection.count() == 0:
        return []

    # 获取查询的 embedding
    query_embedding = get_embedding(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            'content': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'distance': results['distances'][0][i]
        })

    return retrieved


def rag_query(query: str) -> str:
    """RAG 问答：检索相关知识 -> 构建 prompt -> 模型生成回答"""
    # 1. 检索
    console.print(f"\n[dim]检索相关知识...[/dim]")
    results = search_knowledge(query, top_k=3)

    if not results:
        return "知识库为空，请先运行构建知识库。"

    # 2. 构建上下文
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"【来源: {r['source']}】\n{r['content']}")

    context = "\n\n---\n\n".join(context_parts)

    # 3. 构建 prompt
    prompt = f"""请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请说明"根据现有知识库无法回答"。
回答时请引用信息来源。

参考资料：
{context}

用户问题：{query}"""

    # 4. 调用模型
    console.print(f"[dim]生成回答...[/dim]")
    response = llm_chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        backend='local',
    )

    # 5. 附加来源信息
    sources = set(r['source'] for r in results)
    source_info = "\n\n[dim]参考来源: " + ", ".join(sources) + "[/dim]"

    return response['content'] + source_info


# ==================== 主程序 ====================

def main():
    console.print(Panel.fit(
        "[bold cyan]RAG 知识库问答系统[/bold cyan]\n"
        f"基于 LLM + ChromaDB\n[dim]{get_backend_info()}[/dim]\n"
        "输入 [bold]rebuild[/bold] 重建知识库\n"
        "输入 [bold]quit[/bold] 退出",
        border_style="cyan"
    ))

    # 首次运行自动构建知识库
    _, collection = init_chroma_collection()
    if collection.count() == 0:
        console.print("[yellow]知识库为空，自动构建...[/yellow]\n")
        build_knowledge_base()

    while True:
        try:
            user_input = console.input("\n[bold green]你的问题:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            console.print("[dim]再见！[/dim]")
            break
        if user_input.lower() == 'rebuild':
            # 清空并重建
            _, col = init_chroma_collection()
            col.delete()
            build_knowledge_base()
            continue

        # RAG 问答
        answer = rag_query(user_input)
        console.print(f"\n[bold blue]回答:[/bold blue]")
        console.print(Markdown(answer))


if __name__ == '__main__':
    main()
