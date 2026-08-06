"""
阶段四：记忆管理系统
- 用独立的 ChromaDB collection 存储用户记忆
- 支持：提取经验、添加记忆、检索记忆、更新记忆、收集反馈
- 支持 Ollama / OpenAI 双后端（通过环境变量 LLM_BACKEND 切换）
"""

import os
import json
import datetime
import chromadb
from rich.console import Console

from llm_client import llm_chat, llm_embed

console = Console()

# ==================== 配置 ====================

MODEL = 'qwen3:8b'
EMBED_MODEL = 'nomic-embed-text'
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_db')
MEMORY_COLLECTION = 'user_memories'

# ==================== ChromaDB 操作 ====================

def get_memory_collection():
    """获取记忆集合"""
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embedding(text: str) -> list[float]:
    """获取文本的 embedding"""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response['embeddings'][0]


# ==================== 记忆管理 ====================

def add_memory(content: str, category: str = "general", importance: int = 5) -> str:
    """
    添加一条新记忆
    - content: 记忆内容
    - category: 分类 (preference/fact/experience/correction/general)
    - importance: 重要程度 1-10
    """
    collection = get_memory_collection()

    # 生成唯一 ID
    memory_id = f"mem_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{collection.count()}"

    # 获取 embedding
    embedding = get_embedding(content)

    # 存储
    metadata = {
        "category": category,
        "importance": importance,
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
    }

    collection.add(
        ids=[memory_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[metadata]
    )

    console.print(f"  [green]记忆已保存:[/green] [{category}] {content[:60]}...")
    return memory_id


def search_memories(query: str, top_k: int = 5) -> list[dict]:
    """根据查询检索相关记忆"""
    collection = get_memory_collection()

    if collection.count() == 0:
        return []

    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count())
    )

    memories = []
    for i in range(len(results['documents'][0])):
        memories.append({
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'category': results['metadatas'][0][i].get('category', 'general'),
            'importance': results['metadatas'][0][i].get('importance', 5),
            'created_at': results['metadatas'][0][i].get('created_at', ''),
            'distance': results['distances'][0][i]
        })

    return memories


def update_memory(memory_id: str, new_content: str) -> bool:
    """更新一条已有记忆（用于纠正场景）"""
    collection = get_memory_collection()

    try:
        # 获取旧记忆的元数据
        old = collection.get(ids=[memory_id])
        if not old or not old['ids']:
            return False

        old_metadata = old['metadatas'][0]
        old_metadata['updated_at'] = datetime.datetime.now().isoformat()
        old_metadata['category'] = 'correction'

        # 用新内容替换
        new_embedding = get_embedding(new_content)
        collection.update(
            ids=[memory_id],
            documents=[new_content],
            embeddings=[new_embedding],
            metadatas=[old_metadata]
        )
        console.print(f"  [yellow]记忆已更新:[/yellow] {new_content[:60]}...")
        return True
    except Exception as e:
        console.print(f"  [red]更新失败:[/red] {e}")
        return False


def delete_memory(memory_id: str) -> bool:
    """删除一条记忆"""
    collection = get_memory_collection()
    try:
        collection.delete(ids=[memory_id])
        return True
    except Exception:
        return False


def get_all_memories() -> list[dict]:
    """获取所有记忆"""
    collection = get_memory_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get()
    memories = []
    for i in range(len(all_data['ids'])):
        memories.append({
            'id': all_data['ids'][i],
            'content': all_data['documents'][i],
            'category': all_data['metadatas'][i].get('category', 'general'),
            'importance': all_data['metadatas'][i].get('importance', 5),
            'created_at': all_data['metadatas'][i].get('created_at', ''),
        })
    return memories


# ==================== 经验提取 ====================

def extract_experience(conversation: list[dict]) -> list[dict]:
    """
    从对话中提取关键经验/偏好/事实
    返回提取到的经验列表
    """
    # 构建对话摘要
    conv_text = ""
    for msg in conversation:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            conv_text += f"用户: {content}\n"
        elif role == 'assistant':
            conv_text += f"助手: {content[:200]}\n"

    prompt = f"""请从以下对话中提取值得记住的信息。包括：
- 用户表达的偏好或习惯
- 用户分享的个人事实或经验
- 用户对助手回答的纠正
- 重要的知识点或结论

如果没有值得记住的内容，返回空列表。

请以 JSON 格式返回，每个条目包含：
- content: 记忆内容（简洁的一句话）
- category: 分类（preference/fact/experience/correction）
- importance: 重要程度 1-10

对话内容：
{conv_text}

请只返回 JSON 数组，不要其他文字。例如：
[{{"content": "用户偏好简洁的回答风格", "category": "preference", "importance": 7}}]"""

    response = ollama.chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )

    # 解析 JSON
    text = response.message.content.strip()
    # 尝试提取 JSON 部分
    try:
        # 处理模型可能包裹在 ```json ``` 中的情况
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        experiences = json.loads(text)
        if isinstance(experiences, list):
            return experiences
    except (json.JSONDecodeError, IndexError):
        pass

    return []


# ==================== 反馈收集 ====================

def collect_feedback() -> dict:
    """
    收集用户对本次回答的反馈
    返回: {'rating': 'good/bad/correct', 'correction': str or None}
    """
    console.print("\n[bold]请对本次回答给出反馈：[/bold]")
    console.print("  [green]g[/green] - 好的/满意")
    console.print("  [red]b[/red] - 不好/不满意")
    console.print("  [yellow]c[/red] - 需要纠正（请输入正确内容）")
    console.print("  [dim]s[/dim] - 跳过")

    while True:
        try:
            choice = console.input("[bold]你的反馈:[/bold] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return {'rating': 'skip', 'correction': None}

        if choice == 'g':
            return {'rating': 'good', 'correction': None}
        elif choice == 'b':
            return {'rating': 'bad', 'correction': None}
        elif choice == 'c':
            correction = console.input("[bold]请输入正确内容:[/bold] ").strip()
            return {'rating': 'correct', 'correction': correction}
        elif choice == 's':
            return {'rating': 'skip', 'correction': None}
        else:
            console.print("[yellow]请输入 g/b/c/s[/yellow]")


# ==================== 记忆注入 Prompt ====================

def build_memory_context(query: str) -> str:
    """根据用户问题检索相关记忆，构建注入 system prompt 的上下文"""
    memories = search_memories(query, top_k=3)

    if not memories:
        return ""

    memory_text = "以下是你对该用户的了解（来自历史记忆）：\n"
    for m in memories:
        if m['distance'] < 0.7:  # 只使用相关度较高的记忆
            memory_text += f"- [{m['category']}] {m['content']}\n"

    return memory_text
