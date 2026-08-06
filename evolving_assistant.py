"""
阶段四：自我进化助手（最终产品）
整合 Agent + RAG + 记忆系统
对话流程：
1. 加载相关记忆 -> 注入 system prompt
2. Agent 正常对话（可使用所有工具 + RAG）
3. 对话结束后，收集用户反馈
4. 根据反馈更新记忆库
5. 定期提取对话中的经验
"""

import ollama
import datetime
import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# 导入各模块
from rag import search_knowledge
from memory_system import (
    add_memory, search_memories, update_memory,
    extract_experience, collect_feedback, build_memory_context,
    get_all_memories
)

console = Console()

# ==================== 工具定义 ====================

def get_current_time() -> str:
    """获取当前的日期和时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


def search_in_files(query: str) -> str:
    """在当前目录下的文本文件中搜索包含关键词的内容。参数 query 是要搜索的关键词。"""
    results = []
    search_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
        for fname in files:
            if fname.endswith(('.txt', '.md', '.py', '.csv')):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                rel_path = os.path.relpath(fpath, search_dir)
                                results.append(f"[{rel_path}:{i}] {line.strip()}")
                except Exception:
                    pass
    if results:
        return f"找到 {len(results)} 处匹配:\n" + "\n".join(results[:20])
    return f"未找到包含 '{query}' 的内容"


def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression 是数学表达式字符串。"""
    try:
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*]+$', expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def read_note(filename: str) -> str:
    """读取指定笔记文件的内容。参数 filename 是文件名。"""
    search_dir = os.path.dirname(os.path.abspath(__file__))
    for subdir in ['', 'docs']:
        fpath = os.path.join(search_dir, subdir, filename)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > 2000:
                    content = content[:2000] + "\n...(内容过长，已截断)"
                return f"文件 {filename} 的内容:\n{content}"
            except Exception as e:
                return f"读取失败：{e}"
    return f"文件 '{filename}' 不存在"


def search_knowledge_base(query: str) -> str:
    """从个人知识库中语义检索相关信息。适合查询概念、知识、经验等。"""
    results = search_knowledge(query, top_k=3)
    if not results:
        return "知识库中没有找到相关内容。"
    output_parts = []
    for i, r in enumerate(results, 1):
        output_parts.append(f"[来源: {r['source']}] (相关度: {1-r['distance']:.2f})\n{r['content']}")
    return "\n\n---\n\n".join(output_parts)


def search_my_memories(query: str) -> str:
    """搜索用户的历史记忆和偏好。适合查询之前告诉过助手的信息。"""
    memories = search_memories(query, top_k=5)
    if not memories:
        return "没有找到相关记忆。"
    output_parts = []
    for m in memories:
        output_parts.append(f"[{m['category']}] (相关度: {1-m['distance']:.2f}) {m['content']}")
    return "\n".join(output_parts)


# 工具注册表
TOOLS = {
    'get_current_time': get_current_time,
    'search_in_files': search_in_files,
    'calculator': calculator,
    'read_note': read_note,
    'search_knowledge_base': search_knowledge_base,
    'search_my_memories': search_my_memories,
}

# ==================== 动态 System Prompt ====================

def build_system_prompt(user_input: str) -> str:
    """构建包含记忆上下文的 system prompt"""
    memory_context = build_memory_context(user_input)

    base_prompt = """你是一个会不断学习和进化的智能助手。你可以使用多种工具来帮助用户。
你可以使用以下工具：
- get_current_time: 获取当前时间
- search_in_files: 在本地文件中搜索关键词
- calculator: 计算数学表达式
- read_note: 读取笔记文件
- search_knowledge_base: 从知识库中语义检索知识
- search_my_memories: 搜索你记住的关于用户的记忆和偏好

重要：你正在不断进化。每次对话后，你会记住用户的偏好、经验和纠正。
请在回答时考虑你记住的关于用户的信息，提供个性化的回答。
回答请使用中文。"""

    if memory_context:
        return base_prompt + "\n\n" + memory_context
    return base_prompt


# ==================== Agent 循环 ====================

def run_agent(user_input: str, conversation_history: list) -> str:
    """Agent 主循环"""
    system_prompt = build_system_prompt(user_input)
    messages = [{'role': 'system', 'content': system_prompt}] + conversation_history + [
        {'role': 'user', 'content': user_input}
    ]

    for step in range(5):
        console.print(f"\n[dim]--- Agent 思考第 {step + 1} 轮 ---[/dim]")

        response = ollama.chat(
            model='qwen3:8b',
            messages=messages,
            tools=list(TOOLS.values()),
        )

        if response.message.tool_calls:
            messages.append(response.message)

            for tool_call in response.message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments

                console.print(f"  [yellow]调用工具:[/yellow] {func_name}({func_args})")

                if func_name in TOOLS:
                    tool_func = TOOLS[func_name]
                    if isinstance(func_args, dict):
                        result = tool_func(**func_args)
                    else:
                        result = tool_func(str(func_args))
                else:
                    result = f"未知工具: {func_name}"

                console.print(f"  [green]工具结果:[/green] {result[:200]}...")
                messages.append({'role': 'tool', 'content': result})
        else:
            return response.message.content

    return "抱歉，我尝试了多次但未能得出答案。"


# ==================== 对话后处理 ====================

def post_conversation_processing(conversation: list[dict], feedback: dict):
    """对话后处理：根据反馈更新记忆"""

    # 1. 如果用户给出纠正，直接添加纠正记忆
    if feedback['rating'] == 'correct' and feedback['correction']:
        add_memory(
            content=f"用户纠正: {feedback['correction']}",
            category='correction',
            importance=9
        )

    # 2. 如果用户满意，提取对话中的经验
    if feedback['rating'] == 'good':
        experiences = extract_experience(conversation)
        for exp in experiences:
            add_memory(
                content=exp.get('content', ''),
                category=exp.get('category', 'general'),
                importance=exp.get('importance', 5)
            )

    # 3. 如果用户不满意，也提取经验（下次改进）
    if feedback['rating'] == 'bad':
        add_memory(
            content="用户对上次关于某个话题的回答不满意，需要注意改进",
            category='experience',
            importance=6
        )


# ==================== 主程序 ====================

def show_stats():
    """显示系统状态"""
    memories = get_all_memories()
    categories = {}
    for m in memories:
        cat = m['category']
        categories[cat] = categories.get(cat, 0) + 1

    console.print(f"\n[dim]当前记忆数: {len(memories)}", end="")
    if categories:
        cat_str = " | ".join(f"{k}: {v}" for k, v in categories.items())
        console.print(f" ({cat_str})[/dim]")
    else:
        console.print("[/dim]")


def main():
    console.print(Panel.fit(
        "[bold cyan]自我进化助手[/bold cyan]\n"
        "Agent + RAG + 记忆系统\n"
        "每次对话后都会学习和进化\n\n"
        "命令:\n"
        "  [bold]stats[/bold] - 查看记忆统计\n"
        "  [bold]memories[/bold] - 查看所有记忆\n"
        "  [bold]quit[/bold] - 退出",
        border_style="cyan"
    ))

    conversation_history = []
    turn_count = 0

    while True:
        try:
            user_input = console.input("\n[bold green]你:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！期待下次交流。[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            console.print("[dim]再见！我会记住我们的对话。[/dim]")
            break
        if user_input.lower() == 'stats':
            show_stats()
            continue
        if user_input.lower() == 'memories':
            memories = get_all_memories()
            if not memories:
                console.print("[dim]还没有任何记忆[/dim]")
            else:
                for m in memories:
                    console.print(f"  [{m['category']}] {m['content'][:80]}")
            continue

        # 运行 Agent
        answer = run_agent(user_input, conversation_history)

        console.print(f"\n[bold blue]助手:[/bold blue]")
        console.print(Markdown(answer))

        # 更新对话历史
        current_turn = [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': answer}
        ]
        conversation_history.extend(current_turn)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        # 每 3 轮收集一次反馈（避免太频繁打扰用户）
        turn_count += 1
        if turn_count % 3 == 0:
            console.print("\n[dim]--- 反馈时间 ---[/dim]")
            feedback = collect_feedback()
            post_conversation_processing(current_turn, feedback)
            show_stats()


if __name__ == '__main__':
    main()
