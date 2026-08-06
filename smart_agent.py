"""
阶段三：Agent + RAG 融合智能助手
- 复用 agent.py 的所有工具
- 新增 search_knowledge_base 工具（调用 rag.py 的检索能力）
- Agent 自主判断何时查知识库、何时用其他工具
- 支持 Ollama / OpenAI 双后端（通过环境变量 LLM_BACKEND 切换）
"""

import datetime
import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llm_client import llm_chat, build_assistant_message, get_backend_info

# 复用 rag.py 的检索能力
from rag import search_knowledge

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
    else:
        return f"未找到包含 '{query}' 的内容"


def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression 是数学表达式字符串，例如 '123 * 456' 或 '2 ** 10'。"""
    try:
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*]+$', expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def read_note(filename: str) -> str:
    """读取指定笔记文件的内容。参数 filename 是文件名（如 'example.md'）。"""
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
    """从个人知识库中语义检索相关信息。知识库包含编程知识、AI/ML概念等笔记内容。
    参数 query 是自然语言查询，例如 '什么是RAG' 或 'Python装饰器的用法'。"""
    results = search_knowledge(query, top_k=3)
    if not results:
        return "知识库为空或没有找到相关内容。"

    output_parts = []
    for i, r in enumerate(results, 1):
        output_parts.append(f"[来源: {r['source']}] (相关度: {1-r['distance']:.2f})\n{r['content']}")

    return "\n\n---\n\n".join(output_parts)


# 工具注册表
TOOLS = {
    'get_current_time': get_current_time,
    'search_in_files': search_in_files,
    'calculator': calculator,
    'read_note': read_note,
    'search_knowledge_base': search_knowledge_base,
}

# ==================== Agent 核心循环 ====================

SYSTEM_PROMPT = """你是一个智能助手，拥有多种工具可以使用。请根据用户的问题，判断是否需要调用工具来获取信息。
你可以使用以下工具：
- get_current_time: 获取当前时间
- search_in_files: 在本地文件中搜索关键词（精确匹配）
- calculator: 计算数学表达式
- read_note: 读取指定笔记文件
- search_knowledge_base: 从个人知识库中语义检索（适合查询概念、知识、经验等）

使用建议：
- 用户问概念性、知识性问题时，优先使用 search_knowledge_base
- 用户要找特定文件中的内容时，使用 search_in_files
- 需要计算时使用 calculator
- 可以组合使用多个工具

回答请使用中文。"""


def run_agent(user_input: str, conversation_history: list) -> str:
    """Agent 主循环，最多迭代 5 次"""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + conversation_history + [
        {'role': 'user', 'content': user_input}
    ]

    for step in range(5):
        console.print(f"\n[dim]--- Agent 思考第 {step + 1} 轮 ---[/dim]")

        response = llm_chat(
            model='qwen3:8b',
            messages=messages,
            tools=list(TOOLS.values()),
        )

        if response['tool_calls']:
            messages.append(build_assistant_message(response))

            for tool_call in response['tool_calls']:
                func_name = tool_call['function']['name']
                func_args = tool_call['function']['arguments']

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

                messages.append({
                    'role': 'tool',
                    'content': result,
                })
        else:
            return response['content']

    return "抱歉，我尝试了多次但未能得出答案。"


# ==================== 主程序 ====================

def main():
    console.print(Panel.fit(
        "[bold cyan]智能助手 (Agent + RAG)[/bold cyan]\n"
        "支持：查时间 | 搜索文件 | 计算器 | 读笔记 | 知识库问答\n"
        f"[dim]{get_backend_info()}[/dim]\n"
        "输入 [bold]quit[/bold] 或 [bold]exit[/bold] 退出",
        border_style="cyan"
    ))

    conversation_history = []

    while True:
        try:
            user_input = console.input("\n[bold green]你:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            console.print("[dim]再见！[/dim]")
            break

        answer = run_agent(user_input, conversation_history)

        console.print(f"\n[bold blue]助手:[/bold blue]")
        console.print(Markdown(answer))

        conversation_history.append({'role': 'user', 'content': user_input})
        conversation_history.append({'role': 'assistant', 'content': answer})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


if __name__ == '__main__':
    main()
