"""
阶段零：最小 Agent Demo
演示 LLM 调用单个工具（获取时间）
支持 Ollama / OpenAI 双后端（通过环境变量 LLM_BACKEND 切换）
"""

import datetime
import json
import sys

from llm_client import llm_chat, build_assistant_message, get_backend_info

# 1. 定义工具：获取当前时间
def get_current_time() -> str:
    """获取当前的日期和时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

# 2. 第一次请求（非流式），检测是否需要调用工具
print(f"🧠 询问模型：现在几点了？  [{get_backend_info()}]")
response = llm_chat(
    model='qwen3:8b',
    messages=[{'role': 'user', 'content': '现在几点了？'}],
    tools=[get_current_time],
)

# 3. 检查模型是否要求调用工具
if response['tool_calls']:
    print("🔧 检测到工具调用请求：")
    for tool in response['tool_calls']:
        func_name = tool['function']['name']
        print(f"   - 工具名称: {func_name}")
        if func_name == 'get_current_time':
            # 执行工具
            current_time = get_current_time()
            print(f"   - 工具执行结果: {current_time}")

            # 4. 第二次请求（流式），将工具结果返回给模型并实时打印最终回答
            print("\n💬 模型最终回答（流式输出）：")
            stream = llm_chat(
                model='qwen3:8b',
                messages=[
                    {'role': 'user', 'content': '现在几点了？'},
                    build_assistant_message(response),   # 模型的原始回复（含 tool_calls）
                    {
                        'role': 'tool',
                        'content': current_time,
                    }
                ],
                stream=True,  # 开启流式输出
            )

            # 实时打印每个 chunk
            for chunk in stream:
                # 打印内容，flush=True 确保立即输出
                print(chunk['content'], end='', flush=True)
            print()  # 最后换行
else:
    # 模型直接回答了问题（没有调用工具）
    print("📝 模型直接回答：")
    print(response['content'])
