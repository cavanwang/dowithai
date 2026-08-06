"""
llm_client.py - 统一 LLM 客户端层
支持 Ollama 原生 API 和 OpenAI 兼容 API 双后端

配置方式（环境变量）：
  LLM_BACKEND=ollama|openai     （默认 ollama）
  OLLAMA_LOCAL=http://localhost:11434
  OLLAMA_REMOTE=http://10.209.42.118:11434
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_API_KEY=sk-xxx

统一接口：
  llm_chat(messages, model, tools=None, stream=False, backend='remote')
  llm_embed(texts, model, backend='local')

流式输出统一格式（每个 chunk 为 dict）：
  {'content': str, 'thinking': str, 'tool_calls': list|None, 'done': bool}
"""

import os
import json
from typing import Iterator

# ==================== 配置 ====================
LLM_BACKEND = os.environ.get('LLM_BACKEND', 'ollama').lower()  # 'ollama' | 'openai'
OLLAMA_LOCAL = os.environ.get('OLLAMA_LOCAL', 'http://localhost:11434')
OLLAMA_REMOTE = os.environ.get('OLLAMA_REMOTE', 'http://10.209.42.118:11434')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')


# ==================== Ollama 后端 ====================

_ollama_local = None
_ollama_remote = None


def _get_ollama_client(backend: str = 'remote'):
    """获取 Ollama 客户端（懒加载）"""
    global _ollama_local, _ollama_remote
    from ollama import Client
    if backend == 'local':
        if _ollama_local is None:
            _ollama_local = Client(host=OLLAMA_LOCAL)
        return _ollama_local
    else:
        if _ollama_remote is None:
            _ollama_remote = Client(host=OLLAMA_REMOTE)
        return _ollama_remote


def _ollama_chat(messages, model, tools=None, stream=False, backend='remote'):
    """Ollama 后端对话"""
    client = _get_ollama_client(backend)
    kwargs = {'model': model, 'messages': messages}
    if tools:
        kwargs['tools'] = tools
    if stream:
        kwargs['stream'] = True
    response = client.chat(**kwargs)
    if stream:
        return _normalize_ollama_stream(response)
    else:
        return _normalize_ollama_response(response)


def _normalize_ollama_stream(stream) -> Iterator[dict]:
    """将 Ollama 流式输出转为统一格式"""
    for chunk in stream:
        msg = chunk.get('message', {}) if isinstance(chunk, dict) else chunk.message.__dict__
        if isinstance(msg, dict):
            content = msg.get('content', '')
            thinking = msg.get('thinking', '')
            tool_calls = msg.get('tool_calls', None)
        else:
            content = getattr(msg, 'content', '') or ''
            thinking = getattr(msg, 'thinking', '') or ''
            tool_calls = getattr(msg, 'tool_calls', None)

        # 标准化 tool_calls 格式
        normalized_tools = None
        if tool_calls:
            normalized_tools = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get('function', tc)
                    normalized_tools.append({
                        'function': {
                            'name': func.get('name', ''),
                            'arguments': func.get('arguments', {})
                        }
                    })
                else:
                    normalized_tools.append({
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    })

        yield {
            'content': content,
            'thinking': thinking,
            'tool_calls': normalized_tools,
            'done': False
        }
    # 最后一个标记
    yield {'content': '', 'thinking': '', 'tool_calls': None, 'done': True}


def _normalize_ollama_response(response) -> dict:
    """将 Ollama 非流式响应转为统一格式"""
    if isinstance(response, dict):
        msg = response.get('message', {})
    else:
        msg = response.message

    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '') or ''
    thinking = msg.get('thinking', '') if isinstance(msg, dict) else getattr(msg, 'thinking', '') or ''
    tool_calls = msg.get('tool_calls', None) if isinstance(msg, dict) else getattr(msg, 'tool_calls', None)

    normalized_tools = None
    if tool_calls:
        normalized_tools = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get('function', tc)
                normalized_tools.append({
                    'function': {
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', {})
                    }
                })
            else:
                normalized_tools.append({
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments
                    }
                })

    return {
        'content': content,
        'thinking': thinking,
        'tool_calls': normalized_tools,
        'done': True
    }


def _ollama_embed(texts, model, backend='local') -> list[list[float]]:
    """Ollama 后端嵌入"""
    client = _get_ollama_client(backend)
    response = client.embed(model=model, input=texts)
    return response['embeddings']


# ==================== OpenAI 后端 ====================

_openai_client = None


def _get_openai_client():
    """获取 OpenAI 客户端（懒加载）"""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    return _openai_client


def _convert_tools_to_openai(tools) -> list:
    """将 Ollama 格式的工具定义转为 OpenAI 格式"""
    openai_tools = []
    for tool in tools:
        # 如果已经是 OpenAI 格式
        if isinstance(tool, dict) and 'type' in tool:
            openai_tools.append(tool)
            continue
        # Ollama 格式：直接是 function schema
        if isinstance(tool, dict) and 'function' in tool:
            openai_tools.append({'type': 'function', 'function': tool['function']})
            continue
        # Ollama SDK 传入的函数对象（自动转换）
        if callable(tool):
            openai_tools.append({'type': 'function', 'function': _func_to_tool_schema(tool)})
            continue
        # 假设是裸 schema
        openai_tools.append({'type': 'function', 'function': tool})
    return openai_tools


def _func_to_tool_schema(func) -> dict:
    """将 Python 函数转为 OpenAI tool schema"""
    import inspect
    sig = inspect.signature(func)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        prop = {'type': 'string', 'description': ''}
        if param.annotation == int:
            prop['type'] = 'integer'
        elif param.annotation == float:
            prop['type'] = 'number'
        elif param.annotation == bool:
            prop['type'] = 'boolean'
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        'name': func.__name__,
        'description': func.__doc__ or '',
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }


def _convert_messages_to_openai(messages) -> list:
    """将 Ollama 格式消息转为 OpenAI 格式（处理 images / tool_calls / tool 角色）"""
    converted = []
    _id_counter = 0
    pending_tool_call_ids = []  # assistant tool_calls 生成的 id，供后续 tool 消息消费

    for msg in messages:
        if isinstance(msg, str):
            converted.append({'role': 'user', 'content': msg})
            continue

        new_msg = dict(msg)

        # 处理 Ollama 的 images 字段 → OpenAI 的 content 数组格式
        if 'images' in new_msg:
            images = new_msg.pop('images')
            text_content = new_msg.get('content', '')
            content_parts = []
            if text_content:
                content_parts.append({'type': 'text', 'text': text_content})
            for img_b64 in images:
                content_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{img_b64}'}
                })
            new_msg['content'] = content_parts

        # 处理 assistant 消息中的 tool_calls（Ollama 格式 → OpenAI 格式）
        if new_msg.get('role') == 'assistant' and 'tool_calls' in new_msg:
            openai_tool_calls = []
            pending_tool_call_ids = []
            for tc in new_msg['tool_calls']:
                func = tc.get('function', tc)
                tool_call_id = tc.get('id', f'call_{_id_counter}')
                _id_counter += 1
                pending_tool_call_ids.append(tool_call_id)
                args = func.get('arguments', {})
                openai_tool_calls.append({
                    'id': tool_call_id,
                    'type': 'function',
                    'function': {
                        'name': func.get('name', ''),
                        'arguments': json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    }
                })
            new_msg['tool_calls'] = openai_tool_calls
            # OpenAI 要求 content 不能为 None
            if new_msg.get('content') is None:
                new_msg['content'] = ''

        # 处理 tool 角色消息：分配与前面 tool_calls 匹配的 tool_call_id
        if new_msg.get('role') == 'tool':
            if 'tool_call_id' not in new_msg:
                if pending_tool_call_ids:
                    new_msg['tool_call_id'] = pending_tool_call_ids.pop(0)
                else:
                    new_msg['tool_call_id'] = f'call_{_id_counter}'
                    _id_counter += 1

        converted.append(new_msg)
    return converted


def _openai_chat(messages, model, tools=None, stream=False, backend='remote'):
    """OpenAI 后端对话"""
    client = _get_openai_client()
    kwargs = {
        'model': model,
        'messages': _convert_messages_to_openai(messages),
    }
    if tools:
        kwargs['tools'] = _convert_tools_to_openai(tools)
    if stream:
        kwargs['stream'] = True

    response = client.chat.completions.create(**kwargs)
    if stream:
        return _normalize_openai_stream(response)
    else:
        return _normalize_openai_response(response)


def _normalize_openai_stream(stream) -> Iterator[dict]:
    """将 OpenAI 流式输出转为统一格式"""
    pending_tool_calls = {}  # index -> {id, name, arguments}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        content = delta.content or ''
        # OpenAI 兼容服务的思考内容字段（DeepSeek 等用 reasoning_content）
        thinking = getattr(delta, 'reasoning_content', '') or ''

        # 处理工具调用的增量拼接
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in pending_tool_calls:
                    pending_tool_calls[idx] = {
                        'id': tc_delta.id or '',
                        'name': '',
                        'arguments': ''
                    }
                if tc_delta.function:
                    if tc_delta.function.name:
                        pending_tool_calls[idx]['name'] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        pending_tool_calls[idx]['arguments'] += tc_delta.function.arguments

        # 判断流是否结束
        if chunk.choices[0].finish_reason:
            # 流结束，输出累积的工具调用
            tool_calls_out = None
            if pending_tool_calls:
                tool_calls_out = []
                for idx in sorted(pending_tool_calls.keys()):
                    tc = pending_tool_calls[idx]
                    try:
                        args = json.loads(tc['arguments']) if tc['arguments'] else {}
                    except json.JSONDecodeError:
                        args = {'_raw': tc['arguments']}
                    tool_calls_out.append({
                        'function': {'name': tc['name'], 'arguments': args}
                    })
            yield {'content': content, 'thinking': thinking, 'tool_calls': tool_calls_out, 'done': True}
            return

        yield {'content': content, 'thinking': thinking, 'tool_calls': None, 'done': False}

    # 安全兜底
    yield {'content': '', 'thinking': '', 'tool_calls': None, 'done': True}


def _normalize_openai_response(response) -> dict:
    """将 OpenAI 非流式响应转为统一格式"""
    choice = response.choices[0]
    msg = choice.message

    tool_calls_out = None
    if msg.tool_calls:
        tool_calls_out = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {'_raw': tc.function.arguments}
            tool_calls_out.append({
                'function': {'name': tc.function.name, 'arguments': args}
            })

    return {
        'content': msg.content or '',
        'thinking': getattr(msg, 'reasoning_content', '') or '',
        'tool_calls': tool_calls_out,
        'done': True
    }


def _openai_embed(texts, model, backend='local') -> list[list[float]]:
    """OpenAI 后端嵌入"""
    client = _get_openai_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


# ==================== 统一入口 ====================

def llm_chat(messages, model, tools=None, stream=False, backend='remote') -> dict | Iterator[dict]:
    """
    统一对话接口

    参数：
      messages: 消息列表（Ollama 格式，含 images 字段会自动转换）
      model: 模型名称
      tools: 工具列表（支持函数/字典/OpenAI格式，自动转换）
      stream: 是否流式
      backend: 'local' | 'remote'（仅 Ollama 模式有意义）

    返回：
      非流式: {'content': str, 'thinking': str, 'tool_calls': list|None, 'done': True}
      流式: Iterator[{'content': str, 'thinking': str, 'tool_calls': list|None, 'done': bool}]
    """
    if LLM_BACKEND == 'openai':
        return _openai_chat(messages, model, tools=tools, stream=stream, backend=backend)
    else:
        return _ollama_chat(messages, model, tools=tools, stream=stream, backend=backend)


def llm_embed(texts, model, backend='local') -> list[list[float]]:
    """
    统一嵌入接口

    参数：
      texts: 文本或文本列表
      model: 嵌入模型名称
      backend: 'local' | 'remote'

    返回：
      list[list[float]] - 嵌入向量列表
    """
    if isinstance(texts, str):
        texts = [texts]
    if LLM_BACKEND == 'openai':
        return _openai_embed(texts, model, backend=backend)
    else:
        return _ollama_embed(texts, model, backend=backend)


def build_assistant_message(response: dict) -> dict:
    """
    从 llm_chat 的响应构建 assistant 消息（用于追加到 messages 历史）
    适用于 Agent 循环中：模型返回 tool_calls 时，需要将 assistant 消息加入对话
    """
    msg = {'role': 'assistant', 'content': response.get('content', '')}
    if response.get('tool_calls'):
        msg['tool_calls'] = response['tool_calls']
    return msg


def get_backend_info() -> str:
    """返回当前后端配置信息"""
    if LLM_BACKEND == 'openai':
        return f"OpenAI 兼容模式 | base_url={OPENAI_BASE_URL}"
    else:
        return f"Ollama 模式 | local={OLLAMA_LOCAL} | remote={OLLAMA_REMOTE}"
