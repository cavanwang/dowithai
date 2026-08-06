"""
Web 版自我进化助手
基于 Gradio 6.x，整合 Agent + RAG + 记忆系统
支持：对话、反馈（点赞/点踩/纠正）、记忆查看
"""

import gradio as gr
import ollama
from ollama import Client
import datetime
import os
import re
import time
import subprocess
import requests
import base64

# ==================== Ollama 双客户端配置 ====================
# 本地 Ollama：视觉模型 + RAG 嵌入模型
OLLAMA_LOCAL = 'http://localhost:11434'
# 远程 Ollama：文本聊天模型
OLLAMA_REMOTE = 'http://10.209.42.118:11434'

ollama_local = Client(host=OLLAMA_LOCAL)   # 本地客户端（视觉+嵌入）
ollama_remote = Client(host=OLLAMA_REMOTE) # 远程客户端（文本聊天）

from rag import search_knowledge, build_knowledge_base, init_chroma_collection
from memory_system import (
    add_memory, search_memories, update_memory,
    extract_experience, get_all_memories, build_memory_context
)
from ddgs import DDGS
from image_editor import edit_image

# ==================== 工具定义 ====================

def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


def format_timestamp(timestamp_ms: int, timezone: str) -> str:
    """将时间戳转换为指定时区的可读时间格式。自动识别秒级（10位）和毫秒级（13位）。
    参数 timestamp_ms: 时间戳（支持秒级如 1785316420 或毫秒级如 1785316420000）
    参数 timezone: IANA 时区名（如 'Asia/Shanghai'、'America/New_York'、'UTC'）
    返回格式如: 2026-11-12 13:14:15 (CST, UTC+8)"""
    from zoneinfo import ZoneInfo
    try:
        # 自动识别秒级/毫秒级：小于 10^12 视为秒级，自动乘 1000
        if timestamp_ms < 1e12:
            ts_seconds = float(timestamp_ms)
            ts_display = f"{timestamp_ms}（秒级，已自动×1000）"
        else:
            ts_seconds = timestamp_ms / 1000.0
            ts_display = f"{timestamp_ms}（毫秒级）"
        dt = datetime.datetime.fromtimestamp(ts_seconds, tz=ZoneInfo(timezone))
        formatted = dt.strftime('%Y-%m-%d %H:%M:%S.') + f"{dt.microsecond // 1000:03d}"
        tz_abbr = dt.strftime('%Z')
        utc_offset = dt.strftime('%z')
        # 格式化 UTC 偏移为 UTC+8 形式
        if utc_offset:
            sign = utc_offset[0]
            hours = utc_offset[1:3]
            minutes = utc_offset[3:5]
            if minutes == '00':
                offset_str = f"UTC{sign}{int(hours)}"
            else:
                offset_str = f"UTC{sign}{int(hours)}:{minutes}"
        else:
            offset_str = "UTC"
        return f"{formatted} ({tz_abbr}, {offset_str})\n原始时间戳: {ts_display}"
    except Exception as e:
        return f"转换失败: {e}。请检查时间戳和时区是否正确。常见时区: Asia/Shanghai, Asia/Tokyo, America/New_York, Europe/London, UTC"


def search_in_files(query: str) -> str:
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
    try:
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*]+$', expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# ==================== Docker 容器化代码执行 ====================
DOCKER_PYTHON_IMAGE = 'python:3.11-slim'
DOCKER_TIMEOUT = 10  # 容器执行超时（秒）


def _ensure_docker_ready():
    """项目启动时确保 Docker Desktop 已运行，并预拉取 Python 镜像"""
    print('🔍 检查 Docker 状态...')
    
    # 检查 Docker 是否已运行
    check = subprocess.run(['docker', 'info'], capture_output=True, timeout=5)
    if check.returncode == 0:
        print('✅ Docker 已运行')
    else:
        print('🚀 启动 Docker Desktop...')
        subprocess.run(['open', '-a', 'Docker'], capture_output=True)
        # 等待 Docker 就绪（最多 60 秒）
        for i in range(30):
            import time as _time
            _time.sleep(2)
            try:
                r = subprocess.run(['docker', 'info'], capture_output=True, timeout=5)
                if r.returncode == 0:
                    print(f'✅ Docker 已就绪（等待了 {(i+1)*2} 秒）')
                    break
            except Exception:
                pass
        else:
            print('⚠️ Docker 启动超时（60秒），run_python 可能无法使用')
            return
    
    # 检查并拉取 Python 镜像
    img_check = subprocess.run(
        ['docker', 'images', '-q', DOCKER_PYTHON_IMAGE],
        capture_output=True, text=True, timeout=5
    )
    if not img_check.stdout.strip():
        print(f'📦 拉取镜像 {DOCKER_PYTHON_IMAGE}...')
        pull = subprocess.run(
            ['docker', 'pull', DOCKER_PYTHON_IMAGE],
            capture_output=True, text=True, timeout=180
        )
        if pull.returncode == 0:
            print(f'✅ 镜像拉取完成')
        else:
            print(f'⚠️ 镜像拉取失败: {pull.stderr[:200]}')
    else:
        print(f'✅ 镜像 {DOCKER_PYTHON_IMAGE} 已存在')


def run_python(code: str) -> str:
    """在 Docker 容器中安全执行 Python 代码并返回结果。
    容器完全隔离：无网络、无磁盘挂载、只读文件系统、资源限制。"""
    try:
        result = subprocess.run(
            [
                'docker', 'run', '--rm',
                '--network=none',              # 禁止网络
                '--read-only',                 # 只读文件系统
                '--memory=256m',               # 内存限制
                '--cpus=1',                    # CPU 限制
                '--pids-limit=64',             # 进程数限制
                '--security-opt=no-new-privileges',  # 禁止提权
                DOCKER_PYTHON_IMAGE,
                'python3', '-c', code,
            ],
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        
        if result.returncode == 0:
            if output:
                if len(output) > 2000:
                    output = output[:2000] + '\n...(输出过长，已截断)'
                return output
            return "代码执行成功（无输出）"
        else:
            # 过滤掉 Docker 相关的 stderr 信息，只保留 Python 错误
            py_error = '\n'.join(
                line for line in error.split('\n')
                if not line.startswith('docker:') and line.strip()
            )
            return f"执行错误:\n{py_error or output}"
    except subprocess.TimeoutExpired:
        return f"执行超时（超过{DOCKER_TIMEOUT}秒），请检查代码是否有死循环。"
    except Exception as e:
        return f"执行失败: {e}"


def read_note(filename: str) -> str:
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
    results = search_knowledge(query, top_k=3)
    if not results:
        return "知识库中没有找到相关内容。"
    output_parts = []
    for i, r in enumerate(results, 1):
        output_parts.append(f"[来源: {r['source']}] (相关度: {1-r['distance']:.2f})\n{r['content']}")
    return "\n\n---\n\n".join(output_parts)


def search_my_memories(query: str) -> str:
    memories = search_memories(query, top_k=5)
    if not memories:
        return "没有找到相关记忆。"
    output_parts = []
    for m in memories:
        output_parts.append(f"[{m['category']}] (相关度: {1-m['distance']:.2f}) {m['content']}")
    return "\n".join(output_parts)


def web_search(query: str) -> str:
    """联网搜索最新信息。参数 query 是搜索关键词。当需要获取最新资讯、实时数据或知识库中没有的信息时使用。"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"未找到关于 '{query}' 的搜索结果。"
        output_parts = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '无标题')
            body = r.get('body', '无摘要')
            href = r.get('href', '')
            output_parts.append(f"{i}. {title}\n   {body}\n   链接: {href}")
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"搜索失败: {e}"


# ==================== 和风天气 API 配置 ====================
QWEATHER_API_KEY = '3e66c51469db4e19903d47dc137973c9'
QWEATHER_API_HOST = 'p94wcw5vpk.re.qweatherapi.com'


def _qweather_city_lookup(city: str):
    """和风天气城市搜索，返回 (loc_id, city_name, country) 或 None"""
    import re as _re
    city_variants = [city.strip()]
    v1 = _re.sub(r'[市县区]+$', '', city.strip())
    if v1 and v1 != city.strip():
        city_variants.append(v1)
    v2 = _re.sub(r'^(.+?省|.+?自治区)', '', city.strip())
    v2 = _re.sub(r'[市县区]+$', '', v2)
    if v2 and v2 != city.strip() and v2 not in city_variants:
        city_variants.append(v2)
    core = _re.sub(r'^(.+?省|.+?自治区|.+?市)', '', city.strip())
    core = _re.sub(r'[市县区]+$', '', core)
    if core and len(core) >= 2 and core not in city_variants:
        city_variants.append(core)
    
    headers = {'X-QW-Api-Key': QWEATHER_API_KEY}
    for name in city_variants:
        url = f'https://{QWEATHER_API_HOST}/geo/v2/city/lookup?location={requests.utils.quote(name)}&number=1'
        try:
            resp = requests.get(url, headers=headers, timeout=10).json()
            if resp.get('code') == '200' and resp.get('location'):
                loc = resp['location'][0]
                return loc['id'], loc['name'], loc.get('country', '')
        except Exception:
            continue
    return None


def _qweather_get_weather(loc_id: str, city_name: str, date: str) -> str:
    """用和风天气 API 查询天气"""
    headers = {'X-QW-Api-Key': QWEATHER_API_KEY}
    
    if date == 'today':
        # 实时天气 + 3天预报（取今天）
        now_url = f'https://{QWEATHER_API_HOST}/v7/weather/now?location={loc_id}'
        daily_url = f'https://{QWEATHER_API_HOST}/v7/weather/3d?location={loc_id}'
        
        now_resp = requests.get(now_url, headers=headers, timeout=10).json()
        daily_resp = requests.get(daily_url, headers=headers, timeout=10).json()
        
        if now_resp.get('code') != '200':
            return None
        
        now = now_resp.get('now', {})
        daily = daily_resp.get('daily', [{}])[0] if daily_resp.get('code') == '200' else {}
        
        return (f"🌍 {city_name} 今日天气:\n"
                f"   天气: {now.get('text', 'N/A')}\n"
                f"   当前温度: {now.get('temp', 'N/A')}°C\n"
                f"   最高温: {daily.get('tempMax', 'N/A')}°C / 最低温: {daily.get('tempMin', 'N/A')}°C\n"
                f"   湿度: {now.get('humidity', 'N/A')}%\n"
                f"   风速: {now.get('windSpeed', 'N/A')} km/h ({now.get('windDir', '')}{now.get('windScale', '')}级)")
    else:
        # 查询指定日期（3天预报内）
        from datetime import datetime as dt
        target = dt.strptime(date, '%Y-%m-%d').date()
        today = dt.now().date()
        days_ahead = (target - today).days
        
        if days_ahead < 0 or days_ahead > 2:
            return None  # 和风天气免费版只支持3天预报
        
        daily_url = f'https://{QWEATHER_API_HOST}/v7/weather/3d?location={loc_id}'
        daily_resp = requests.get(daily_url, headers=headers, timeout=10).json()
        
        if daily_resp.get('code') != '200':
            return None
        
        daily_list = daily_resp.get('daily', [])
        for day in daily_list:
            if day.get('fxDate') == date:
                return (f"🌍 {city_name} {date} 天气(预报):\n"
                        f"   天气: {day.get('textDay', 'N/A')}\n"
                        f"   最高温: {day.get('tempMax', 'N/A')}°C / 最低温: {day.get('tempMin', 'N/A')}°C\n"
                        f"   风速: {day.get('windSpeedDay', 'N/A')} km/h ({day.get('windDirDay', '')}{day.get('windScaleDay', '')}级)\n"
                        f"   降水量: {day.get('precip', 'N/A')} mm")
        return None


def get_weather(city: str, date: str = "today") -> str:
    """查询指定城市指定日期的天气。参数 city 是城市名（如'北京'、'Tokyo'），date 是日期（格式 YYYY-MM-DD 或 'today'）。"""
    try:
        # 优先尝试和风天气（国内城市）
        qweather_result = _qweather_city_lookup(city)
        if qweather_result:
            loc_id, city_name, country = qweather_result
            if country == '中国':
                result = _qweather_get_weather(loc_id, city_name, date)
                if result:
                    return result
        
        # 回退到 Open-Meteo（国际城市或和风天气失败）
        import re as _re
        city_variants = [city.strip()]
        v1 = _re.sub(r'[市县区]+$', '', city.strip())
        if v1 and v1 != city.strip():
            city_variants.append(v1)
        v2 = _re.sub(r'^(.+?省|.+?自治区)', '', city.strip())
        v2 = _re.sub(r'[市县区]+$', '', v2)
        if v2 and v2 != city.strip() and v2 not in city_variants:
            city_variants.append(v2)
        v3 = _re.sub(r'[省市区县自治区]', '', city.strip())
        if v3 and v3 != city.strip() and v3 not in city_variants:
            city_variants.append(v3)
        core = city.strip()
        core = _re.sub(r'^(.+?省|.+?自治区|.+?市)', '', core)
        core = _re.sub(r'[市县区]+$', '', core)
        if core and len(core) >= 2 and core not in city_variants:
            city_variants.append(core)
        
        # 地理编码
        geo_resp = None
        for name in city_variants:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(name)}&count=1&language=zh"
            geo_resp = requests.get(geo_url, timeout=10).json()
            if 'results' in geo_resp and geo_resp['results']:
                break
        
        if not geo_resp or 'results' not in geo_resp or not geo_resp['results']:
            return f"未找到城市 '{city}'，请检查城市名是否正确。"
        loc = geo_resp['results'][0]
        lat, lon = loc['latitude'], loc['longitude']
        city_name = loc.get('name', city)

        # 查询天气
        if date == "today":
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,weather_code&timezone=auto&forecast_days=1"
            resp = requests.get(weather_url, timeout=10).json()
            cur = resp.get('current', {})
            daily = resp.get('daily', {})
            weather_desc = _weather_code_to_desc(cur.get('weather_code', -1))
            t_max = daily.get('temperature_2m_max', ['N/A'])[0] if daily.get('temperature_2m_max') else 'N/A'
            t_min = daily.get('temperature_2m_min', ['N/A'])[0] if daily.get('temperature_2m_min') else 'N/A'
            return (f"🌍 {city_name} 今日天气:\n"
                    f"   天气: {weather_desc}\n"
                    f"   当前温度: {cur.get('temperature_2m', 'N/A')}°C\n"
                    f"   最高温: {t_max}°C / 最低温: {t_min}°C\n"
                    f"   湿度: {cur.get('relative_humidity_2m', 'N/A')}%\n"
                    f"   风速: {cur.get('wind_speed_10m', 'N/A')} km/h")
        else:
            from datetime import datetime as dt
            target = dt.strptime(date, '%Y-%m-%d').date()
            today = dt.now().date()
            if target > today:
                days_ahead = (target - today).days
                if days_ahead > 16:
                    return f"最多只能预报未来 16 天的天气，{date} 超出范围。"
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum&timezone=auto&forecast_days=16"
            else:
                weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum&timezone=auto"
            resp = requests.get(weather_url, timeout=10).json()
            daily = resp.get('daily', {})
            dates = daily.get('time', [])
            if date not in dates:
                return f"未找到 {city_name} {date} 的天气数据。"
            idx = dates.index(date)
            t_max = daily['temperature_2m_max'][idx]
            t_min = daily['temperature_2m_min'][idx]
            code = daily['weather_code'][idx]
            precip = daily['precipitation_sum'][idx]
            weather_desc = _weather_code_to_desc(code)
            period = "预报" if target > today else "历史"
            return (f"🌍 {city_name} {date} 天气({period}):\n"
                    f"   天气: {weather_desc}\n"
                    f"   最高温: {t_max}°C / 最低温: {t_min}°C\n"
                    f"   降水量: {precip} mm")
    except Exception as e:
        return f"天气查询失败: {e}"


def _weather_code_to_desc(code: int) -> str:
    """WMO 天气代码转描述"""
    codes = {
        0: "☀️ 晴", 1: "🌤️ 大部晴朗", 2: "⛅ 多云", 3: "☁️ 阴",
        45: "🌫️ 雾", 48: "🌫️ 雾凇", 51: "🌦️ 小毛毛雨", 53: "🌦️ 中毛毛雨",
        55: "🌦️ 大毛毛雨", 61: "🌧️ 小雨", 63: "🌧️ 中雨", 65: "🌧️ 大雨",
        71: "🌨️ 小雪", 73: "🌨️ 中雪", 75: "🌨️ 大雪", 77: "🌨️ 雪粒",
        80: "🌦️ 小阵雨", 81: "🌦️ 中阵雨", 82: "🌧️ 大阵雨",
        85: "🌨️ 小阵雪", 86: "🌨️ 大阵雪",
        95: "⛈️ 雷暴", 96: "⛈️ 雷暴+小冰雹", 99: "⛈️ 雷暴+大冰雹"
    }
    return codes.get(code, f"未知({code})")


# ==================== 股票行情查询（新浪财经 API） ====================

# 常用指数/股票代码映射
STOCK_ALIASES = {
    '上证指数': 'sh000001', '上证': 'sh000001',
    '深证成指': 'sz399001', '深证': 'sz399001',
    '创业板指': 'sz399006', '创业板': 'sz399006',
    '沪深300': 'sh000300',
    '科创50': 'sh000688',
}


def search_stock_code(keyword: str) -> str:
    """根据中文名称或拼音搜索股票代码。当用户提到个股名称（如'格力'、'茅台'）但不知道代码时使用。
    参数 keyword: 股票名称或关键词，如 '格力'、'茅台'、'宁德时代'。
    返回匹配的股票列表（含代码），如果只有一个精确匹配则直接返回代码。"""
    try:
        url = f'https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key={keyword}'
        headers = {'Referer': 'https://finance.sina.com.cn'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'gbk'
        # 解析: var suggestvalue="代码,名称,显示代码,拼音,...;代码2,名称2,..."
        import re as _re
        match = _re.search(r'"(.+)"', resp.text)
        if not match or not match.group(1).strip():
            return f"未找到与 '{keyword}' 匹配的股票。"
        items = match.group(1).strip().split(';')
        results = []
        for item in items:
            fields = item.split(',')
            if len(fields) >= 4:
                name = fields[0].strip()          # 如 贵州茅台
                display_code = fields[2].strip()  # 如 600519
                code_raw = fields[3].strip()      # 如 sh600519
                # 只保留 A 股个股（sh6xx 上海主板, sz0xx 深圳主板, sz3xx 创业板）
                if code_raw.startswith(('sh6', 'sz0', 'sz3')):
                    results.append((code_raw, name, display_code))
        if not results:
            return f"未找到与 '{keyword}' 匹配的股票。"
        if len(results) == 1:
            code_raw, name, display_code = results[0]
            return f"精确匹配: {name} ({code_raw})"
        # 多个结果，返回候选列表让模型询问用户
        lines = [f"找到 {len(results)} 个匹配结果，请确认您要查询哪一只："]
        for i, (code_raw, name, display_code) in enumerate(results[:8], 1):
            lines.append(f"  {i}. {name} ({code_raw})")
        return '\n'.join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


def get_stock_price(symbol: str) -> str:
    """查询股票/指数实时行情（数据来源：新浪财经）。
    参数 symbol: 股票或指数代码，如 'sh000001'(上证指数)、'sh600519'(贵州茅台)、'sz000858'(五粮液)。
    也支持中文名称如 '上证指数'、'贵州茅台'。"""
    try:
        # 支持中文名称映射
        raw_symbol = symbol.strip()
        code = STOCK_ALIASES.get(raw_symbol, raw_symbol)
        
        # 确保代码格式正确（sh/sz + 6位数字）
        code = code.lower().strip()
        if not code.startswith(('sh', 'sz')):
            # 尝试自动判断：6开头=上海，其他=深圳
            if code.isdigit() and len(code) == 6:
                code = ('sh' if code.startswith('6') else 'sz') + code
            else:
                return f"无法识别的代码 '{symbol}'，请使用完整代码（如 sh000001、sh600519）"
        
        # 调用新浪财经 API
        url = f'https://hq.sinajs.cn/list={code}'
        headers = {'Referer': 'https://finance.sina.com.cn'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'gbk'
        
        # 解析返回数据
        # 格式: var hq_str_sh000001="名称,开盘价,昨收,当前价,最高,最低,...";
        import re as _re
        match = _re.search(r'"(.+)"', resp.text)
        if not match:
            return f"未找到 '{symbol}' 的行情数据，请检查代码是否正确。"
        
        fields = match.group(1).split(',')
        if len(fields) < 32:
            return f"行情数据格式异常，请稍后重试。"
        
        name = fields[0]
        open_price = float(fields[1])     # 今日开盘价
        prev_close = float(fields[2])     # 昨日收盘价
        current = float(fields[3])        # 当前价格
        high = float(fields[4])           # 今日最高
        low = float(fields[5])            # 今日最低价
        volume = fields[8]                # 成交量（手）
        amount = fields[9]                # 成交额
        date = fields[30]                 # 日期
        time_str = fields[31]             # 时间
        
        # 计算涨跌
        change = current - prev_close
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
        arrow = '📈' if change >= 0 else '📉'
        sign = '+' if change >= 0 else ''
        
        # 格式化成交额
        amount_f = float(amount)
        if amount_f >= 1e8:
            amount_str = f"{amount_f/1e8:.2f} 亿"
        elif amount_f >= 1e4:
            amount_str = f"{amount_f/1e4:.2f} 万"
        else:
            amount_str = f"{amount_f:.2f}"
        
        return (f"{arrow} **{name}** ({code.upper()})\n"
                f"   当前价格: {current:.2f}\n"
                f"   涨跌: {sign}{change:.2f} ({sign}{change_pct:.2f}%)\n"
                f"   今日开盘: {open_price:.2f}\n"
                f"   最高: {high:.2f} / 最低: {low:.2f}\n"
                f"   昨收: {prev_close:.2f}\n"
                f"   成交额: {amount_str}\n"
                f"   更新时间: {date} {time_str}")
    except requests.exceptions.Timeout:
        return "查询超时，请稍后重试。"
    except Exception as e:
        return f"查询失败: {e}"


def image_edit_tool(image_path: str, instruction: str) -> str:
    """编辑图片：根据文字指令修改图片内容。
    参数 image_path: 原始图片的文件路径
    参数 instruction: 英文编辑指令（如 'Replace the red hat with a blue cap'）
    返回: 修改后的图片文件路径
    """
    result = edit_image(image_path, instruction)
    if result.startswith('❌'):
        return result
    return f"✅ 图片编辑完成！\n修改后的图片已保存: {result}"


TOOLS = {
    'get_current_time': get_current_time,
    'search_in_files': search_in_files,
    'calculator': calculator,
    'run_python': run_python,
    'read_note': read_note,
    'search_knowledge_base': search_knowledge_base,
    'search_my_memories': search_my_memories,
    'web_search': web_search,
    'get_weather': get_weather,
    'get_stock_price': get_stock_price,
    'search_stock_code': search_stock_code,
    'format_timestamp': format_timestamp,
    'image_edit': image_edit_tool,
    'image_edit_tool': image_edit_tool,  # 别名，防止模型用函数名调用时报未知工具
}

# ==================== Agent 逻辑 ====================

def build_system_prompt(user_input: str) -> str:
    memory_context = build_memory_context(user_input)
    # 获取当前日期，帮助模型理解"今天"、"明天"等相对时间
    from datetime import datetime as _dt
    today_str = _dt.now().strftime('%Y年%m月%d日')
    today_iso = _dt.now().strftime('%Y-%m-%d')
    
    base_prompt = f"""你是一个会不断学习和进化的智能助手。你可以使用多种工具来帮助用户。
今天是 {today_str}（{today_iso}）。

你可以使用以下工具：
- get_current_time: 获取当前时间
- search_in_files: 在本地文件中搜索关键词
- calculator: 计算数学表达式
- run_python: 执行 Python 代码（适合闰年判断、斐波那契数列、时间戳转换、时区转换等编程计算）
- read_note: 读取笔记文件
- search_knowledge_base: 从知识库中语义检索知识
- search_my_memories: 搜索你记住的关于用户的记忆和偏好
- web_search: 联网搜索最新信息（适合查询新闻、实时数据、知识库中没有的内容）
- get_weather: 查询指定城市指定日期的天气（参数: city=城市名, date=日期YYYY-MM-DD或today）
- get_stock_price: 查询股票/指数实时行情（参数: symbol=股票代码如sh000001、sh600519，也支持中文如'上证指数'）
- search_stock_code: 根据中文名称搜索股票代码（参数: keyword=股票名称如'格力'、'茅台'）
- format_timestamp: 将毫秒时间戳转换为可读时间（参数: timestamp_ms=毫秒时间戳, timezone=IANA时区如'Asia/Shanghai'）
- image_edit: 根据文字指令编辑图片（参数: image_path=图片路径, instruction=英文编辑指令如'Replace the hat with a blue cap'）

重要：查询个股行情时，如果不知道股票代码，请先用 search_stock_code 搜索代码。
如果 search_stock_code 返回多个匹配结果，你必须将候选列表展示给用户，询问用户具体要查询哪一只股票，等用户确认后再调用 get_stock_price。不要自行猜测。

重要：你正在不断进化。每次对话后，你会记住用户的偏好、经验和纠正。
请在回答时考虑你记住的关于用户的信息，提供个性化的回答。
回答请使用中文。提及城市名称时请使用完整名称（如“乌鲁木齐”而非“乌鲁木”）。
重要：对于以下类型的任务，优先使用 run_python 工具执行代码来完成，不要手动计算：
- 时间日期类：时间戳转换、时区转换、日期加减、计算两个日期的间隔等
- 数学求解类：斐波那契数列、素数判断、排列组合、方程求解、矩阵运算等
- 进制转换类：二进制/十进制/十六进制转换、编码解码等
- 算法验证类：排序结果、递归计算、算法复杂度分析等
- 数据处理类：统计分析、数据筛选、格式化转换等
简单四则运算可以使用 calculator 工具，但涉及多步骤推理或容易出错的计算，请使用 run_python。
重要：对于时间戳转换的任务，必须使用 format_timestamp 工具，先从用户消息中提取时区信息（如"北京时间"对应 Asia/Shanghai，"纽约时间"对应 America/New_York，"东京时间"对应 Asia/Tokyo），然后调用该工具。不要手动计算。
重要：当用户提到"知识库"、"需求文档"、"文档内容"、"已保存的资料"等，或者问题可能涉及本地知识库中已有的内容时，必须优先调用 search_knowledge_base 工具检索知识库，不要依赖模型自身的知识来回答。如果你不确定是否需要检索知识库，请先询问用户是否需要查阅本地知识库，等用户确认后再决定是否调用。
重要：你具备图片编辑能力！当用户上传图片并要求修改图片内容时（如换帽子、改颜色、加物体、修改文字、改变背景等），你必须调用 image_edit 工具来完成编辑。绝对不要回复"无法编辑图片"或"没有图片编辑能力"——你有这个能力。
使用方式：将用户的中文编辑意图翻译为英文指令，然后调用 image_edit(image_path=图片路径, instruction=英文指令)。
示例：用户说"把帽子换成蓝色的" → 调用 image_edit(image_path="/tmp/xxx.png", instruction="Replace the hat with a blue cap")
注意：当用户消息中包含"[用户上传了图片]"标记时，说明用户确实上传了图片，你必须积极使用 image_edit 工具。
重要：每次工具调用返回结果后，你必须向用户输出明确的回复，说明工具执行的结果。如果工具执行成功，告诉用户结果；如果工具执行失败或无法完成用户的请求，也要如实告知用户原因并给出建议。绝对不允许在工具返回结果后保持沉默。
"""
    if memory_context:
        return base_prompt + "\n\n" + memory_context
    return base_prompt


def extract_text(content) -> str:
    """从 Gradio 消息格式中提取纯文本
    Gradio 6.x 的 content 可能是 [{'text': '...', 'type': 'text'}] 列表
    Ollama 期望 content 是纯字符串
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(
            item.get('text', '') if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


# 思考内容显示缓冲区（滞后截断：0→256→截断到128→增长到256→再截断到128）
_thinking_display_state = {'buffer': '', 'last_full_len': 0}


def _fmt_thinking(thinking: str) -> str:
    """格式化思考内容（滞后截断：达到256时截断到128，继续增长到256再截断）"""
    if not thinking:
        _thinking_display_state['buffer'] = ''
        _thinking_display_state['last_full_len'] = 0
        return ''
    
    text = thinking.strip().replace('\n', ' ')
    last_len = _thinking_display_state['last_full_len']
    buffer = _thinking_display_state['buffer']
    
    # 追加新增内容
    if len(text) > last_len:
        buffer += text[last_len:]
    else:
        # 思考内容被重置（新一轮对话）
        buffer = text
    
    # 超过256时截断到128
    if len(buffer) > 256:
        buffer = buffer[-128:]
    
    _thinking_display_state['buffer'] = buffer
    _thinking_display_state['last_full_len'] = len(text)
    
    return buffer


def _fmt_tool_args(args) -> str:
    """格式化工具参数为可读字符串"""
    if isinstance(args, dict):
        return ', '.join(f'{k}={v}' for k, v in args.items())
    return str(args)


def _extract_clean_answer(content: str) -> str:
    """从完整展示内容中提取纯文本回答（去掉步骤展示 + 统计后缀）"""
    if not content:
        return ''
    # 去掉前面的步骤展示（✅ 第X步...）和分隔线
    if '\n---\n\n' in content:
        clean = content.split('\n---\n\n')[-1]
    else:
        clean = content
    # 去掉末尾的统计块（---\n🧠思考/💬回答/⚡总计）
    if '\n\n---\n' in clean:
        clean = clean.split('\n\n---\n')[0]
    return clean.strip()


# 滑动窗口：只保留最近 N 轮对话（1轮 = 1条user + 1条assistant = 2条消息）
_MAX_HISTORY_ROUNDS = 10


def run_agent_stream(user_input: str, history: list):
    """流式 Agent 循环 - 展示思考过程、工具调用和最终回答"""
    # 构建消息：去噪 + 滑动窗口
    conversation_history = []
    for msg in history:
        raw_content = extract_text(msg['content'])
        clean_content = _extract_clean_answer(raw_content)
        if clean_content:  # 跳过空消息
            conversation_history.append({'role': msg['role'], 'content': clean_content})

    # 滑动窗口：只保留最近 N 轮（每条 user+assistant 算 1 轮）
    max_messages = _MAX_HISTORY_ROUNDS * 2
    if len(conversation_history) > max_messages:
        conversation_history = conversation_history[-max_messages:]

    system_prompt = build_system_prompt(user_input)
    messages = [{'role': 'system', 'content': system_prompt}] + conversation_history + [
        {'role': 'user', 'content': user_input}
    ]

    completed_steps = []  # 已完成的步骤描述列表
    current_thinking = ''  # 当前思考内容
    step_num = 0
    # 流式统计：用于计算 token 生成速度
    _stream_start = None  # 回答开始时间
    _stream_chunks = 0  # 回答 token 数（chunk 计数）
    _stream_chars = 0
    _think_start = None  # 思考开始时间
    _think_chunks = 0  # 思考 token 数
    _think_chars = 0
    # P0: 死循环检测 — 记录最近工具调用签名
    _recent_tool_sigs = []  # 最近 N 次的 (工具名, 参数) 签名
    _MAX_REPEAT = 3  # 连续 3 次相同调用 → 强制停止
    # P0: 跨轮总计统计
    _total_think_chunks = 0
    _total_stream_chunks = 0
    _total_tool_calls = 0
    _agent_start = time.time()
    _thinking_only_retries = 0  # 思考后未执行工具/回答的自动重试计数

    MAX_STEPS = 5

    for step in range(MAX_STEPS):
        step_num += 1
        current_thinking = ''
        current_content = ''
        tool_calls_data = []
        has_tool_calls = False

        # 构建已完成步骤的展示
        def _build_display(current_status='', answer_text=''):
            parts = []
            for cs in completed_steps:
                parts.append(cs)
            if current_status:
                parts.append(current_status)
            header = '\n\n'.join(parts) if parts else ''
            if answer_text:
                if header:
                    return f"{header}\n\n---\n\n{answer_text}"
                return answer_text
            return header

        # P0: 接近上限时注入提示，让模型尽快给结论
        if step == MAX_STEPS - 2:
            messages.append({
                'role': 'user',
                'content': '[系统提示] 你只剩最后 1 轮机会。请基于已获取的信息，尽快给出最终回答，不要再调用工具。'
            })

        # 流式调用模型
        _stream_start = time.time()
        _stream_chunks = 0
        _stream_chars = 0
        _think_start = time.time()
        _think_chunks = 0
        _think_chars = 0
        try:
            stream = ollama_remote.chat(
                model=TEXT_MODEL,
                messages=messages,
                tools=list(TOOLS.values()),
                stream=True,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if 'connection' in err_msg or 'connect' in err_msg:
                yield _build_display(answer_text=f"⚠️ **连接中断**：与模型服务的连接已断开（可能是页面刷新或网络中断）。\n\n请重新发送您的请求。")
            else:
                yield _build_display(answer_text=f"⚠️ **模型服务异常**：{e}\n\n请稍后重试。")
            return

        try:
          for chunk in stream:
            msg = chunk.get('message', {})

            # 捕获思考内容
            thinking = msg.get('thinking', '')
            if thinking:
                current_thinking += thinking
                _think_chunks += 1
                _think_chars += len(thinking)
                think_elapsed = time.time() - _think_start
                think_speed = _think_chunks / think_elapsed if think_elapsed > 0 else 0
                think_speed_info = f"⚡ 思考: {think_speed:.1f} tokens/s | {_think_chunks} tokens"
                thinking_display = _fmt_thinking(current_thinking)
                if current_content:
                    status = f"**🧠 思考中（第{step_num}步）：** {thinking_display}\n\n{think_speed_info}"
                    yield _build_display(current_status=status, answer_text=current_content)
                else:
                    status = f"**🧠 思考中（第{step_num}步）：** {thinking_display}\n\n{think_speed_info}"
                    yield _build_display(current_status=status)

            # 捕获回答内容
            content = msg.get('content', '')
            if content:
                current_content += content
                _stream_chunks += 1
                _stream_chars += len(content)
                elapsed = time.time() - _stream_start
                speed = _stream_chunks / elapsed if elapsed > 0 else 0
                speed_info = f"⚡ {speed:.1f} tokens/s | {_stream_chunks} tokens | {elapsed:.1f}s"
                if current_thinking and not has_tool_calls:
                    thinking_display = _fmt_thinking(current_thinking)
                    status = f"**🧠 思考：** {thinking_display}\n\n{speed_info}"
                    yield _build_display(current_status=status, answer_text=current_content)
                else:
                    yield _build_display(current_status=speed_info, answer_text=current_content)

            # 捕获工具调用
            chunk_tool_calls = msg.get('tool_calls', None)
            if chunk_tool_calls:
                has_tool_calls = True
                for tc in chunk_tool_calls:
                    func = tc.get('function', tc)
                    name = func.get('name', 'unknown')
                    args = func.get('arguments', {})
                    tool_calls_data.append((name, args))
        except (ConnectionError, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, GeneratorExit) as e:
            # 流式传输中断（客户端断连或网络问题）
            yield _build_display(answer_text=f"⚠️ **连接中断**：模型响应流已断开（可能是页面刷新或网络中断）。\n\n请重新发送您的请求。")
            return
        except Exception as e:
            yield _build_display(answer_text=f"⚠️ **模型响应异常**：{e}\n\n请稍后重试。")
            return

        # P0: 死循环检测 — 检查是否连续调用相同工具
        if has_tool_calls:
            for name, args in tool_calls_data:
                sig = (name, str(sorted(args.items()) if isinstance(args, dict) else args))
                _recent_tool_sigs.append(sig)
            # 检查最近 _MAX_REPEAT 次是否完全相同
            if len(_recent_tool_sigs) >= _MAX_REPEAT:
                last_sigs = _recent_tool_sigs[-_MAX_REPEAT:]
                if all(s == last_sigs[0] for s in last_sigs):
                    yield _build_display(
                        answer_text=f"⚠️ 检测到连续 {_MAX_REPEAT} 次相同的工具调用，已强制停止。请尝试换一种方式提问。"
                    )
                    return

        # 处理工具调用
        if has_tool_calls:
            # 显示思考 + 工具调用
            thinking_line = ''
            if current_thinking:
                thinking_line = f"**🧠 思考：** {_fmt_thinking(current_thinking)}\n\n"

            tool_lines = []
            for name, args in tool_calls_data:
                tool_lines.append(f"**🔧 调用工具：`{name}`**（参数：{_fmt_tool_args(args)}）")

            step_display = thinking_line + '\n'.join(tool_lines)
            status = f"✅ **第{step_num}步**\n{step_display}\n\n⏳ 执行中..."
            yield _build_display(current_status=status)

            # 执行工具
            assistant_msg = {'role': 'assistant', 'content': ''}
            assistant_msg['tool_calls'] = [
                {'function': {'name': name, 'arguments': args}}
                for name, args in tool_calls_data
            ]
            messages.append(assistant_msg)

            for name, args in tool_calls_data:
                if name in TOOLS:
                    try:
                        if isinstance(args, dict):
                            result = TOOLS[name](**args)
                        else:
                            result = TOOLS[name](str(args))
                    except Exception as e:
                        result = f"执行错误: {e}"
                else:
                    result = f"未知工具: {name}"
                messages.append({'role': 'tool', 'content': result})
                # 检测图片编辑结果：如果返回的是文件路径，存储用于展示
                global _last_edited_image
                if name in ('image_edit', 'image_edit_tool') and not result.startswith('❌') and os.path.exists(result):
                    _last_edited_image = result

            # 累计跨轮统计
            _total_think_chunks += _think_chunks
            _total_stream_chunks += _stream_chunks
            _total_tool_calls += len(tool_calls_data)

            # 记录完成步骤
            for name, args in tool_calls_data:
                completed_steps.append(
                    f"✅ **第{step_num}步**\n"
                    f"🧠 思考：{_fmt_thinking(current_thinking)}\n\n"
                    f"🔧 调用工具：`{name}`（参数：{_fmt_tool_args(args)}）"
                )
            continue
        else:
            # 无工具调用 = 最终回答已流式输出完成，附加最终速度统计
            # 检测：模型只有思考但没有工具调用也没有回答内容（thinking-only 死循环）
            if current_thinking and not current_content and _thinking_only_retries < 2:
                _thinking_only_retries += 1
                messages.append({'role': 'assistant', 'content': ''})
                # 查找最近一次工具返回结果，帮助模型理解上下文
                _last_tool_result = ''
                for _m in reversed(messages):
                    if _m.get('role') == 'tool':
                        _last_tool_result = _m['content'][:200]
                        break
                _retry_hint = '请根据工具返回的结果向用户说明情况。' if _last_tool_result else '请立即调用工具完成任务，不要再思考。'
                messages.append({
                    'role': 'user',
                    'content': f'[系统提示] 你刚才描述了计划但没有实际执行。{_retry_hint}' + (f'工具返回结果：{_last_tool_result}' if _last_tool_result else '')
                })
                yield _build_display(current_status=f'⚠️ 模型思考后未执行操作，自动重试（{_thinking_only_retries}/2）...')
                continue

            # thinking-only重试耗尽且无内容：生成兜底回复
            if current_thinking and not current_content:
                _fallback_parts = ['⚠️ **模型响应异常**：模型思考后未能生成有效回复。\n']
                for _m in reversed(messages):
                    if _m.get('role') == 'tool':
                        _tr = _m['content']
                        if '❌' in _tr or '错误' in _tr or '失败' in _tr:
                            _fallback_parts.append(f'\n工具执行结果：{_tr}')
                        else:
                            _fallback_parts.append(f'\n工具已执行完成，返回：{_tr}')
                        break
                _fallback_parts.append('\n\n请尝试换一种方式描述您的需求，或稍后重试。')
                current_content = '\n'.join(_fallback_parts)

            # 跨轮总计统计
            _total_think_chunks += _think_chunks
            _total_stream_chunks += _stream_chunks
            _agent_elapsed = time.time() - _agent_start

            if _total_think_chunks > 0 or _total_stream_chunks > 0:
                think_elapsed = time.time() - _think_start
                stream_elapsed = time.time() - _stream_start
                think_speed = _think_chunks / think_elapsed if think_elapsed > 0 else 0
                stream_speed = _stream_chunks / stream_elapsed if stream_elapsed > 0 else 0
                total_tokens = _total_think_chunks + _total_stream_chunks
                
                stats_lines = ["\n\n---"]
                if _total_think_chunks > 0:
                    stats_lines.append(f"🧠 思考: **{think_speed:.1f} tokens/s** | {_total_think_chunks} tokens")
                if _total_stream_chunks > 0:
                    stats_lines.append(f"💬 回答: **{stream_speed:.1f} tokens/s** | {_total_stream_chunks} tokens")
                stats_lines.append(f"⚡ 总计: **{total_tokens} tokens** | 🛠️ {_total_tool_calls} 次工具调用 | ⏱️ {_agent_elapsed:.1f}s")
                final_stats = '\n'.join(stats_lines)
                
                current_content += final_stats
                # 如果有编辑后的图片，嵌入图片预览
                if _last_edited_image and os.path.exists(_last_edited_image):
                    current_content += '\n\n---\n\n**🖼️ 编辑后的图片：**\n\n' + _embed_image_markdown(_last_edited_image)
                if current_thinking:
                    thinking_display = _fmt_thinking(current_thinking)
                    status = f"**🧠 思考：** {thinking_display}"
                    yield _build_display(current_status=status, answer_text=current_content)
                else:
                    yield _build_display(answer_text=current_content)
            return

    # MAX_STEPS耗尽：尝试从最近工具结果生成有意义的回复
    _agent_elapsed = time.time() - _agent_start
    _fallback_text = ''
    for _m in reversed(messages):
        if _m.get('role') == 'tool':
            _tr = _m['content']
            if '❌' in _tr or '错误' in _tr or '失败' in _tr:
                _fallback_text = f'⚠️ 工具执行遇到问题：{_tr}\n\n请尝试换一种方式描述您的需求。'
            else:
                _fallback_text = f'工具执行完成，结果如下：{_tr}'
            break
    if not _fallback_text:
        _fallback_text = f"⚠️ 已达最大推理轮次（{MAX_STEPS}轮），无法得出最终答案。\n🛠️ 共调用工具 {_total_tool_calls} 次 | ⏱️ 耗时 {_agent_elapsed:.1f}s"
    # 如果有编辑后的图片，嵌入图片预览
    if _last_edited_image and os.path.exists(_last_edited_image):
        _fallback_text += '\n\n---\n\n**🖼️ 编辑后的图片：**\n\n' + _embed_image_markdown(_last_edited_image)
    yield _build_display(answer_text=_fallback_text)


# ==================== 反馈处理 ====================

# 全局状态：存储最后一轮对话用于反馈处理
_last_turn = {"user": "", "assistant": ""}
# 存储最后一次编辑后的图片路径
_last_edited_image = ""


def _embed_image_markdown(image_path: str) -> str:
    """将图片文件转为 base64 并生成 markdown 图片标签"""
    import base64
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{data}" style="max-width:512px;max-height:512px;border-radius:8px;margin:8px 0;" />'
    except Exception as e:
        return f"⚠️ 图片加载失败: {e}"


def copy_image_to_clipboard():
    """将编辑后的图片复制到剪贴板"""
    global _last_edited_image
    if not _last_edited_image or not os.path.exists(_last_edited_image):
        return gr.update(value="⚠️ 没有可复制的图片", visible=True)
    try:
        subprocess.run([
            'osascript', '-e',
            f'set the clipboard to (read (POSIX file "{_last_edited_image}") as «class PNGf")'
        ], check=True)
        return gr.update(value="✅ 已复制到剪贴板", visible=True)
    except Exception as e:
        return gr.update(value=f"❌ 复制失败: {e}", visible=True)


def open_image_in_finder():
    """在 Finder 中显示编辑后的图片"""
    global _last_edited_image
    if not _last_edited_image or not os.path.exists(_last_edited_image):
        return gr.update(value="⚠️ 没有可显示的图片", visible=True)
    try:
        subprocess.run(['open', '-R', _last_edited_image], check=True)
        return gr.update(value=f"✅ 已在 Finder 中显示", visible=True)
    except Exception as e:
        return gr.update(value=f"❌ 打开失败: {e}", visible=True)


def process_feedback(message, history, feedback_value):
    """处理用户对回答的反馈"""
    if not _last_turn["user"]:
        return

    if feedback_value == "like":
        # 提取经验并保存
        conversation = [
            {'role': 'user', 'content': _last_turn["user"]},
            {'role': 'assistant', 'content': _last_turn["assistant"][:500]}
        ]
        experiences = extract_experience(conversation)
        saved = 0
        for exp in experiences:
            add_memory(
                content=exp.get('content', ''),
                category=exp.get('category', 'general'),
                importance=exp.get('importance', 5)
            )
            saved += 1
        if saved == 0:
            add_memory(
                content=f"用户认可了关于「{_last_turn['user'][:30]}」的回答",
                category='experience',
                importance=5
            )
            saved = 1
        return f"已记住 {saved} 条经验"

    elif feedback_value == "dislike":
        add_memory(
            content=f"用户对「{_last_turn['user'][:30]}」的回答不满意，需要改进",
            category='experience',
            importance=6
        )
        return "已记录，下次会改进"

    return None


def handle_correction(message, history, correction_text):
    """处理用户的纠正输入"""
    if not correction_text or not _last_turn["user"]:
        return "请输入纠正内容"

    add_memory(
        content=f"用户纠正: {correction_text} (原问题: {_last_turn['user'][:50]})",
        category='correction',
        importance=9
    )
    return f"已记住纠正内容"


# ==================== 知识库管理 ====================

def get_knowledge_base_status():
    """获取知识库状态信息"""
    try:
        _, col = init_chroma_collection()
        total = col.count()
        if total == 0:
            return "知识库为空（0 条记录）"
        all_data = col.get(include=['metadatas', 'documents'])
        sources = {}
        for meta, doc in zip(all_data['metadatas'], all_data['documents']):
            src = meta.get('source', '未知')
            if src not in sources:
                sources[src] = {'count': 0, 'chars': 0}
            sources[src]['count'] += 1
            sources[src]['chars'] += len(doc)
        lines = [f"共 {total} 条记录 | {len(sources)} 个文档"]
        for src, info in sources.items():
            lines.append(f"  📄 {src}: {info['count']} 块, {info['chars']:,} 字符")
        return '\n'.join(lines)
    except Exception as e:
        return f"获取知识库状态失败: {e}"


def rebuild_knowledge_base():
    """重建知识库：清空旧数据并重新索引 docs/ 目录"""
    try:
        _, col = init_chroma_collection()
        existing = col.get()
        if existing['ids']:
            col.delete(ids=existing['ids'])
        build_knowledge_base()
        return f"✅ 知识库重建完成！\n\n{get_knowledge_base_status()}"
    except Exception as e:
        return f"❌ 重建失败: {e}"


def get_kb_document_list():
    """获取知识库中的文档列表，返回 [(文件名, 显示名)] 格式"""
    try:
        _, col = init_chroma_collection()
        if col.count() == 0:
            return []
        all_data = col.get(include=['metadatas', 'documents'])
        sources = {}
        for meta, doc in zip(all_data['metadatas'], all_data['documents']):
            src = meta.get('source', '未知')
            if src not in sources:
                sources[src] = {'count': 0, 'chars': 0}
            sources[src]['count'] += 1
            sources[src]['chars'] += len(doc)
        return [(f"{src} ({info['count']}块, {info['chars']:,}字符)", src) for src, info in sorted(sources.items())]
    except Exception:
        return []


def get_document_chunks(doc_name: str):
    """获取指定文档的所有文本块详情"""
    if not doc_name:
        return "请先选择一个文档"
    try:
        _, col = init_chroma_collection()
        all_data = col.get(include=['documents', 'metadatas'])
        chunks = []
        for doc, meta in zip(all_data['documents'], all_data['metadatas']):
            if meta.get('source') == doc_name:
                chunks.append(doc)
        if not chunks:
            return f"未找到文档 '{doc_name}' 的分块记录"
        lines = [f"=== {doc_name} 共 {len(chunks)} 个文本块 ===\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"【块 {i}】({len(chunk)} 字符)")
            lines.append(chunk)
            lines.append("")  # 空行分隔
        return "\n".join(lines)
    except Exception as e:
        return f"获取分块失败: {e}"


# ==================== 记忆面板 ====================

def get_memory_display():
    """获取记忆展示文本"""
    memories = get_all_memories()
    if not memories:
        return "暂无记忆"
    lines = []
    categories = {}
    for m in memories:
        cat = m['category']
        categories[cat] = categories.get(cat, 0) + 1
        lines.append(f"[{cat}] {m['content'][:80]}")
    header = f"共 {len(memories)} 条记忆 | " + " | ".join(f"{k}: {v}" for k, v in categories.items())
    return header + "\n\n" + "\n".join(lines)


def save_memories_from_text(text: str) -> str:
    """从文本解析记忆并重建记忆库"""
    import re as _re
    from memory_system import get_memory_collection

    lines = text.strip().split('\n')
    entries = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('共 ') or line.startswith('暂无'):
            continue
        # 解析格式: [category] content
        match = _re.match(r'^\[(\w+)\]\s*(.+)$', line)
        if match:
            entries.append({'category': match.group(1), 'content': match.group(2)})
        elif line:
            entries.append({'category': 'general', 'content': line})

    if not entries:
        return "未解析到有效记忆条目"

    # 清空旧记忆并重建
    collection = get_memory_collection()
    all_ids = collection.get()['ids']
    if all_ids:
        collection.delete(ids=all_ids)

    for entry in entries:
        add_memory(
            content=entry['content'],
            category=entry['category'],
            importance=5
        )

    return f"已保存 {len(entries)} 条记忆"


# ==================== 视觉模型（图片理解）====================

TEXT_MODEL = 'qwen3:8b'
VISION_MODEL = 'qwen2.5vl:7b'

def run_vision_stream(text: str, image_paths: list):
    """调用视觉模型处理图片+文字，流式返回结果"""
    # 将图片转为 base64
    images_b64 = []
    for img_path in image_paths:
        try:
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            images_b64.append(img_data)
        except Exception as e:
            yield f"❌ 无法读取图片 {img_path}: {e}"
            return

    if not images_b64:
        yield "❌ 未找到有效图片"
        return

    # 构造视觉模型消息（支持多图）
    messages = [{
        'role': 'user',
        'content': text or '请描述这张图片的内容',
        'images': images_b64
    }]

    result = f"🤖 **{VISION_MODEL}**\n\n"
    try:
        stream = ollama_local.chat(
            model=VISION_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            msg = chunk.get('message', {})
            content = msg.get('content', '')
            if content:
                result += content
                yield result
    except Exception as e:
        yield f"❌ 视觉模型调用失败: {e}\n\n请确认已下载模型: ollama pull {VISION_MODEL}"


def describe_image(image_path: str) -> str:
    """使用视觉模型描述图片内容（非流式，返回完整描述）"""
    try:
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = ollama_local.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': '请详细描述这张图片的内容，包括：图片中有什么物体、人物、场景、文字、颜色等。描述要尽量详细，以便后续进行图片编辑操作。',
                'images': [img_data]
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"[图片描述失败: {e}]"


# ==================== 聊天函数 ====================

def chat_fn(message, history):
    """聊天主函数 - 支持多模态（文字+图片）"""
    # 多模态模式：message 是字典 {"text": "...", "files": ["路径1", ...]}
    if isinstance(message, dict):
        text = message.get('text', '')
        files = message.get('files', [])
    else:
        text = str(message)
        files = []

    _last_turn["user"] = text

    # 有图片时：先用视觉模型描述图片，然后交给 Agent 处理
    if files:
        # 将图片描述和文件路径注入到用户消息中
        image_descriptions = []
        image_paths = []
        for img_path in files:
            desc = describe_image(img_path)
            image_descriptions.append(f"图片文件: {img_path}\n内容描述: {desc}")
            image_paths.append(img_path)
                
        img_context = "\n\n".join(image_descriptions)
        paths_str = ", ".join(image_paths)
        enriched_text = f"[用户上传了图片，文件路径: {paths_str}]\n\n[图片内容描述]\n{img_context}\n\n[用户的请求]\n{text}\n\n[提示: 如需编辑图片，请使用 image_edit 工具，image_path 参数使用上述图片文件路径]"
        
        # 交给 Agent 流程处理（Agent 可调用 image_edit 等工具）
        response_text = ""
        first_yield = True
        for partial in run_agent_stream(enriched_text, history):
            if first_yield:
                partial = f"🤖 **{TEXT_MODEL}**\n\n{partial}"
                first_yield = False
            response_text = partial
            yield partial
        # 提取纯回答存入记忆
        if '\n---\n\n' in response_text:
            clean_answer = response_text.split('\n---\n\n')[-1]
        else:
            clean_answer = response_text
        if '\n\n---\n' in clean_answer:
            clean_answer = clean_answer.split('\n\n---\n')[0]
        if clean_answer.startswith('🤖 **'):
            parts = clean_answer.split('\n\n', 1)
            if len(parts) > 1:
                clean_answer = parts[1]
        _last_turn["assistant"] = clean_answer
        return

    # 纯文字 → 走 Agent 流程
    response_text = ""
    first_yield = True
    for partial in run_agent_stream(text, history):
        if first_yield:
            # 在第一次输出前添加模型标签
            partial = f"🤖 **{TEXT_MODEL}**\n\n{partial}"
            first_yield = False
        response_text = partial
        yield partial
    # 提取纯回答（去掉思考/工具调用前缀 + 速度统计后缀）
    if '\n---\n\n' in response_text:
        clean_answer = response_text.split('\n---\n\n')[-1]
    else:
        clean_answer = response_text
    # 去掉末尾的速度统计块（🧠思考/💬回答/⚡总计）
    if '\n\n---\n' in clean_answer:
        clean_answer = clean_answer.split('\n\n---\n')[0]
    # 去掉开头的模型标签（🤖 **qwen3:8b**\n\n）
    if clean_answer.startswith('🤖 **'):
        # 找到第一个双换行之后的内容
        parts = clean_answer.split('\n\n', 1)
        if len(parts) > 1:
            clean_answer = parts[1]
    _last_turn["assistant"] = clean_answer


def create_app():
    with gr.Blocks(
        title="自我进化助手",
    ) as app:
        with gr.Row():
            # ===== 左侧导航栏 =====
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("<div class='nav-header'>🤖 自我进化助手</div>")
                nav_chat_btn = gr.Button("💬 聊天", elem_id="nav-chat", elem_classes=["sidebar-btn", "active"])
                nav_memory_btn = gr.Button("🧠 记忆管理", elem_id="nav-memory", elem_classes=["sidebar-btn"])
                nav_kb_btn = gr.Button("📚 知识库", elem_id="nav-kb", elem_classes=["sidebar-btn"])
                gr.Markdown("<hr><div style='font-size:12px;color:#888;padding:8px 0;'>Agent + RAG + 记忆系统<br>每次对话都会学习和进化</div>")

            # ===== 右侧内容区 =====
            with gr.Column(scale=5):

                # --- Tab 1: 聊天 ---
                with gr.Group(visible=True) as chat_panel:
                    chatbot = gr.ChatInterface(
                        fn=chat_fn,
                        multimodal=True,
                        examples=[
                            {"text": "根据需求文档，模型平台有哪些核心问题？"},
                            {"text": "帮我算一下 2048 * 512"},
                            {"text": "你还记得我的背景吗？"},
                        ],
                        cache_examples=False,
                    )
                    # 反馈区域
                    with gr.Group():
                        gr.Markdown("### 对上一条回答的反馈")
                        with gr.Row():
                            like_btn = gr.Button("👍 满意", variant="secondary")
                            dislike_btn = gr.Button("👎 不满意", variant="secondary")
                        correction_input = gr.Textbox(
                            label="纠正（如果回答有误，请输入正确内容）",
                            placeholder="输入纠正内容...",
                            lines=2
                        )
                        correction_btn = gr.Button("提交纠正", variant="primary")
                        feedback_status = gr.Textbox(label="反馈状态", interactive=False)
                        clear_feedback = gr.Button("清除反馈状态", variant="secondary")

                    # 图片编辑操作区域
                    with gr.Group(visible=True) as image_action_panel:
                        with gr.Row():
                            copy_img_btn = gr.Button("📋 复制图片到剪贴板", variant="secondary", size="sm")
                            open_dir_btn = gr.Button("📂 在 Finder 中显示", variant="secondary", size="sm")
                        image_action_status = gr.Textbox(label="操作状态", interactive=False, visible=False)

                # --- Tab 2: 记忆管理 ---
                with gr.Group(visible=False) as memory_panel:
                    gr.Markdown("### 🧠 记忆状态管理")
                    memory_display = gr.Textbox(
                        value=get_memory_display,
                        label="当前记忆（可编辑后保存）",
                        lines=20,
                        interactive=True,
                        elem_classes=["memory-panel"]
                    )
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新记忆")
                        save_btn = gr.Button("💾 保存记忆", variant="primary")
                    memory_status = gr.Textbox(label="操作状态", interactive=False)

                # --- Tab 3: 知识库管理 ---
                with gr.Group(visible=False) as kb_panel:
                    gr.Markdown("### 📚 知识库管理")
                    kb_status = gr.Textbox(
                        value=get_knowledge_base_status,
                        label="知识库状态",
                        lines=8,
                        interactive=False,
                        elem_classes=["memory-panel"]
                    )
                    with gr.Row():
                        kb_refresh_btn = gr.Button("🔄 刷新状态")
                        kb_rebuild_btn = gr.Button("🔨 重建知识库", variant="primary")
                    kb_rebuild_status = gr.Textbox(label="重建状态", interactive=False, lines=3)

                    gr.Markdown("---")
                    gr.Markdown("### 🔍 查看文档分块详情")
                    kb_doc_dropdown = gr.Dropdown(
                        choices=get_kb_document_list(),
                        label="选择文档",
                        value=None,
                        interactive=True
                    )
                    with gr.Row():
                        kb_view_chunks_btn = gr.Button("📖 查看分块", variant="secondary")
                    kb_chunks_display = gr.Textbox(
                        label="分块详情",
                        lines=20,
                        interactive=False,
                        elem_classes=["memory-panel"],
                        placeholder="选择一个文档后点击「查看分块」按钮"
                    )

        # ===== 导航切换 =====
        ACTIVE = ["sidebar-btn", "active"]
        INACTIVE = ["sidebar-btn"]

        def switch_to_chat():
            return (
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(elem_classes=ACTIVE), gr.update(elem_classes=INACTIVE), gr.update(elem_classes=INACTIVE)
            )

        def switch_to_memory():
            return (
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(elem_classes=INACTIVE), gr.update(elem_classes=ACTIVE), gr.update(elem_classes=INACTIVE)
            )

        def switch_to_kb():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(elem_classes=INACTIVE), gr.update(elem_classes=INACTIVE), gr.update(elem_classes=ACTIVE)
            )

        nav_outputs = [chat_panel, memory_panel, kb_panel, nav_chat_btn, nav_memory_btn, nav_kb_btn]
        nav_chat_btn.click(fn=switch_to_chat, outputs=nav_outputs)
        nav_memory_btn.click(fn=switch_to_memory, outputs=nav_outputs)
        nav_kb_btn.click(fn=switch_to_kb, outputs=nav_outputs)

        # ===== 聊天反馈事件 =====
        like_btn.click(
            fn=lambda msg, hist: process_feedback(msg, hist, "like"),
            inputs=[chatbot.textbox, chatbot.chatbot_value],
            outputs=[feedback_status]
        )
        dislike_btn.click(
            fn=lambda msg, hist: process_feedback(msg, hist, "dislike"),
            inputs=[chatbot.textbox, chatbot.chatbot_value],
            outputs=[feedback_status]
        )
        correction_btn.click(
            fn=handle_correction,
            inputs=[chatbot.textbox, chatbot.chatbot_value, correction_input],
            outputs=[feedback_status]
        )
        clear_feedback.click(
            fn=lambda: "",
            outputs=[feedback_status]
        )

        # ===== 图片编辑操作事件 =====
        copy_img_btn.click(
            fn=copy_image_to_clipboard,
            outputs=[image_action_status]
        )
        open_dir_btn.click(
            fn=open_image_in_finder,
            outputs=[image_action_status]
        )

        # ===== 记忆管理事件 =====
        refresh_btn.click(
            fn=get_memory_display,
            outputs=[memory_display]
        )
        save_btn.click(
            fn=save_memories_from_text,
            inputs=[memory_display],
            outputs=[memory_status]
        )

        # ===== 知识库管理事件 =====
        kb_refresh_btn.click(
            fn=get_knowledge_base_status,
            outputs=[kb_status]
        )
        kb_rebuild_btn.click(
            fn=rebuild_knowledge_base,
            outputs=[kb_rebuild_status]
        )
        kb_view_chunks_btn.click(
            fn=get_document_chunks,
            inputs=[kb_doc_dropdown],
            outputs=[kb_chunks_display]
        )

    return app


# ==================== 启动 ====================

if __name__ == '__main__':
    _ensure_docker_ready()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css="""
        .feedback-row { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
        .memory-panel { font-size: 13px; }
        .sidebar-btn { 
            width: 100%; 
            text-align: left; 
            justify-content: flex-start; 
            font-size: 15px;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 4px;
            transition: all 0.2s ease;
        }
        .sidebar-btn.active { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
        }
        .sidebar-btn:not(.active) {
            background: #f5f5f5 !important;
            color: #333 !important;
            border: 1px solid #e0e0e0 !important;
        }
        .sidebar-btn:not(.active):hover {
            background: #e8e8e8 !important;
        }
        .nav-header { 
            font-size: 18px; 
            font-weight: bold; 
            padding: 8px 0 16px 0;
            border-bottom: 1px solid #e5e5e5;
            margin-bottom: 12px;
        }
        /* 右侧区域所有按钮默认深灰色 */
        .gradio-col button:not(#nav-chat):not(#nav-memory):not(#nav-kb),
        .gradio-col button:not(#nav-chat):not(#nav-memory):not(#nav-kb) > span {
            background: #4a4a4a !important;
            color: #fff !important;
            border: none !important;
        }
        .gradio-col button:not(#nav-chat):not(#nav-memory):not(#nav-kb):hover {
            background: #333 !important;
        }
        """
    )
