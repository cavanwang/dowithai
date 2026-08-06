"""
从 Beem Docs MHTML 文件中提取文档正文，生成打印友好的纯 HTML 文件。
彻底解决 SPA 布局（100vh + overflow:hidden + content-visibility:auto）导致的打印空白问题。
"""

import quopri
import re

INPUT_FILE = "/Users/cavan/ai_learn/xuqiu.mhtml"
OUTPUT_FILE = "/Users/cavan/ai_learn/xuqiu_printable.html"


def extract_html_from_mhtml(filepath):
    """从 MHTML 中提取 HTML 部分并解码 quoted-printable"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 找到 HTML 部分的边界
    # MHTML 格式: 先有 MIME 头，然后是 boundary 分隔的各部分
    # 第一个 text/html 部分就是我们需要的
    html_start = content.find("Content-Type: text/html")
    if html_start == -1:
        raise ValueError("无法在 MHTML 中找到 text/html 部分")

    # 跳过 Content-Type 行，找到空行后的正文
    body_start = content.find("\n\n", html_start)
    if body_start == -1:
        body_start = content.find("\r\n\r\n", html_start)
    body_start += 2  # 跳过空行

    # 找到下一个 boundary（表示这一部分结束）
    boundary = content.find("------MultipartBoundary", body_start)
    if boundary == -1:
        html_raw = content[body_start:]
    else:
        html_raw = content[body_start:boundary]

    # 解码 quoted-printable
    html_decoded = quopri.decodestring(html_raw.encode("utf-8")).decode("utf-8", errors="replace")
    return html_decoded


def extract_document_content(html):
    """从解码后的 HTML 中提取标题和正文内容"""

    # 提取标题 - 从 title-container 内的 data-type="title" 元素中提取
    title_match = re.search(
        r'data-type=3D"title"[^>]*><p[^>]*>(.*?)</p>',
        html, re.DOTALL
    )
    if not title_match:
        # 备用：从解码后的 HTML 中查找
        title_match = re.search(
            r'data-type="title"[^>]*><p[^>]*>(.*?)</p>',
            html, re.DOTALL
        )
    title = ""
    if title_match:
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
    if not title:
        title = "模型平台 v1技术需求文档"  # 从 MHTML 头解析的默认标题

    # 提取正文内容 (main-editor-content 内的 tiptap ProseMirror div)
    content_match = re.search(
        r'class="main-editor-content[^"]*"[^>]*><div[^>]*class="tiptap ProseMirror[^"]*"[^>]*>(.*?)</div><div class="react-renderer bubble-menu"',
        html, re.DOTALL
    )
    if not content_match:
        # 备用：尝试匹配到 drag-handle 或 react-renderer
        content_match = re.search(
            r'class="main-editor-content[^"]*"[^>]*><div[^>]*class="tiptap ProseMirror[^"]*"[^>]*>(.*?)</div>\s*<div[^>]*(?:bubble-menu|drag-handle)',
            html, re.DOTALL
        )

    body_html = ""
    if content_match:
        body_html = content_match.group(1)
        # 清理不需要的 UI 元素
        body_html = re.sub(r'<div class="react-renderer[^"]*"[^>]*>.*?</div>', '', body_html, flags=re.DOTALL)
        body_html = re.sub(r'<div style="pointer-events:\s*none[^>]*>.*?</div>\s*$', '', body_html, flags=re.DOTALL)

    return title, body_html


def build_printable_html(title, body_html):
    """构建打印友好的 HTML 文件"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{
    content-visibility: visible !important;
    contain: none !important;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
  }}

  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  h3 {{ font-size: 18px; margin-top: 24px; }}
  h4 {{ font-size: 16px; margin-top: 20px; }}

  p {{ margin: 8px 0; }}

  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{ background: #f5f5f5; font-weight: 600; }}

  pre {{
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
  }}
  code {{
    font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
    font-size: 0.9em;
  }}
  :not(pre) > code {{
    background: #f0f0f0;
    padding: 2px 5px;
    border-radius: 3px;
  }}

  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  ul, ol {{ padding-left: 24px; }}
  li {{ margin: 4px 0; }}

  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}

  @media print {{
    body {{ padding: 0; max-width: 100%; }}
    pre {{ white-space: pre-wrap; word-break: break-all; }}
    h2, h3 {{ page-break-after: avoid; }}
    table, pre, img {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">提取自 Beem Docs MHTML 文件</p>
{body_html}
</body>
</html>"""


def main():
    print("📖 正在解析 MHTML 文件...")
    html = extract_html_from_mhtml(INPUT_FILE)
    print(f"   HTML 大小: {len(html):,} 字符")

    print("📝 正在提取文档正文...")
    title, body_html = extract_document_content(html)
    print(f"   标题: {title}")
    print(f"   正文大小: {len(body_html):,} 字符")

    if not body_html:
        print("❌ 未能提取到文档正文，请检查文件结构")
        return

    print("🔨 正在生成打印友好 HTML...")
    output = build_printable_html(title, body_html)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ 已生成: {OUTPUT_FILE}")
    print("📄 请在浏览器中打开该文件，然后 Cmd+P 打印")


if __name__ == "__main__":
    main()
