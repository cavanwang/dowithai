"""
从 Beem Docs MHTML 提取文档正文，生成彻底清洗后的打印友好 HTML。
去除所有 SPA 内联样式、content-visibility、虚拟滚动等干扰打印的属性。
"""

import quopri
import re
from html.parser import HTMLParser

INPUT_FILE = "/Users/cavan/ai_learn/xuqiu.mhtml"
OUTPUT_FILE = "/Users/cavan/ai_learn/xuqiu_printable.html"


def extract_html_from_mhtml(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    html_start = content.find("Content-Type: text/html")
    body_start = content.find("\n\n", html_start) + 2
    boundary = content.find("------MultipartBoundary", body_start)
    html_raw = content[body_start:boundary] if boundary != -1 else content[body_start:]
    return quopri.decodestring(html_raw.encode("utf-8")).decode("utf-8", errors="replace")


class ContentCleaner(HTMLParser):
    """清洗 HTML：保留语义结构，去除所有 SPA 样式干扰"""

    SKIP_TAGS = {"script", "style", "svg", "button"}
    SKIP_CLASSES_WITH_CONTENT = {
        "react-renderer", "drag-handle-root", "table-menu-mount",
        "table-shadow-right", "table-shadow-left", "table-add-button-mount",
        "table-scrollbar", "codeblock-actions", "codeblock-line-gutter",
    }

    def __init__(self):
        super().__init__()
        self.output = []
        self.skip_stack = []  # 跟踪需要跳过的标签层级
        self.tag_stack = []

    def _should_skip(self):
        return len(self.skip_stack) > 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        # 检查是否需要跳过整个子树
        for skip_cls in self.SKIP_CLASSES_WITH_CONTENT:
            if skip_cls in cls:
                self.skip_stack.append(tag)
                return

        if tag in self.SKIP_TAGS:
            self.skip_stack.append(tag)
            return

        self.tag_stack.append(tag)

        # 简化标签
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.output.append(f"<{tag}>")
        elif tag == "p":
            self.output.append("<p>")
        elif tag == "br":
            self.output.append("<br>\n")
        elif tag == "pre":
            self.output.append("<pre>")
        elif tag == "code":
            self.output.append("<code>")
        elif tag == "strong" or tag == "b":
            self.output.append("<strong>")
        elif tag == "em" or tag == "i":
            self.output.append("<em>")
        elif tag == "a":
            href = attrs_dict.get("href", "")
            self.output.append(f'<a href="{href}">')
        elif tag == "table":
            self.output.append("<table>")
        elif tag == "thead":
            self.output.append("<thead>")
        elif tag == "tbody":
            self.output.append("<tbody>")
        elif tag == "tr":
            self.output.append("<tr>")
        elif tag == "th":
            self.output.append("<th>")
        elif tag == "td":
            self.output.append("<td>")
        elif tag == "ul":
            self.output.append("<ul>")
        elif tag == "ol":
            self.output.append("<ol>")
        elif tag == "li":
            self.output.append("<li>")
        elif tag == "div":
            # 将特定 div 转换为语义标签
            if "advanced-list-item" in cls and "marker" not in cls:
                if "list-type-bullet" in cls:
                    # 无序列表项 - 只输出内容，li 标签由外层处理
                    pass
                elif "list-type-ordered" in cls:
                    pass
            elif "marker-bullet" in cls or "advanced-list-item-marker" in cls:
                self.skip_stack.append(tag)  # 跳过列表标记
            elif "tableContainer" in cls or "tableWrapper" in cls:
                pass  # 表格容器，直接透传内部 table
            elif "codeblock-body" in cls:
                self.output.append("<pre>")
            elif "codeblock-line-gutter" in cls:
                self.skip_stack.append(tag)
            else:
                self.output.append("<div>")
        elif tag == "colgroup" or tag == "col":
            pass  # 跳过 colgroup

    def handle_endtag(self, tag):
        # 检查是否在跳过状态
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
            return

        if self._should_skip():
            return

        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "code",
                    "strong", "b", "em", "i", "a", "table", "thead", "tbody",
                    "tr", "th", "td", "ul", "ol", "li"):
            self.output.append(f"</{tag}>")
        elif tag == "div":
            self.output.append("</div>")
        elif tag == "br":
            pass  # br 是自闭合

    def handle_data(self, data):
        if not self._should_skip():
            self.output.append(data)

    def get_result(self):
        return "".join(self.output)


def clean_html(raw_html):
    """清洗 HTML 内容"""
    cleaner = ContentCleaner()
    cleaner.feed(raw_html)
    result = cleaner.get_result()

    # 后处理：清理多余空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    # 清理空 div
    result = re.sub(r'<div>\s*</div>', '', result)
    return result.strip()


def build_printable_html(title, body_html):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 860px;
    margin: 0 auto;
    padding: 40px 24px;
    font-size: 15px;
  }}
  h1 {{ font-size: 26px; margin: 32px 0 12px; }}
  h2 {{ font-size: 21px; margin: 28px 0 10px; border-bottom: 1px solid #e5e5e5; padding-bottom: 6px; }}
  h3 {{ font-size: 17px; margin: 22px 0 8px; }}
  h4 {{ font-size: 15px; margin: 18px 0 6px; font-weight: 600; }}
  p {{ margin: 6px 0; }}
  a {{ color: #0969da; text-decoration: none; word-break: break-all; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d0d0; padding: 7px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  pre {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4px; padding: 10px 14px; overflow-x: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
  code {{ font-family: Menlo, Consolas, monospace; font-size: 0.88em; }}
  :not(pre) > code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
  ul, ol {{ padding-left: 22px; margin: 6px 0; }}
  li {{ margin: 3px 0; }}

  @media print {{
    body {{ padding: 0; max-width: 100%; font-size: 12px; }}
    h2, h3 {{ page-break-after: avoid; }}
    table, pre {{ page-break-inside: avoid; }}
    @page {{ margin: 12mm; }}
  }}
</style>
</head>
<body>
<h1>{title}</h1>
{body_html}
</body>
</html>"""


def main():
    print("📖 正在解析 MHTML...")
    html = extract_html_from_mhtml(INPUT_FILE)
    print(f"   HTML 大小: {len(html):,} 字符")

    # 提取标题
    title_match = re.search(r'data-type="title"[^>]*><p[^>]*>(.*?)</p>', html, re.DOTALL)
    title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else "模型平台 v1技术需求文档"
    print(f"   标题: {title}")

    # 提取正文
    content_match = re.search(
        r'class="main-editor-content[^"]*"[^>]*><div[^>]*class="tiptap ProseMirror[^"]*"[^>]*>(.*?)</div>\s*<div[^>]*(?:bubble-menu|drag-handle)',
        html, re.DOTALL
    )
    if not content_match:
        print("❌ 未能提取到文档正文")
        return

    raw_body = content_match.group(1)
    print(f"   原始正文: {len(raw_body):,} 字符")

    # 清洗
    print("🧹 正在清洗 SPA 样式...")
    clean_body = clean_html(raw_body)
    print(f"   清洗后: {len(clean_body):,} 字符")

    # 生成
    output = build_printable_html(title, clean_body)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ 已生成: {OUTPUT_FILE}")
    print("📄 请在浏览器中打开，Cmd+P 打印")


if __name__ == "__main__":
    main()
