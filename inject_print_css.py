"""
为 MHTML 文件注入 @media print CSS，解除 SPA 布局的高度和 overflow 限制，
使其在浏览器打印时能正常分页显示所有内容。
"""

import re

INPUT_FILE = "/Users/cavan/ai_learn/xuqiu.mhtml"
OUTPUT_FILE = "/Users/cavan/ai_learn/xuqiu_printable.mhtml"

# 打印修复 CSS：解除 SPA 布局限制，让内容自然流分页
PRINT_FIX_CSS = """
<style>
@media print {
  /* 解除最外层 100vh + overflow:hidden */
  .h-100vh,
  #app > div {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
  }

  /* 解除所有 overflow:hidden 容器 */
  .overflow-hidden,
  .flex-col.overflow-hidden,
  div[class*="overflow-hidden"] {
    overflow: visible !important;
    overflow-y: visible !important;
  }

  /* 解除 header 固定高度 */
  .drive-header,
  header[class*="h-64px"] {
    height: auto !important;
    min-height: 0 !important;
  }

  /* 解除 flex-1 区域的 overflow */
  .flex-1.flex.relative.overflow-hidden {
    overflow: visible !important;
    height: auto !important;
    flex: none !important;
  }

  /* 解除 react-container 限制 */
  .react-container,
  .docx-header-offset {
    height: auto !important;
    overflow: visible !important;
  }

  /* 解除 docx-root 内的 scroll-area-viewport 固定高度 */
  .docx-root div[data-slot="scroll-area-viewport"],
  div[data-radix-scroll-area-viewport] {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    overflow-y: visible !important;
  }

  /* 解除 content-wrapper 限制 */
  .docx-root .content-wrapper,
  .content-wrapper[data-slot="scroll-area"] {
    overflow: visible !important;
    overflow-y: visible !important;
    height: auto !important;
  }

  /* 解除 docx-root 本身的限制 */
  .docx-root {
    height: auto !important;
    overflow: visible !important;
  }

  /* 解除内部 scroll 容器的限制 */
  .content-wrapper > div[data-slot="scroll-area-viewport"] > div {
    max-width: 100% !important;
    min-height: auto !important;
  }

  /* 隐藏不需要打印的元素 */
  header.drive-header,
  .table-of-contents,
  .toc-wrapper,
  [data-slot="scroll-area-scrollbar"] {
    display: none !important;
  }

  /* 确保文档内容区域占满宽度 */
  .docx-root .docx-editor-content {
    width: 100% !important;
    margin-inline: 0 !important;
  }

  /* 打印时页面边距 */
  @page {
    margin: 15mm;
  }

  /* 确保 body 和 html 不限制 */
  html, body {
    height: auto !important;
    overflow: visible !important;
  }
}
</style>
"""

def inject_print_css():
    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 在 </head> 前注入 CSS
    # MHTML 中的 HTML 使用 quoted-printable 编码，</head> 可能表示为 </head>
    # 但由于是保存的 MHTML，浏览器通常已经解码了部分内容
    # 尝试多种可能的 </head> 表示方式
    injected = False

    # 方式1: 直接匹配 </head>
    if "</head>" in content:
        content = content.replace("</head>", PRINT_FIX_CSS + "</head>", 1)
        injected = True
    else:
        # 方式2: MHTML quoted-printable 中可能没有编码 </head>
        # 尝试在第一个 CSS link 之前注入（在 <head> 内部）
        head_match = re.search(r"<head[^>]*>", content)
        if head_match:
            insert_pos = head_match.end()
            content = content[:insert_pos] + PRINT_FIX_CSS + content[insert_pos:]
            injected = True

    if not injected:
        print("❌ 无法找到注入 CSS 的位置，请检查文件格式")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ 已生成可打印版本: {OUTPUT_FILE}")
    print("📄 请在浏览器中打开该文件，然后 Ctrl+P 打印")


if __name__ == "__main__":
    inject_print_css()
