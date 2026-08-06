"""
通过 Chrome CDP：在已加载的 MHTML 页面中提取文档内容，
写入一个全新的空白页面，然后导出 PDF。
这样完全避开 SPA 布局问题。
"""

import subprocess
import json
import time
import http.client
import base64
import tempfile
import shutil

MHTML_FILE = "/Users/cavan/ai_learn/xuqiu.mhtml"
OUTPUT_PDF = "/Users/cavan/ai_learn/xuqiu_output.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEBUG_PORT = 9224


def main():
    tmp_profile = tempfile.mkdtemp(prefix="chrome_headless_")

    print("🚀 启动 Chrome headless...")
    proc = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu",
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={tmp_profile}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-extensions", "--remote-allow-origins=*",
        f"file://{MHTML_FILE}"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(5)

    try:
        print("📡 连接 CDP...")
        conn = http.client.HTTPConnection("localhost", DEBUG_PORT)
        conn.request("GET", "/json")
        targets = json.loads(conn.getresponse().read())
        conn.close()

        target = None
        for t in targets:
            if "mhtml" in t.get("url", "").lower() or "模型" in t.get("title", ""):
                target = t
                break
        if not target:
            target = targets[0]

        import websocket
        ws = websocket.create_connection(target["webSocketDebuggerUrl"])
        msg_id = 0

        def send_cmd(method, params=None):
            nonlocal msg_id
            msg_id += 1
            cmd = {"id": msg_id, "method": method}
            if params:
                cmd["params"] = params
            ws.send(json.dumps(cmd))
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == msg_id:
                    return r

        send_cmd("Page.enable")
        send_cmd("Runtime.enable")
        time.sleep(2)

        # 1. 先检查原始页面内容
        check = send_cmd("Runtime.evaluate", {
            "expression": "document.body.innerText.length",
            "returnByValue": True
        })
        orig_chars = check.get("result", {}).get("result", {}).get("value", 0)
        print(f"   原始页面可见字符: {orig_chars:,}")

        # 2. 提取文档正文 innerHTML（在 SPA 渲染正常的状态下提取）
        print("📝 提取文档正文...")
        extract_js = """
        (function() {
            // 提取标题
            var titleEl = document.querySelector('[data-type="title"] p');
            var title = titleEl ? titleEl.innerText : '未知标题';

            // 提取正文 - 从 main-editor-content 获取
            var mainContent = document.querySelector('.main-editor-content .tiptap.ProseMirror');
            if (!mainContent) {
                return JSON.stringify({error: 'main-editor-content not found'});
            }

            // 获取纯文本内容（保留换行）
            var text = mainContent.innerText;

            return JSON.stringify({
                title: title,
                textLength: text.length,
                text: text
            });
        })()
        """
        result = send_cmd("Runtime.evaluate", {
            "expression": extract_js,
            "returnByValue": True
        })
        raw = result.get("result", {}).get("result", {}).get("value", "")
        data = json.loads(raw)

        if "error" in data:
            print(f"❌ 提取失败: {data['error']}")
            return

        title = data["title"]
        text = data["text"]
        print(f"   标题: {title}")
        print(f"   正文长度: {len(text):,} 字符")

        if len(text) < 100:
            print("❌ 提取内容太少，可能提取失败")
            return

        # 3. 转义文本用于 JS
        escaped_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        # 4. 导航到空白页，写入纯净 HTML
        print("📄 创建纯净打印页面...")
        send_cmd("Page.navigate", {"url": "about:blank"})
        time.sleep(1)

        # 将长文本分块写入（避免 JS 字符串长度限制）
        chunk_size = 50000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # 构建 HTML
        html_parts = []
        html_parts.append(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.7; color: #1a1a1a; max-width: 800px; margin: 0 auto; padding: 30px 20px; font-size: 14px; }}
h1 {{ font-size: 24px; margin: 24px 0 10px; }}
h2 {{ font-size: 20px; margin: 22px 0 8px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
h3 {{ font-size: 17px; margin: 18px 0 6px; }}
h4 {{ font-size: 15px; margin: 14px 0 4px; }}
p {{ margin: 5px 0; }}
pre {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4px; padding: 10px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }}
code {{ font-family: Menlo, Consolas, monospace; font-size: 0.88em; background: #f0f0f0; padding: 1px 3px; border-radius: 2px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
ul, ol {{ padding-left: 20px; }}
a {{ color: #0969da; }}
@media print {{ body {{ padding: 0; max-width: 100%; }} @page {{ margin: 12mm; }} }}
</style></head><body>""")

        # 将文本内容按行处理，识别标题和段落
        lines = text.split("\n")
        body_lines = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 简单的启发式：识别标题
            if stripped.startswith("## ") or stripped.startswith("# "):
                level = stripped.count("#")
                heading_text = stripped.lstrip("# ").strip()
                escaped = heading_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                body_lines.append(f"<h{level}>{escaped}</h{level}>")
            elif stripped.startswith("### "):
                escaped = stripped[4:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                body_lines.append(f"<h3>{escaped}</h3>")
            elif stripped.startswith("#### "):
                escaped = stripped[5:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                body_lines.append(f"<h4>{escaped}</h4>")
            else:
                escaped = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # 处理 `code` 格式
                import re
                escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
                body_lines.append(f"<p>{escaped}</p>")

        html_parts.append("\n".join(body_lines))
        html_parts.append("</body></html>")

        full_html = "\n".join(html_parts)
        escaped_html = full_html.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        # 通过 JS 写入
        write_js = f"""
        document.open();
        document.write(`{escaped_html}`);
        document.close();
        document.body.innerText.length
        """
        check = send_cmd("Runtime.evaluate", {"expression": write_js, "returnByValue": True})
        chars = check.get("result", {}).get("result", {}).get("value", 0)
        print(f"   新页面字符数: {chars:,}")

        time.sleep(1)

        # 5. 导出 PDF
        print("📄 导出 PDF...")
        pdf_result = send_cmd("Page.printToPDF", {
            "landscape": False,
            "displayHeaderFooter": False,
            "printBackground": False,
            "paperWidth": 8.27,
            "paperHeight": 11.69,
            "marginTop": 0.5,
            "marginBottom": 0.5,
            "marginLeft": 0.5,
            "marginRight": 0.5,
        })

        if "result" in pdf_result and "data" in pdf_result["result"]:
            pdf_data = base64.b64decode(pdf_result["result"]["data"])
            with open(OUTPUT_PDF, "wb") as f:
                f.write(pdf_data)
            print(f"✅ PDF 已生成: {OUTPUT_PDF} ({len(pdf_data):,} bytes)")
        else:
            print(f"❌ 导出失败: {json.dumps(pdf_result)[:200]}")

        ws.close()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        proc.terminate()
        proc.wait()
        shutil.rmtree(tmp_profile, ignore_errors=True)


if __name__ == "__main__":
    main()
