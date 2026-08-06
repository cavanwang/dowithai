import email
import base64
import re

with open('xuqiu.mhtml', 'rb') as f:
    msg = email.message_from_binary_file(f)

html_content = ""
resources = {}  # {content-id: data-uri}

for part in msg.walk():
    content_type = part.get_content_type()
    content_id = part.get('Content-Location', '')
    
    if content_type == 'text/html':
        charset = part.get_content_charset() or 'utf-8'
        html_content = part.get_payload(decode=True).decode(charset)
    elif content_type.startswith('image/') or content_type in ('text/css',):
        data = part.get_payload(decode=True)
        b64 = base64.b64encode(data).decode()
        resources[content_id] = f"data:{content_type};base64,{b64}"

# 把外部引用替换为内联data URI
for url, data_uri in resources.items():
    html_content = html_content.replace(url, data_uri)

with open('xuqiu.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("完成")
