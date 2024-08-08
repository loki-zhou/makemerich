import requests
import html2text
import random
import time
from fake_useragent import UserAgent


def url_to_markdown(url, output_file=None):
    ua = UserAgent()

    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    time.sleep(random.uniform(1, 3))

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        markdown_content = h.handle(response.text)

        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Markdown content has been saved to {output_file}")
            except IOError as e:
                print(f"Error saving file: {e}")

        return markdown_content

    except requests.RequestException as e:
        error_message = f"Error fetching URL: {e}"
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(error_message)
                print(f"Error message has been saved to {output_file}")
            except IOError as e:
                print(f"Error saving file: {e}")
        return error_message


# 使用示例
url = "https://mp.weixin.qq.com/s/Ls1Xwb-54GS4sUBI6Hs9ZQ"  # 替换为您想要转换的URL
output_file = "output.md"  # 指定输出文件名，如果不想保存文件，可以设为None

markdown_result = url_to_markdown(url, output_file)
print(markdown_result)
