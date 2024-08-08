import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import html2text

def url_to_markdown(url, output_file=None, wait_time=5):
    ua = UserAgent()
    user_agent = ua.random

    # 设置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_argument('--headless')  # 无头模式，不显示浏览器窗口
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # 初始化WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 添加随机延迟
        time.sleep(random.uniform(1, 3))

        # 访问URL
        driver.get(url)

        # 等待页面加载（可以根据需要调整等待时间）
        time.sleep(wait_time)

        # 获取页面源代码
        page_source = driver.page_source

        # 使用html2text转换HTML为Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        markdown_content = h.handle(page_source)

        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Markdown content has been saved to {output_file}")
            except IOError as e:
                print(f"Error saving file: {e}")

        return markdown_content

    except Exception as e:
        error_message = f"Error fetching URL: {e}"
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(error_message)
                print(f"Error message has been saved to {output_file}")
            except IOError as e:
                print(f"Error saving file: {e}")
        return error_message

    finally:
        driver.quit()

# 使用示例
url = "https://zhuanlan.zhihu.com/p/508517779"  # 替换为您想要转换的URL
output_file = "output_headless.md"  # 指定输出文件名，如果不想保存文件，可以设为None
wait_time = 5  # 等待页面加载的时间（秒），可以根据需要调整

markdown_result = url_to_markdown(url, output_file, wait_time)
print(markdown_result)