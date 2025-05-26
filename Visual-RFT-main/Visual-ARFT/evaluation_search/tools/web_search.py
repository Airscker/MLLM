import os
from duckduckgo_search import DDGS
import json
import requests
from typing import List, Literal, Optional, Union
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
load_dotenv()

# 使用免费的 duck duck go 进行网页检索，使用 firecrawl 将网页转化为markdown格式
def web_search_DDG(query: Optional[str], search_num: int = 2, search_mode: str = 'fast') -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    if search_mode == 'fast':
        assert type(query)==str
        results = DDGS().text(query, max_results=search_num)
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        results = DDGS().text(query, max_results=search_num)
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results

def web_search_SERPER_API(query: Optional[str], search_num=2, search_mode: str = 'fast') -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    if search_mode == 'fast':
        assert type(query)==str
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results