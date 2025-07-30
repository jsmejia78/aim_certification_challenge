import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArticleBot/1.0)"
}

def clean_text(text):
    # Replace non-breaking spaces and odd unicode with regular spaces
    text = text.replace('\xa0', ' ').replace('\u200b', ' ')
    # Split into paragraphs by double newlines or blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    # Join lines within paragraphs, then join paragraphs with double newlines
    cleaned_paragraphs = [' '.join(p.splitlines()) for p in paragraphs]
    cleaned = '\n\n'.join([re.sub(r' +', ' ', p).strip() for p in cleaned_paragraphs if p.strip()])
    # Remove excessive newlines (more than 2)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned

def get_article_urls(page_url, base_url, page_num):
    url = page_url.format(page_num)
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select('a.search-card--link'):
        href = a.get('href')
        if href and href.startswith('/read/'):
            links.append(urljoin(base_url, href))
    if not links:
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/read/'):
                links.append(urljoin(base_url, href))
    print(f"[DEBUG] Found {len(links)} article links on page {page_num}")
    return links

def get_article_text(article_url):
    resp = requests.get(article_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Untitled"
    content = ""
    # Try extracting from <div id="main-content">
    main_div = soup.find('div', id='main-content')
    if main_div:
        raw_text = main_div.get_text(separator='\n', strip=True)
    else:
        # Try itemprop and common class names as fallback
        body = soup.find(itemprop="articleBody")
        if body:
            raw_text = body.get_text(separator='\n', strip=True)
        else:
            raw_text = ""
            for class_name in [
                'article-content', 'post-content', 'entry-content', 'content', 'main-content', 'sc-a3fdd2f3-0']:
                div = soup.find('div', class_=class_name)
                if div:
                    raw_text = div.get_text(separator='\n', strip=True)
                    break
    if not raw_text:
        paragraphs = soup.find_all('p')
        raw_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    # Clean up spacing: remove empty lines and strip each line
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    content = '\n'.join(lines)
    # --- Clean the content further ---
    content = clean_text(content)
    return title, content

def save_txt(title, content, folder, idx):
    safe = ''.join(c for c in title if c.isalnum() or c in " _-")[:50].rstrip()
    filename = f"{idx:03d}_{safe}.txt"
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + content)

def download_articles(max_articles, folder, page_url, base_url):
    os.makedirs(folder, exist_ok=True)
    count = 0
    page = 1

    unlimited = max_articles <= 0

    while unlimited or count < max_articles:
        urls = get_article_urls(page_url, base_url, page)
        if not urls:
            print("🔍 No more articles found.")
            break

        for url in urls:
            if not unlimited and count >= max_articles:
                break
            try:
                print(f"📥 Downloading ({count+1}/{max_articles if not unlimited else '∞'}): {url}")
                title, content = get_article_text(url)
                save_txt(title, content, folder, count + 1)
                count += 1
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print("❗ Error:", e)
        page += 1

    print(f"✅ Done! Downloaded {count} articles into '{folder}'")

# --- CLI usage ---
if __name__ == "__main__":
    import argparse
    N = -1
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num", type=int, default=N, help="Number of articles to download per URL")
    args = p.parse_args()
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    url_folder_map = {
        "data_toddlers_1p5-3": {
            "base_url": "https://www.peacefulparenthappykids.com",
            "page_url": "https://www.peacefulparenthappykids.com/search-results?query=&age[]=770114&type=20&page={}"
        },
        "data_preschoolers_3-5": {
            "base_url": "https://www.peacefulparenthappykids.com",
            "page_url": "https://www.peacefulparenthappykids.com/search-results?query=&age[]=770115&type=20&page={}"
        },
        "data_school_age_5-9": {
            "base_url": "https://www.peacefulparenthappykids.com",
            "page_url": "https://www.peacefulparenthappykids.com/search-results?query=&age[]=770116&type=20&page={}"
        },
        "data_tweens_and_preteens_10-12": {
            "base_url": "https://www.peacefulparenthappykids.com",
            "page_url": "https://www.peacefulparenthappykids.com/search-results?query=&age[]=770117&type=20&page={}"
        },
    }

    for folder, params in url_folder_map.items():
        print(f"\n=== Downloading to folder '{folder}' from '{params['page_url']}' ===")
        download_articles(args.num, os.path.join(parent_dir, folder), params['page_url'], params['base_url'])
