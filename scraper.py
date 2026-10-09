import os
import re
import json
import time
import hashlib
import requests

from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag, unquote
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.spit.ac.in/"
ALLOWED_DOMAINS = {
    "spit.ac.in",
    "www.spit.ac.in",
}

OUTPUT_DIR = "data"
PAGES_DIR = os.path.join(OUTPUT_DIR, "website_pages")
PDF_DIR = os.path.join(OUTPUT_DIR, "institutional_pdfs")
METADATA_DIR = "metadata"

for directory in [PAGES_DIR, PDF_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

REQUEST_DELAY = 0.3
TIMEOUT = 30
MAX_RETRIES = 3

session = requests.Session()
session.headers.update({
    "User-Agent": "SPIT-RAG-ResearchBot/1.0 (academic research)"
})

visited = set()
queued = set()
page_records = []
pdf_records = []
failed_urls = []


# ============================================================
# URL UTILITIES
# ============================================================

def normalize_url(url):
    url = urldefrag(url)[0].strip()

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.netloc.lower() not in ALLOWED_DOMAINS:
        return None

    # Ignore common tracking parameters.
    from urllib.parse import parse_qsl, urlencode, urlunparse

    ignored = {
        "utm_source", "utm_medium", "utm_campaign",
        "utm_term", "utm_content", "fbclid", "gclid"
    }

    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in ignored
    ])

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        parsed.params,
        query,
        ""
    ))


def is_pdf_url(url):
    return urlparse(url).path.lower().endswith(".pdf")


def safe_filename(name):
    name = unquote(name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name[:150] or "document"


def request_url(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
                allow_redirects=True
            )

            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            return response

        except requests.RequestException as error:
            if attempt == MAX_RETRIES - 1:
                failed_urls.append({
                    "url": url,
                    "error": str(error)
                })
                return None

            time.sleep(2 ** attempt)

    return None


# ============================================================
# ROBOTS.TXT
# ============================================================

robots = RobotFileParser()
robots_url = urljoin(BASE_URL, "robots.txt")

robots_response = request_url(robots_url)

if robots_response and robots_response.status_code == 200:
    robots.parse(robots_response.text.splitlines())
else:
    robots = None

def can_fetch(url):
    if robots is None:
        return True

    try:
        return robots.can_fetch(
            session.headers["User-Agent"],
            url
        )
    except Exception:
        return False


# ============================================================
# PDF DOWNLOADER
# ============================================================

def download_pdf(url):
    if any(record["url"] == url for record in pdf_records):
        return

    if not can_fetch(url):
        print(f"ROBOTS DISALLOW: {url}")
        return

    response = request_url(url)

    if response is None or response.status_code != 200:
        return

    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()

    # Validate actual PDF content.
    if not content.startswith(b"%PDF"):
        return

    digest = hashlib.sha256(content).hexdigest()

    # Avoid downloading duplicate file contents.
    for record in pdf_records:
        if record["sha256"] == digest:
            record.setdefault("duplicate_urls", []).append(url)
            return

    original_name = os.path.basename(urlparse(url).path)
    filename = f"{digest[:12]}_{safe_filename(original_name)}"

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    filepath = os.path.join(PDF_DIR, filename)

    with open(filepath, "wb") as file:
        file.write(content)

    pdf_records.append({
        "url": url,
        "filename": filename,
        "local_path": filepath,
        "sha256": digest,
        "size_bytes": len(content),
        "content_type": content_type
    })

    print(f"PDF DOWNLOADED: {url}")


# ============================================================
# SITEMAP DISCOVERY
# ============================================================

def discover_sitemaps():
    sitemap_urls = {
        urljoin(BASE_URL, "sitemap.xml"),
        urljoin(BASE_URL, "wp-sitemap.xml"),
    }

    if robots_response and robots_response.status_code == 200:
        for line in robots_response.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_urls.add(line.split(":", 1)[1].strip())

    discovered = set()
    sitemap_queue = deque(sitemap_urls)

    while sitemap_queue:
        sitemap_url = sitemap_queue.popleft()

        response = request_url(sitemap_url)

        if response is None or response.status_code != 200:
            continue

        try:
            soup = BeautifulSoup(response.content, "xml")
        except Exception:
            continue

        for loc in soup.find_all("loc"):
            target = normalize_url(loc.get_text(strip=True))

            if not target:
                continue

            if target.endswith(".xml"):
                sitemap_queue.append(target)
            elif is_pdf_url(target):
                discovered.add(target)
            else:
                discovered.add(target)

    return discovered


# ============================================================
# CRAWLER
# ============================================================

def crawl():
    initial_urls = discover_sitemaps()

    queue = deque()
    queued.add(BASE_URL)
    queue.append(BASE_URL)

    for url in initial_urls:
        if url not in queued:
            queue.append(url)
            queued.add(url)

    print(f"Initial URLs discovered: {len(queue)}")

    while queue:
        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        if not can_fetch(url):
            print(f"ROBOTS DISALLOW: {url}")
            continue

        if is_pdf_url(url):
            download_pdf(url)
            continue

        response = request_url(url)

        if response is None:
            continue

        if response.status_code != 200:
            failed_urls.append({
                "url": url,
                "error": f"HTTP {response.status_code}"
            })
            continue

        # Stay on approved domains after redirects.
        final_url = normalize_url(response.url)

        if not final_url:
            continue

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            if "pdf" in content_type or response.content.startswith(b"%PDF"):
                download_pdf(final_url)
            continue

        if final_url != url and final_url in visited:
            continue

        visited.add(final_url)

        soup = BeautifulSoup(response.text, "html.parser")

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title else final_url
        )

        # Remove irrelevant HTML elements after collecting links.
        links = []

        for anchor in soup.find_all("a", href=True):
            target = normalize_url(urljoin(final_url, anchor["href"]))

            if target:
                links.append(target)

        for element in soup([
            "script", "style", "noscript", "svg"
        ]):
            element.decompose()

        main_content = soup.find("main") or soup.find("article") or soup.body or soup

        text = main_content.get_text("\n", strip=True)

        digest = hashlib.sha256(
            final_url.encode("utf-8")
        ).hexdigest()[:16]

        filepath = os.path.join(
            PAGES_DIR,
            f"{digest}.txt"
        )

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"TITLE: {title}\n")
            file.write(f"URL: {final_url}\n\n")
            file.write(text)

        page_records.append({
            "title": title,
            "url": final_url,
            "local_path": filepath,
            "sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
        })

        print(f"PAGE [{len(page_records)}]: {final_url}")

        for target in links:
            if is_pdf_url(target):
                download_pdf(target)

            elif target not in visited and target not in queued:
                queue.append(target)
                queued.add(target)

        time.sleep(REQUEST_DELAY)

        # Periodically save progress.
        if len(page_records) % 25 == 0:
            save_metadata()


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata():
    with open(
        os.path.join(METADATA_DIR, "pages_metadata.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(page_records, file, indent=2, ensure_ascii=False)

    with open(
        os.path.join(METADATA_DIR, "pdf_metadata.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(pdf_records, file, indent=2, ensure_ascii=False)

    with open(
        os.path.join(METADATA_DIR, "failed_urls.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(failed_urls, file, indent=2, ensure_ascii=False)

    with open(
        os.path.join(METADATA_DIR, "crawl_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "pages_collected": len(page_records),
            "pdfs_downloaded": len(pdf_records),
            "unique_urls_visited": len(visited),
            "failed_requests": len(failed_urls)
        }, file, indent=2)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        crawl()
    except KeyboardInterrupt:
        print("\nCrawl interrupted. Saving progress...")
    finally:
        save_metadata()

        print("\n========== CRAWL SUMMARY ==========")
        print(f"Pages collected : {len(page_records)}")
        print(f"PDFs downloaded : {len(pdf_records)}")
        print(f"URLs visited    : {len(visited)}")
        print(f"Failed requests : {len(failed_urls)}")
        print("Metadata saved in:", METADATA_DIR)