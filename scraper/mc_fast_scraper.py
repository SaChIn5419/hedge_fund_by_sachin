"""
MoneyControl Fast Parallel Scraper v2
======================================
Async scraper with proper error categorization and encoding handling.

Usage:
    python scraper/mc_fast_scraper.py --max-articles 10000 --workers 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRAPER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRAPER_DIR.parent
URL_CACHE_PATH = SCRAPER_DIR / "url_cache.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "news" / "raw" / "moneycontrol_scraped.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _parse_mc_date(raw: str) -> str:
    if not raw:
        return ""
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})", raw
    )
    if match:
        try:
            dt = datetime.strptime(
                f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y"
            )
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw[:10] if len(raw) >= 10 else ""


def _parse_article_html(html: str, url: str) -> dict | None:
    """Extract article data from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.text.strip() if h1 else ""
    if not title or len(title) < 10:
        return None

    # Date
    date_str = ""
    date_div = soup.find("div", class_="article_schedule")
    if date_div:
        date_str = date_div.text.strip()
    if not date_str:
        meta_date = soup.find("meta", attrs={"property": "article:published_time"})
        if meta_date:
            date_str = meta_date.get("content", "")[:10]

    # Description
    description = ""
    meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc:
        description = meta_desc.get("content", "").strip()

    # Author
    author = ""
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author:
        author = meta_author.get("content", "").strip()
    if not author:
        author_div = soup.find("div", class_="article_author")
        if author_div:
            author = author_div.text.strip()

    # Keywords
    keywords = ""
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        keywords = meta_kw.get("content", "").strip()

    # Content
    content = ""
    content_div = soup.find("div", class_="content_wrapper")
    if content_div:
        paragraphs = content_div.find_all("p")
        content = " ".join(p.text.strip() for p in paragraphs if p.text.strip())
    if not content:
        for cls in ["article_content", "artText", "arti-flow"]:
            el = soup.find("div", class_=cls)
            if el:
                content = el.text.strip()
                break

    # Must have at least a title and some description or content
    if not title:
        return None
    if not content and not description:
        return None

    return {
        "Title": title,
        "Date": _parse_mc_date(date_str),
        "Description": description,
        "Author": author,
        "Content": content[:5000],
        "Keywords": keywords,
        "URL": url,
    }


# Counters for error categorization
class Stats:
    def __init__(self):
        self.ok = 0
        self.http_fail = 0
        self.parse_fail = 0
        self.timeout = 0
        self.encoding_err = 0


async def fetch_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    fallback_date: str,
    stats: Stats,
) -> dict | None:
    """Fetch and parse one article."""
    async with semaphore:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    stats.http_fail += 1
                    return None
                try:
                    html = await resp.text(encoding="utf-8", errors="replace")
                except UnicodeDecodeError:
                    # Fallback: read as bytes and decode leniently
                    raw = await resp.read()
                    html = raw.decode("utf-8", errors="replace")
                    stats.encoding_err += 1
        except asyncio.TimeoutError:
            stats.timeout += 1
            return None
        except Exception:
            stats.http_fail += 1
            return None

        article = _parse_article_html(html, url)
        if article:
            if not article["Date"]:
                article["Date"] = fallback_date
            stats.ok += 1
            return article
        else:
            stats.parse_fail += 1
            return None


async def scrape_batch(
    urls: list[dict],
    workers: int = 20,
    save_every: int = 500,
    existing: list[dict] | None = None,
) -> list[dict]:
    """Scrape a batch with full progress + error categorization."""
    existing = existing or []
    semaphore = asyncio.Semaphore(workers)
    connector = aiohttp.TCPConnector(limit=workers * 2, limit_per_host=workers)
    stats = Stats()
    new_articles = []
    t0 = time.time()

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        chunk_size = save_every
        for chunk_start in range(0, len(urls), chunk_size):
            chunk = urls[chunk_start : chunk_start + chunk_size]

            tasks = [
                fetch_one(session, semaphore, item["url"], item["sitemap_date"], stats)
                for item in chunk
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    new_articles.append(result)

            elapsed = time.time() - t0
            done = chunk_start + len(chunk)
            rate = done / elapsed if elapsed > 0 else 0
            total = len(existing) + len(new_articles)
            eta = (len(urls) - done) / rate if rate > 0 else 0
            print(
                f"  [{done:>5}/{len(urls)}] "
                f"ok={stats.ok} parse_skip={stats.parse_fail} "
                f"http_err={stats.http_fail} timeout={stats.timeout} "
                f"| total={total} "
                f"| {rate:.1f}/s ETA={eta/60:.0f}min"
            )

            _save_progress(existing, new_articles)
            await asyncio.sleep(0.2)

    return new_articles


def _save_progress(existing: list[dict], new: list[dict]) -> list[dict]:
    all_articles = existing + new
    seen = set()
    unique = []
    for a in all_articles:
        if a["URL"] not in seen:
            seen.add(a["URL"])
            unique.append(a)
    unique.sort(key=lambda x: x.get("Date", ""))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    return unique


def sample_urls(
    url_cache: list[dict],
    max_articles: int,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:
    """Sample URLs evenly across months for balanced coverage."""
    filtered = url_cache
    if from_date:
        filtered = [u for u in filtered if u["sitemap_date"] >= from_date]
    if to_date:
        filtered = [u for u in filtered if u["sitemap_date"] <= to_date]

    if len(filtered) <= max_articles:
        return filtered

    by_month: dict[str, list[dict]] = {}
    for u in filtered:
        month = u["sitemap_date"][:7]
        by_month.setdefault(month, []).append(u)

    per_month = max(1, max_articles // len(by_month))
    sampled = []
    for month in sorted(by_month):
        month_urls = by_month[month]
        if len(month_urls) <= per_month:
            sampled.extend(month_urls)
        else:
            step = len(month_urls) / per_month
            sampled.extend(month_urls[int(i * step)] for i in range(per_month))

    if len(sampled) > max_articles:
        sampled = sampled[:max_articles]
    return sampled


def main():
    parser = argparse.ArgumentParser(description="MoneyControl Fast Parallel Scraper v2")
    parser.add_argument("--max-articles", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--from-date", type=str, default=None)
    parser.add_argument("--to-date", type=str, default=None)
    args = parser.parse_args()

    if not URL_CACHE_PATH.exists():
        raise SystemExit(f"No URL cache at {URL_CACHE_PATH}. Run mc_scraper.py collect first.")

    with open(URL_CACHE_PATH) as f:
        url_cache = json.load(f)
    print(f"URL cache: {len(url_cache)} URLs")

    existing = []
    existing_urls = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
            existing_urls = {a["URL"] for a in existing}
        print(f"Already scraped: {len(existing)} articles")

    unseen = [u for u in url_cache if u["url"] not in existing_urls]
    print(f"Unseen URLs: {len(unseen)}")

    to_scrape = sample_urls(unseen, args.max_articles, args.from_date, args.to_date)
    print(
        f"Sampling {len(to_scrape)} URLs across "
        f"{to_scrape[0]['sitemap_date']} → {to_scrape[-1]['sitemap_date']}"
    )
    print(f"Workers: {args.workers} concurrent connections")
    print()

    new = asyncio.run(
        scrape_batch(to_scrape, args.workers, save_every=500, existing=existing)
    )

    total = len(existing) + len(new)
    print(f"\nDone! Scraped {len(new)} new articles. Total: {total}")


if __name__ == "__main__":
    main()
