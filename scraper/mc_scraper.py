"""
MoneyControl News Scraper
=========================
Two-phase scraper that produces IN-FINews-quality data:

Phase 1 (fast):  Pull article URLs + dates from MoneyControl sitemaps
Phase 2 (slow):  Visit each article page and extract full content

Output format matches IN-FINews: {Title, Date, Description, Author, Content, Keywords, URL}

Usage:
    # Phase 1: Collect URLs from sitemaps for a date range
    python scraper/mc_scraper.py collect --from-date 2019-12-01 --to-date 2026-04-12

    # Phase 2: Scrape full article content (with rate limiting)
    python scraper/mc_scraper.py scrape --max-articles 500 --delay 1.5

    # Both phases at once
    python scraper/mc_scraper.py full --from-date 2024-01-01 --to-date 2026-04-12 --max-articles 2000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRAPER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRAPER_DIR.parent
URL_CACHE_PATH = SCRAPER_DIR / "url_cache.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "news" / "raw" / "moneycontrol_scraped.json"

# ── Constants ──────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Only scrape articles in these MoneyControl sections (financial relevance)
BUSINESS_SECTIONS = {
    "/news/business/",
    "/news/economy/",
    "/news/markets/",
    "/news/india/",
}

# Keywords in URL or title that indicate financial relevance
FINANCIAL_KEYWORDS = {
    "market", "stock", "share", "nifty", "sensex", "rbi", "rupee", "economy",
    "gdp", "inflation", "rate", "bank", "earnings", "ipo", "tariff", "crude",
    "oil", "trade", "export", "import", "fiscal", "budget", "policy", "mutual",
    "fund", "forex", "fii", "dii", "sebi", "nse", "bse", "profit", "revenue",
    "growth", "deficit", "surplus", "bond", "yield", "credit", "debt", "loan",
    "tax", "gst", "pmi", "manufacturing", "services", "retail", "consumer",
    "commodity", "gold", "silver", "copper", "steel", "power", "energy",
    "pharma", "auto", "infra", "realty", "cement", "fmcg", "it", "tech",
    "telecom", "insurance", "nbfc", "npa", "liquidity", "repo", "cpi", "wpi",
    "iip", "capex", "disinvestment", "privatisation", "subsidy",
}


def _is_financial_url(url: str) -> bool:
    """Check if a URL looks like a financial/business news article."""
    url_lower = url.lower()
    # Must be in a business/economy/markets section OR contain financial keywords
    in_section = any(sec in url_lower for sec in BUSINESS_SECTIONS)
    has_keyword = any(kw in url_lower for kw in FINANCIAL_KEYWORDS)
    is_article = url_lower.endswith(".html") and "/news/" in url_lower
    # Exclude non-article pages
    excludes = ["/tags/", "/photos/", "/videos/", "/webstories/", "/podcast", "/slideshows/"]
    is_excluded = any(ex in url_lower for ex in excludes)
    return is_article and (in_section or has_keyword) and not is_excluded


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Collect URLs from Sitemaps
# ═══════════════════════════════════════════════════════════════════════════════

def get_monthly_sitemap_urls(from_date: str, to_date: str) -> list[str]:
    """Generate sitemap URLs for the given date range."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")

    urls = []
    year, month = start.year, start.month
    while datetime(year, month, 1) <= end:
        urls.append(
            f"https://www.moneycontrol.com/news/sitemap/sitemap-post-{year}-{month:02d}.xml"
        )
        month += 1
        if month > 12:
            month = 1
            year += 1
    return urls


def fetch_sitemap_articles(sitemap_url: str) -> list[dict]:
    """Parse a monthly sitemap and return list of {url, date} dicts."""
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠ Failed to fetch {sitemap_url}: {e}")
        return []

    locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
    lastmods = re.findall(r"<lastmod>(.*?)</lastmod>", resp.text)

    articles = []
    for url, date_str in zip(locs, lastmods):
        if _is_financial_url(url):
            articles.append({
                "url": url,
                "sitemap_date": date_str[:10],  # YYYY-MM-DD
            })

    return articles


def collect_urls(from_date: str, to_date: str) -> list[dict]:
    """Phase 1: Collect all financial article URLs from sitemaps."""
    sitemap_urls = get_monthly_sitemap_urls(from_date, to_date)
    print(f"Fetching {len(sitemap_urls)} monthly sitemaps ({from_date} → {to_date})...")

    all_articles = []
    for i, smap_url in enumerate(sitemap_urls):
        month_label = smap_url.split("sitemap-post-")[1].replace(".xml", "")
        articles = fetch_sitemap_articles(smap_url)
        all_articles.extend(articles)
        print(f"  [{i+1}/{len(sitemap_urls)}] {month_label}: {len(articles)} financial articles")
        time.sleep(0.5)  # Be polite

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in all_articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    # Filter by date range
    unique = [a for a in unique if from_date <= a["sitemap_date"] <= to_date]
    unique.sort(key=lambda x: x["sitemap_date"])

    # Save URL cache
    URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(URL_CACHE_PATH, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"\nCollected {len(unique)} unique financial article URLs → {URL_CACHE_PATH}")
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Scrape Article Content
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_article(url: str) -> dict | None:
    """Visit a single article page and extract IN-FINews-format data."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.text.strip() if h1 else ""
    if not title:
        return None  # Not a real article page

    # Date — from the page or meta tags
    date_str = ""
    date_div = soup.find("div", class_="article_schedule")
    if date_div:
        date_str = date_div.text.strip()
    if not date_str:
        meta_date = soup.find("meta", attrs={"property": "article:published_time"})
        if meta_date:
            date_str = meta_date.get("content", "")[:10]

    # Normalize date to YYYY-MM-DD
    parsed_date = _parse_mc_date(date_str)

    # Description — from og:description meta tag
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

    # Keywords — from meta keywords tag
    keywords = ""
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        keywords = meta_kw.get("content", "").strip()

    # Content — from article body
    content = ""
    content_div = soup.find("div", class_="content_wrapper")
    if content_div:
        paragraphs = content_div.find_all("p")
        content = " ".join(p.text.strip() for p in paragraphs if p.text.strip())

    if not content:
        # Fallback: try other common content selectors
        for cls in ["article_content", "artText", "arti-flow"]:
            el = soup.find("div", class_=cls)
            if el:
                content = el.text.strip()
                break

    if not title or not content:
        return None

    return {
        "Title": title,
        "Date": parsed_date,
        "Description": description,
        "Author": author,
        "Content": content[:5000],  # Cap to avoid bloat
        "Keywords": keywords,
        "URL": url,
    }


def _parse_mc_date(raw: str) -> str:
    """Parse MoneyControl's various date formats to YYYY-MM-DD."""
    if not raw:
        return ""
    # Try ISO format first (from meta tag)
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # Try "April 10, 2026 / 09:24 IST" format
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})", raw
    )
    if match:
        try:
            dt = datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw[:10] if len(raw) >= 10 else ""


def scrape_articles(max_articles: int = 500, delay: float = 1.5) -> list[dict]:
    """Phase 2: Scrape full content for cached URLs."""
    if not URL_CACHE_PATH.exists():
        raise SystemExit(f"No URL cache found at {URL_CACHE_PATH}. Run 'collect' first.")

    with open(URL_CACHE_PATH) as f:
        url_cache = json.load(f)

    # Load existing scraped data to skip already-done URLs
    existing_urls = set()
    existing_articles = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing_articles = json.load(f)
            existing_urls = {a["URL"] for a in existing_articles}
        print(f"Loaded {len(existing_articles)} already-scraped articles")

    # Filter to unseen URLs
    to_scrape = [u for u in url_cache if u["url"] not in existing_urls]
    to_scrape = to_scrape[:max_articles]
    print(f"Scraping {len(to_scrape)} articles (delay={delay}s between requests)...")

    new_articles = []
    failed = 0
    for i, item in enumerate(to_scrape):
        url = item["url"]
        article = scrape_article(url)

        if article:
            # Use sitemap date as fallback if page date extraction failed
            if not article["Date"]:
                article["Date"] = item["sitemap_date"]
            new_articles.append(article)
        else:
            failed += 1

        if (i + 1) % 50 == 0 or i == len(to_scrape) - 1:
            total = len(existing_articles) + len(new_articles)
            print(f"  [{i+1}/{len(to_scrape)}] scraped={len(new_articles)} failed={failed} total={total}")

            # Save progress every 50 articles
            _save_progress(existing_articles, new_articles)

        # Rate limiting with jitter
        time.sleep(delay + random.uniform(0, 0.5))

    # Final save
    all_articles = _save_progress(existing_articles, new_articles)
    print(f"\nDone! Total articles: {len(all_articles)} → {OUTPUT_PATH}")
    return all_articles


def _save_progress(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge and save articles to disk."""
    all_articles = existing + new
    # Deduplicate by URL
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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MoneyControl News Scraper")
    sub = parser.add_subparsers(dest="command", required=True)

    # Collect command
    collect_p = sub.add_parser("collect", help="Phase 1: Collect article URLs from sitemaps")
    collect_p.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD")
    collect_p.add_argument("--to-date", required=True, help="End date YYYY-MM-DD")

    # Scrape command
    scrape_p = sub.add_parser("scrape", help="Phase 2: Scrape full article content")
    scrape_p.add_argument("--max-articles", type=int, default=500, help="Max articles to scrape")
    scrape_p.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")

    # Full command (both phases)
    full_p = sub.add_parser("full", help="Run both phases")
    full_p.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD")
    full_p.add_argument("--to-date", required=True, help="End date YYYY-MM-DD")
    full_p.add_argument("--max-articles", type=int, default=500, help="Max articles to scrape")
    full_p.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")

    args = parser.parse_args()

    if args.command == "collect":
        collect_urls(args.from_date, args.to_date)

    elif args.command == "scrape":
        scrape_articles(args.max_articles, args.delay)

    elif args.command == "full":
        collect_urls(args.from_date, args.to_date)
        scrape_articles(args.max_articles, args.delay)


if __name__ == "__main__":
    main()
