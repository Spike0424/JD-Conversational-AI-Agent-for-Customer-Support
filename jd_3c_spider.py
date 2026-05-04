#!/usr/bin/env python3
"""
JD 3C digital products spider (search-page based).

Usage:
  1) pip install playwright
  2) playwright install chromium
  3) python jd_3c_spider.py --keywords 手机,笔记本,耳机 --pages 2 --format json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote_plus

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


DEFAULT_KEYWORDS = ["手机", "笔记本", "平板", "耳机", "显示器"]
DEBUG_LOG_PATH = Path("/home/spike/Workspace/Custom_answer_agent/.cursor/debug-2d86db.log")


# region agent log
def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "2d86db",
        "id": f"log_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[agent-debug-log-error] {exc}")


# endregion agent log


@dataclass
class Product:
    keyword: str
    sku: str
    title: str
    price: str
    shop: str
    comments: str
    product_url: str
    page_no: int


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_search_url(keyword: str, page_no: int) -> str:
    # JD search pagination is odd numbers: 1,3,5...
    page_param = 2 * page_no - 1
    s_param = (page_no - 1) * 60 + 1
    return (
        "https://search.jd.com/Search"
        f"?keyword={quote_plus(keyword)}&enc=utf-8&page={page_param}&s={s_param}&click=0"
    )


def _wait_for_page_ready(page) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_selector("body", timeout=15000)
    except PlaywrightTimeoutError:
        # JD can keep background requests open; a body is enough for best-effort scraping.
        pass


def _auto_scroll(page, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            _wait_for_page_ready(page)
            page.evaluate(
                """
                () => new Promise((resolve) => {
                  let total = 0;
                  const step = 600;
                  const timer = setInterval(() => {
                    window.scrollBy(0, step);
                    total += step;
                    if (total >= document.body.scrollHeight * 1.2) {
                      clearInterval(timer);
                      resolve();
                    }
                  }, 200);
                })
                """
            )
            return
        except PlaywrightError as exc:
            message = str(exc)
            is_navigation_race = "Execution context was destroyed" in message
            if not is_navigation_race or attempt == retries:
                raise
            page.wait_for_timeout(1000 * attempt)


def crawl_keyword(
    keyword: str,
    pages: int,
    max_items: int,
    headless: bool = True,
    timeout_ms: int = 45000,
) -> List[Product]:
    products: List[Product] = []
    dedupe_keys = set()
    _debug_log(
        "initial",
        "H4",
        "jd_3c_spider.py:crawl_keyword",
        "crawl_keyword started",
        {"keyword": keyword, "pages": pages, "max_items": max_items, "headless": headless},
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            viewport={"width": 1600, "height": 1000},
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        for page_no in range(1, pages + 1):
            url = build_search_url(keyword, page_no)
            _debug_log(
                "initial",
                "H4",
                "jd_3c_spider.py:crawl_keyword.before_goto",
                "navigating to JD search page",
                {"keyword": keyword, "page_no": page_no, "target_url": url},
            )
            page.goto(url, wait_until="domcontentloaded")
            _wait_for_page_ready(page)
            pre_scroll_counts = page.evaluate(
                """
                () => ({
                  title: document.title,
                  href: location.href,
                  bodyText: document.body ? document.body.innerText.slice(0, 500) : "",
                  glItems: document.querySelectorAll("li.gl-item[data-sku]").length,
                  dataSku: document.querySelectorAll("[data-sku]").length,
                  productLinks: document.querySelectorAll("a[href*='item.jd.com']").length,
                  captchaHints: /验证码|安全验证|滑块|访问验证|风险|频繁/.test(document.body ? document.body.innerText : "")
                })
                """
            )
            _debug_log(
                "initial",
                "H1,H2,H3,H4",
                "jd_3c_spider.py:crawl_keyword.after_goto",
                "page state before scroll",
                {"keyword": keyword, "page_no": page_no, "state": pre_scroll_counts},
            )

            # Trigger lazy-loaded items.
            _auto_scroll(page)
            page.wait_for_timeout(1200)
            post_scroll_counts = page.evaluate(
                """
                () => ({
                  href: location.href,
                  scrollY: window.scrollY,
                  scrollHeight: document.body ? document.body.scrollHeight : 0,
                  glItems: document.querySelectorAll("li.gl-item[data-sku]").length,
                  dataSku: document.querySelectorAll("[data-sku]").length,
                  productLinks: document.querySelectorAll("a[href*='item.jd.com']").length
                })
                """
            )
            _debug_log(
                "initial",
                "H1,H3",
                "jd_3c_spider.py:crawl_keyword.after_scroll",
                "page state after scroll",
                {"keyword": keyword, "page_no": page_no, "state": post_scroll_counts},
            )

            items = page.query_selector_all("li.gl-item[data-sku]")
            if not items:
                # Some pages can be anti-bot blocked or layout changed.
                _debug_log(
                    "initial",
                    "H1,H2",
                    "jd_3c_spider.py:crawl_keyword.no_items",
                    "primary selector returned zero products",
                    {"keyword": keyword, "page_no": page_no, "current_url": page.url, "state": post_scroll_counts},
                )
                break

            page_added = 0
            skipped_empty_key = 0
            skipped_duplicate = 0
            sample_items = []
            for item in items:
                sku = clean_text(item.get_attribute("data-sku") or "")

                title = clean_text(
                    item.query_selector(".p-name em").inner_text()
                    if item.query_selector(".p-name em")
                    else ""
                )
                price = clean_text(
                    item.query_selector(".p-price i").inner_text()
                    if item.query_selector(".p-price i")
                    else ""
                )
                shop = clean_text(
                    item.query_selector(".p-shop a").inner_text()
                    if item.query_selector(".p-shop a")
                    else ""
                )
                comments = clean_text(
                    item.query_selector(".p-commit a").inner_text()
                    if item.query_selector(".p-commit a")
                    else ""
                )

                url_node = item.query_selector(".p-name a")
                product_url = clean_text(url_node.get_attribute("href") or "") if url_node else ""
                if product_url.startswith("//"):
                    product_url = f"https:{product_url}"

                dedupe_key = sku or product_url or title
                if not dedupe_key or dedupe_key in dedupe_keys:
                    if not dedupe_key:
                        skipped_empty_key += 1
                    else:
                        skipped_duplicate += 1
                    continue
                dedupe_keys.add(dedupe_key)

                if len(sample_items) < 3:
                    sample_items.append(
                        {
                            "sku": sku,
                            "title": title[:120],
                            "price": price,
                            "shop": shop,
                            "comments": comments,
                            "product_url": product_url,
                        }
                    )

                products.append(
                    Product(
                        keyword=keyword,
                        sku=sku,
                        title=title,
                        price=price,
                        shop=shop,
                        comments=comments,
                        product_url=product_url,
                        page_no=page_no,
                    )
                )
                page_added += 1

                if max_items > 0 and len(products) >= max_items:
                    _debug_log(
                        "initial",
                        "H5",
                        "jd_3c_spider.py:crawl_keyword.max_items",
                        "max item limit reached",
                        {
                            "keyword": keyword,
                            "page_no": page_no,
                            "items_found": len(items),
                            "page_added": page_added,
                            "total_products": len(products),
                            "sample_items": sample_items,
                        },
                    )
                    context.close()
                    browser.close()
                    return products

            _debug_log(
                "initial",
                "H5",
                "jd_3c_spider.py:crawl_keyword.page_summary",
                "product extraction summary for page",
                {
                    "keyword": keyword,
                    "page_no": page_no,
                    "items_found": len(items),
                    "page_added": page_added,
                    "skipped_empty_key": skipped_empty_key,
                    "skipped_duplicate": skipped_duplicate,
                    "total_products": len(products),
                    "sample_items": sample_items,
                },
            )

            # Randomized pacing to reduce blocking probability.
            page.wait_for_timeout(random.randint(1000, 2200))

        context.close()
        browser.close()

    return products


def save_json(items: Iterable[Product], output: Path) -> None:
    data = [asdict(x) for x in items]
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(items: Iterable[Product], output: Path) -> None:
    rows = [asdict(x) for x in items]
    fields = ["keyword", "sku", "title", "price", "shop", "comments", "product_url", "page_no"]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl JD 3C digital products from search pages.")
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keywords, e.g. 手机,笔记本,耳机",
    )
    parser.add_argument("--pages", type=int, default=1, help="Pages per keyword.")
    parser.add_argument(
        "--max-items-per-keyword",
        type=int,
        default=60,
        help="Limit items for each keyword (0 means no limit).",
    )
    parser.add_argument("--output", type=str, default="jd_3c_products.json", help="Output file path.")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format.")
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window for debugging (headless=False).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # region agent log
    _debug_log(
        "initial",
        "H0",
        "jd_3c_spider.py:main",
        "script main started",
        {"script_path": str(Path(__file__).resolve()), "cwd": str(Path.cwd()), "args": vars(args)},
    )
    # endregion agent log
    keywords = [clean_text(k) for k in args.keywords.split(",") if clean_text(k)]
    if not keywords:
        raise ValueError("No valid keyword provided.")
    if args.pages <= 0:
        raise ValueError("--pages must be > 0")

    all_products: List[Product] = []
    for kw in keywords:
        items = crawl_keyword(
            keyword=kw,
            pages=args.pages,
            max_items=args.max_items_per_keyword,
            headless=not args.show_browser,
        )
        all_products.extend(items)

    output = Path(args.output)
    if args.format == "json":
        save_json(all_products, output)
    else:
        save_csv(all_products, output)

    print(f"Done. Crawled {len(all_products)} products.")
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
