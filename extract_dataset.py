"""
extract_dataset.py — Export all forum threads to a plaintext training dataset.

Output format:
    === Thread Title ===
    username: post content
    username2: reply content
    ...

Threads are separated by a blank line. One post per line.
"""

import re
import sys
from pathlib import Path

from nodebb import NodeBB
import secrets

ADMIN_UID   = 1
OUTPUT_FILE = Path("/code/forumbots/dataset.txt")


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&nbsp;", " "))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_all_category_ids(api: NodeBB) -> list[tuple[int, str]]:
    """Return flat list of (cid, name) for all categories including children."""
    data = api._get("/api/categories", uid=ADMIN_UID)
    cats = data.get("categories", [])

    result = []

    def collect(cat_list):
        for cat in cat_list:
            cid  = cat.get("cid")
            name = cat.get("name", f"cid {cid}")
            if cid:
                result.append((cid, name))
            children = cat.get("children", [])
            if children:
                collect(children)

    collect(cats)
    return result


def get_topic_ids_in_category(api: NodeBB, cid: int) -> list[int]:
    """Paginate through a category and collect all tids."""
    tids = []
    page = 1
    while True:
        data       = api._get(f"/api/category/{cid}", uid=ADMIN_UID, params={"page": page})
        topics     = data.get("topics", [])
        tids.extend(t["tid"] for t in topics if t.get("tid"))
        page_count = data.get("pagination", {}).get("pageCount", 1)
        if page >= page_count:
            break
        page += 1
    return tids


def get_full_topic(api: NodeBB, tid: int) -> tuple[str, list[dict]]:
    """Return (title, all_posts) for a topic, handling pagination."""
    all_posts = []
    title     = "untitled"
    page      = 1
    while True:
        data       = api._get(f"/api/topic/{tid}/_", uid=ADMIN_UID, params={"page": page})
        title      = data.get("title", title)
        all_posts.extend(data.get("posts", []))
        page_count = data.get("pagination", {}).get("pageCount", 1)
        if page >= page_count:
            break
        page += 1
    return title, all_posts


def format_thread(title: str, posts: list[dict]) -> str | None:
    lines = [f"=== {title} ==="]
    for post in posts:
        if post.get("deleted"):
            continue
        username = (post.get("user", {}).get("username")
                    or post.get("username", "unknown"))
        content  = strip_html(post.get("content", ""))
        if not content:
            continue
        lines.append(f"{username}: {content}")

    # Need at least the OP (header + 1 post line)
    if len(lines) < 2:
        return None
    return "\n".join(lines)


def main():
    api = NodeBB(secrets.NODEBB_URL, secrets.NODEBB_MASTER_TOKEN)

    print("Discovering categories...")
    categories = get_all_category_ids(api)
    print(f"Found {len(categories)} categories: {[name for _, name in categories]}")

    seen_tids   = set()
    all_threads = []

    for cid, name in categories:
        print(f"\nCategory: {name!r} (cid={cid})")
        try:
            tids = get_topic_ids_in_category(api, cid)
        except Exception as e:
            print(f"  Could not list topics: {e}")
            continue
        print(f"  {len(tids)} topics")

        for tid in tids:
            if tid in seen_tids:
                continue
            seen_tids.add(tid)
            try:
                title, posts = get_full_topic(api, tid)
                block = format_thread(title, posts)
                if block:
                    all_threads.append(block)
                    print(f"  tid={tid}: {title!r} ({len(posts)} posts)")
            except Exception as e:
                print(f"  tid={tid}: error - {e}")

    output = "\n\n".join(all_threads)
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    total_posts = sum(b.count("\n") for b in all_threads)  # each newline = one post line
    print(f"\nDone. {len(all_threads)} threads, ~{total_posts} posts -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
