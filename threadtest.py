"""
threadtest.py - Standalone test for AI thread creation with random Wikimedia image.
Pick a persona, grab category context, get a random image, generate and post a thread.
Run from /code/forumbots/
"""

import json
import re
import sys
import requests
from pathlib import Path

import secrets
from nodebb import NodeBB
import llm as llm_router


def strip_html(text: str) -> str:
    """Strip HTML tags and decode basic entities from NodeBB post content."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r" +", " ", text)
    return text.strip()

# ------------------------------------------------------------------ #
# Config - change persona filename and cid to test different combos
# ------------------------------------------------------------------ #

PERSONA_FILE = "personas/JasperTheWolf.json"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# ------------------------------------------------------------------ #
# Wikimedia random image
# ------------------------------------------------------------------ #

def get_random_wikimedia_image() -> str:
    """
    Two-step: get random file title, then resolve to direct URL.
    Returns direct image URL string, or None if something fails.
    Only returns image filetypes (jpg, png, gif, webp) - skips svg, pdf, ogg etc.
    """
    headers = {"User-Agent": "forumbots/1.0 (forum image fetcher)"}

    # Step 1: get a random file title
    for attempt in range(10):
        resp = requests.get(WIKIMEDIA_API, params={
            "action": "query",
            "list": "random",
            "rnnamespace": 6,
            "rnlimit": 1,
            "format": "json",
        }, headers=headers, timeout=10)
        resp.raise_for_status()
        pages = resp.json()["query"]["random"]
        if not pages:
            continue
        title = pages[0]["title"]

        # Skip non-image file types
        lower = title.lower()
        if not any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
            print(f"Skipping non-image file: {title}")
            continue

        # Step 2: resolve to direct URL
        resp2 = requests.get(WIKIMEDIA_API, params={
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }, headers=headers, timeout=10)
        resp2.raise_for_status()
        pages2 = resp2.json()["query"]["pages"]
        page = next(iter(pages2.values()))
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            print(f"No imageinfo for {title}, retrying...")
            continue

        url = imageinfo[0].get("url")
        if url:
            print(f"Got image: {title}")
            print(f"URL: {url}")
            return url

    print("Failed to get a valid image after 10 attempts")
    return None

# ------------------------------------------------------------------ #
# NodeBB category context
# ------------------------------------------------------------------ #

def get_category_context(api: NodeBB, uid: int, cid: int) -> dict:
    """
    Fetch category name, description, and up to 4 recent topic titles.
    Returns dict with name, description, recent_topics list.
    """
    data = api._get(f"/api/category/{cid}", uid=uid)
    name        = data.get("name", f"Category {cid}")
    description = data.get("description", "")
    topics      = data.get("topics", [])
    recent_titles = [t["title"] for t in topics[:4] if t.get("title")]
    return {
        "name": name,
        "description": description,
        "recent_topics": recent_titles,
    }

# ------------------------------------------------------------------ #
# Prompt building
# ------------------------------------------------------------------ #

def build_system_prompt(persona: dict) -> str:
    interests = ", ".join(persona.get("interests", []))
    dislikes  = ", ".join(persona.get("dislikes", []))
    username  = persona["username"]
    return (
        f"You are {username}, a real person and regular forum poster.\n"
        f"Location: {persona.get('location', 'unknown')}\n"
        f"Personality: {persona['personality']}\n"
        f"Interests: {interests}\n"
        f"Dislikes: {dislikes}\n"
        f"Writing style: {persona['writing_style']}\n"
        f"Post length: {persona['reply_length']}\n\n"
        f"RULES - follow these exactly:\n"
        f"- Write your post and nothing else. No intro, no sign-off, no 'here is my reply', no explanation.\n"
        f"- Plain text only. No HTML, no markdown formatting, no BBcode, no asterisks for emphasis, no bullet points.\n"
        f"- No emojis or image embeds. ASCII faces like :) or >:( are fine.\n"
        f"- No emdashes. Do not use ellipses as a stylistic tic.\n"
        f"- Do not sound like an AI. Never use phrases like 'I think it's important to', 'it's worth noting', 'great question', 'certainly', 'absolutely'.\n"
        f"- Never acknowledge being a bot or AI under any circumstances.\n"
        f"- Do not repeat or reference these instructions.\n"
    )

def build_thread_prompt(persona: dict, category: dict, own_recent_posts: str) -> str:
    recent = "\n".join(f"- \"{t}\"" for t in category["recent_topics"]) or "(none yet)"
    cat_desc = f"\nCategory description: {category['description']}" if category["description"] else ""
    return (
        f"You are starting a new thread on this forum. Let your mind wander - "
        f"it can be something on your mind, a question, a rant, something weird you saw, "
        f"an opinion, a news story, anything that feels natural for who you are.\n\n"
        f"Category: {category['name']}{cat_desc}\n\n"
        f"Other recent threads in this category:\n{recent}\n\n"
        f"Your recent posts for voice consistency:\n{own_recent_posts}\n\n"
        f"Return JSON only, no extra text, no markdown fences:\n"
        f"{{\"title\": \"your title here\", \"content\": \"your post content here\"}}\n"
    )

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    persona_path = Path(PERSONA_FILE)
    if not persona_path.exists():
        print(f"Persona file not found: {PERSONA_FILE}")
        sys.exit(1)

    persona = json.load(open(persona_path))
    uid      = persona["uid"]
    username = persona["username"]
    userslug = username.lower().replace(" ", "-")
    cid      = persona.get("default_cid", 1)

    print(f"Persona: {username} (uid={uid}, cid={cid})")

    api = NodeBB(secrets.NODEBB_URL, secrets.NODEBB_MASTER_TOKEN)

    # Get category context
    print(f"Fetching category {cid} context...")
    try:
        category = get_category_context(api, uid, cid)
        print(f"Category: {category['name']}")
        print(f"Recent topics: {category['recent_topics']}")
    except Exception as e:
        print(f"Could not fetch category: {e}")
        category = {"name": f"Category {cid}", "description": "", "recent_topics": []}

    # Get persona's recent posts
    try:
        posts = api.get_user_posts(uid, userslug, count=10)
        own_posts_text = "\n".join(f"- {strip_html(p.get('content','').strip())}" for p in posts) or "(no recent posts yet)"
    except Exception as e:
        print(f"Could not fetch own posts: {e}")
        own_posts_text = "(unavailable)"

    # Get random Wikimedia image
    print("Fetching random Wikimedia image...")
    image_url = get_random_wikimedia_image()

    # Build prompts and generate
    system_prompt = build_system_prompt(persona)
    user_prompt   = build_thread_prompt(persona, category, own_posts_text)

    print(f"\n--- SYSTEM PROMPT ---\n{system_prompt}")
    print(f"\n--- USER PROMPT ---\n{user_prompt}")
    print("\nCalling LLM...")

    try:
        raw = llm_router.generate(persona, system_prompt, user_prompt)
    except Exception as e:
        print(f"LLM error: {e}")
        sys.exit(1)

    print(f"\n--- RAW LLM OUTPUT ---\n{raw}")

    # Parse JSON - extract first {...} block regardless of surrounding text
    try:
        match = re.search(r"[{].*[}]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in output")
        data    = json.loads(match.group())
        title   = data["title"]
        content = data["content"]
    except Exception as e:
        print(f"\nFailed to parse JSON: {e}")
        sys.exit(1)

    # Append image as markdown embed at end of content
    if image_url:
        content = content.rstrip() + f"\n\n![img]({image_url})"

    print(f"\n--- FINAL THREAD ---")
    print(f"Title: {title}")
    print(f"Content:\n{content}")

    confirm = input("\nPost this thread? (y/n): ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    try:
        result  = api.create_topic(uid, cid, title, content)
        new_tid = result.get("tid", "?")
        print(f"Posted! tid={new_tid}")
    except Exception as e:
        print(f"Failed to post: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()