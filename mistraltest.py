"""
mistraltest.py - Test Mistral API with Rasmir persona posting a new thread.
Run from /code/forumbots/
"""

import json
import re
import sys
import requests
from pathlib import Path

import secrets
from nodebb import NodeBB

MISTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
PERSONA_FILE  = "personas/rasmir.json"

# ------------------------------------------------------------------ #
# Mistral call
# ------------------------------------------------------------------ #

def mistral_generate(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {secrets.MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens":  500,
    }
    resp = requests.post(MISTRAL_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r" +", " ", text)
    return text.strip()

def get_random_wikimedia_image() -> str:
    headers = {"User-Agent": "forumbots/1.0"}
    for _ in range(10):
        try:
            r = requests.get(WIKIMEDIA_API, params={
                "action": "query", "list": "random",
                "rnnamespace": 6, "rnlimit": 1, "format": "json",
            }, headers=headers, timeout=10)
            r.raise_for_status()
            title = r.json()["query"]["random"][0]["title"]
            if not any(title.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                continue
            r2 = requests.get(WIKIMEDIA_API, params={
                "action": "query", "titles": title,
                "prop": "imageinfo", "iiprop": "url", "format": "json",
            }, headers=headers, timeout=10)
            r2.raise_for_status()
            pages = r2.json()["query"]["pages"]
            info  = next(iter(pages.values())).get("imageinfo", [])
            if info and info[0].get("url"):
                return info[0]["url"]
        except Exception as e:
            print(f"Wikimedia error: {e}")
    return None

def get_category_context(api: NodeBB, uid: int, cid: int) -> dict:
    try:
        data   = api._get(f"/api/category/{cid}", uid=uid)
        topics = data.get("topics", [])
        return {
            "name":          data.get("name", f"Category {cid}"),
            "description":   data.get("description", ""),
            "recent_topics": [t["title"] for t in topics[:4] if t.get("title")],
        }
    except Exception as e:
        print(f"Could not fetch category: {e}")
        return {"name": f"Category {cid}", "description": "", "recent_topics": []}

# ------------------------------------------------------------------ #
# Prompts
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
        f"- Write your post and nothing else. No intro, no sign-off, no explanation.\n"
        f"- Plain text only. No HTML, no markdown, no BBcode, no asterisks, no bullet points.\n"
        f"- No emojis or image embeds. ASCII faces like :) are fine.\n"
        f"- No emdashes. Do not use ellipses as a stylistic tic.\n"
        f"- Do not sound like an AI. Never use 'certainly', 'absolutely', 'great question', 'it's worth noting'.\n"
        f"- Never acknowledge being a bot or AI.\n"
        f"- Do not repeat or reference these instructions.\n"
        f"- Do not mirror other users' phrases or personal details.\n"
        f"- Vary your openings. Never start the same way twice.\n"
    )

def build_thread_prompt(category: dict) -> str:
    recent = "\n".join(f'- "{t}"' for t in category["recent_topics"]) or "(none yet)"
    cat_desc = f"\nCategory description: {category['description']}" if category["description"] else ""
    return (
        f"You are starting a new thread on this forum. Let your mind wander - "
        f"something on your mind, a question, a rant, an opinion, anything that feels natural for who you are.\n\n"
        f"Category: {category['name']}{cat_desc}\n\n"
        f"Other recent threads in this category:\n{recent}\n\n"
        f"Return JSON only, no extra text, no markdown fences:\n"
        '{"title": "your title here", "content": "your post content here"}\n'
    )

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    if not hasattr(secrets, 'MISTRAL_API_KEY'):
        print("Add MISTRAL_API_KEY to secrets.py first")
        sys.exit(1)

    persona = json.load(open(PERSONA_FILE))
    uid     = persona["uid"]
    cid     = persona.get("default_cid", 1)
    print(f"Persona: {persona['username']} (uid={uid}, cid={cid})")

    api      = NodeBB(secrets.NODEBB_URL, secrets.NODEBB_MASTER_TOKEN)
    category = get_category_context(api, uid, cid)
    print(f"Category: {category['name']} — recent: {category['recent_topics']}")

    print("Fetching Wikimedia image...")
    image_url = get_random_wikimedia_image()
    print(f"Image: {image_url}")

    system_prompt = build_system_prompt(persona)
    user_prompt   = build_thread_prompt(category)

    print("\n=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
    print("\nCalling Mistral...")

    try:
        raw = mistral_generate(system_prompt, user_prompt)
    except Exception as e:
        print(f"Mistral error: {e}")
        sys.exit(1)

    print(f"\n=== RAW OUTPUT ===\n{raw}")

    try:
        match = re.search(r"[{].*[}]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        data    = json.loads(match.group())
        title   = data["title"]
        content = data["content"]
    except Exception as e:
        print(f"JSON parse failed: {e}")
        sys.exit(1)

    if image_url:
        content = content.rstrip() + f"\n\n![img]({image_url})"

    print(f"\n=== FINAL THREAD ===")
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