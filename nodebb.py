"""
nodebb.py — NodeBB API wrapper for forumbots
Covers all read/write operations the runner needs.
"""

import requests
import logging

log = logging.getLogger(__name__)


class NodeBB:
    def __init__(self, base_url: str, master_token: str):
        """
        base_url     : e.g. "http://localhost:4567" — no trailing slash
        master_token : admin master token from ACP > Settings > API Access
        """
        self.base_url = base_url.rstrip("/")
        self.master_token = master_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {master_token}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get(self, path: str, uid: int, params: dict = None) -> dict:
        """
        GET from the Read API, impersonating uid via _uid query param.
        NodeBB read API accepts _uid as a query param when using a master token.
        """
        params = params or {}
        params["_uid"] = uid
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, uid: int, payload: dict = None) -> dict:
        """POST to the Write API (v3), injecting _uid into the body."""
        body = payload or {}
        body["_uid"] = uid
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, uid: int, payload: dict = None) -> dict:
        """PUT to the Write API (v3), injecting _uid into the body."""
        body = payload or {}
        body["_uid"] = uid
        url = f"{self.base_url}{path}"
        resp = self.session.put(url, json=body)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Notifications
    # ------------------------------------------------------------------ #

    def get_notifications(self, uid: int) -> list[dict]:
        """
        Returns unread notifications for a user.
        GET /api/notifications
        Response shape: { notifications: [...], hasMore: bool }
        Each notification includes: nid, tid, pid, type, bodyShort, read, etc.
        """
        data = self._get("/api/notifications", uid=uid)
        return data.get("notifications", [])

    def mark_topic_read(self, uid: int, tid: int) -> None:
        """
        Marks all posts in a topic as read for this user.
        PUT /api/v3/topics/:tid/read
        """
        try:
            self._put(f"/api/v3/topics/{tid}/read", uid=uid)
        except Exception as e:
            log.warning(f"Failed to mark tid {tid} read for uid {uid}: {e}")

    # ------------------------------------------------------------------ #
    # Topics
    # ------------------------------------------------------------------ #

    def get_topic(self, uid: int, tid: int, slug: str = "_") -> dict:
        """
        Fetch a full topic with its posts.
        GET /api/topic/:tid/:slug
        Returns topic object with .posts list.
        Each post: { pid, uid, content, timestamp, username, ... }
        """
        data = self._get(f"/api/topic/{tid}/{slug}", uid=uid)
        return data

    def get_recent_topics(self, uid: int) -> list[dict]:
        """
        Fetch recently active topics across the forum.
        GET /api/recent
        Returns list of topic objects with tid, title, cid, postcount, etc.
        """
        data = self._get("/api/recent", uid=uid)
        return data.get("topics", [])

    def create_topic(self, uid: int, cid: int, title: str, content: str) -> dict:
        """
        Post a new topic to a category.
        POST /api/v3/topics
        Returns the created topic object including tid.
        """
        payload = {
            "cid": cid,
            "title": title,
            "content": content,
        }
        data = self._post("/api/v3/topics", uid=uid, payload=payload)
        return data.get("response", data)

    # ------------------------------------------------------------------ #
    # Posts / Replies
    # ------------------------------------------------------------------ #

    def reply_to_topic(self, uid: int, tid: int, content: str, to_pid: int = None) -> dict:
        """
        Post a reply in an existing topic.
        POST /api/v3/topics/:tid
        to_pid: if set, reply is directed at that specific post.
        Returns created post object including pid.
        """
        payload = {"content": content}
        if to_pid:
            payload["toPid"] = to_pid
        data = self._post(f"/api/v3/topics/{tid}", uid=uid, payload=payload)
        return data.get("response", data)

    def get_post(self, uid: int, pid: int) -> dict:
        """
        Fetch a single post by pid.
        GET /api/post/:pid
        Returns post object: { pid, tid, uid, content, username, timestamp, ... }
        """
        data = self._get(f"/api/post/{pid}", uid=uid)
        return data.get("post", data)

    # ------------------------------------------------------------------ #
    # User
    # ------------------------------------------------------------------ #

    def get_user_posts(self, uid: int, userslug: str, count: int = 10) -> list[dict]:
        """
        Fetch recent posts by a user (for voice consistency context).
        GET /api/user/:userslug/posts
        Returns list of post objects, most recent first.
        """
        data = self._get(f"/api/user/{userslug}/posts", uid=uid)
        posts = data.get("posts", [])
        return posts[:count]