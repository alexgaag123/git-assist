"""One-shot GitLab setup script: wait for API, create PAT, create project, register webhook."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

GITLAB_URL = os.environ.get("GITLAB_URL", "http://gitlab")
GITLAB_PROJECT_NAME = os.environ.get("GITLAB_PROJECT_NAME", "git-assist-repo")
WORKER_WEBHOOK_URL = os.environ.get("WORKER_WEBHOOK_URL", "http://worker:8001/webhook")
GITLAB_SETUP_TOKEN = os.environ.get("GITLAB_SETUP_TOKEN", "")
GITLAB_ROOT_PASSWORD = os.environ.get("GITLAB_ROOT_PASSWORD", "")

MAX_WAIT_ATTEMPTS = 60
WAIT_SLEEP_SECONDS = 5


def _webhook_secret(env: dict[str, str]) -> str:
    """Return the GitLab hook token, accepting the worker-prefixed env as fallback."""
    return env.get("GITLAB_WEBHOOK_SECRET") or env.get("GIT_ASSIST_WEBHOOK_SECRET", "")


GITLAB_WEBHOOK_SECRET = _webhook_secret(dict(os.environ))


def _request(
    method: str,
    url: str,
    token: str | None = None,
    body: Any = None,
) -> tuple[int, Any]:
    """Make a JSON HTTP request; return (status_code, parsed_body) without raising on errors."""
    data: bytes | None = None
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        # ponytail: Bearer works for both OAuth tokens and PATs in GitLab
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body_bytes = e.read()
            parsed = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            parsed = {}
        return e.code, parsed


def _wait_for_api(base: str) -> None:
    """Poll GET /api/v4/version until HTTP 200 or 401; exit non-zero if it never comes up."""
    url = f"{base}/api/v4/version"
    for attempt in range(1, MAX_WAIT_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    print(f"[gitlab-setup] GitLab API ready (attempt {attempt})")
                    return
        except urllib.error.HTTPError as e:
            # ponytail: 401 means the API is up (just needs auth) — treat as ready
            if e.code == 401:
                print(f"[gitlab-setup] GitLab API ready/401 (attempt {attempt})")
                return
            print(f"[gitlab-setup] attempt {attempt}/{MAX_WAIT_ATTEMPTS}: HTTP {e.code}")
        except Exception as exc:
            print(f"[gitlab-setup] attempt {attempt}/{MAX_WAIT_ATTEMPTS}: {exc}")
        time.sleep(WAIT_SLEEP_SECONDS)
    print("[gitlab-setup] ERROR: GitLab API never became ready", file=sys.stderr)
    raise SystemExit(1)


def _oauth_token(base: str, password: str) -> str:
    """Get OAuth access token via resource-owner password grant."""
    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "username": "root",
            "password": password,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/oauth/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["access_token"]  # type: ignore[no-any-return]


def main() -> int:
    _wait_for_api(GITLAB_URL)

    # Obtain working token: prefer GITLAB_SETUP_TOKEN, fall back to OAuth password grant.
    token: str
    if GITLAB_SETUP_TOKEN:
        token = GITLAB_SETUP_TOKEN
        print("[gitlab-setup] Using GITLAB_SETUP_TOKEN")
    elif GITLAB_ROOT_PASSWORD:
        print("[gitlab-setup] No GITLAB_SETUP_TOKEN; obtaining OAuth token via root password ...")
        try:
            token = _oauth_token(GITLAB_URL, GITLAB_ROOT_PASSWORD)
            print("[gitlab-setup] OAuth token obtained")
        except Exception as exc:
            print(f"[gitlab-setup] ERROR: OAuth token failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "[gitlab-setup] ERROR: neither GITLAB_SETUP_TOKEN nor GITLAB_ROOT_PASSWORD set",
            file=sys.stderr,
        )
        return 1

    # Allow webhooks to internal/local network (needed for worker on Docker network).
    _request(
        "PUT",
        f"{GITLAB_URL}/api/v4/application/settings",
        token=token,
        body={"allow_local_requests_from_web_hooks_and_services": True},
    )
    print("[gitlab-setup] Local network webhooks enabled")

    # Ensure project exists.
    encoded = urllib.parse.quote(f"root/{GITLAB_PROJECT_NAME}", safe="")
    status, resp = _request("GET", f"{GITLAB_URL}/api/v4/projects/{encoded}", token=token)
    if status == 200 and isinstance(resp, dict):
        project_id: int = resp["id"]
        print(f"[gitlab-setup] Project '{GITLAB_PROJECT_NAME}' already exists (id={project_id})")
    elif status == 404:
        print(f"[gitlab-setup] Creating project '{GITLAB_PROJECT_NAME}' ...")
        status, resp = _request(
            "POST",
            f"{GITLAB_URL}/api/v4/projects",
            token=token,
            body={"name": GITLAB_PROJECT_NAME, "visibility": "internal"},
        )
        if status not in (200, 201) or not isinstance(resp, dict):
            print(
                f"[gitlab-setup] ERROR: project creation failed ({status}): {resp}",
                file=sys.stderr,
            )
            return 1
        project_id = resp["id"]
        print(f"[gitlab-setup] Project created (id={project_id})")
    else:
        print(f"[gitlab-setup] ERROR: unexpected status {status} checking project", file=sys.stderr)
        return 1

    # Idempotently register push webhook.
    status, hooks = _request("GET", f"{GITLAB_URL}/api/v4/projects/{project_id}/hooks", token=token)
    if status != 200 or not isinstance(hooks, list):
        print(f"[gitlab-setup] ERROR: failed to list hooks ({status})", file=sys.stderr)
        return 1

    existing_hook = next((h for h in hooks if h.get("url") == WORKER_WEBHOOK_URL), None)
    if isinstance(existing_hook, dict) and existing_hook.get("id") is not None:
        print("[gitlab-setup] webhook already registered; refreshing settings")
        status, resp = _request(
            "PUT",
            f"{GITLAB_URL}/api/v4/projects/{project_id}/hooks/{existing_hook['id']}",
            token=token,
            body={
                "url": WORKER_WEBHOOK_URL,
                "push_events": True,
                "token": GITLAB_WEBHOOK_SECRET,
            },
        )
        if status not in (200, 201):
            print(
                f"[gitlab-setup] ERROR: webhook update failed ({status}): {resp}",
                file=sys.stderr,
            )
            return 1
        print("[gitlab-setup] webhook settings refreshed")
    else:
        print(f"[gitlab-setup] Registering webhook -> {WORKER_WEBHOOK_URL} ...")
        status, resp = _request(
            "POST",
            f"{GITLAB_URL}/api/v4/projects/{project_id}/hooks",
            token=token,
            body={
                "url": WORKER_WEBHOOK_URL,
                "push_events": True,
                "token": GITLAB_WEBHOOK_SECRET,
            },
        )
        if status not in (200, 201):
            print(
                f"[gitlab-setup] ERROR: webhook creation failed ({status}): {resp}",
                file=sys.stderr,
            )
            return 1
        print("[gitlab-setup] webhook registered")

    # Create a PAT for worker repo cloning; write to shared secrets volume.
    secrets_path = pathlib.Path("/run/secrets/gitlab_clone_token")
    if secrets_path.exists() and secrets_path.read_text().strip():
        print("[gitlab-setup] Clone PAT file already exists, skipping")
    else:
        status, resp = _request(
            "POST",
            f"{GITLAB_URL}/api/v4/users/1/personal_access_tokens",
            token=token,
            body={
                "name": "worker-clone",
                "scopes": ["read_repository"],
                "expires_at": "2027-06-01",
            },
        )
        if status in (200, 201) and isinstance(resp, dict) and "token" in resp:
            pat_value = str(resp["token"])
            secrets_path.parent.mkdir(parents=True, exist_ok=True)
            secrets_path.write_text(pat_value)
            print("[gitlab-setup] Clone PAT created and written to /run/secrets/gitlab_clone_token")
        else:
            print(
                f"[gitlab-setup] WARNING: PAT creation failed ({status}): {resp}", file=sys.stderr
            )

    clone_url = f"http://localhost:8929/root/{GITLAB_PROJECT_NAME}.git"
    print(f"[gitlab-setup] Clone URL: {clone_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
