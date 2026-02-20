#!/usr/bin/env python3
"""Upload report assets to Notion from a markdown upload block.

Usage example:
  export NOTION_TOKEN="ntn_xxx"
  python3 unitree_go2_phm/scripts/rsl_rl/reports/upload_to_notion.py \
    --upload-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_notion_upload_block.md \
    --page-title "2026-02-20 새벽 실험 업데이트 (TB+영상)" \
    --report-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_realobs_growth_journal.md
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

NOTION_VERSION = "2022-06-28"
DEFAULT_PARENT_WEEK_TITLE = "일주일"
DEFAULT_PARENT_MONTH_TITLE = "2월"
DEFAULT_PARENT_YEAR_TITLE = "2026년"
UPLOADABLE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mov", ".webm", ".pdf"}


def _api_request(token: str, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _search_page_by_title(token: str, title: str) -> dict[str, Any] | None:
    out = _api_request(token, "POST", "https://api.notion.com/v1/search", {"query": title, "page_size": 100})
    for item in out.get("results", []):
        if item.get("object") != "page":
            continue
        props = item.get("properties", {})
        page_title = ""
        for v in props.values():
            if v.get("type") == "title":
                page_title = "".join(x.get("plain_text", "") for x in v.get("title", []))
                break
        if page_title == title:
            return item
    return None


def _list_children(token: str, block_id: str) -> list[dict[str, Any]]:
    out = _api_request(
        token,
        "GET",
        f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100",
    )
    return out.get("results", [])


def _find_child_page_by_title(token: str, parent_id: str, title: str) -> str | None:
    for block in _list_children(token, parent_id):
        if block.get("type") == "child_page" and block.get("child_page", {}).get("title") == title:
            return block.get("id")
    return None


def _resolve_parent_week_page_id(
    token: str,
    year_title: str,
    month_title: str,
    week_title: str,
) -> str:
    year_page = _search_page_by_title(token, year_title)
    if not year_page:
        raise RuntimeError(f"Cannot find year page: {year_title}")
    year_id = year_page["id"]

    month_id = _find_child_page_by_title(token, year_id, month_title)
    if not month_id:
        raise RuntimeError(f"Cannot find month page '{month_title}' under '{year_title}'")

    week_id = _find_child_page_by_title(token, month_id, week_title)
    if not week_id:
        raise RuntimeError(f"Cannot find week page '{week_title}' under '{month_title}'")
    return week_id


def _create_child_page(token: str, parent_page_id: str, title: str) -> dict[str, Any]:
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}],
            }
        },
    }
    return _api_request(token, "POST", "https://api.notion.com/v1/pages", payload)


def _append_blocks(token: str, block_id: str, children: list[dict[str, Any]]) -> None:
    if not children:
        return
    # Notion limit per append request.
    chunk_size = 50
    for i in range(0, len(children), chunk_size):
        _api_request(
            token,
            "PATCH",
            f"https://api.notion.com/v1/blocks/{block_id}/children",
            {"children": children[i : i + chunk_size]},
        )


def _rt(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}}]


def _upload_file_to_notion(token: str, file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if file_path.suffix.lower() not in UPLOADABLE_EXTS:
        raise RuntimeError(f"unsupported file extension for upload: {file_path.suffix}")
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        # Ensure mp4 works, fall back for other unknowns.
        if file_path.suffix.lower() == ".mp4":
            mime = "video/mp4"
        elif file_path.suffix.lower() in {".png"}:
            mime = "image/png"
        elif file_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        else:
            mime = "application/octet-stream"

    slot = _api_request(
        token,
        "POST",
        "https://api.notion.com/v1/file_uploads",
        {
            "filename": file_path.name,
            "content_type": mime,
            "content_length": file_path.stat().st_size,
        },
    )
    file_id = slot["id"]
    send_url = f"https://api.notion.com/v1/file_uploads/{file_id}/send"

    # Use curl multipart for binary upload.
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        send_url,
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        f"Notion-Version: {NOTION_VERSION}",
        "-F",
        f"file=@{file_path};type={mime}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"file upload failed: {file_path}\n{proc.stderr}")
    # Return created upload id, status validated by attach attempt.
    return file_id


def _file_block_from_upload_id(upload_id: str, caption: str, local_path: Path) -> dict[str, Any]:
    mime, _ = mimetypes.guess_type(str(local_path))
    if not mime and local_path.suffix.lower() == ".mp4":
        mime = "video/mp4"
    if mime and mime.startswith("image/"):
        block_type = "image"
    elif mime and mime.startswith("video/"):
        block_type = "video"
    else:
        block_type = "file"

    return {
        "object": "block",
        "type": block_type,
        block_type: {
            "type": "file_upload",
            "file_upload": {"id": upload_id},
            "caption": _rt(caption),
        },
    }


def _parse_upload_markdown(upload_md: Path) -> list[tuple[str, str | None]]:
    # Returns list of (line_text, file_path_or_none)
    out: list[tuple[str, str | None]] = []
    path_re = re.compile(r"`([^`]+)`")
    for raw in upload_md.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            out.append((line, None))
            continue
        m = path_re.search(line)
        if m:
            out.append((line, m.group(1)))
        else:
            out.append((line, None))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload report assets to Notion page.")
    parser.add_argument("--upload-md", required=True, help="Markdown file containing upload asset list.")
    parser.add_argument(
        "--page-title",
        default="",
        help="Child page title (default mode) or section heading text (when --append-to-week-page).",
    )
    parser.add_argument("--report-md", default="", help="Main report markdown path to mention in page.")
    parser.add_argument("--token", default="", help="Notion token (or use env var).")
    parser.add_argument("--token-env", default="NOTION_TOKEN", help="Token env var name.")
    parser.add_argument("--week-page-id", default="", help="Direct target parent week page id.")
    parser.add_argument(
        "--append-to-week-page",
        action="store_true",
        help="Append blocks directly into week page body (no child page creation).",
    )
    parser.add_argument("--year-title", default=DEFAULT_PARENT_YEAR_TITLE)
    parser.add_argument("--month-title", default=DEFAULT_PARENT_MONTH_TITLE)
    parser.add_argument("--week-title", default=DEFAULT_PARENT_WEEK_TITLE)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded.")
    args = parser.parse_args()

    token = args.token or os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError("Notion token missing. Set --token or NOTION_TOKEN env var.")

    upload_md = Path(args.upload_md).resolve()
    if not upload_md.exists():
        raise FileNotFoundError(str(upload_md))

    parent_week_id = args.week_page_id
    if not parent_week_id:
        parent_week_id = _resolve_parent_week_page_id(token, args.year_title, args.month_title, args.week_title)

    entries = _parse_upload_markdown(upload_md)

    if args.dry_run:
        print("parent_week_id:", parent_week_id)
        print("append_to_week_page:", args.append_to_week_page)
        print("page_title:", args.page_title)
        for line, p in entries:
            print("-", line, "=>", p or "(text)")
        return 0

    if args.append_to_week_page:
        page_id = parent_week_id
        page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
    else:
        if not args.page_title:
            raise RuntimeError("--page-title is required unless --append-to-week-page is set.")
        page = _create_child_page(token, parent_week_id, args.page_title)
        page_id = page["id"]
        page_url = page.get("url", "")

    pre_blocks = []
    if args.page_title:
        pre_blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rt(args.page_title)}})
    pre_blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(f"source: {upload_md}")}})
    if args.report_md:
        pre_blocks.append(
            {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(f"report: {Path(args.report_md).resolve()}")}}
        )
    _append_blocks(token, page_id, pre_blocks)

    pending_blocks: list[dict[str, Any]] = []
    for line, p in entries:
        if line.startswith("## "):
            pending_blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rt(line[3:])}})
            continue
        if line.startswith("# "):
            pending_blocks.append({"object": "block", "type": "heading_1", "heading_1": {"rich_text": _rt(line[2:])}})
            continue

        if p:
            local_path = Path(p)
            if not local_path.is_absolute():
                local_path = (Path.cwd() / local_path).resolve()
            if local_path.exists():
                try:
                    _append_blocks(token, page_id, pending_blocks)
                    pending_blocks = []
                    upload_id = _upload_file_to_notion(token, local_path)
                    caption = line.split("`", 1)[0].strip("- ").strip()
                    if not caption:
                        caption = local_path.name
                    block = _file_block_from_upload_id(upload_id, caption, local_path)
                    _append_blocks(token, page_id, [block])
                except Exception:
                    pending_blocks.append(
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": _rt(f"[path] {local_path}")},
                        }
                    )
            else:
                pending_blocks.append(
                    {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt(f"[missing] {local_path}")}}
                )
        else:
            txt = line.lstrip("- ").strip()
            pending_blocks.append(
                {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt(txt)}}
            )
    _append_blocks(token, page_id, pending_blocks)

    print("DONE")
    print("PAGE_ID", page_id)
    print("URL", page_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
