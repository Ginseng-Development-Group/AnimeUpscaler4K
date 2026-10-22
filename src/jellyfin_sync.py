#!/usr/bin/env python3
"""
Jellyfin Library Sync & Metadata Manager
========================================
Triggers a Jellyfin library scan and optionally populates & locks localized
metadata (Titles, Overviews, Genres, Studios, Cast/Crew, Ratings, IMDb/TMDB IDs,
and Primary/Backdrop/Thumb images) via the Jellyfin REST API.
"""
import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path


class JellyfinClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f'MediaBrowser Token="{self.api_key}"',
            "Content-Type": "application/json",
        }

    def get(self, path: str):
        req = urllib.request.Request(f"{self.base_url}{path}", headers=self.headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post(self, path: str, payload: dict = None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else b""
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status

    def upload_image(self, item_id: str, image_type: str, file_path: Path):
        mime = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(file_path.read_bytes())
        req = urllib.request.Request(
            f"{self.base_url}/Items/{item_id}/Images/{image_type}",
            data=b64,
            headers={
                "Authorization": f'MediaBrowser Token="{self.api_key}"',
                "Content-Type": mime,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status

    def refresh_library(self):
        status = self.post("/Library/Refresh")
        print(f"[Jellyfin] Library refresh triggered (HTTP {status})")

    def apply_metadata(self, metadata_items: list):
        users = self.get("/Users")
        if not users:
            raise RuntimeError("No Jellyfin users found")
        uid = users[0]["Id"]

        for entry in metadata_items:
            item_id = entry["item_id"]
            item = self.get(f"/Users/{uid}/Items/{item_id}")
            for field in (
                "Name", "OriginalTitle", "ForcedSortName", "SortName",
                "Overview", "ProductionYear", "PremiereDate",
                "IndexNumber", "ParentIndexNumber",
                "CommunityRating", "CriticRating", "OfficialRating",
                "Genres", "Studios", "People", "ProviderIds",
            ):
                if field in entry:
                    item[field] = entry[field]

            if entry.get("lock", True):
                item["LockData"] = True
                item["LockedFields"] = [
                    "Name", "Overview", "Genres", "Studios", "Cast", "OfficialRating"
                ]

            status = self.post(f"/Items/{item_id}", item)
            print(f"[Jellyfin] Updated '{item.get('Name')}' ({item_id}) -> HTTP {status}")

            for img_type, img_path in entry.get("images", {}).items():
                p = Path(img_path)
                if p.exists():
                    img_status = self.upload_image(item_id, img_type, p)
                    print(f"[Jellyfin] Uploaded {img_type} for {item_id} -> HTTP {img_status}")


def main():
    parser = argparse.ArgumentParser(description="Jellyfin Library Refresh & Metadata Populator")
    parser.add_argument("--url", default=os.environ.get("JELLYFIN_URL", "http://127.0.0.1:8096"))
    parser.add_argument("--api-key", default=os.environ.get("JELLYFIN_API_KEY", ""))
    parser.add_argument("--refresh-only", action="store_true", help="Only trigger a library scan")
    parser.add_argument("--metadata-json", type=Path, help="Optional JSON file with item metadata updates")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Error: Provide --api-key or set JELLYFIN_API_KEY environment variable.")

    client = JellyfinClient(args.url, args.api_key)
    client.refresh_library()

    if not args.refresh_only and args.metadata_json and args.metadata_json.exists():
        data = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        client.apply_metadata(data.get("items", data if isinstance(data, list) else []))


if __name__ == "__main__":
    main()
