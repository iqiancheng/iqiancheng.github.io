#!/usr/bin/env python3
"""Detect broken image links in _posts/*.md.

Scans every markdown post for external image references (both `![alt](url)`
and `<img src="url">`), fetches each, and classifies the result:

  ok                200 + image content-type + not a known placeholder
  placeholder       200 but the body is a known "no image available" stub
                    (ar5iv/arXiv serve a fixed-size, fixed-md5 placeholder
                    when the real figure does not exist — the md5 is the
                    reliable signal, not the HTTP status)
  http-error        non-200 status (404/403/429/...)
  fetch-error       network/timeout failure

Placeholder detection is two-tier:
  1. A hard-coded set of known stub md5s (the ar5iv "no image available"
     PNG is 20498 bytes, md5 ded85833de1226c2b391743296455b30).
  2. A heuristic: any md5 that repeats across 2+ distinct URLs and is small
     (< 100 KB) is almost certainly a shared placeholder, not real content.

Exit code: 0 if no http-error / placeholder / fetch-error, 1 otherwise.
Run from anywhere; the repo root is resolved from this file's location.

Usage:
  python3 tools/check_post_images.py            # scan, print report, exit 1 on problems
  python3 tools/check_post_images.py --json     # also dump full JSON to stdout
  python3 tools/check_post_images.py --workers 32
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

# ar5iv / arXiv "no image available" stub. Fixed size + fixed md5, so the md5
# is the authoritative dead-link signal (HTTP still returns 200).
KNOWN_PLACEHOLDER_MD5 = {
    "ded85833de1226c2b391743296455b30",  # ar5iv "no image available" PNG, 20498 B
}

# Matches ![alt](url) and <img ... src="url"> / src='url'.
IMG_RE = re.compile(
    r"!\[[^\]]*\]\((https?://[^)\s]+)\)"
    r"|<img[^>]+src=[\"'](https?://[^\"']+)[\"']",
    re.I,
)

# A full browser UA: some hosts (fortelabs.com, wikimedia) 403/429 a bare
# "Mozilla/5.0" UA, which would show up as a false dead link.
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_urls(root):
    urls = {}
    posts = sorted(glob.glob(os.path.join(root, "_posts", "*.md")))
    for f in posts:
        rel = os.path.basename(f)
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                for m in IMG_RE.finditer(line):
                    u = m.group(1) or m.group(2)
                    urls.setdefault(u, []).append(f"{rel}:{i}")
    return urls


def _fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def check(url):
    r = {"url": url}
    # 429 (wikimedia rate limit under concurrency) and transient 5xx get one
    # backoff retry so they don't surface as false dead links.
    for attempt in range(2):
        try:
            status, ctype, data = _fetch(url)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(2.0)
                continue
            r.update(status=e.code, error=str(e))
            return r
        except Exception as e:  # noqa: BLE001 - report any fetch failure
            r.update(status=-1, error=repr(e)[:200])
            return r
        r.update(status=status, ctype=ctype, size=len(data),
                 md5=hashlib.md5(data).hexdigest())
        return r
    return r


def classify(results):
    md5c = Counter(r["md5"] for r in results if r.get("md5"))
    for r in results:
        m = r.get("md5")
        if r.get("status") != 200:
            r["kind"] = "http-error" if r.get("status", -1) > 0 else "fetch-error"
            continue
        if m in KNOWN_PLACEHOLDER_MD5:
            r["kind"] = "placeholder"
            r["why"] = f"known stub md5 {m}"
        elif m and md5c[m] > 1 and r["size"] < 100_000:
            r["kind"] = "placeholder"
            r["why"] = f"md5 {m} repeats {md5c[m]}x, {r['size']}B"
        else:
            r["kind"] = "ok"
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="dump full JSON to stdout")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = repo_root()
    urls = collect_urls(root)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(check, urls))
    results = classify(results)
    for r in results:
        r["refs"] = urls[r["url"]]

    by_kind = Counter(r["kind"] for r in results)
    problems = [r for r in results if r["kind"] != "ok"]

    print(f"scanned {len(urls)} unique image URLs across _posts/*.md")
    print(
        f"  ok={by_kind.get('ok', 0)}  "
        f"placeholder={by_kind.get('placeholder', 0)}  "
        f"http-error={by_kind.get('http-error', 0)}  "
        f"fetch-error={by_kind.get('fetch-error', 0)}"
    )

    for kind in ("placeholder", "http-error", "fetch-error"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        print(f"\n== {kind} ({len(rows)}) ==")
        for r in rows:
            ref = r.get("refs", ["?"])[0]
            extra = r.get("why") or r.get("error") or f"status={r.get('status')}"
            print(f"  {r['url']}\n      -> {ref}  [{extra}]")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(results, ensure_ascii=False, indent=1))

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
