"""Download one audio sample from each data/*.zip of a Hugging Face dataset.

Only the zip central directory and the chosen member are fetched via HTTP
range requests, so the full (100-400 MB) archives are never downloaded.

Usage:
    python tools/fetch_opentts_samples.py
    python tools/fetch_opentts_samples.py --out samples --pick random
"""

import argparse
import io
import os
import random
import zipfile

import requests

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file object backed by HTTP range requests."""

    def __init__(self, url, session):
        self.session = session
        resp = session.head(url, allow_redirects=True)
        resp.raise_for_status()
        # Use the final (CDN) URL so redirects are not repeated per read.
        self.url = resp.url
        self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60
        )
        resp.raise_for_status()
        if resp.status_code != 206:
            raise IOError("Server ignored Range header")
        data = resp.content
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def list_zip_paths(session, repo, revision, folder):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return [
        item["path"]
        for item in resp.json()
        if item["type"] == "file" and item["path"].endswith(".zip")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="Yehor/opentts-uk")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--folder", default="data")
    parser.add_argument("--out", default="opentts_uk_samples")
    parser.add_argument("--pick", choices=["first", "random"], default="first")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    session = requests.Session()
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"

    os.makedirs(args.out, exist_ok=True)
    zip_paths = list_zip_paths(session, args.repo, args.revision, args.folder)
    if not zip_paths:
        raise SystemExit(f"No zip files found in {args.repo}/{args.folder}")

    for path in zip_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        url = f"https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}/{path}"
        remote = io.BufferedReader(HttpRangeFile(url, session), buffer_size=1 << 20)
        with zipfile.ZipFile(remote) as zf:
            audio = [
                info
                for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith(AUDIO_EXTS)
            ]
            if not audio:
                print(f"[{name}] no audio files found, skipping")
                continue
            info = random.choice(audio) if args.pick == "random" else audio[0]
            ext = os.path.splitext(info.filename)[1]
            target = os.path.join(args.out, f"{name}{ext}")
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"[{name}] {info.filename} -> {target} ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
