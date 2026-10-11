"""
Back up a finished RVC model (inference files only) as one zip that can be
stored anywhere (e.g. cloud storage) and later restored by unpacking it at the
repository root.

Each backup contains, with their repo-relative paths preserved:
  assets/weights/<model>.pth                       the model used for conversion
  assets/indices/<model>_added_*.index            the feature index (search feature ratio)
      (or logs/<model>/added_*.index if the index only exists there)
  logs/<model>/config.json                        training config (sample rate, architecture)
  MODEL_BACKUP_<model>.json                       manifest: files, sizes, SHA-256, notes
Training checkpoints (G_*.pth / D_*.pth) are NOT included.

Usage (from anywhere; paths are resolved relative to the repo root):
  python tools/model_backup.py --list
  python tools/model_backup.py v2_HuggingKaterina
  python tools/model_backup.py v2_HuggingKaterina --notes "transpose +12, protect 0.33"
  python tools/model_backup.py v2_HuggingKaterina --overwrite
  python tools/model_backup.py --verify assets/model_backups/v2_HuggingKaterina.zip

Restore: unpack the zip at the repo root (keep folder structure), restart the WebUI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "assets" / "weights"
INDICES = ROOT / "assets" / "indices"
LOGS = ROOT / "logs"
OUT_DIR = ROOT / "assets" / "model_backups"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_index(name: str) -> Path | None:
    """The 'added' index for this model (never the 'trained' one)."""
    cands = sorted(INDICES.glob(f"{name}_added_*.index")) if INDICES.is_dir() else []
    # guard against name prefixes (e.g. 'Model' vs 'Model2'): the index file ends with '_<name>_v1/v2.index'
    cands = [p for p in cands if p.stem.endswith((f"_{name}_v1", f"_{name}_v2"))] or cands
    if not cands and (LOGS / name).is_dir():
        cands = sorted((LOGS / name).glob("added_*.index"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)  # newest if several


def collect(name: str) -> tuple[list[Path], list[str]]:
    files, missing = [], []
    pth = WEIGHTS / f"{name}.pth"
    (files if pth.is_file() else missing).append(pth)
    idx = find_index(name)
    if idx:
        files.append(idx)
    else:
        missing.append(INDICES / f"{name}_added_*.index")
    cfg = LOGS / name / "config.json"
    (files if cfg.is_file() else missing).append(cfg)
    return files, missing


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def mb(n: int) -> str:
    return f"{n / 1048576:.1f} MB"


def cmd_list() -> int:
    if not WEIGHTS.is_dir():
        print(f"no models folder: {WEIGHTS}")
        return 1
    print(f"models in {rel(WEIGHTS)}:")
    for pth in sorted(WEIGHTS.glob("*.pth")):
        name = pth.stem
        idx = find_index(name)
        cfg = (LOGS / name / "config.json").is_file()
        print(f"  {name:<32} pth {mb(pth.stat().st_size):>9} | index {'yes' if idx else 'MISSING':<7} | config {'yes' if cfg else 'MISSING'}")
    return 0


def cmd_backup(name: str, notes: str | None, overwrite: bool, allow_missing: bool) -> int:
    files, missing = collect(name)
    if missing:
        print("missing files:")
        for m in missing:
            print(f"  {m}")
        if (WEIGHTS / f"{name}.pth") in missing:
            print(f"error: model '{name}' not found (use --list).", file=sys.stderr)
            return 2
        if not allow_missing:
            print("error: refusing an incomplete backup (use --allow-missing to back up anyway).", file=sys.stderr)
            return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.zip"
    if out.exists() and not overwrite:
        out = OUT_DIR / f"{name}_{dt.datetime.now():%Y%m%d-%H%M%S}.zip"

    manifest = {
        "model": name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "restore": "Unpack this zip at the RVC repository root, keeping folder structure; restart the WebUI.",
        "notes": notes or "",
        "files": [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }
    tmp = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, rel(p))
        z.writestr(f"MODEL_BACKUP_{name}.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(out)

    print(f"backup of '{name}':")
    for f in manifest["files"]:
        print(f"  {f['path']}  ({mb(f['bytes'])})")
    print(f"wrote {rel(out)}  ({mb(out.stat().st_size)})")
    return 0 if verify(out, quiet=True) == 0 else 3


def verify(zip_path: Path, quiet: bool = False) -> int:
    zip_path = zip_path if zip_path.is_absolute() else (Path.cwd() / zip_path)
    if not zip_path.is_file():
        zip_path = ROOT / zip_path.name if (ROOT / zip_path.name).is_file() else zip_path
    with zipfile.ZipFile(zip_path) as z:
        man_names = [n for n in z.namelist() if n.startswith("MODEL_BACKUP_") and n.endswith(".json")]
        if not man_names:
            print("error: no manifest in zip", file=sys.stderr)
            return 4
        man = json.loads(z.read(man_names[0]))
        bad = []
        for f in man["files"]:
            h = hashlib.sha256()
            with z.open(f["path"]) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                bad.append(f["path"])
    if bad:
        print("VERIFY FAILED:", ", ".join(bad), file=sys.stderr)
        return 3
    if not quiet:
        print(f"OK: {zip_path.name} — model '{man['model']}', {len(man['files'])} files, created {man['created']}")
        if man.get("notes"):
            print(f"notes: {man['notes']}")
    else:
        print("verified: all checksums match")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Back up a finished RVC model (.pth + index + config) as a restorable zip.")
    ap.add_argument("model", nargs="?", help="model name = file name in assets/weights without .pth")
    ap.add_argument("--list", action="store_true", help="list models and whether their files are complete")
    ap.add_argument("--notes", help="free text stored in the manifest (e.g. working conversion settings)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing <model>.zip instead of adding a timestamp")
    ap.add_argument("--allow-missing", action="store_true", help="back up even if index or config is missing")
    ap.add_argument("--verify", type=Path, metavar="ZIP", help="check a backup's checksums")
    a = ap.parse_args(argv)
    if a.list:
        return cmd_list()
    if a.verify:
        return verify(a.verify)
    if not a.model:
        ap.error("give a model name, or --list / --verify")
    return cmd_backup(a.model, a.notes, a.overwrite, a.allow_missing)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
