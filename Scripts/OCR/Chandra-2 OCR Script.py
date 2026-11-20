"""
Chandra-2 OCR Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VLLM_HEALTH_URL = "http://localhost:8000/v1/models"


def check_vllm_server() -> bool:
    try:
        resp = requests.get(VLLM_HEALTH_URL, timeout=5)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def find_date_folders(scans_root: Path):
    """Yield (entity_name, date_name, path) for every leaf folder containing images."""
    for entity_dir in sorted(p for p in scans_root.iterdir() if p.is_dir()):
        for date_dir in sorted(p for p in entity_dir.iterdir() if p.is_dir()):
            has_images = any(
                f.suffix.lower() in IMAGE_EXTENSIONS
                for f in date_dir.iterdir() if f.is_file()
            )
            if has_images:
                yield entity_dir.name, date_dir.name, date_dir


def already_processed(out_dir: Path) -> bool:
    """Treat a folder as done if it exists and isn't empty."""
    return out_dir.exists() and any(out_dir.iterdir())


def process_folder(src_dir: Path, out_dir: Path, max_workers: int, max_retries: int = 2) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "chandra", str(src_dir), str(out_dir),
        "--method", "vllm",
        "--max-workers", str(max_workers),
    ]
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        print(f"    attempt {attempt}/{max_retries} failed (exit {result.returncode})")
        if result.stderr:
            print(f"    stderr (last 500 chars): {result.stderr[-500:]}")
        if attempt < max_retries:
            time.sleep(5 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch OCR a Scans/<entity>/<date>/ tree via Chandra + vLLM")
    parser.add_argument("--scans-dir", default="Scans", help="Root input folder (default: Scans)")
    parser.add_argument("--output-dir", default="output", help="Root output folder (default: output)")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel vLLM workers per folder (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be processed, without running")
    args = parser.parse_args()

    scans_root = Path(args.scans_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not scans_root.is_dir():
        sys.exit(f"Scans folder not found: {scans_root}")

    folders = list(find_date_folders(scans_root))
    if not folders:
        sys.exit(f"No date folders containing images found under {scans_root}")

    n_entities = len({entity for entity, _, _ in folders})
    total_images = sum(
        sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        for _, _, d in folders
    )
    print(f"Found {len(folders)} date folders across {n_entities} entity "
          f"subfolders ({total_images} images total).\n")

    if not args.dry_run and not check_vllm_server():
        sys.exit(
            f"Could not reach the vLLM server at {VLLM_HEALTH_URL}\n"
            "Start it first in its own tmux session (so it survives SSH "
            "disconnects), then re-run this script:\n"
            "    tmux new -s vllm\n"
            "    chandra_vllm --gpu <your_gpu>\n"
            "    (press Ctrl+B then D to detach)"
        )

    start_time = time.time()
    done = skipped = failed = 0

    for entity, date, src_dir in folders:
        out_dir = output_root / entity / date
        label = f"{entity}/{date}"

        if already_processed(out_dir):
            print(f"[skip]  {label} (output already exists)")
            skipped += 1
            continue

        if args.dry_run:
            n_images = sum(1 for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"[would process] {label}  ({n_images} images) -> {out_dir}")
            continue

        print(f"[run]   {label} ...", flush=True)
        t0 = time.time()
        ok = process_folder(src_dir, out_dir, args.max_workers)
        elapsed = time.time() - t0

        if ok:
            print(f"        done in {elapsed:.1f}s")
            done += 1
        else:
            print("        FAILED after retries -- see stderr above")
            failed += 1
            if out_dir.exists() and not any(out_dir.iterdir()):
                shutil.rmtree(out_dir, ignore_errors=True)

    if args.dry_run:
        return

    total_elapsed = time.time() - start_time
    print(f"\nDone in {total_elapsed/60:.1f} min. "
          f"{done} processed, {skipped} skipped, {failed} failed.")
    if failed:
        print("Re-run the same command to retry failed folders "
              "(completed ones are skipped automatically).")


if __name__ == "__main__":
    main()
