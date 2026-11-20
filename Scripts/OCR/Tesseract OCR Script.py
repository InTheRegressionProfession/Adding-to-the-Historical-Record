"""
Tesseract OCR Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Source folder name -> filename prefix
SOURCES = {
    "High Rez Gallica": "HRG",
    "High Rez ScanTent": "HRS",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".bmp", ".webp"}
WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
FALLBACK_DPI = 300

def find_tesseract(explicit=None):
    candidates = [explicit] if explicit else [shutil.which("tesseract"), str(WINDOWS_DEFAULT)]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    sys.exit("Tesseract not found. Install it. "
             "or pass --tesseract <path to tesseract executable>.")


def tesseract_version(exe):
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    text = (out.stdout or "") + (out.stderr or "")
    m = re.search(r"tesseract\s+v?(\d+)\.(\d+)(?:\.(\d+))?", text, re.I)
    if not m:
        sys.exit(f"Couldn't read the Tesseract version from: {text[:200]}")
    return tuple(int(x or 0) for x in m.groups()), text.splitlines()[0].strip()


def check_language(exe, lang, tessdata_dir):
    cmd = [exe, "--list-langs"] + (["--tessdata-dir", tessdata_dir] if tessdata_dir else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    langs = {l.strip() for l in ((out.stdout or "") + (out.stderr or "")).splitlines()}
    missing = [l for l in lang.split("+") if l not in langs]
    if missing:
        sys.exit(f"Tesseract has no language data for: {', '.join(missing)}. "
                 "Install the French language data."
                 + (f" or put {missing[0]}.traineddata in {tessdata_dir}" if tessdata_dir else "") + ".")


# Finding the pages

def find_pages(scans_root, sources, limit):
    """-> list of (source_folder_name, prefix, image_path)"""
    pages = []
    for name in sources:
        folder = scans_root / name
        if not folder.is_dir():
            print(f"  Note: no folder {folder} -- skipped")
            continue
        found = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if limit is not None:
            found = found[:limit]
        pages.extend((name, SOURCES.get(name, ""), p) for p in found)
    return pages


def output_stem(image_path, prefix):
    stem = image_path.stem
    if prefix and not stem.upper().startswith(prefix.upper() + "_"):
        stem = f"{prefix}_{stem}"
    return stem


# Image handling

def image_dpi_and_size(path):
    try:
        from PIL import Image
    except ImportError:
        return None, None
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            size = im.size
    except Exception:
        return None, None
    if dpi:
        try:
            d = int(round(float(dpi[0])))
            return (d if 100 <= d <= 2400 else None), size
        except (TypeError, ValueError):
            pass
    return None, size


def scaled_copy(path, scale, tmpdir):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("L").resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        out = Path(tmpdir) / (path.stem + "_scaled.png")
        im.save(out)
    return out

_HYPHEN_BREAK = re.compile(r"(\w)[-\u00AD\u2010\u00AC]\s*\n\s*(\w)")
_HYPHEN_END = re.compile(r"\w[-\u00AD\u2010\u00AC]$")


def _join_hyphen(m):
    left, right = m.group(1), m.group(2)
    return left + right if right.islower() else left + "-" + right


def reflow(raw_text):
    """Join lines within each paragraph, repair end-of-line hyphenation, keep
    blank lines between paragraphs/blocks."""
    text = raw_text.replace("\r\n", "\n").replace("\f", "\n\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged = []
    for para in paragraphs:
        if merged and _HYPHEN_END.search(merged[-1]) and para[:1].islower():
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)
    out = []
    for para in merged:
        para = _HYPHEN_BREAK.sub(_join_hyphen, para)
        para = re.sub(r"[ \t]*\n[ \t]*", " ", para)
        para = re.sub(r"[ \t]{2,}", " ", para)
        out.append(para)
    return "\n\n".join(out) + "\n"


def tsv_stats(tsv_text):
    """Word count and mean confidence from Tesseract's TSV output."""
    confs, words = [], 0
    for line in tsv_text.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or cols[0] != "5":  # level 5 = word
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if cols[11].strip() and conf >= 0:
            words += 1
            confs.append(conf)
    mean = round(sum(confs) / len(confs), 2) if confs else None
    low = sum(1 for c in confs if c < 50)
    return words, mean, low


def ocr_page(exe, image_path, out_dir, stem, settings, save_hocr, keep_lines):
    t0 = time.time()
    dpi, size = image_dpi_and_size(image_path)
    use_dpi = dpi or FALLBACK_DPI
    with tempfile.TemporaryDirectory() as tmp:
        src = image_path
        if settings["scale"] and settings["scale"] != 1.0:
            src = scaled_copy(image_path, settings["scale"], tmp)
            use_dpi = int(round(use_dpi * settings["scale"]))
        base = Path(tmp) / "page"
        cmd = [exe, str(src), str(base), "-l", settings["lang"], "--oem", "1",
               "--psm", str(settings["psm"]), "--dpi", str(use_dpi)]
        if settings["tessdata_dir"]:
            cmd += ["--tessdata-dir", settings["tessdata_dir"]]
        for kv in settings["config"]:
            cmd += ["-c", kv]
        cmd += ["txt", "tsv"] + (["hocr"] if save_hocr else [])
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        txt_path = base.with_suffix(".txt")
        if proc.returncode != 0 or not txt_path.exists():
            err = (proc.stderr or "").strip().splitlines()
            raise RuntimeError(err[-1] if err else f"tesseract exited with code {proc.returncode}")
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        tsv = base.with_suffix(".tsv").read_text(encoding="utf-8", errors="replace")
        hocr = base.with_suffix(".hocr").read_text(encoding="utf-8", errors="replace") if save_hocr else None

    words, mean_conf, low = tsv_stats(tsv)
    md = raw.replace("\f", "").strip() + "\n" if keep_lines else reflow(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_raw.txt").write_text(raw, encoding="utf-8")
    if hocr is not None:
        (out_dir / f"{stem}.hocr").write_text(hocr, encoding="utf-8")
    meta = {
        "source_image": str(image_path),
        "image_size": list(size) if size else None,
        "engine": settings["version_string"],
        "language": settings["lang"],
        "tessdata_dir": settings["tessdata_dir"],
        "oem": 1, "psm": settings["psm"], "dpi_used": use_dpi,
        "dpi_from_image": dpi, "scale": settings["scale"],
        "config": settings["config"],
        "reflowed": not keep_lines,
        "words": words, "mean_word_confidence": mean_conf, "words_below_50_confidence": low,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")  # written last: marks the page done
    return md, meta


# Main


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch OCR of the pipeline scans with Tesseract")
    parser.add_argument("--scans-dir", default="Pipeline Scans",
                        help='Root folder with the source sub-folders (default: "Pipeline Scans")')
    parser.add_argument("--out-dir", default="Tesseract Output",
                        help='Where to write results (default: "Tesseract Output")')
    parser.add_argument("--sources", nargs="+", default=list(SOURCES),
                        help='Source sub-folders to process (default: "High Rez Gallica" "High Rez ScanTent")')
    parser.add_argument("--export-rawtext",
                        help="Also copy each .md to <folder>/HRG/ and <folder>/HRS/ for the NER scripts")
    parser.add_argument("--lang", default="fra", help="Tesseract language(s) (default: fra)")
    parser.add_argument("--psm", type=int, default=3,
                        help="Page segmentation mode (default 3: automatic, multi-column; 1 adds "
                             "orientation detection)")
    parser.add_argument("--no-sauvola", action="store_true",
                        help="Use Tesseract's default Otsu binarisation instead of Sauvola")
    parser.add_argument("--tessdata-dir", help="Folder with traineddata files (e.g. tessdata_best)")
    parser.add_argument("--tesseract", help="Path to the tesseract executable (if not found automatically)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize pages before OCR (e.g. 0.5 halves very large scans; needs Pillow)")
    parser.add_argument("--keep-lines", action="store_true",
                        help="Don't join lines or repair hyphenation in the .md")
    parser.add_argument("--save-hocr", action="store_true", help="Also save hOCR (word boxes + confidences)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)),
                        help="Pages processed in parallel (default: number of CPU cores)")
    parser.add_argument("--limit", type=int, help="Only the first N pages of each source (quick test)")
    parser.add_argument("--overwrite", action="store_true", help="Redo pages that already have output")
    args = parser.parse_args()

    scans_root = Path(args.scans_dir)
    if not scans_root.is_dir() and (here / args.scans_dir).is_dir():
        scans_root = here / args.scans_dir
    if not scans_root.is_dir():
        sys.exit(f"Scans folder not found: {Path(args.scans_dir).resolve()}")
    out_root = Path(args.out_dir)
    export_root = Path(args.export_rawtext) if args.export_rawtext else None

    exe = find_tesseract(args.tesseract)
    version, version_string = tesseract_version(exe)
    check_language(exe, args.lang, args.tessdata_dir)
    config = []
    if not args.no_sauvola:
        if version >= (5, 0, 0):
            config.append("thresholding_method=2")
        else:
            print(f"Warning: {version_string} has no Sauvola binarisation (needs Tesseract 5); "
                  "using the default.")
    if args.scale != 1.0 and importlib.util.find_spec("PIL") is None:
        sys.exit("--scale needs Pillow: pip install pillow")
    settings = {"lang": args.lang, "psm": args.psm, "tessdata_dir": args.tessdata_dir,
                "config": config, "scale": args.scale, "version_string": version_string}

    pages = find_pages(scans_root, args.sources, args.limit)
    if not pages:
        sys.exit(f"No page images found under {scans_root} in: {', '.join(args.sources)}")

    jobs, skipped = [], 0
    for source, prefix, img in pages:
        stem = output_stem(img, prefix)
        rel_parent = img.parent.relative_to(scans_root / source)
        out_dir = out_root / source / rel_parent / stem
        if (out_dir / f"{stem}.md").exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((source, prefix, img, stem, out_dir))

    print(f"{version_string} | lang {args.lang} | --oem 1 --psm {args.psm}"
          + (" | Sauvola binarisation" if config else "") + f" | {args.workers} worker(s)")
    print(f"{len(pages)} page(s) found, {skipped} already done, {len(jobs)} to process -> {out_root}")

    log_path = out_root / "tesseract_run_log.csv"
    log_fields = ["source", "page", "status", "words", "mean_word_confidence",
                  "words_below_50_confidence", "dpi_used", "seconds", "error"]
    out_root.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    done = failed = 0
    t_start = time.time()
    confs_by_source = {}

    with open(log_path, "a", newline="", encoding="utf-8-sig") as logf, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        log = csv.DictWriter(logf, fieldnames=log_fields)
        if new_log:
            log.writeheader()
        futures = {pool.submit(ocr_page, exe, img, out_dir, stem, settings, args.save_hocr, args.keep_lines):
                   (source, prefix, img, stem) for source, prefix, img, stem, out_dir in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            source, prefix, img, stem = futures[fut]
            try:
                md, meta = fut.result()
            except Exception as exc:
                failed += 1
                print(f"  [{i}/{len(jobs)}] {stem}: FAILED -- {exc}")
                log.writerow({"source": source, "page": stem, "status": "failed", "error": str(exc)[:300]})
                logf.flush()
                continue
            done += 1
            if export_root is not None and prefix:
                dest = export_root / prefix
                dest.mkdir(parents=True, exist_ok=True)
                (dest / f"{stem}.md").write_text(md, encoding="utf-8")
            if meta["mean_word_confidence"] is not None:
                confs_by_source.setdefault(source, []).append(meta["mean_word_confidence"])
            print(f"  [{i}/{len(jobs)}] {stem}: {meta['words']} words, mean confidence "
                  f"{meta['mean_word_confidence']} ({meta['seconds']}s)")
            log.writerow({"source": source, "page": stem, "status": "ok", "words": meta["words"],
                          "mean_word_confidence": meta["mean_word_confidence"],
                          "words_below_50_confidence": meta["words_below_50_confidence"],
                          "dpi_used": meta["dpi_used"], "seconds": meta["seconds"], "error": ""})
            logf.flush()

    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min: {done} done, {skipped} skipped "
          f"(already done), {failed} failed.")
    for source, confs in confs_by_source.items():
        print(f"  {source}: mean word confidence {sum(confs) / len(confs):.1f} over {len(confs)} page(s) "
              f"this run (lowest {min(confs):.1f})")
    if failed:
        print("Re-run the same command to retry the failed pages.")
    print(f"Per-page log: {log_path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
