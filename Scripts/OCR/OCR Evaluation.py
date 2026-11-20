"""
OCR Evaluation Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import math
import re
import sys
from itertools import combinations
from pathlib import Path

from rapidfuzz.distance import Levenshtein

GOLD_FOLDER = "HumanValidated"
GOLD_SUFFIX = "_Human"
# Source folder -> short code, and the code's display name.
FOLDER_TO_GROUP = {
    "hrg": "HRG",
    "hrs": "HRS",
    "lrs": "LRS",
}
GROUP_NAMES = {"HRG": "High Rez Gallica", "HRS": "High Rez ScanTent", "LRS": "Low Rez ScanTent"}
GROUP_ORDER = ["HRG", "HRS", "LRS"]

# Normalization

TAG_RE = re.compile(r"</?(sup|sub|i|b|em|strong)>", re.IGNORECASE)
EMPHASIS_RE = re.compile(r"\*{1,2}")
SPACE_VARIANTS_RE = re.compile("[\u00A0\u202F\u2009\u200A\t]")
INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def strip_markup(text):
    return EMPHASIS_RE.sub("", TAG_RE.sub("", text))


def normalize_for_scoring(text):
    text = INVISIBLE_RE.sub("", strip_markup(text))
    text = SPACE_VARIANTS_RE.sub(" ", text)
    return re.sub(r"[ \t]*\n[ \t\n]*", " ", text).strip()


# Metrics

def compute_metrics(ref, hyp):
    ref_chars, hyp_chars = list(ref), list(hyp)
    ref_words, hyp_words = ref.split(), hyp.split()
    char_dist = Levenshtein.distance(ref_chars, hyp_chars)
    word_dist = Levenshtein.distance(ref_words, hyp_words)
    return {
        "cer": char_dist / len(ref_chars) if ref_chars else float("nan"),
        "wer": word_dist / len(ref_words) if ref_words else float("nan"),
        "char_edit_distance": char_dist, "word_edit_distance": word_dist,
        "ref_char_count": len(ref_chars), "hyp_char_count": len(hyp_chars),
        "ref_word_count": len(ref_words), "hyp_word_count": len(hyp_words),
    }


def aggregate(metrics_list):
    """Micro (pooled) CER/WER and macro (mean per page) CER/WER."""
    n = len(metrics_list)
    if not n:
        return None
    rc = sum(m["ref_char_count"] for m in metrics_list)
    rw = sum(m["ref_word_count"] for m in metrics_list)
    return {
        "docs": n,
        "micro_cer": sum(m["char_edit_distance"] for m in metrics_list) / rc if rc else float("nan"),
        "micro_wer": sum(m["word_edit_distance"] for m in metrics_list) / rw if rw else float("nan"),
        "macro_cer": sum(m["cer"] for m in metrics_list) / n,
        "macro_wer": sum(m["wer"] for m in metrics_list) / n,
    }


# Finding Files 

KEY_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_[A-Za-z0-9]+)")


def doc_key(stem):
    m = KEY_RE.search(stem)
    if m:
        return m.group(1)
    return stem.split("_", 1)[1] if "_" in stem else stem


def group_of(md_path, engine_dir):
    rel = md_path.relative_to(engine_dir)
    if len(rel.parts) > 1:
        g = FOLDER_TO_GROUP.get(rel.parts[0].strip().casefold())
        if g:
            return g
    prefix = md_path.stem.split("_", 1)[0].upper()
    return prefix if prefix and "_" in md_path.stem else None


def index_engine(engine_dir):
    """-> {group: {doc_key: path}} for every .md under one engine folder."""
    out, unplaced = {}, []
    for p in sorted(engine_dir.rglob("*.md")):
        g = group_of(p, engine_dir)
        if not g:
            unplaced.append(p.name)
            continue
        cell = out.setdefault(g, {})
        k = doc_key(p.stem)
        if k in cell:
            print(f"  Warning: {engine_dir.name}/{g}: two files for {k} -- keeping {cell[k].name}, "
                  f"ignoring {p.name}")
            continue
        cell[k] = p
    if unplaced:
        print(f"  Warning: {engine_dir.name}: {len(unplaced)} file(s) with no recognizable source "
              f"(folder or prefix), ignored: {', '.join(unplaced[:5])}")
    return out


def index_gold(gold_dir):
    out = {}
    for p in sorted(gold_dir.glob("*.md")):
        stem = p.stem
        if stem.casefold().endswith(GOLD_SUFFIX.casefold()):
            stem = stem[: -len(GOLD_SUFFIX)]
        out.setdefault(doc_key(stem), p)
    return out


def gname(g):
    return GROUP_NAMES.get(g, g)


def order_groups(groups):
    return sorted(groups, key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, g))


# Main 

def fmt(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare OCR engines and scan sources against human transcriptions")
    parser.add_argument("--ocr-root", default="OCR", help='Folder holding the engine folders and HumanValidated (default: "OCR")')
    parser.add_argument("--gold-dir", help=f"Gold folder (default: <ocr-root>/{GOLD_FOLDER})")
    parser.add_argument("--engines", nargs="+", help="Only these engine folders (default: all)")
    parser.add_argument("--groups", nargs="+", help="Only these sources, as codes: HRG HRS LRS (default: all)")
    parser.add_argument("--common-docs", action="store_true",
                        help="Score the matrix on pages present in EVERY engine x source cell "
                             "(paired comparisons are always on shared pages anyway)")
    parser.add_argument("--results-dir", help="Output folder (default: ocr_results next to the OCR folder)")
    args = parser.parse_args()

    ocr_root = Path(args.ocr_root)
    if not ocr_root.is_dir() and (here / args.ocr_root).is_dir():
        ocr_root = here / args.ocr_root
    if not ocr_root.is_dir():
        sys.exit(f"OCR folder not found: {Path(args.ocr_root).resolve()}")
    gold_dir = Path(args.gold_dir) if args.gold_dir else ocr_root / GOLD_FOLDER
    if not gold_dir.is_dir():
        sys.exit(f"Gold folder not found: {gold_dir}")
    results_dir = Path(args.results_dir) if args.results_dir else ocr_root.resolve().parent / "ocr_results"

    gold = index_gold(gold_dir)
    if not gold:
        sys.exit(f"No *{GOLD_SUFFIX}.md files in {gold_dir}")

    engine_dirs = [d for d in sorted(ocr_root.iterdir()) if d.is_dir()
                   and d.resolve() != gold_dir.resolve() and not d.name.startswith((".", "_"))]
    if args.engines:
        wanted = {e.casefold() for e in args.engines}
        engine_dirs = [d for d in engine_dirs if d.name.casefold() in wanted]
    engines = {}
    for d in engine_dirs:
        idx = index_engine(d)
        if args.groups:
            idx = {g: v for g, v in idx.items() if g in args.groups}
        if idx:
            engines[d.name] = idx
    if not engines:
        sys.exit(f"No OCR .md files found in the engine folders under {ocr_root}")

    print(f"Gold: {len(gold)} page(s) in {gold_dir.name}/")
    for e, idx in engines.items():
        print(f"  {e}: " + ", ".join(f"{gname(g)} {len(v)}" for g, v in
                                    ((g, idx[g]) for g in order_groups(idx))))

    # Score every (engine, source, page) that has a gold file.
    results = {}  # (engine, group) -> {doc: metrics}
    gold_text = {}
    no_gold = set()
    for e, idx in engines.items():
        for g, cell in idx.items():
            for d, path in cell.items():
                if d not in gold:
                    no_gold.add(d)
                    continue
                if d not in gold_text:
                    gold_text[d] = gold[d].read_text(encoding="utf-8", errors="replace")
                hyp_raw = path.read_text(encoding="utf-8", errors="replace")
                m = compute_metrics(normalize_for_scoring(gold_text[d]), normalize_for_scoring(hyp_raw))
                m["path"] = str(path)
                results.setdefault((e, g), {})[d] = m
    if no_gold:
        print(f"  {len(no_gold)} OCR page(s) have no gold file and are not scored: "
              f"{', '.join(sorted(no_gold)[:6])}{' ...' if len(no_gold) > 6 else ''}")
    if not results:
        sys.exit("No OCR page matched a gold file -- check that names share the <date>_<letter> part.")

    engine_order = list(engines)
    cells = sorted(results, key=lambda c: (engine_order.index(c[0]),
                                           GROUP_ORDER.index(c[1]) if c[1] in GROUP_ORDER else 99, c[1]))
    all_groups = order_groups({g for _, g in cells})
    common = set.intersection(*(set(results[c]) for c in cells)) if args.common_docs else None
    if args.common_docs:
        print(f"  --common-docs: matrix uses the {len(common)} page(s) present in every cell")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-document CSV
    fields = ["engine", "source", "source_name", "document", "cer", "wer", "char_edit_distance",
              "word_edit_distance", "ref_char_count", "hyp_char_count", "ref_word_count", "hyp_word_count", "path"]
    with open(results_dir / "Per_Document_Scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (e, g) in cells:
            for d, m in sorted(results[(e, g)].items()):
                w.writerow({"engine": e, "source": g, "source_name": gname(g), "document": d,
                            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}})

    # Matrix
    cell_stats = {}
    for c in cells:
        docs = results[c] if common is None else {d: results[c][d] for d in common}
        cell_stats[c] = aggregate(list(docs.values()))
    with open(results_dir / "Cell_Summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["engine", "source", "source_name", "docs", "micro_cer", "micro_wer",
                    "macro_cer", "macro_wer"])
        for (e, g) in cells:
            s = cell_stats[(e, g)]
            if s:
                w.writerow([e, g, gname(g), s["docs"]] + [round(s[k], 6) for k in
                           ("micro_cer", "micro_wer", "macro_cer", "macro_wer")])

    def matrix(metric, label):
        width = max(len(e) for e in engines) + 2
        lines = [label, "".ljust(width) + "".join(gname(g).rjust(22) for g in all_groups)]
        for e in engines:
            row = e.ljust(width)
            for g in all_groups:
                s = cell_stats.get((e, g))
                row += (f"{fmt(s[metric])} (n={s['docs']})" if s else "-").rjust(22)
            lines.append(row)
        return "\n".join(lines)

    # Paired comparisons
    comparisons = []

    def compare(kind, label_a, label_b, cell_a, cell_b):
        docs = sorted(set(results[cell_a]) & set(results[cell_b]))
        if not docs:
            return
        a = aggregate([results[cell_a][d] for d in docs])
        b = aggregate([results[cell_b][d] for d in docs])
        for metric in ("cer", "wer"):
            wins_a = sum(1 for d in docs if results[cell_a][d][metric] < results[cell_b][d][metric])
            wins_b = sum(1 for d in docs if results[cell_a][d][metric] > results[cell_b][d][metric])
            comparisons.append({
                "comparison": kind, "metric": metric.upper(), "a": label_a, "b": label_b, "docs": len(docs),
                "a_micro": a[f"micro_{metric}"], "b_micro": b[f"micro_{metric}"],
                "difference_b_minus_a": b[f"micro_{metric}"] - a[f"micro_{metric}"],
                "a_macro": a[f"macro_{metric}"], "b_macro": b[f"macro_{metric}"],
                "pages_a_better": wins_a, "pages_b_better": wins_b, "ties": len(docs) - wins_a - wins_b,
            })

    for g in all_groups:  # Engine vs engine, same source
        es = [e for e in engines if (e, g) in results]
        for e1, e2 in combinations(es, 2):
            compare(f"engines on {gname(g)}", e1, e2, (e1, g), (e2, g))
    for e in engines:     # Source vs source, same engine
        gs = [g for g in all_groups if (e, g) in results]
        for g1, g2 in combinations(gs, 2):
            compare(f"sources with {e}", gname(g1), gname(g2), (e, g1), (e, g2))

    with open(results_dir / "Paired_Comparisons.csv", "w", newline="", encoding="utf-8-sig") as f:
        cf = ["comparison", "metric", "a", "b", "docs", "a_micro", "b_micro", "difference_b_minus_a",
              "a_macro", "b_macro", "pages_a_better", "pages_b_better", "ties"]
        w = csv.DictWriter(f, fieldnames=cf)
        w.writeheader()
        for c in comparisons:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()})

    def comparison_lines(kind_prefix):
        lines = []
        for c in comparisons:
            if not c["comparison"].startswith(kind_prefix) or c["metric"] != "CER":
                continue
            wer = next(x for x in comparisons if x["comparison"] == c["comparison"] and x["a"] == c["a"]
                       and x["b"] == c["b"] and x["metric"] == "WER")
            better = c["a"] if c["a_micro"] < c["b_micro"] else c["b"]
            lines.append(
                f"  {c['comparison']}: {c['a']} vs {c['b']}, {c['docs']} shared page(s)\n"
                f"      CER {fmt(c['a_micro'])} vs {fmt(c['b_micro'])}   WER {fmt(wer['a_micro'])} vs "
                f"{fmt(wer['b_micro'])}   -> lower CER: {better}\n"
                f"      pages with lower CER: {c['a']} {c['pages_a_better']}, {c['b']} {c['pages_b_better']}, "
                f"tied {c['ties']}")
        return lines or ["  (nothing to compare)"]

    parts = [
        "OCR Quality Assessment",
        f"Engines: {', '.join(engines)}   Sources: {', '.join(gname(g) for g in all_groups)}   "
        f"Gold pages: {len(gold)}",
        ("Matrix uses only pages present in every cell (--common-docs)." if args.common_docs else
         "Matrix: each cell on all its pages (n shown). Comparisons below use only shared pages."),
        "",
        matrix("micro_cer", "Micro CER"), "",
        matrix("micro_wer", "Micro WER"), "",
        "Engine vs engine, same source (paired):",
        *comparison_lines("engines"), "",
        "Source vs source, same engine (paired):",
        *comparison_lines("sources"), "",
        "Micro = pooled over all characters/words. Per-page and macro",
        "Figures are in the CSV files.",
    ]
    summary = "\n".join(parts) + "\n"
    (results_dir / "Summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Results in {results_dir}")


if __name__ == "__main__":
    main()
