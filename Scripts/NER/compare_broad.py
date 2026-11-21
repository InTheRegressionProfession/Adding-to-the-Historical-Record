"""
Comparison Script for Coarse-level entities

Thesis document contains final data used. Fuzzy matching and models approximating labels they were not trained for (e.g. SpaCy and Products) were kept on as initial reference, but not present in the final analysis


Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import io
import re
import sys
import unicodedata
from pathlib import Path

from ner_utils import COARSE_CATEGORIES, GOLD_DIR, NER_DIR, doc_key

METHODS = {  # name: (folder, file suffix)
    "spaCy": ("SpaCy_broad", "_spacy_broad"),
    "impresso": ("impresso_broad", "_impresso_broad"),
    "CamemBERT": ("camembert_broad", "_camembert_broad"),
    "Mistral": ("mistral_broad", "_mistral_broad"),
    "Qwen": ("qwen_broad", "_qwen_broad"),
}
CORE3 = ["Person", "Organization", "Location"]
CANNOT_OUTPUT = {"spaCy": {"Time"}}
PROXY_ONLY = {"spaCy": {"Product"}, "CamemBERT": {"Product"}}

_TYPE_ALIASES = {
    "person": "Person", "pers": "Person", "per": "Person", "personne": "Person",
    "organization": "Organization", "organisation": "Organization", "org": "Organization",
    "location": "Location", "loc": "Location", "lieu": "Location",
    "time": "Time", "date": "Time", "temps": "Time",
    "product": "Product", "prod": "Product", "produit": "Product",
}
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_EDGE = " \t\n.,;:!?()[]{}'\"-–—…«»“”"
_ARTICLES = re.compile(r"^(?:(?:le|la|les|du|des|de)\s+|[ld]')+")


def coarse_from(coarse_label, fine_label=""):
    for label in (coarse_label, fine_label):
        key = (label or "").strip().casefold()
        if key:
            hit = _TYPE_ALIASES.get(key) or _TYPE_ALIASES.get(key.split(".")[0])
            if hit:
                return hit
    return None


_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def strip_invisible(s):
    return _INVISIBLE_RE.sub("", s or "")


def norm(s):
    s = unicodedata.normalize("NFC", strip_invisible(s)).translate(_APOS)
    return re.sub(r"\s+", " ", s).strip(_EDGE).casefold()


def _col(fieldnames, *names):
    lookup = {(f or "").strip().casefold(): f for f in fieldnames or []}
    for n in names:
        if n in lookup:
            return lookup[n]
    return None


def read_csv_rows(path):
    """-> (fieldnames, list of row dicts). UTF-8 (with or without BOM); falls
    back to Windows-1252, which is what Excel's plain "CSV" option saves."""
    raw = Path(path).read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("cp1252", errors="replace")
        print(f"  Note: {Path(path).name} is not UTF-8; read it as Windows-1252 "
              "(save as 'CSV UTF-8' to avoid this)")
    reader = csv.DictReader(io.StringIO(content, newline=""))
    return reader.fieldnames or [], list(reader)


def read_entity_csv(path, is_gold):
    """-> (set of (normalized entity, coarse type), {normalized entity: text as written})"""
    entities, display, bad = set(), {}, 0
    fieldnames, rows = read_csv_rows(path)
    ent_col = _col(fieldnames, "entity", "surface", "text")
    coarse_col = _col(fieldnames, "coarse_type", "type")
    fine_col = _col(fieldnames, "fine_type")
    if ent_col is None:
        sys.exit(f"{path} has no 'entity' column (columns: {', '.join(fieldnames)})")
    for row in rows:
        text = row.get(ent_col) or ""
        n = norm(text)
        if not n:
            continue
        t = coarse_from(row.get(coarse_col, "") if coarse_col else "",
                        row.get(fine_col, "") if fine_col else "")
        if t is None:
            bad += 1
            continue
        entities.add((n, t))
        display.setdefault(n, text.strip())
    if bad and is_gold:
        print(f"  Warning: {path.name}: {bad} row(s) with no recognizable type were ignored")
    return entities, display


def _matches(p, suffix):
    stem = p.stem.strip("[] ")  
    return stem if stem.casefold().endswith(suffix.casefold()) else None


def index_folder(folder, suffix):
    """Gold files: {doc_key: path} for <document><suffix>.csv directly in folder."""
    out = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.csv")):
            stem = _matches(p, suffix)
            if stem:
                out.setdefault(doc_key(stem[: -len(suffix)]), p)
    return out


def index_method(folder, suffix):
    """Method outputs: {group: {doc_key: path}}; group is the sub-folder
    ('HRG', 'HRS'), or '' for files directly in the method folder."""
    out = {}
    if folder.is_dir():
        for p in sorted(folder.rglob("*.csv")):
            if p.name.startswith("_"):     
                continue
            stem = _matches(p, suffix)
            if not stem:
                continue
            rel = p.parent.relative_to(folder).as_posix()
            group = "" if rel == "." else rel
            out.setdefault(group, {}).setdefault(doc_key(stem[: -len(suffix)]), p)
    return out


# Matching

def match_pairs(pred, gold, fuzzy):
    """One-to-one (pred, gold) pairs. Exact matches first, then (if fuzzy) same
    type with word-boundary containment after dropping leading articles."""
    exact = pred & gold
    pairs = [(x, x) for x in sorted(exact)]
    if fuzzy:
        rest_gold = sorted(gold - exact)
        used = set()
        for p in sorted(pred - exact):
            pf = f" {_ARTICLES.sub('', p[0]).strip()} "
            if not pf.strip():
                continue
            for j, g in enumerate(rest_gold):
                if j in used or g[1] != p[1]:
                    continue
                gf = f" {_ARTICLES.sub('', g[0]).strip()} "
                if gf.strip() and (pf in gf or gf in pf):
                    used.add(j)
                    pairs.append((p, g))
                    break
    return pairs


def counts(pred, gold, fuzzy):
    m = len(match_pairs(pred, gold, fuzzy))
    return m, len(pred) - m, len(gold) - m


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return p, r, f


def r4(x):
    return "" if x is None else round(x, 4)


def score_document(pred, gold, fuzzy):
    """{category: (tp, fp, fn)} for the five categories, CORE3, ALL and untyped."""
    out = {}
    for c in COARSE_CATEGORIES:
        out[c] = counts({x for x in pred if x[1] == c}, {x for x in gold if x[1] == c}, fuzzy)
    for name, cats in (("CORE3", CORE3), ("ALL", COARSE_CATEGORIES)):
        out[name] = tuple(sum(out[c][i] for c in cats) for i in range(3))
    out["ANY (untyped)"] = counts({(s, "*") for s, _ in pred}, {(s, "*") for s, _ in gold}, fuzzy)
    return out


CATEGORY_ORDER = COARSE_CATEGORIES + ["CORE3", "ALL", "ANY (untyped)"]


def category_note(method, category):
    if category in CANNOT_OUTPUT.get(method, ()):
        return "method has no label for this category"
    if category in PROXY_ONLY.get(method, ()):
        return "MISC used as a rough stand-in"
    return ""


def detail_rows(doc, method, pred, gold, pdisp, gdisp):
    strict = {p for p, _ in match_pairs(pred, gold, False)}
    fuzzy_pairs = dict(match_pairs(pred, gold, True))
    fuzzy_gold = {g: p for p, g in fuzzy_pairs.items()}
    gold_types, pred_types = {}, {}
    for s, t in gold:
        gold_types.setdefault(s, set()).add(t)
    for s, t in pred:
        pred_types.setdefault(s, set()).add(t)
    rows = []
    for p in sorted(pred):
        note = ""
        if p not in strict:
            other = gold_types.get(p[0], set()) - {p[1]}
            if other:
                note = f"in gold as {'/'.join(sorted(other))}"
        fm = fuzzy_pairs.get(p)
        rows.append({"document": doc, "method": method, "entity": pdisp.get(p[0], p[0]), "type": p[1],
                     "result": "TP" if p in strict else "FP",
                     "fuzzy_match": "" if p in strict or fm is None else gdisp.get(fm[0], fm[0]),
                     "note": note})
    for g in sorted(gold - strict):
        other = pred_types.get(g[0], set()) - {g[1]}
        fm = fuzzy_gold.get(g)
        rows.append({"document": doc, "method": method, "entity": gdisp.get(g[0], g[0]), "type": g[1],
                     "result": "FN", "fuzzy_match": "" if fm is None else pdisp.get(fm[0], fm[0]),
                     "note": f"found but typed {'/'.join(sorted(other))}" if other else ""})
    return rows


def table(overall, group, methods, match, value="f1"):
    width = max(len(m) for m in methods) + 2
    lines = ["".ljust(width) + "".join(c[:13].rjust(14) for c in CATEGORY_ORDER)]
    for m in methods:
        cells = []
        for c in CATEGORY_ORDER:
            row = overall[(group, m, match, c)]
            v = row[value]
            mark = "*" if category_note(m, c).startswith("method has no") else (
                "~" if category_note(m, c) else "")
            cells.append((f"{v:.3f}" if isinstance(v, float) else "-") + mark)
        lines.append(m.ljust(width) + "".join(x.rjust(14) for x in cells))
    return "\n".join(lines)


def group_label(group):
    return group or "(no sub-folder)"


def main():
    parser = argparse.ArgumentParser(description="Score the broad NER outputs against the gold standard")
    parser.add_argument("--gold-dir", default=str(GOLD_DIR))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--groups", nargs="+",
                        help="Only score these output sub-folders, e.g. --groups HRG (default: all found)")
    parser.add_argument("--per-group-docs", action="store_true",
                        help="Score each sub-folder on every document it has.")
    parser.add_argument("--out-dir", default=str(NER_DIR / "broad_comparison"))
    args = parser.parse_args()

    gold_dir, out_dir = Path(args.gold_dir), Path(args.out_dir)
    gold_files = index_folder(gold_dir, "_Human")
    if not gold_files:
        sys.exit(f"No *_Human.csv files found in {gold_dir}")
    print(f"Gold: {len(gold_files)} document(s) in {gold_dir.name}/")

    outputs = {}  
    for m in args.methods:
        folder, suffix = METHODS[m]
        found = index_method(NER_DIR / folder, suffix)
        if args.groups:
            found = {g: v for g, v in found.items() if g in args.groups}
        if not found:
            print(f"  {m}: no output files in {folder}/ yet -- left out")
            continue
        outputs[m] = found
        print(f"  {m}: " + ", ".join(f"{group_label(g)} {len(v)} file(s)" for g, v in sorted(found.items())))
    if not outputs:
        sys.exit("No method outputs found to compare.")
    groups = sorted({g for found in outputs.values() for g in found})

    # Which methods and documents are scored in each group
    plan = {}
    for g in groups:
        methods = [m for m in outputs if g in outputs[m]]
        absent = [m for m in outputs if g not in outputs[m]]
        if absent:
            print(f"  {group_label(g)}: no output from {', '.join(absent)} -- left out of this group")
        docs = set(gold_files).intersection(*(set(outputs[m][g]) for m in methods))
        plan[g] = {"methods": methods, "docs": docs}
    if not args.per_group_docs and len(groups) > 1:
        shared = set.intersection(*(plan[g]["docs"] for g in groups))
        for g in groups:
            plan[g]["docs"] = shared
    for g in groups:
        plan[g]["docs"] = sorted(plan[g]["docs"])
        left = sorted(set(gold_files) - set(plan[g]["docs"]))
        plan[g]["left_out"] = left
        print(f"  {group_label(g)}: scoring {len(plan[g]['docs'])} document(s)"
              + (f"; {len(left)} gold document(s) left out: {', '.join(left[:6])}"
                 + (" ..." if len(left) > 6 else "") if left else ""))
    if not any(plan[g]["docs"] for g in groups):
        sys.exit("No document has a gold file plus output from every method"
                 + ("" if args.per_group_docs else " in every sub-folder (try --per-group-docs)") + ".")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc_rows, details = [], []
    totals, doc_f1s = {}, {}  # (group, method, match, category) -> counts / per-document F1s
    gold_cache = {}

    for g in groups:
        for d in plan[g]["docs"]:
            if d not in gold_cache:
                gold_cache[d] = read_entity_csv(gold_files[d], is_gold=True)
            gold, gdisp = gold_cache[d]
            for m in plan[g]["methods"]:
                pred, pdisp = read_entity_csv(outputs[m][g][d], is_gold=False)
                for row in detail_rows(d, m, pred, gold, pdisp, gdisp):
                    details.append({"group": g, **row})
                for match in ("strict", "fuzzy"):
                    for c, (tp, fp, fn) in score_document(pred, gold, match == "fuzzy").items():
                        p, r, f = prf(tp, fp, fn)
                        per_doc_rows.append({"group": g, "document": d, "method": m, "match": match,
                                             "category": c, "tp": tp, "fp": fp, "fn": fn,
                                             "precision": r4(p), "recall": r4(r), "f1": r4(f)})
                        t = totals.setdefault((g, m, match, c), [0, 0, 0])
                        t[0], t[1], t[2] = t[0] + tp, t[1] + fp, t[2] + fn
                        if f is not None:
                            doc_f1s.setdefault((g, m, match, c), []).append(f)

    overall, overall_rows = {}, []
    for g in groups:
        if not plan[g]["docs"]:
            continue
        for m in plan[g]["methods"]:
            for match in ("strict", "fuzzy"):
                for c in CATEGORY_ORDER:
                    tp, fp, fn = totals[(g, m, match, c)]
                    p, r, f = prf(tp, fp, fn)
                    f1s = doc_f1s.get((g, m, match, c), [])
                    row = {"group": g, "method": m, "match": match, "category": c,
                           "tp": tp, "fp": fp, "fn": fn, "precision": r4(p), "recall": r4(r),
                           "f1": r4(f), "macro_f1": r4(sum(f1s) / len(f1s)) if f1s else "",
                           "docs": len(plan[g]["docs"]), "note": category_note(m, c)}
                    overall[(g, m, match, c)] = row
                    overall_rows.append(row)

    def write(name, rows, fields):
        with open(out_dir / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write("overall_scores.csv", overall_rows, list(overall_rows[0]))
    write("per_document_scores.csv", per_doc_rows, list(per_doc_rows[0]))
    write("match_details.csv", details,
          ["group", "document", "method", "entity", "type", "result", "fuzzy_match", "note"])

    all_methods = list(outputs)
    parts = ["Broad NER vs human gold standard",
             f"Methods: {', '.join(all_methods)}   Sub-folders: {', '.join(group_label(g) for g in groups)}",
             ("Each sub-folder is scored on the documents it has."
              if args.per_group_docs or len(groups) == 1 else
              "All sub-folders are scored on the same documents, so their results are directly comparable."),
             ""]

    scored = [g for g in groups if plan[g]["docs"]]
    if len(scored) > 1:
        width = max(len(m) for m in all_methods) + 2
        head = "".ljust(width)
        for c in ("ALL", "CORE3"):
            for g in scored:
                head += f"{c} {group_label(g)}"[:13].rjust(14)
        if len(scored) == 2:
            head += f"ALL {scored[1]}-{scored[0]}"[:15].rjust(16)
        lines = [head]
        for m in all_methods:
            line = m.ljust(width)
            vals = {}
            for c in ("ALL", "CORE3"):
                for g in scored:
                    row = overall.get((g, m, "strict", c))
                    v = row["f1"] if row else None
                    vals[(c, g)] = v
                    line += (f"{v:.3f}" if isinstance(v, float) else "-").rjust(14)
            if len(scored) == 2:
                a, b = vals[("ALL", scored[0])], vals[("ALL", scored[1])]
                line += (f"{b - a:+.3f}" if isinstance(a, float) and isinstance(b, float) else "-").rjust(16)
            lines.append(line)
        parts += ["Side by side -- F1, strict match", *lines, ""]

    for g in scored:
        methods, docs = plan[g]["methods"], plan[g]["docs"]
        n_gold = sum(totals[(g, methods[0], "strict", "ALL")][i] for i in (0, 2))
        wrong_type = {m: sum(1 for r in details if r["group"] == g and r["method"] == m
                             and r["note"].startswith("in gold as")) for m in methods}
        parts += [
            "=" * 78,
            f"{group_label(g)} -- {len(docs)} document(s), {n_gold} gold entities (distinct per document)",
            "=" * 78, "",
            "F1, strict match (pooled over all documents)",
            table(overall, g, methods, "strict"), "",
            "F1, fuzzy match",
            table(overall, g, methods, "fuzzy"), "",
            "Precision, strict match",
            table(overall, g, methods, "strict", "precision"), "",
            "Recall, strict match",
            table(overall, g, methods, "strict", "recall"), "",
            "Right entity, wrong type (strict FPs whose text is in the gold under another type):",
            "  " + ", ".join(f"{m} {n}" for m, n in wrong_type.items()),
            "Mean per-document F1 (ALL, strict) -- weights every page equally:",
            "  " + ", ".join(f"{m} {overall[(g, m, 'strict', 'ALL')]['macro_f1']}" for m in methods),
        ]
        if plan[g]["left_out"]:
            parts += [f"Gold documents left out here: {', '.join(plan[g]['left_out'])}"]
        parts += [""]

    parts += ["*  the method has no label for this category (spaCy: no dates)",
              "~  MISC used as a rough stand-in for Product",
              "CORE3 = Person + Organization + Location"]
    summary = "\n".join(parts) + "\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
