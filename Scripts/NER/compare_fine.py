"""
Comparison Script for Fine-level entities

Thesis document contains final data used. Fuzzy matching was kept on as initial reference, but not present in the final analysis


Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from compare_broad import (_col, counts, detail_rows, group_label, index_folder, index_method, norm,
                           prf, r4, read_csv_rows, strip_invisible)
from ner_utils import (FINE_TYPES, GOLD_DIR, IMPRESSO_UNSUPPORTED_FINE, NER_DIR, fine_to_coarse)

METHODS = {  # name: (folder, file suffix)
    "impresso": ("impresso_fine", "_impresso_fine"),
    "Mistral": ("mistral_fine", "_mistral_fine"),
    "Qwen": ("qwen_fine", "_qwen_fine"),
}
CANNOT_OUTPUT = {"impresso": IMPRESSO_UNSUPPORTED_FINE}
FAMILIES = ["pers", "org", "loc", "time", "prod"]
SUMMARY_COLUMNS = [f"{f}.*" for f in FAMILIES] + ["ALL", "ANY (untyped)", "COARSE"]


def read_fine_csv(path):
    """-> (set of (normalized entity, fine type), {normalized: as written},
    problem_rows, unknown_labels). problem_rows lists rows whose fine_type is blank
    or unrecognizable, as readable descriptions."""
    fieldnames, rows = read_csv_rows(path)
    ent_col = _col(fieldnames, "entity", "surface", "text")
    fine_col = _col(fieldnames, "fine_type")
    if ent_col is None or fine_col is None:
        sys.exit(f"{path} needs 'entity' and 'fine_type' columns (has: {', '.join(fieldnames)})")
    entities, display, missing, unknown = set(), {}, [], set()
    for line, row in enumerate(rows, start=2):  
        text = row.get(ent_col) or ""
        n = norm(text)
        if not n:
            continue
        fine = strip_invisible(row.get(fine_col) or "").strip().casefold()
        if not fine or not fine_to_coarse(fine):
            raw = row.get(fine_col) or ""
            missing.append(f"line {line} '{text.strip()}': "
                           + ("no fine_type" if not raw.strip() else f"fine_type {raw!r} not recognized"))
            continue
        if fine not in FINE_TYPES:
            unknown.add(fine)
        entities.add((n, fine))
        display.setdefault(n, text.strip())
    return entities, display, missing, unknown


def score_document(pred, gold, fuzzy, categories):
    out = {}
    for c in categories:
        out[c] = counts({x for x in pred if x[1] == c}, {x for x in gold if x[1] == c}, fuzzy)

    def total(cats):
        return tuple(sum(out[c][i] for c in cats) for i in range(3))

    for fam in FAMILIES:
        out[f"{fam}.*"] = total([c for c in categories if c.split(".")[0] == fam])
    out["ALL"] = total(categories)
    out["ANY (untyped)"] = counts({(s, "*") for s, _ in pred}, {(s, "*") for s, _ in gold}, fuzzy)
    out["COARSE"] = counts({(s, fine_to_coarse(t)) for s, t in pred},
                           {(s, fine_to_coarse(t)) for s, t in gold}, fuzzy)
    return out


def category_note(method, category):
    if category in CANNOT_OUTPUT.get(method, ()):
        return "method cannot output this type"
    return ""


def confusions(pred, gold):
    """(gold type, predicted type) pairs for entities whose text matches exactly
    but whose fine type differs."""
    g_types, p_types = {}, {}
    for s, t in gold:
        g_types.setdefault(s, set()).add(t)
    for s, t in pred:
        p_types.setdefault(s, set()).add(t)
    pairs = []
    for s in set(g_types) & set(p_types):
        for gt in g_types[s] - p_types[s]:
            for pt in p_types[s] - g_types[s]:
                pairs.append((gt, pt))
    return pairs


def summary_table(overall, group, methods, match, value="f1"):
    width = max(len(m) for m in methods) + 2
    lines = ["".ljust(width) + "".join(c[:13].rjust(14) for c in SUMMARY_COLUMNS)]
    for m in methods:
        cells = []
        for c in SUMMARY_COLUMNS:
            v = overall[(group, m, match, c)][value]
            cells.append(f"{v:.3f}" if isinstance(v, float) else "-")
        lines.append(m.ljust(width) + "".join(x.rjust(14) for x in cells))
    return "\n".join(lines)


def per_type_table(overall, group, methods, categories):
    w = max(14, max(len(m) for m in methods) + 5) 
    head = "type".ljust(24) + "gold".rjust(6)
    for m in methods:
        head += f"{m} F1".rjust(w) + f"{m} P/R".rjust(w + 2)
    lines = [head]
    for c in categories:
        first = overall[(group, methods[0], "strict", c)]
        n_gold = first["tp"] + first["fn"]
        n_pred = [overall[(group, m, "strict", c)]["tp"] + overall[(group, m, "strict", c)]["fp"]
                  for m in methods]
        if not n_gold and not any(n_pred):
            continue
        line = (c + ("  (not HIPE)" if c not in FINE_TYPES else "")).ljust(24) + str(n_gold).rjust(6)
        for m in methods:
            row = overall[(group, m, "strict", c)]
            mark = "*" if c in CANNOT_OUTPUT.get(m, ()) else ""
            f1 = (f"{row['f1']:.3f}" if isinstance(row["f1"], float) else "-") + mark
            p = f"{row['precision']:.2f}" if isinstance(row["precision"], float) else "-"
            r = f"{row['recall']:.2f}" if isinstance(row["recall"], float) else "-"
            line += f1.rjust(w) + f"{p}/{r}".rjust(w + 2)
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score the fine-grained NER outputs against the gold standard")
    parser.add_argument("--gold-dir", default=str(GOLD_DIR))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--groups", nargs="+", help="Only score these sub-folders, e.g. --groups HRG")
    parser.add_argument("--per-group-docs", action="store_true",
                        help="Score each sub-folder on every document it has (default: HRG and HRS "
                             "are scored on the same documents)")
    parser.add_argument("--out-dir", default=str(NER_DIR / "fine_comparison"))
    args = parser.parse_args()

    gold_dir, out_dir = Path(args.gold_dir), Path(args.out_dir)
    gold_paths = index_folder(gold_dir, "_Human")
    if not gold_paths:
        sys.exit(f"No *_Human.csv files found in {gold_dir}")

    gold, incomplete, unknown_gold = {}, [], set()
    for key, path in sorted(gold_paths.items()):
        ents, disp, missing, unknown = read_fine_csv(path)
        unknown_gold |= unknown
        if missing:
            incomplete.append(f"{key} ({'; '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})")
            continue
        gold[key] = (ents, disp)
    print(f"Gold: {len(gold_paths)} document(s) in {gold_dir.name}/; {len(gold)} usable for fine scoring")
    if incomplete:
        print(f"  Left out -- gold rows without a usable fine_type: {', '.join(incomplete)}")
    if unknown_gold:
        print(f"  Warning: gold uses fine types that aren't in the HIPE list: {', '.join(sorted(unknown_gold))}")
    if not gold:
        sys.exit("No gold file has a fine_type for every entity.")

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

    plan = {}
    for g in groups:
        methods = [m for m in outputs if g in outputs[m]]
        absent = [m for m in outputs if g not in outputs[m]]
        if absent:
            print(f"  {group_label(g)}: no output from {', '.join(absent)} -- left out of this group")
        plan[g] = {"methods": methods,
                   "docs": set(gold).intersection(*(set(outputs[m][g]) for m in methods))}
    if not args.per_group_docs and len(groups) > 1:
        shared = set.intersection(*(plan[g]["docs"] for g in groups))
        for g in groups:
            plan[g]["docs"] = shared
    for g in groups:
        plan[g]["docs"] = sorted(plan[g]["docs"])
        plan[g]["left_out"] = sorted(set(gold) - set(plan[g]["docs"]))
        left = plan[g]["left_out"]
        print(f"  {group_label(g)}: scoring {len(plan[g]['docs'])} document(s)"
              + (f"; left out: {', '.join(left[:6])}{' ...' if len(left) > 6 else ''}" if left else ""))
    if not any(plan[g]["docs"] for g in groups):
        sys.exit("No document has a usable gold file plus output from every method"
                 + ("" if args.per_group_docs else " in every sub-folder (try --per-group-docs)") + ".")

    preds, unknown_pred = {}, Counter()
    for g in groups:
        for d in plan[g]["docs"]:
            for m in plan[g]["methods"]:
                ents, disp, _, unknown = read_fine_csv(outputs[m][g][d])
                preds[(g, m, d)] = (ents, disp)
                for u in unknown:
                    unknown_pred[(m, u)] += 1
    seen = {t for ents, _ in gold.values() for _, t in ents} | {t for ents, _ in preds.values() for _, t in ents}
    categories = FINE_TYPES + sorted(seen - set(FINE_TYPES))
    cat_order = categories + SUMMARY_COLUMNS

    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc_rows, details, conf_counts = [], [], Counter()
    totals, doc_f1s = {}, {}
    for g in groups:
        for d in plan[g]["docs"]:
            g_ents, g_disp = gold[d]
            for m in plan[g]["methods"]:
                p_ents, p_disp = preds[(g, m, d)]
                for row in detail_rows(d, m, p_ents, g_ents, p_disp, g_disp):
                    details.append({"group": g, **row})
                for gt, pt in confusions(p_ents, g_ents):
                    conf_counts[(g, m, gt, pt)] += 1
                for match in ("strict", "fuzzy"):
                    for c, (tp, fp, fn) in score_document(p_ents, g_ents, match == "fuzzy", categories).items():
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
                for c in cat_order:
                    tp, fp, fn = totals[(g, m, match, c)]
                    p, r, f = prf(tp, fp, fn)
                    f1s = doc_f1s.get((g, m, match, c), [])
                    row = {"group": g, "method": m, "match": match, "category": c, "tp": tp, "fp": fp,
                           "fn": fn, "precision": r4(p), "recall": r4(r), "f1": r4(f),
                           "macro_f1": r4(sum(f1s) / len(f1s)) if f1s else "",
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
    conf_rows = [{"group": g, "method": m, "gold_type": gt, "predicted_type": pt, "count": n}
                 for (g, m, gt, pt), n in sorted(conf_counts.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1]))]
    write("type_confusions.csv", conf_rows, ["group", "method", "gold_type", "predicted_type", "count"])

    all_methods = list(outputs)
    scored = [g for g in groups if plan[g]["docs"]]
    parts = ["Fine-grained NER vs human gold standard",
             f"Methods: {', '.join(all_methods)}   Sub-folders: {', '.join(group_label(g) for g in groups)}",
             ("Each sub-folder is scored on the documents it has." if args.per_group_docs or len(groups) == 1
              else "All sub-folders are scored on the same documents, so their results are directly comparable."),
             ""]

    if len(scored) > 1:
        width = max(len(m) for m in all_methods) + 2
        cols = ["ALL", "COARSE"]
        head = "".ljust(width) + "".join(f"{c} {group_label(g)}"[:15].rjust(16) for c in cols for g in scored)
        if len(scored) == 2:
            head += f"ALL {scored[1]}-{scored[0]}"[:17].rjust(18)
        lines = [head]
        for m in all_methods:
            line, vals = m.ljust(width), {}
            for c in cols:
                for g in scored:
                    row = overall.get((g, m, "strict", c))
                    v = row["f1"] if row else None
                    vals[(c, g)] = v
                    line += (f"{v:.3f}" if isinstance(v, float) else "-").rjust(16)
            if len(scored) == 2:
                a, b = vals[("ALL", scored[0])], vals[("ALL", scored[1])]
                line += (f"{b - a:+.3f}" if isinstance(a, float) and isinstance(b, float) else "-").rjust(18)
            lines.append(line)
        parts += ["Side by side -- F1, strict match", *lines, ""]

    for g in scored:
        methods, docs = plan[g]["methods"], plan[g]["docs"]
        n_gold = sum(totals[(g, methods[0], "strict", "ALL")][i] for i in (0, 2))
        parts += ["=" * 78,
                  f"{group_label(g)} -- {len(docs)} document(s), {n_gold} gold entities (distinct per document)",
                  "=" * 78, "",
                  "F1, strict match (pooled over all documents)",
                  summary_table(overall, g, methods, "strict"), "",
                  "F1, fuzzy match",
                  summary_table(overall, g, methods, "fuzzy"), "",
                  "Precision, strict match",
                  summary_table(overall, g, methods, "strict", "precision"), "",
                  "Recall, strict match",
                  summary_table(overall, g, methods, "strict", "recall"), "",
                  "Per fine type, strict match (types with no gold and no predictions omitted)",
                  per_type_table(overall, g, methods, categories), ""]
        for m in methods:
            top = [(gt, pt, n) for (gg, mm, gt, pt), n in conf_counts.items() if gg == g and mm == m]
            top.sort(key=lambda x: -x[2])
            parts.append(f"Most common wrong subtypes, {m} (gold -> predicted): "
                         + (", ".join(f"{gt} -> {pt} ({n})" for gt, pt, n in top[:8]) or "none"))
        parts.append("Mean per-document F1 (ALL, strict): "
                     + ", ".join(f"{m} {overall[(g, m, 'strict', 'ALL')]['macro_f1']}" for m in methods))
        if plan[g]["left_out"]:
            parts.append(f"Gold documents left out here: {', '.join(plan[g]['left_out'])}")
        parts.append("")

    # Only explain the * if a type impresso cannot output actually occurs in the data.
    unsupported_seen = sorted(set(categories) & set(IMPRESSO_UNSUPPORTED_FINE)
                              & {c for (g, m, match, c), (tp, fp, fn) in totals.items() if tp + fp + fn})
    if unsupported_seen:
        parts += [f"*  impresso cannot output this type ({', '.join(unsupported_seen)})"]
    parts += ["COARSE = fine types collapsed to Person / Organization / Location / Time / Product."]
    if incomplete:
        parts += [f"Gold files left out for missing fine types: {', '.join(incomplete)}"]
    if unknown_gold or unknown_pred:
        parts += ["Labels not in the HIPE list -- gold: " + (", ".join(sorted(unknown_gold)) or "none")
                  + "; outputs: " + (", ".join(f"{m} {u}" for (m, u) in sorted(unknown_pred)) or "none")]
    summary = "\n".join(parts) + "\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
