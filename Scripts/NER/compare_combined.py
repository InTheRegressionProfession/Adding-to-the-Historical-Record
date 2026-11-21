"""
Comparison Script for the model supplementation experiment

Thesis document contains final data used. Fuzzy matching was kept on as initial reference, but not present in the final analysis


Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import compare_fine
from compare_broad import group_label, index_folder, index_method
from ner_utils import GOLD_DIR, NER_DIR

IMPRESSO = "impresso"
IMPRESSO_SRC = ("impresso_fine", "_impresso_fine")
PIPELINES = {  # --llms choice: LLM name, pipeline name, pipeline output, LLM-alone output, script
    "mistral": ("Mistral", "Mistral+imp", ("combined_pipeline", "_combined_pipeline"),
                ("mistral_fine", "_mistral_fine"), "ner_combined_pipeline.py"),
    "qwen": ("Qwen", "Qwen+imp", ("qwen_combined", "_qwen_combined"),
             ("qwen_fine", "_qwen_fine"), "ner_qwen_combined.py"),
}
DELTA_CATEGORIES = ["ALL", "COARSE"]


def judge(change, before, after, gold):
    """Verdict for one entity text: impresso's types (before), the pipeline's
    types (after), gold types (None if the text isn't a gold entity)."""
    in_gold = gold is not None
    right_after = bool(in_gold and after and after & gold)
    right_before = bool(in_gold and before and before & gold)
    if change == "kept":
        return "correct" if right_after else ("wrong type" if in_gold else "false positive")
    if change == "retyped":
        if not in_gold:
            return "false positive"
        if right_after and not right_before:
            return "fixed type"
        if right_before and not right_after:
            return "broke type"
        return "correct" if right_after else "still wrong type"
    if change == "removed":
        return "bad removal" if in_gold else "good removal"
    if not in_gold:
        return "bad addition"
    return "good addition" if right_after else "added, wrong type"


def by_text(entities):
    out = {}
    for s, t in entities:
        out.setdefault(s, set()).add(t)
    return out


def is_right(types, gold):
    """Is a pipeline's answer for one entity text right? Found with a gold type if
    it's a gold entity; not output at all if it isn't."""
    if gold is None:
        return not types
    return bool(types and types & gold)


def main():
    parser = argparse.ArgumentParser(description="Score the Mistral and Qwen combined pipelines against the gold")
    parser.add_argument("--llms", "--llm", nargs="+", choices=list(PIPELINES), default=list(PIPELINES),
                        help="Which combined pipelines to score (default: mistral qwen)")
    parser.add_argument("--groups", nargs="+", help="Only score these sub-folders, e.g. --groups HRG")
    parser.add_argument("--per-group-docs", action="store_true",
                        help="Score each sub-folder on every document it has (default: HRG and HRS on the "
                             "same documents)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Score only the combined pipelines, without impresso / the LLMs alone")
    parser.add_argument("--gold-dir", default=str(GOLD_DIR))
    parser.add_argument("--out-dir", default=str(NER_DIR / "combined_comparison"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    pipes = []  # (llm name, pipeline name, pipeline src, llm-alone src)
    for key in dict.fromkeys(args.llms):
        llm, name, src, alone, script = PIPELINES[key]
        if index_method(NER_DIR / src[0], src[1]):
            pipes.append((llm, name, src, alone))
        else:
            print(f"  {name}: no output in {src[0]}/ yet (run {script}) -- left out")
    if not pipes:
        sys.exit("No combined pipeline output found.")

    methods = {name: src for _, name, src, _ in pipes}
    if not args.no_baselines:
        methods[IMPRESSO] = IMPRESSO_SRC
        for llm, _, _, alone in pipes:
            methods[llm] = alone
    for m in [m for m in methods if not index_method(NER_DIR / methods[m][0], methods[m][1])]:
        print(f"  {m}: no output in {methods[m][0]}/ -- left out as a baseline")
        del methods[m]

   
    compare_fine.METHODS = methods
    argv = ["compare_fine.py", "--out-dir", str(out_dir), "--gold-dir", args.gold_dir]
    if args.groups:
        argv += ["--groups", *args.groups]
    if args.per_group_docs:
        argv.append("--per-group-docs")
    saved, sys.argv = sys.argv, argv
    try:
        compare_fine.main()
    finally:
        sys.argv = saved

    with open(out_dir / "overall_scores.csv", newline="", encoding="utf-8-sig") as f:
        overall = {(r["group"], r["method"], r["match"], r["category"]): r for r in csv.DictReader(f)}
    groups = sorted({k[0] for k in overall})
    scored_methods = [m for m in methods if any(k[1] == m for k in overall)]
    pipes = [p for p in pipes if p[1] in scored_methods]

    def f1(g, m, match, c):
        r = overall.get((g, m, match, c))
        return float(r["f1"]) if r and r["f1"] not in ("", None) else None

    def fmt(x):
        return f"{x:.3f}" if x is not None else "-"

    def diff(a, b):
        return a - b if a is not None and b is not None else None

    def fmt_d(x):
        return f"{x:+.3f}" if x is not None else "-"

    # Head-to-head F1 table 
    lines = [f"Head-to-head F1 ({', '.join(scored_methods)})", ""]
    col = max(12, max(len(m) for m in scored_methods) + 2)
    for match in ("strict", "fuzzy"):
        lines.append(f"{match} match:")
        lines.append("  " + "".ljust(21) + "".join(m.rjust(col) for m in scored_methods))
        for g in groups:
            for c in DELTA_CATEGORIES:
                lines.append(f"  {group_label(g):<8} {c:<11} "
                             + "".join(fmt(f1(g, m, match, c)).rjust(col) for m in scored_methods))
        lines.append("")

    # Gains 
    gain_cols = []  # (label, a, b): F1(a) - F1(b)
    for llm, name, _, _ in pipes:
        if IMPRESSO in scored_methods:
            gain_cols.append((f"{name} vs impresso", name, IMPRESSO))
        if llm in scored_methods:
            gain_cols.append((f"{name} vs {llm}", name, llm))
    names = [p[1] for p in pipes]
    if len(names) == 2:
        gain_cols.append((f"{names[1]} vs {names[0]}", names[1], names[0]))
    gain_rows = []
    if gain_cols:
        lines += ["Gains in F1 (positive = the first method is better):", ""]
        gw = max(len(lbl) for lbl, _, _ in gain_cols) + 2
        for match in ("strict", "fuzzy"):
            lines.append(f"{match} match:")
            lines.append("  " + "".ljust(21) + "".join(lbl.rjust(gw) for lbl, _, _ in gain_cols))
            for g in groups:
                for c in DELTA_CATEGORIES:
                    vals = [diff(f1(g, a, match, c), f1(g, b, match, c)) for _, a, b in gain_cols]
                    lines.append(f"  {group_label(g):<8} {c:<11} " + "".join(fmt_d(v).rjust(gw) for v in vals))
                    for (lbl, a, b), v in zip(gain_cols, vals):
                        gain_rows.append({"group": g, "match": match, "category": c, "comparison": lbl,
                                          "f1_first": fmt(f1(g, a, match, c)),
                                          "f1_second": fmt(f1(g, b, match, c)),
                                          "gain": f"{v:.4f}" if v is not None else ""})
            lines.append("")
        with open(out_dir / "combined_gains.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["group", "match", "category", "comparison", "f1_first",
                                              "f1_second", "gain"])
            w.writeheader()
            w.writerows(gain_rows)
    lines += ["COARSE collapses fine types to the five broad categories.", ""]

    # Entity-level analysis
    gold_paths = index_folder(Path(args.gold_dir), "_Human")
    with open(out_dir / "per_document_scores.csv", newline="", encoding="utf-8-sig") as f:
        scored_docs = sorted({(r["group"], r["document"]) for r in csv.DictReader(f)})
    idx = {m: index_method(NER_DIR / methods[m][0], methods[m][1]) for m in scored_methods}
    gold_cache, read_cache = {}, {}

    def gold_for(d):
        if d not in gold_cache:
            gold_cache[d] = by_text(compare_fine.read_fine_csv(gold_paths[d])[0])
        return gold_cache[d]

    def output(m, g, d):
        if (m, g, d) not in read_cache:
            ents, disp, _, _ = compare_fine.read_fine_csv(idx[m][g][d])
            read_cache[(m, g, d)] = (by_text(ents), disp)
        return read_cache[(m, g, d)]

    # What each LLM did to the Impresso BERT Model's tags.
    if IMPRESSO in scored_methods:
        change_rows = []
        for llm, name, _, _ in pipes:
            for g, d in scored_docs:
                gold = gold_for(d)
                before, before_disp = output(IMPRESSO, g, d)
                after, after_disp = output(name, g, d)
                for s in sorted(set(before) | set(after)):
                    if s in before and s in after:
                        change = "kept" if before[s] == after[s] else "retyped"
                    else:
                        change = "removed" if s in before else "added"
                    change_rows.append({
                        "pipeline": name, "group": g, "document": d,
                        "entity": after_disp.get(s) or before_disp.get(s) or s,
                        "impresso_type": "|".join(sorted(before.get(s, ()))),
                        "combined_type": "|".join(sorted(after.get(s, ()))),
                        "gold_type": "|".join(sorted(gold.get(s, ()))),
                        "change": change, "verdict": judge(change, before.get(s), after.get(s), gold.get(s))})
        with open(out_dir / "changes_vs_impresso.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["pipeline", "group", "document", "entity", "impresso_type",
                                              "combined_type", "gold_type", "change", "verdict"])
            w.writeheader()
            w.writerows(change_rows)

        lines.append("What each LLM did to Impresso's tags (distinct entities per document), checked against gold:")
        for _, name, _, _ in pipes:
            for g in groups:
                sub = [r for r in change_rows if r["pipeline"] == name and r["group"] == g]
                c = Counter(r["change"] for r in sub)
                v = Counter(r["verdict"] for r in sub)
                helpful = v["fixed type"] + v["good removal"] + v["good addition"]
                harmful = v["broke type"] + v["bad removal"] + v["bad addition"] + v["added, wrong type"]
                lines.append(f"  {name:<12} {group_label(g):<8} kept {c['kept']}, retyped {c['retyped']}, "
                             f"removed {c['removed']}, added {c['added']}")
                lines.append(f"  {'':<21} helpful {helpful} (fixed type {v['fixed type']}, good removal "
                             f"{v['good removal']}, good addition {v['good addition']}); harmful {harmful} "
                             f"(broke type {v['broke type']}, bad removal {v['bad removal']}, bad addition "
                             f"{v['bad addition']}, added with wrong type {v['added, wrong type']})")
        lines += ["  (details per entity in changes_vs_impresso.csv; a boundary fix counts as one removal + "
                  "one addition)", ""]

    # Where the two pipelines disagree.
    if len(names) == 2:
        a_name, b_name = names
        dis_rows = []
        for g, d in scored_docs:
            gold = gold_for(d)
            a, a_disp = output(a_name, g, d)
            b, b_disp = output(b_name, g, d)
            for s in sorted(set(a) | set(b)):
                if a.get(s) == b.get(s):
                    continue
                ra, rb = is_right(a.get(s), gold.get(s)), is_right(b.get(s), gold.get(s))
                verdict = ("both right" if ra and rb else f"only {a_name} right" if ra
                           else f"only {b_name} right" if rb else "both wrong")
                dis_rows.append({"group": g, "document": d, "entity": a_disp.get(s) or b_disp.get(s) or s,
                                 f"{a_name}_type": "|".join(sorted(a.get(s, ()))),
                                 f"{b_name}_type": "|".join(sorted(b.get(s, ()))),
                                 "gold_type": "|".join(sorted(gold.get(s, ()))) or "(not in gold)",
                                 "verdict": verdict})
        with open(out_dir / "pipeline_disagreements.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["group", "document", "entity", f"{a_name}_type", f"{b_name}_type",
                                              "gold_type", "verdict"])
            w.writeheader()
            w.writerows(dis_rows)
        lines.append(f"Where {a_name} and {b_name} disagree (distinct entities per document), checked against gold:")
        for g in groups:
            v = Counter(r["verdict"] for r in dis_rows if r["group"] == g)
            lines.append(f"  {group_label(g):<8} {sum(v.values())} disagreements: only {a_name} right "
                         f"{v[f'only {a_name} right']}, only {b_name} right {v[f'only {b_name} right']}, "
                         f"both wrong {v['both wrong']}, both right {v['both right']}")
        lines += ["  (details in pipeline_disagreements.csv; 'right' = found with a gold type for a gold "
                  "entity, or left out for a non-gold one; 'both right' = both include a gold type but "
                  "list different extra types)", ""]

    extra = "\n".join(lines)
    summary_path = out_dir / "summary.txt"
    # Put the new title and the head-to-head sections on top of compare_fine's summary.
    _, _, rest = summary_path.read_text(encoding="utf-8").partition("\n")   
    header, _, body = rest.partition("\n\n")                                 
    title = " vs ".join(f"{p[1]} (impresso -> {p[0]})" for p in pipes)
    summary_path.write_text(f"Combined pipelines: {title} -- against the human gold standard\n"
                            + header + "\n\n" + extra + "\n" + body, encoding="utf-8")
    print("\n" + extra)
    print(f"Full tables in {summary_path}")


if __name__ == "__main__":
    main()
