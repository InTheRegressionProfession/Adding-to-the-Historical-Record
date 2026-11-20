"""
Mistral Supplemented NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import csv
import sys
import time
from pathlib import Path

from compare_broad import norm
# ner_mistral_broad renamed to Mistral NER Run Script (Coarse) on Github for file clarity, rename back for file link
from ner_mistral_broad import LMStudio, build_system_prompt, locate, parse_response, pick_model, response_schema
from ner_utils import (FINE_TYPE_DEFINITIONS, FINE_TYPES, NER_DIR, chunk_spans, fine_to_coarse,
                       list_documents, make_parser, read_document, write_entities)

OUTPUT_FIELDS = ["entity", "fine_type", "coarse_type", "impresso_type", "change",
                 "model_label", "start", "end", "confidence", "context"]

PREAMBLE = {
    "inline": (
        "The text has been pre-annotated by an automatic named entity recognition system. Each entity tag takes the following form, "
        "<TYPE conf=\"0.93\">entity form</TYPE>, where conf is the system's confidence between 0 and 1."
    ),
    "list": (
        "The text is followed by a list of named entities that an automatic named entity recognition system found in it, "
        "each with its type and the system's confidence between 0 and 1."
    ),
}
PREAMBLE_COMMON = (
    " The named entity recognition system is a useful guide, however, it may be incorrect. Some of its assignments may be incorrectly assigned or some entities in the original text may be missed altogether. "
    "Assignments with lesser confidence are less reliable. Use the existing entity assignments to assist with your task, but do not rely on them. \n\n"
    "Conduct your own complete annotation of the original text, correctly identifying named entities per the rules you have been given. Return the full, corrected list of distinct "
    "entities including the valid pre-identified entities you confirm (with boundaries or types corrected where "
    "needed) plus any entities the original named entity recognition system missed. Leave out pre-identified items that are not "
    "valid. \n\n"
)


def system_prompt(tag_format):
    """The fine-grained prompt from ner_mistral_broad.build_system_prompt(), with the
    pre-annotation preamble inserted right after the list of entity types (so it works
    whatever wording surrounds that list). If the list can't be found, the preamble goes
    first."""
    base = build_system_prompt(FINE_TYPE_DEFINITIONS)
    preamble = PREAMBLE[tag_format] + PREAMBLE_COMMON
    types_block = "\n".join(f"- {t}: {d}" for t, d in FINE_TYPE_DEFINITIONS.items())
    if types_block in base:
        head, _, tail = base.partition(types_block)
        tail = tail.lstrip("\n")
        return head + types_block + "\n\n" + preamble + tail
    return preamble + base


def load_impresso_tags(path, text):
    """Read an impresso_fine CSV. Offsets are checked against the current text;
    if the rawtext changed since impresso ran, tags are re-located by their text.
    -> (tags, n_unplaceable)"""
    tags, lost = [], 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ent, fine = row.get("entity") or "", (row.get("fine_type") or "").strip()
            if not ent.strip() or not fine:
                continue
            try:
                s, e = int(row["start"]), int(row["end"])
            except (KeyError, ValueError):
                s = e = -1
            if not (0 <= s < e <= len(text) and text[s:e] == ent):
                loc = locate(ent, text)
                if loc is None:
                    lost += 1
                    continue
                s, e = loc
            try:
                conf = float(row.get("confidence") or "")
            except ValueError:
                conf = None
            tags.append({"start": s, "end": e, "entity": text[s:e], "fine_type": fine, "confidence": conf})
    tags.sort(key=lambda t: (t["start"], -t["end"]))
    return tags, lost


def user_message(chunk, start, end, tags, tag_format):
    inside = [t for t in tags if t["start"] >= start and t["end"] <= end]
    if tag_format == "list":
        seen, lines = set(), []
        for t in inside:
            key = (t["entity"], t["fine_type"])
            if key in seen:
                continue
            seen.add(key)
            conf = f", confidence {t['confidence']:.2f}" if t["confidence"] is not None else ""
            lines.append(f"- {t['entity']} ({t['fine_type']}{conf})")
        return ("Text:\n<<<\n" + chunk + "\n>>>\n\nEntities pre-identified by the automatic tagger in "
                "this text:\n" + ("\n".join(lines) if lines else "(none)"))
    pieces, pos = [], start
    for t in inside:
        if t["start"] < pos:
            continue
        conf = f' conf="{t["confidence"]:.2f}"' if t["confidence"] is not None else ""
        pieces.append(chunk[pos - start:t["start"] - start])
        pieces.append(f'<{t["fine_type"]}{conf}>{chunk[t["start"] - start:t["end"] - start]}</{t["fine_type"]}>')
        pos = t["end"]
    pieces.append(chunk[pos - start:])
    return "Pre-annotated text:\n<<<\n" + "".join(pieces) + "\n>>>"


def main():
    parser = make_parser("impresso fine tags -> Mistral review (combined pipeline)", "combined_pipeline")
    parser.add_argument("--impresso-dir", default=str(NER_DIR / "impresso_fine"),
                        help="Where ner_impresso_fine.py wrote its output (default: impresso_fine/)")
    parser.add_argument("--tag-format", choices=["inline", "list"], default="inline",
                        help="Show impresso's tags inline in the text (default) or as a list after it")
    parser.add_argument("--min-tag-confidence", type=float, default=0.0,
                        help="Only show Mistral impresso tags at or above this confidence (default 0 = all)")
    parser.add_argument("--base-url", default="http://localhost:1234/v1",
                        help="LM Studio server (default http://localhost:1234/v1)")
    parser.add_argument("--model", help="Model name as LM Studio lists it (default: the loaded Mistral Small model)")
    parser.add_argument("--chunk-chars", type=int, default=2500, help="Characters of text per request (default 2500)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max answer tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    impresso_dir, out_root = Path(args.impresso_dir), Path(args.out_dir)
    if not impresso_dir.is_dir():
        sys.exit(f"No impresso output folder at {impresso_dir}. Run ner_impresso_fine.py first.")

    docs = list_documents(args.rawtext_dir, args.groups)
    if args.limit is not None:
        kept, per_group = [], {}
        for group, path in docs:
            per_group[group] = per_group.get(group, 0) + 1
            if per_group[group] <= args.limit:
                kept.append((group, path))
        docs = kept

    client = LMStudio(args.base_url, None, args.max_tokens, args.temperature,
                      system_prompt=system_prompt(args.tag_format), schema=response_schema(FINE_TYPES))
    client.model = pick_model(client, args.model)
    allowed = set(FINE_TYPES)
    print(f"Using model: {client.model}  |  tags shown {args.tag_format}, confidence >= {args.min_tag_confidence}")
    print(f"Combined pipeline: {len(docs)} document(s) -> {out_root}")

    log_path = out_root / "_combined_pipeline_run_log.csv"
    log_fields = ["group", "document", "model", "tag_format", "min_tag_confidence", "impresso_distinct",
                  "tags_shown_occurrences", "confirmed", "retyped", "added", "removed", "invalid_type", "not_in_text",
                  "chunks", "seconds"]
    done = skipped = failed = 0
    t_start = time.time()

    for i, (group, path) in enumerate(docs, 1):
        out_dir = out_root / group if group else out_root
        name = f"{group}/{path.stem}" if group else path.stem
        out_path = out_dir / f"{path.stem}_combined_pipeline.csv"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(docs)}] {name}: already done, skipping")
            skipped += 1
            continue
        imp_path = (impresso_dir / group if group else impresso_dir) / f"{path.stem}_impresso_fine.csv"
        if not imp_path.exists():
            print(f"  [{i}/{len(docs)}] {name}: no impresso_fine output ({imp_path.name}) -- "
                  "run ner_impresso_fine.py for it first")
            failed += 1
            continue

        text = read_document(path)
        all_tags, lost = load_impresso_tags(imp_path, text)
        if lost:
            print(f"      note: {lost} impresso entity(ies) no longer found in the text (rawtext edited?)")
        shown = [t for t in all_tags if t["confidence"] is None or t["confidence"] >= args.min_tag_confidence]

        t0 = time.time()
        st = {"chunks": 0, "invalid_type": 0, "not_in_text": 0}
        entities = []
        try:
            for s, e in chunk_spans(text, args.chunk_chars):
                chunk = text[s:e]
                st["chunks"] += 1
                parsed, _ = parse_response(client.chat(user_message(chunk, s, e, shown, args.tag_format)))
                if parsed is None:
                    raise RuntimeError(f"unreadable answer for characters {s}-{e}")
                for surface, raw_type in parsed:
                    fine = raw_type.strip().casefold()
                    if fine not in allowed:
                        st["invalid_type"] += 1
                        continue
                    loc = locate(surface, chunk)
                    if loc is None:
                        st["not_in_text"] += 1
                        continue
                    a, b = s + loc[0], s + loc[1]
                    entities.append({"entity": text[a:b], "fine_type": fine, "coarse_type": fine_to_coarse(fine),
                                     "model_label": raw_type, "start": a, "end": b, "confidence": ""})
        except Exception as exc:
            print(f"  [{i}/{len(docs)}] {name}: FAILED -- {type(exc).__name__}: {exc}")
            failed += 1
            continue

        imp_types = {}
        for t in all_tags:
            imp_types.setdefault(norm(t["entity"]), set()).add(t["fine_type"])
        found = set()
        counts = {"confirmed": 0, "retyped": 0, "added": 0}
        seen_pairs = set()
        for ent in entities:
            n = norm(ent["entity"])
            found.add(n)
            before = imp_types.get(n)
            ent["impresso_type"] = "|".join(sorted(before)) if before else ""
            ent["change"] = ("added" if not before else
                             "confirmed" if ent["fine_type"] in before else "retyped")
            if (n, ent["fine_type"]) not in seen_pairs:
                seen_pairs.add((n, ent["fine_type"]))
                counts[ent["change"]] += 1
        removed = len(set(imp_types) - found)

        out_dir.mkdir(parents=True, exist_ok=True)
        write_entities(out_path, text, entities, OUTPUT_FIELDS)
        done += 1
        elapsed = time.time() - t0
        print(f"  [{i}/{len(docs)}] {name}: {len(entities)} entities ({elapsed:.1f}s) -- distinct vs impresso: "
              f"{counts['confirmed']} confirmed, {counts['retyped']} retyped, {counts['added']} added, "
              f"{removed} removed")
        if st["invalid_type"] or st["not_in_text"]:
            print(f"      dropped {st['invalid_type']} with an invalid type, {st['not_in_text']} not found in the text")

        new_log = not log_path.exists()
        out_root.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if new_log:
                w.writeheader()
            w.writerow({"group": group, "document": path.stem, "model": client.model,
                        "tag_format": args.tag_format, "min_tag_confidence": args.min_tag_confidence,
                        "impresso_distinct": len(imp_types), "tags_shown_occurrences": len(shown), **counts,
                        "removed": removed, "invalid_type": st["invalid_type"],
                        "not_in_text": st["not_in_text"], "chunks": st["chunks"], "seconds": round(elapsed, 1)})

    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min: {done} written, "
          f"{skipped} skipped (already done), {failed} failed or missing impresso output.")
    if failed:
        print("Re-run the same command to retry them.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
