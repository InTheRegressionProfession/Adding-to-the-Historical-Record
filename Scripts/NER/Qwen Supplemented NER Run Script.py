"""
Qwen Supplemented NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys
from pathlib import Path

from compare_broad import norm
# ner_mistral_broad renamed to Mistral NER Run Script (Coarse) on Github for file clarity, rename back for file link
# ner_combined_pipeline renamed to Mistral Supplemented NER Run Script on Github for file clarity, rename back for file link
from ner_combined_pipeline import OUTPUT_FIELDS, load_impresso_tags, system_prompt, user_message
from ner_mistral_broad import response_schema
from ner_utils import FINE_TYPES, NER_DIR, chunk_spans, fine_to_coarse
from qwen_common import extract_chunks, make_parser, make_server, run_parallel, select_docs

LOG_FIELDS = ["group", "document", "model", "tag_format", "min_tag_confidence", "impresso_distinct",
              "tags_shown_occurrences", "confirmed", "retyped", "added", "removed", "invalid_type",
              "not_in_text", "chunks", "salvaged_chunks", "seconds"]


def main():
    parser = make_parser("Combined pipeline (impresso tags -> Qwen review) via vLLM", "qwen_combined")
    parser.add_argument("--impresso-dir", default=str(NER_DIR / "impresso_fine"),
                        help="Where ner_impresso_fine.py wrote its output (default: impresso_fine/)")
    parser.add_argument("--tag-format", choices=["inline", "list"], default="inline",
                        help="Show impresso's tags inline in the text (default) or as a list after it")
    parser.add_argument("--min-tag-confidence", type=float, default=0.0,
                        help="Only show tags at or above this confidence (default 0 = all)")
    args = parser.parse_args()

    impresso_dir = Path(args.impresso_dir)
    if not impresso_dir.is_dir():
        sys.exit(f"No impresso output folder at {impresso_dir}. Run ner_impresso_fine.py first.")
    server = make_server(args, system_prompt(args.tag_format), response_schema(FINE_TYPES))
    docs = select_docs(args)
    print(f"Tags shown {args.tag_format}, confidence >= {args.min_tag_confidence}")

    def process(group, path, text):
        imp_path = (impresso_dir / group if group else impresso_dir) / f"{path.stem}_impresso_fine.csv"
        if not imp_path.exists():
            raise RuntimeError(f"no impresso_fine output ({imp_path.name}) -- run ner_impresso_fine.py first")
        all_tags, _ = load_impresso_tags(imp_path, text)
        shown = [t for t in all_tags if t["confidence"] is None or t["confidence"] >= args.min_tag_confidence]

        entities, st = extract_chunks(
            server, text, args.chunk_chars,
            user_builder=lambda chunk, s, e: user_message(chunk, s, e, shown, args.tag_format),
            allowed=set(FINE_TYPES),
            normalise_type=lambda raw: raw.strip().casefold(),
            make_entity=lambda ent, t, raw, a, b: {"entity": ent, "fine_type": t, "coarse_type": fine_to_coarse(t),
                                                   "model_label": raw, "start": a, "end": b, "confidence": ""},
            chunk_spans=chunk_spans)


        imp_types = {}
        for t in all_tags:
            imp_types.setdefault(norm(t["entity"]), set()).add(t["fine_type"])
        found, seen = set(), set()
        counts = {"confirmed": 0, "retyped": 0, "added": 0}
        for ent in entities:
            n = norm(ent["entity"])
            found.add(n)
            before = imp_types.get(n)
            ent["impresso_type"] = "|".join(sorted(before)) if before else ""
            ent["change"] = "added" if not before else ("confirmed" if ent["fine_type"] in before else "retyped")
            if (n, ent["fine_type"]) not in seen:
                seen.add((n, ent["fine_type"]))
                counts[ent["change"]] += 1
        st.update(counts)
        st.update({"removed": len(set(imp_types) - found), "impresso_distinct": len(imp_types),
                   "tags_shown_occurrences": len(shown), "tag_format": args.tag_format,
                   "min_tag_confidence": args.min_tag_confidence})
        return entities, st

    failed = run_parallel(args, docs, "qwen_combined", OUTPUT_FIELDS, process, "_qwen_combined_run_log.csv",
                          LOG_FIELDS, "Qwen combined", server)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
