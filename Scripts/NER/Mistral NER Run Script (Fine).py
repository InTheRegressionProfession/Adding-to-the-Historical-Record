"""
Mistral NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import csv
import sys
from pathlib import Path

# ner_mistral_broad renamed to Mistral NER Run Script (Fine) on Github for file clarity, rename back for file link
from ner_mistral_broad import LMStudio, build_system_prompt, locate, parse_response, pick_model, response_schema
from ner_utils import (FINE_ENTITY_FIELDS, FINE_TYPE_DEFINITIONS, FINE_TYPES, chunk_spans,
                       fine_to_coarse, make_parser, run_extractor)

SYSTEM_PROMPT_FINE = build_system_prompt(FINE_TYPE_DEFINITIONS)
SCHEMA_FINE = response_schema(FINE_TYPES)


def main():
    parser = make_parser("Fine-grained NER with Mistral-Small-3.2 via LM Studio", "mistral_fine")
    parser.add_argument("--base-url", default="http://localhost:1234/v1",
                        help="LM Studio server (default http://localhost:1234/v1)")
    parser.add_argument("--model", help="Model name as LM Studio lists it (default: the loaded "
                                        "Mistral Small model)")
    parser.add_argument("--chunk-chars", type=int, default=2500,
                        help="Characters of text per request (default 2500)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max answer tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    client = LMStudio(args.base_url, None, args.max_tokens, args.temperature,
                      system_prompt=SYSTEM_PROMPT_FINE, schema=SCHEMA_FINE)
    client.model = pick_model(client, args.model)
    print(f"Using model: {client.model}")

    allowed = set(FINE_TYPES)
    stats = {}

    def extract(text):
        st = {"chunks": 0, "returned": 0, "invalid_type": 0, "not_in_text": 0, "salvaged_chunks": 0}
        entities = []
        for s, e in chunk_spans(text, args.chunk_chars):
            chunk = text[s:e]
            st["chunks"] += 1
            parsed, salvaged = parse_response(client.chat("Text:\n<<<\n" + chunk + "\n>>>"))
            if parsed is None:
                raise RuntimeError(f"unreadable answer for characters {s}-{e}")
            st["salvaged_chunks"] += int(salvaged)
            st["returned"] += len(parsed)
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
        stats.clear()
        stats.update(st)
        return entities

    log_path = Path(args.out_dir) / "_mistral_fine_run_log.csv"
    log_fields = ["group", "document", "model", "chunks", "returned", "invalid_type", "not_in_text",
                  "salvaged_chunks", "seconds"]

    def after_doc(group, stem, seconds):
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if new:
                w.writeheader()
            w.writerow({"group": group, "document": stem, "model": client.model, **stats,
                        "seconds": round(seconds, 1)})
        if stats.get("invalid_type") or stats.get("not_in_text"):
            print(f"      dropped {stats['invalid_type']} with an invalid type, "
                  f"{stats['not_in_text']} not found in the text")

    sys.exit(1 if run_extractor(args, "mistral_fine", extract, "Mistral fine", after_doc,
                                fields=FINE_ENTITY_FIELDS) else 0)


if __name__ == "__main__":
    main()
