"""
Qwen NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

# ner_mistral_fine renamed to Mistral NER Run Script (Fine) on Github for file clarity, rename back for file link
from ner_mistral_fine import SCHEMA_FINE, SYSTEM_PROMPT_FINE
from ner_utils import FINE_ENTITY_FIELDS, FINE_TYPES, chunk_spans, fine_to_coarse
from qwen_common import extract_chunks, make_parser, make_server, run_parallel, select_docs

LOG_FIELDS = ["group", "document", "model", "chunks", "returned", "invalid_type", "not_in_text",
              "salvaged_chunks", "seconds"]


def main():
    args = make_parser("Fine-grained NER with Qwen via vLLM", "qwen_fine").parse_args()
    server = make_server(args, SYSTEM_PROMPT_FINE, SCHEMA_FINE)
    docs = select_docs(args)

    def process(group, path, text):
        return extract_chunks(
            server, text, args.chunk_chars,
            user_builder=lambda chunk, s, e: "Text:\n<<<\n" + chunk + "\n>>>",
            allowed=set(FINE_TYPES),
            normalise_type=lambda raw: raw.strip().casefold(),
            make_entity=lambda ent, t, raw, a, b: {"entity": ent, "fine_type": t, "coarse_type": fine_to_coarse(t),
                                                   "model_label": raw, "start": a, "end": b, "confidence": ""},
            chunk_spans=chunk_spans)

    failed = run_parallel(args, docs, "qwen_fine", FINE_ENTITY_FIELDS, process, "_qwen_fine_run_log.csv",
                          LOG_FIELDS, "Qwen fine", server)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
