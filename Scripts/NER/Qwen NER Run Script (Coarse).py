"""
Qwen NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

# ner_mistral_broad renamed to Mistral NER Run Script (Coarse) on Github for file clarity, rename back for file link
from ner_mistral_broad import _TYPE_ALIASES, RESPONSE_SCHEMA, SYSTEM_PROMPT
from ner_utils import COARSE_CATEGORIES, ENTITY_FIELDS, chunk_spans
from qwen_common import extract_chunks, make_parser, make_server, run_parallel, select_docs

LOG_FIELDS = ["group", "document", "model", "chunks", "returned", "invalid_type", "not_in_text",
              "salvaged_chunks", "seconds"]


def main():
    args = make_parser("Broad NER with Qwen via vLLM", "qwen_broad").parse_args()
    server = make_server(args, SYSTEM_PROMPT, RESPONSE_SCHEMA)
    docs = select_docs(args)

    def process(group, path, text):
        return extract_chunks(
            server, text, args.chunk_chars,
            user_builder=lambda chunk, s, e: "Text:\n<<<\n" + chunk + "\n>>>",
            allowed=set(COARSE_CATEGORIES),
            normalise_type=lambda raw: _TYPE_ALIASES.get(raw.strip().casefold()),
            make_entity=lambda ent, t, raw, a, b: {"entity": ent, "coarse_type": t, "model_label": raw,
                                                   "start": a, "end": b, "confidence": ""},
            chunk_spans=chunk_spans)

    failed = run_parallel(args, docs, "qwen_broad", ENTITY_FIELDS, process, "_qwen_broad_run_log.csv",
                          LOG_FIELDS, "Qwen broad", server)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
