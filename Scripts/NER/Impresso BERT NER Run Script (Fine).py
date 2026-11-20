"""
Impresso BERT NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

# ner_impresso_broad renamed to Impresso BERT NER Run Script (Fine) on Github for file clarity, rename back for file link
from ner_impresso_broad import MODEL_NAME, decode_chunk, load_model
from ner_utils import (FINE_ENTITY_FIELDS, chunk_spans, fine_to_coarse, fit_to_token_limit,
                       make_parser, resolve_device, run_extractor)

FINE_HEAD = "NE-FINE-LIT"


def main():
    parser = make_parser("Fine-grained NER with impresso", "impresso_fine")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk-chars", type=int, default=800,
                        help="Max characters per chunk (default 800; re-split if over 512 tokens)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Loading {MODEL_NAME} ({FINE_HEAD} head) on {device} ...")
    tokenizer, model, labels, torch = load_model(device, FINE_HEAD)

    def extract(text):
        entities = []
        for s, e in fit_to_token_limit(tokenizer, text, chunk_spans(text, args.chunk_chars)):
            for sp in decode_chunk(text[s:e], tokenizer, model, labels, torch, device, FINE_HEAD):
                coarse = fine_to_coarse(sp["label"])
                if not coarse:
                    continue
                a, b = sp["start"] + s, sp["end"] + s
                entities.append({"entity": text[a:b], "fine_type": sp["label"], "coarse_type": coarse,
                                 "model_label": sp["label"], "start": a, "end": b,
                                 "confidence": round(sum(sp["confs"]) / len(sp["confs"]), 4)})
        return entities

    sys.exit(1 if run_extractor(args, "impresso_fine", extract, "impresso (fine)",
                                fields=FINE_ENTITY_FIELDS) else 0)


if __name__ == "__main__":
    main()
