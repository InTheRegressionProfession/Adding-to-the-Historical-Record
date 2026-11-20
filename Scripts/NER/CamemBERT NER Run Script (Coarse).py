"""
CamemBERT NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

from ner_utils import (chunk_spans, fit_to_token_limit, make_parser, resolve_device,
                       run_extractor)

MODEL_NAME = "Jean-Baptiste/camembert-ner-with-dates"
LABEL_MAP = {"PER": "Person", "ORG": "Organization", "LOC": "Location",
             "DATE": "Time", "MISC": "Product"}


def main():
    parser = make_parser("Broad NER with CamemBERT", "camembert_broad")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk-chars", type=int, default=800,
                        help="Max characters per chunk (default 800; re-split if over 512 tokens)")
    args = parser.parse_args()

    from transformers import pipeline
    device = resolve_device(args.device)
    print(f"Loading {MODEL_NAME} on {device} ...")
    ner = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple",
                   device=0 if device == "cuda" else -1)

    def extract(text):
        entities = []
        for s, e in fit_to_token_limit(ner.tokenizer, text, chunk_spans(text, args.chunk_chars)):
            for ent in ner(text[s:e]):
                coarse = LABEL_MAP.get(ent.get("entity_group"))
                if not coarse or ent.get("start") is None:
                    continue
                a, b = ent["start"] + s, ent["end"] + s
                # trim the leading space the tokenizer sometimes attaches
                while a < b and text[a].isspace():
                    a += 1
                if a < b:
                    entities.append({"entity": text[a:b], "coarse_type": coarse,
                                     "model_label": ent["entity_group"], "start": a, "end": b,
                                     "confidence": round(float(ent["score"]), 4)})
        return entities

    sys.exit(1 if run_extractor(args, "camembert_broad", extract, "CamemBERT") else 0)


if __name__ == "__main__":
    main()
