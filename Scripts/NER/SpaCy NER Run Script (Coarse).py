"""
SpaCy NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

from ner_utils import make_parser, run_extractor

SPACY_LABEL_MAP = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Product"}


def main():
    parser = make_parser("Broad NER with spaCy", "SpaCy_broad")
    parser.add_argument("--model", default="fr_core_news_lg", help="spaCy model (default fr_core_news_lg)")
    args = parser.parse_args()

    import spacy
    try:
        nlp = spacy.load(args.model)
    except OSError:
        sys.exit(f"spaCy model '{args.model}' is not installed. Run:\n"
                 f"    python -m spacy download {args.model}")

    def extract(text):
        nlp.max_length = max(nlp.max_length, len(text) + 1)
        return [{"entity": ent.text, "coarse_type": SPACY_LABEL_MAP[ent.label_],
                 "model_label": ent.label_, "start": ent.start_char, "end": ent.end_char,
                 "confidence": ""}
                for ent in nlp(text).ents if ent.label_ in SPACY_LABEL_MAP]

    sys.exit(1 if run_extractor(args, "spacy_broad", extract, f"spaCy ({args.model})") else 0)


if __name__ == "__main__":
    main()
