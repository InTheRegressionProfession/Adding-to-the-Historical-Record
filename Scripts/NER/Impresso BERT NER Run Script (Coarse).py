"""
Impresso BERT NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import sys

from ner_utils import (chunk_spans, fit_to_token_limit, make_parser, resolve_device,
                       run_extractor)

MODEL_NAME = "impresso-project/ner-stacked-bert-multilingual"
HEAD = "NE-COARSE-LIT"
HIPE_TO_COARSE = {"pers": "Person", "org": "Organization", "loc": "Location",
                  "time": "Time", "prod": "Product"}


def id2label_from_label_map(label_map):
    """config.label_map is {task: {label: id}}; invert it per task."""
    out = {}
    for task, mapping in label_map.items():
        if all(isinstance(v, int) for v in mapping.values()):
            out[task] = {v: k for k, v in mapping.items()}
        else:
            out[task] = {int(k): v for k, v in mapping.items()}
    return out


def load_model(device, head=HEAD):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not getattr(tokenizer, "is_fast", False):
        sys.exit("A fast tokenizer is needed for character offsets: pip install -U tokenizers")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device).eval()
    label_map = getattr(model.config, "label_map", None)
    if not label_map:
        sys.exit("The impresso model config has no label_map -- the model repository may have "
                 "changed; check its Hugging Face page.")
    id2label = id2label_from_label_map(label_map)
    if head not in id2label:
        sys.exit(f"The model has no {head} head (it has: {', '.join(id2label)})")
    return tokenizer, model, id2label[head], torch


def decode_chunk(chunk, tokenizer, model, labels, torch, device, head=HEAD):
    """Run one chunk through the model and turn the head's BIO tags into spans."""
    enc = tokenizer(chunk, return_offsets_mapping=True, truncation=True, max_length=512,
                    return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    word_ids = enc.word_ids(0)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device))
    logits_by_task = out.logits if hasattr(out, "logits") else out[0]
    probs = torch.softmax(logits_by_task[head][0], dim=-1)
    conf, pred = probs.max(dim=-1)

    words, prev = [], None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        s, e = offsets[i]
        if wid == prev and words:
            words[-1][1] = e
        else:
            words.append([s, e, labels.get(int(pred[i]), "O"), float(conf[i])])
        prev = wid

    spans, cur = [], None
    for s, e, label, c in words:
        if label == "O" or "-" not in label:
            if cur:
                spans.append(cur)
            cur = None
            continue
        bio, typ = label.split("-", 1)
        if bio == "B" or cur is None or cur["label"] != typ:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "label": typ, "confs": [c]}
        else:
            cur["end"] = e
            cur["confs"].append(c)
    if cur:
        spans.append(cur)
    return spans


def main():
    parser = make_parser("Broad NER with impresso", "impresso_broad")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk-chars", type=int, default=800,
                        help="Max characters per chunk (default 800; re-split if over 512 tokens)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Loading {MODEL_NAME} on {device} ...")
    tokenizer, model, labels, torch = load_model(device)

    def extract(text):
        entities = []
        for s, e in fit_to_token_limit(tokenizer, text, chunk_spans(text, args.chunk_chars)):
            for sp in decode_chunk(text[s:e], tokenizer, model, labels, torch, device):
                coarse = HIPE_TO_COARSE.get(sp["label"])
                if coarse:
                    a, b = sp["start"] + s, sp["end"] + s
                    entities.append({"entity": text[a:b], "coarse_type": coarse,
                                     "model_label": sp["label"], "start": a, "end": b,
                                     "confidence": round(sum(sp["confs"]) / len(sp["confs"]), 4)})
        return entities

    sys.exit(1 if run_extractor(args, "impresso_broad", extract, "impresso") else 0)


if __name__ == "__main__":
    main()
