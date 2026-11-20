"""
Mistral NER Run Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

from ner_utils import COARSE_CATEGORIES, chunk_spans, make_parser, run_extractor

TYPE_DEFINITIONS = {
    "Person": "a named individual person, or a named collective of people such as a family, "
              "dynasty, or group of people",
    "Organization": "a named organization of any kind: administrative, governmental, political, "
                    "judicial, military, commercial, financial, religious, cultural or "
                    "educational institutions and or associations",
    "Location": "a named place of any kind: towns, regions, countries, "
                "natural features (rivers, seas, mountains, islands), named buildings, and "
                "facilities, streets, and specific addresses",
    "Time": "a date or time expression, absolute only ('12 avril 1903', 'mars 1892', '1870') ",
    "Product": "a named human-made product or broadcast item: newspapers, periodicals, books, doctrines, and other "
               "such items.",
}

RULES = (
    "Rules:\n"
    "- Document each entity EXACTLY as it appears in the text: with the same spelling, accents "
    "and capitalization, OCR errors included. Do not correct, translate, or expand upon it.\n"
    "- Include titles and honorifics written directly before a person's name (\"M. Loubet\", "
    "\"Mme Curie\", \"général Gallieni\"), but nothing further.\n"
    "- Do not include leading articles (le, la, les, l', du, des) unless they are part of the name.\n"
    "- There should be no overlap between different entities (i.e., no nested or intersecting spans). \n"
    "- Only include spans that match one of the specified labels.\n"
    "- Be precise and only extract valid named entities. \n"
    "- List each distinct entity once, even if it occurs several times.\n"
    "- Do not annotate the following: Names of diseases (AIDS, SIDA, etc.), Psychological phenomena (Stockholm syndrome), Political or judicial matters (Watergate, the Dreyfus Affair, etc), however, the components of these expressions may still be annotated (annotate Stockholm within Stockholm Syndrome as a location)\n"
    "- Linguistic units considered as named entities must include a proper name, or a definite description having the status of a proper name. Named entity mentions exclude subordinate clauses, incidental clauses or insertions (if an insertion divides a mention, each part is annotated separately)\n"
    "- Assign exactly one type from the list above to each entity.\n\n"
)


def build_system_prompt(type_definitions):
    return (
        "You are an effective named entity recognition system. Your task is to identify and label named entities within the given set of text using the following entity label set: \n\n"
        + "\n".join(f"- {t}: {d}" for t, d in type_definitions.items()) + "\n\n"
        + "The text was collected by OCR methods and may contain recognition errors; annotate it as it is written. Also follow these important guidelines: \n\n"
        + RULES
        + "Respond with JSON only, in exactly this form:\n"
        '{"entities": [{"text": "<entity as it appears in the text>", "type": "<type>"}]}\n'
        'Most passages contain entities; check carefully before returning an empty list..'
    )


def response_schema(allowed_types):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "entities",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"},
                                           "type": {"type": "string", "enum": list(allowed_types)}},
                            "required": ["text", "type"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["entities"],
                "additionalProperties": False,
            },
        },
    }


SYSTEM_PROMPT = build_system_prompt(TYPE_DEFINITIONS)
RESPONSE_SCHEMA = response_schema(COARSE_CATEGORIES)

_TYPE_ALIASES = {"person": "Person", "personne": "Person", "per": "Person",
                 "organization": "Organization", "organisation": "Organization", "org": "Organization",
                 "location": "Location", "lieu": "Location", "loc": "Location",
                 "time": "Time", "date": "Time", "temps": "Time",
                 "product": "Product", "produit": "Product", "prod": "Product"}
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


class LMStudio:
    def __init__(self, base_url, model, max_tokens, temperature, timeout=600,
                 system_prompt=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA):
        self.base_url = base_url.rstrip("/")
        self.system_prompt, self.schema = system_prompt, schema
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.timeout = timeout
        self.use_schema = True

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/models", timeout=15)
            r.raise_for_status()
        except requests.RequestException as exc:
            sys.exit(f"Can't reach LM Studio at {self.base_url} ({exc}).\n"
                     "In LM Studio: load Mistral Small 3.2, then Developer tab -> Start Server.")
        return [m["id"] for m in r.json().get("data", [])]

    def chat(self, user):
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": user}]}
        if self.use_schema:
            payload["response_format"] = self.schema
        last = None
        for attempt in range(1, 4):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
                if r.status_code == 400 and self.use_schema and "response_format" in r.text:
                    print("    (server rejected structured output; falling back to plain JSON)")
                    self.use_schema = False
                    payload.pop("response_format", None)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = exc
                time.sleep(3 * attempt)
        raise RuntimeError(f"LM Studio request failed after retries: {last}")


def parse_response(content):
    """-> (list of (text, type), salvaged) or (None, False) if unreadable."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    data, salvaged = None, False
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if data is None: 
        items = []
        for m in re.finditer(r"\{[^{}]*\}", content):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
        if not items:
            return None, False
        data, salvaged = items, True
    if isinstance(data, dict):
        data = data.get("entities", [])
    if not isinstance(data, list):
        return None, False
    out = []
    for item in data:
        if isinstance(item, dict):
            text, typ = item.get("text"), item.get("type")
            if isinstance(text, str) and isinstance(typ, str) and text.strip():
                out.append((text.strip(), typ.strip()))
    return out, salvaged


def locate(surface, chunk):
    tokens = surface.translate(_APOS).split()
    if not tokens:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in tokens), chunk.translate(_APOS), re.I)
    return (m.start(), m.end()) if m else None


def pick_model(client, requested):
    available = client.list_models()
    if requested:
        if available and requested not in available:
            print(f"Warning: LM Studio lists {', '.join(available)}; '{requested}' not among them.")
        return requested
    candidates = [m for m in available if "mistral-small" in m.lower()]
    if not candidates:
        sys.exit("No Mistral Small model is loaded in LM Studio. Loaded models: "
                 f"{', '.join(available) or 'none'}.\nLoad it, or pass --model <name>.")
    preferred = [m for m in candidates if "3.2" in m] or candidates
    return preferred[0]


def main():
    parser = make_parser("Broad NER with Mistral-Small-3.2 via LM Studio", "mistral_broad")
    parser.add_argument("--base-url", default="http://localhost:1234/v1",
                        help="LM Studio server (default http://localhost:1234/v1)")
    parser.add_argument("--model", help="Model name as LM Studio lists it (default: the loaded "
                                        "Mistral Small model)")
    parser.add_argument("--chunk-chars", type=int, default=2500,
                        help="Characters of text per request (default 2500)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max answer tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    client = LMStudio(args.base_url, None, args.max_tokens, args.temperature)
    client.model = pick_model(client, args.model)
    print(f"Using model: {client.model}")

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
                coarse = _TYPE_ALIASES.get(raw_type.casefold())
                if coarse is None:
                    st["invalid_type"] += 1
                    continue
                loc = locate(surface, chunk)
                if loc is None:
                    st["not_in_text"] += 1
                    continue
                a, b = s + loc[0], s + loc[1]
                entities.append({"entity": text[a:b], "coarse_type": coarse, "model_label": raw_type,
                                 "start": a, "end": b, "confidence": ""})
        stats.clear()
        stats.update(st)
        return entities

    log_path = Path(args.out_dir) / "_mistral_run_log.csv"
    log_fields = ["group", "document", "model", "chunks", "returned", "invalid_type", "not_in_text",
                  "salvaged_chunks", "seconds"]

    def after_doc(group, stem, seconds):
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            if new:
                w.writeheader()
            w.writerow({"group": group, "document": stem, "model": client.model, **stats, "seconds": round(seconds, 1)})
        if stats.get("invalid_type") or stats.get("not_in_text"):
            print(f"      dropped {stats['invalid_type']} with an invalid type, "
                  f"{stats['not_in_text']} not found in the text")

    sys.exit(1 if run_extractor(args, "mistral_broad", extract, "Mistral", after_doc) else 0)


if __name__ == "__main__":
    main()
