"""
NER Utilities Script
These are shared helper functions and script components to ensure consistency in variables among different script run files.

Code developed with the assistance of Claude Opus 5/5.5

"""


import argparse
import csv
import html
import re
import sys
import time
import unicodedata
from pathlib import Path

NER_DIR = Path(__file__).resolve().parent
RAWTEXT_DIR = NER_DIR / "rawtext"
GOLD_DIR = NER_DIR / "HumanValidated"

COARSE_CATEGORIES = ["Person", "Organization", "Location", "Time", "Product"]

ENTITY_FIELDS = ["entity", "coarse_type", "model_label", "start", "end", "confidence", "context"]
FINE_ENTITY_FIELDS = ["entity", "fine_type", "coarse_type", "model_label", "start", "end",
                      "confidence", "context"]

# Impresso Annotation Guidelines for fine-level entities
FINE_TYPE_DEFINITIONS = {
    "pers.ind": "an individual person, including fictional characters, though not when referring to the work itself",
    "pers.coll": "a named collective of people that is not a formal organization "
                 "(a family, a dynasty, a people referred to as a group)",
    "pers.ind.articleauthor": "the author of the article itself (byline or signature)",
    "org.adm": "an organization that plays mainly an administrative role: governmental, political, or judicial offices"
               "(ministries, parliaments, councils, courts, armies, regiments)",
    "org.ent": "any other organization that sells products or provides services that are not only administrative: companies, banks, associations, societies, clubs, "
               "schools, churches, military units, scientific, and cultural institutions",
    "org.ent.pressagency": "a press or news agency (e.g. AFP, Reuters)",
    "loc.adm.town": "a city, town, village, district, borough, locality, or municipality",
    "loc.adm.reg": "a sub-national administrative region (province, departement, canton, county, state). Entities which used to be a kind of independent states or strong entities but were not real states, or are not anymore today, are annotated as loc.adm.reg",
    "loc.adm.nat": "a country or nation state",
    "loc.adm.sup": "a supranational political or administrative entity like world regions, continents, etc.",
    "loc.phys.geo": "a natural geographical area or feature (islands, plain, "
                    "forest, etc.) Geonyms including names given to natural geographical spaces, such as deserts, mountains, mountain chains, glaciers, plains, chasms, plateaus, valleys, volcanoes, canyons, etc",
    "loc.phys.hydro": "a body of water: river, lake, sea, ocean, canal, bay",
    "loc.phys.astro": "a celestial body or astronomical location: stars, galaxies, and planets",
    "loc.oro": "streets, squares, roads, highways, etc. They are often artificial but sometimes natural, and are not necessarily geopolitical: this justifies not classifying them as ​loc​.adm​ and ​loc​.phys​.",
    "loc.fac": "a named facility or built structure: building, monument, church, station, port, "
               "bridge, or university. Often the physical location of an organization.",
    "loc.add.phys": "a postal or physical address (street and number). As opposed to a ​loc​.oro​, an address is a point in space (e.g. a point in a street)",
    "loc.unk": "a location whose precise type cannot be determined",
    "prod.media": "anything that is broadcast in the press, on radio or television: newspapers, magazines, broadcasts, sales catalogues, etc, not including films",
    "prod.doctr": "a doctrine, ideology, or belief system like socialism, communism, Protestantism, Buddhism, etc.",
    "time.date.abs": "an absolute date as in a calendar ('12 avril 1903', 'mars 1892', '1870')",
}
FINE_TYPES = list(FINE_TYPE_DEFINITIONS)
IMPRESSO_UNSUPPORTED_FINE = {"time.date.rel", "loc.phys.astro"}
HIPE_TO_COARSE = {"pers": "Person", "org": "Organization", "loc": "Location",
                  "time": "Time", "prod": "Product"}


def fine_to_coarse(fine_type):
    """'loc.adm.town' -> 'Location'; None if unrecognised."""
    return HIPE_TO_COARSE.get((fine_type or "").strip().casefold().split(".")[0])


# Document names

_KEY_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_[A-Za-z0-9]+)")


def doc_key(stem):
    """Shared key used to pair a rawtext document, a model output and a gold
    file: the 'YYYY-MM-DD_X' part of the name. 'HRG_1900-01-01_E',
    '1900-01-01_E' and '1900-01-01_E_Human' all give '1900-01-01_E'.
    Names without a date fall back to the full name."""
    m = _KEY_RE.search(stem)
    return m.group(1) if m else stem


def list_documents(rawtext_dir=RAWTEXT_DIR, groups=None):
    """-> sorted list of (group, path). group is the sub-folder of rawtext the
    file sits in ('HRG', 'HRS'), or '' for files directly in rawtext."""
    rawtext_dir = Path(rawtext_dir)
    if not rawtext_dir.is_dir():
        sys.exit(f"Input folder not found: {rawtext_dir}")
    docs = []
    for p in rawtext_dir.rglob("*.md"):
        rel = p.relative_to(rawtext_dir).parent.as_posix()
        group = "" if rel == "." else rel
        if groups and group not in groups:
            continue
        docs.append((group, p))
    docs.sort(key=lambda gp: (gp[0], gp[1].name))
    if not docs:
        sys.exit(f"No .md files found in {rawtext_dir}"
                 + (f" sub-folder(s) {', '.join(groups)}" if groups else ""))
    return docs

# Text cleaning and chunking

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_HTML_TAG_RE = re.compile(r"<[^>\n]+>")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.M)
_TABLE_RULE_RE = re.compile(r"^[ \t]*\|?[ \t:\-|]{3,}\|?[ \t]*$", re.M)
_EMPH_RE = re.compile(r"\*{1,3}|_{2,3}")
_SPACES_RE = re.compile(r"[\u00A0\u202F\u2009\u200A\t]")


def clean_text(raw):
    t = unicodedata.normalize("NFC", raw)
    t = _IMG_RE.sub(" ", t)
    t = _LINK_RE.sub(r"\1", t)
    t = _BR_RE.sub("\n", t)
    t = _HTML_TAG_RE.sub("", t)        
    t = html.unescape(t)
    t = _HEADING_RE.sub("", t)
    t = _TABLE_RULE_RE.sub("", t)
    t = t.replace("|", " ")
    t = _EMPH_RE.sub("", t)
    t = _SPACES_RE.sub(" ", t)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def read_document(path):
    return clean_text(Path(path).read_text(encoding="utf-8", errors="replace"))


_SPLIT_LEVELS = [re.compile(r"\n\s*\n"), re.compile(r"(?<=[.!?;:\u00BB])\s+"), re.compile(r"\s+")]


def _split_by(text, start, end, pattern):
    pieces, pos = [], start
    for m in pattern.finditer(text, start, end):
        if m.start() > pos:
            pieces.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        pieces.append((pos, end))
    return pieces


def _atomize(text, start, end, max_chars, level):
    if end - start <= max_chars:
        return [(start, end)]
    if level >= len(_SPLIT_LEVELS):
        return [(s, min(s + max_chars, end)) for s in range(start, end, max_chars)]
    pieces = _split_by(text, start, end, _SPLIT_LEVELS[level])
    if len(pieces) <= 1:
        return _atomize(text, start, end, max_chars, level + 1)
    out = []
    for s, e in pieces:
        out.extend(_atomize(text, s, e, max_chars, level + 1) if e - s > max_chars else [(s, e)])
    return out


def chunk_spans(text, max_chars, start=0, end=None):
    """(start, end) spans of at most max_chars, split at paragraphs, then
    sentences, then words. Together they cover every non-space character."""
    end = len(text) if end is None else end
    atoms = [(s, e) for s, e in _atomize(text, start, end, max_chars, 0) if text[s:e].strip()]
    chunks, cur = [], None
    for s, e in atoms:
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_chars:
            cur[1] = e
        else:
            chunks.append(tuple(cur))
            cur = [s, e]
    if cur is not None:
        chunks.append(tuple(cur))
    return chunks


def fit_to_token_limit(tokenizer, text, spans, limit=500):
    """Re-split any chunk whose token count is over the model's window."""
    out = []
    for s, e in spans:
        n = len(tokenizer(text[s:e], add_special_tokens=True)["input_ids"])
        if n <= limit or e - s < 40:
            out.append((s, e))
            continue
        sub = chunk_spans(text, max((e - s) // 2, 20), s, e)
        out.extend([(s, e)] if sub == [(s, e)] else fit_to_token_limit(tokenizer, text, sub, limit))
    return out


def resolve_device(choice="auto"):
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu" 
    except ImportError:
        return "cpu"


# Output and the per-document run loop

def _context(text, start, end, width=50):
    left = text[max(0, start - width):start]
    right = text[end:end + width]
    return re.sub(r"\s+", " ", f"{left}[{text[start:end]}]{right}").strip()


def write_entities(path, text, entities, fields=ENTITY_FIELDS):
    entities = sorted(entities, key=lambda e: (e["start"], e["end"]))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entities:
            row = {k: e.get(k, "") for k in fields if k != "context"}
            row["context"] = _context(text, e["start"], e["end"])
            w.writerow(row)


def make_parser(description, default_out):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--rawtext-dir", default=str(RAWTEXT_DIR),
                   help=f"Folder of input .md files (default: {RAWTEXT_DIR.name}/ in the NER folder)")
    p.add_argument("--out-dir", default=str(NER_DIR / default_out),
                   help=f"Output folder (default: {default_out}/ in the NER folder)")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo documents that already have an output file")
    p.add_argument("--limit", type=int,
                   help="Only process the first N documents of each sub-folder (quick test)")
    p.add_argument("--groups", nargs="+",
                   help="Only process these rawtext sub-folders, e.g. --groups HRG (default: all)")
    return p


def run_extractor(args, suffix, extract, label, after_doc=None, fields=ENTITY_FIELDS):
    """extract(text) -> list of entity dicts with keys entity, coarse_type,
    model_label, start, end, confidence. Raise to mark a document failed
    (nothing is written, so a re-run retries it). after_doc(group, stem,
    seconds) is called after each document that was written."""
    out_root = Path(args.out_dir)
    docs = list_documents(args.rawtext_dir, args.groups)
    if args.limit is not None:
        kept, per_group = [], {}
        for group, path in docs:
            per_group[group] = per_group.get(group, 0) + 1
            if per_group[group] <= args.limit:
                kept.append((group, path))
        docs = kept
    counts = {}
    for group, _ in docs:
        counts[group or "(rawtext root)"] = counts.get(group or "(rawtext root)", 0) + 1
    print(f"{label}: {len(docs)} document(s) -- "
          + ", ".join(f"{g}: {n}" for g, n in counts.items()) + f" -> {out_root}")
    done = skipped = failed = 0
    t_start = time.time()
    for i, (group, path) in enumerate(docs, 1):
        out_dir = out_root / group if group else out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{suffix}.csv"
        name = f"{group}/{path.stem}" if group else path.stem
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(docs)}] {name}: already done, skipping")
            skipped += 1
            continue
        text = read_document(path)
        t0 = time.time()
        try:
            entities = extract(text)
        except Exception as exc: 
            print(f"  [{i}/{len(docs)}] {name}: FAILED -- {type(exc).__name__}: {exc}")
            failed += 1
            continue
        write_entities(out_path, text, entities, fields)
        done += 1
        elapsed = time.time() - t0
        print(f"  [{i}/{len(docs)}] {name}: {len(entities)} entities ({elapsed:.1f}s)")
        if after_doc:
            after_doc(group, path.stem, elapsed)
    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min: {done} written, "
          f"{skipped} skipped (already done), {failed} failed.")
    if failed:
        print("Re-run the same command to retry the failed documents.")
    return failed
