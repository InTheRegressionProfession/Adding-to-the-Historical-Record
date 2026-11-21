"""
Impresso NEL Run Script

Code developed with the assistance of Claude Opus 5/5.5

"""

import argparse
import csv
import datetime
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from nel_paths import NEL_DIR  
from compare_broad import (_col, coarse_from, group_label, index_folder, index_method, norm, read_csv_rows,
                           strip_invisible)
from ner_utils import GOLD_DIR, NER_DIR, RAWTEXT_DIR, doc_key, list_documents, read_document, resolve_device

MODEL_NAME = "impresso-project/nel-mgenre-multilingual"
GOLD = "gold"
METHODS = {  # name: (folder, file suffix); "gold" is read from --gold-dir
    GOLD: (None, "_Human"),
    "impresso": ("impresso_fine", "_impresso_fine"),
    "Mistral": ("mistral_fine", "_mistral_fine"),
    "Qwen": ("qwen_fine", "_qwen_fine"),
    "Mistral+imp": ("combined_pipeline", "_combined_pipeline"),
    "Qwen+imp": ("qwen_combined", "_qwen_combined"),
}
LINKABLE = ["Person", "Organization", "Location", "Product"]
QID_RE = re.compile(r"^Q\d+$")
HUMAN = {"Q5", "Q15632617"}  # human, fictional human


# Reading entities

def read_entities(path):
    """-> {normalized text: {"display", "fine": set, "coarse": set, "start", "human_qid"}}"""
    fields, rows = read_csv_rows(path)
    ent_col = _col(fields, "entity", "surface", "text")
    fine_col = _col(fields, "fine_type")
    coarse_col = _col(fields, "coarse_type", "type")
    start_col = _col(fields, "start")
    qid_col = _col(fields, "wikidata", "qid", "wikidata_id", "wkd_id")
    if ent_col is None:
        sys.exit(f"{path} has no 'entity' column")
    out = {}
    for row in rows:
        text = row.get(ent_col) or ""
        n = norm(text)
        if not n:
            continue
        fine = strip_invisible(row.get(fine_col) or "").strip().casefold() if fine_col else ""
        coarse = coarse_from(row.get(coarse_col, "") if coarse_col else "", fine)
        e = out.setdefault(n, {"display": text.strip(), "fine": set(), "coarse": set(), "start": None,
                               "human_qid": None})
        if fine:
            e["fine"].add(fine)
        if coarse:
            e["coarse"].add(coarse)
        if e["start"] is None and start_col and (row.get(start_col) or "").strip().isdigit():
            e["start"] = int(row[start_col])
        if qid_col:
            q = strip_invisible(row.get(qid_col) or "").strip().upper()
            q = q.rsplit("/", 1)[-1]  # accept full URLs
            if QID_RE.match(q) or q == "NIL":
                e["human_qid"] = e["human_qid"] or q
    return out


def find_span(text, display, hint=None):
    """First occurrence of the entity in the page (exact, then case/space-insensitive),
    else the method's own offset."""
    i = text.find(display)
    if i >= 0:
        return i, i + len(display)
    words = display.split()
    if words:
        m = re.search(r"\s+".join(map(re.escape, words)), text, re.IGNORECASE)
        if m:
            return m.span()
    if hint is not None and 0 <= hint < len(text):
        return hint, min(len(text), hint + len(display))
    return None


def model_input(text, span, display, n_words):
    if span is None:
        return f"[START] {' '.join(display.split())} [END]", False
    s, e = span
    left = text[:s].split()[-n_words:] if n_words else []
    right = text[e:].split()[:n_words] if n_words else []
    return " ".join(left + ["[START]", " ".join(text[s:e].split()), "[END]"] + right), True


# NEL Model

class ImpressoNEL:
    def __init__(self, model_name, device, beams, max_input_tokens, max_new_tokens=48):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            sys.exit("Needs transformers, torch and sentencepiece: pip install transformers torch sentencepiece")
        self.torch, self.device = torch, device
        self.beams, self.max_input, self.max_new = beams, max_input_tokens, max_new_tokens
        print(f"Loading {model_name} on {device} (first time: ~2.5 GB download) ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.truncated = 0

    def predict(self, texts):
        """-> list of (candidate strings, confidences), best first."""
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input)
        full = self.tok(texts, truncation=False)["input_ids"]
        self.truncated += sum(len(ids) > self.max_input for ids in full)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, num_beams=self.beams, num_return_sequences=self.beams,
                                      max_new_tokens=self.max_new, return_dict_in_generate=True,
                                      output_scores=True)
        decoded = self.tok.batch_decode(out.sequences, skip_special_tokens=True)
        conf = self.torch.exp(out.sequences_scores).cpu().tolist()
        b = self.beams
        return [([d.strip() for d in decoded[i * b:(i + 1) * b]], [round(c, 4) for c in conf[i * b:(i + 1) * b]])
                for i in range(len(texts))]


def split_prediction(pred):
    """'Émile Loubet >> fr' -> ('Émile Loubet', 'fr')"""
    if ">>" in pred:
        title, lang = pred.rsplit(">>", 1)
        return title.strip(), (lang.strip() or "en")
    return pred.strip(), "en"


# Wikipedia / Wikidata lookups

class Wiki:
    def __init__(self, cache_path, contact=None, offline=False):
        self.cache_path = Path(cache_path)
        self.offline = offline
        self.cache = {"titles": {}, "search": {}, "entities": {}}
        if self.cache_path.exists():
            self.cache.update(json.loads(self.cache_path.read_text(encoding="utf-8")))
        self.session = None
        # Wikimedia's User-Agent policy: a descriptive name plus a way to contact you. Generic
        # library agents ("python-requests/...") are throttled or refused.
        self.ua = ("PetitJournalNEL/1.1 (historical newspaper entity-linking research"
                   + (f"; {contact}" if contact else "") + ")")
        self.calls = 0
        self.blocked = None      # reason, once lookups have been given up for this run
        self.fail_streak = 0
        self.since_save = 0

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def _give_up(self, reason):
        if not self.blocked:
            self.blocked = reason
            print(f"\n    Lookups stopped: {reason}\n"
                  "    Everything looked up so far is cached; the rest is marked 'lookup pending'.\n"
                  "    Re-run the same command later (or with --contact <your e-mail>) to finish.")

    def get(self, url, params):
        if self.offline or self.blocked:
            return None
        import requests
        if self.session is None:
            self.session = requests.Session()
            self.session.headers["User-Agent"] = self.ua
        params = {**params, "format": "json", "maxlag": 5}
        detail = ""
        for attempt in range(4):
            try:
                time.sleep(0.25)
                r = self.session.get(url, params=params, timeout=60)
                self.calls += 1
                body = r.text[:300].replace("\n", " ")
                if r.status_code == 403:
                    self._give_up(f"HTTP 403 from {url.split('/')[2]} -- {body}")
                    return None
                if r.status_code in (429, 500, 502, 503, 504) or '"code":"maxlag"' in body:
                    wait = r.headers.get("Retry-After", "")
                    wait = int(wait) if wait.isdigit() else 5 * 2 ** attempt
                    detail = f"HTTP {r.status_code} -- {body}"
                    time.sleep(min(wait, 120))
                    continue
                r.raise_for_status()
                data = r.json()
                self.fail_streak = 0
                self.since_save += 1
                if self.since_save >= 10:
                    self.save()
                    self.since_save = 0
                return data
            except (requests.RequestException, ValueError) as exc:
                detail = f"{type(exc).__name__}: {exc}"
                time.sleep(5 * 2 ** attempt)
        self.fail_streak += 1
        print(f"    (lookup failed: {detail[:300]})")
        if self.fail_streak >= 3:
            self._give_up(f"3 lookups in a row failed; last error: {detail[:300]}")
        return None

    def titles_to_qids(self, pairs):
        """pairs: iterable of (title, lang). Fills the cache {lang|title: qid or None}."""
        todo = defaultdict(set)
        for title, lang in pairs:
            if title and f"{lang}|{title}" not in self.cache["titles"]:
                todo[lang].add(title)
        n_batches = sum((len(t) + 49) // 50 for t in todo.values())
        done = 0
        for lang, titles in todo.items():
            titles = sorted(titles)
            for i in range(0, len(titles), 50):
                if self.blocked:
                    return
                batch = titles[i:i + 50]
                done += 1
                print(f"    Wikipedia lookups: batch {done}/{n_batches}", end="\r")
                data = self.get(f"https://{lang}.wikipedia.org/w/api.php",
                                {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                                 "redirects": 1, "titles": "|".join(batch), "formatversion": 2})
                if data is None:
                    continue  # not cached: retried on the next run
                q = data.get("query", {})
                step = {}
                for key in ("normalized", "redirects"):
                    for x in q.get(key, []):
                        step[x["from"]] = x["to"]
                page_qid = {p["title"]: p.get("pageprops", {}).get("wikibase_item")
                            for p in q.get("pages", []) if not p.get("missing") and not p.get("invalid")}
                for t in batch:
                    cur, seen = t, set()
                    while cur in step and cur not in seen:
                        seen.add(cur)
                        cur = step[cur]
                    self.cache["titles"][f"{lang}|{t}"] = page_qid.get(cur)

    def search(self, title, lang):
        key = f"{lang}|{title}"
        if key not in self.cache["search"]:
            data = self.get("https://www.wikidata.org/w/api.php",
                            {"action": "wbsearchentities", "search": title, "language": lang, "limit": 1,
                             "uselang": lang})
            if data is None:
                return None
            hits = data.get("search", [])
            self.cache["search"][key] = hits[0]["id"] if hits else None
        return self.cache["search"][key]

    def entity_info(self, qids):
        todo = sorted({q for q in qids if q and q not in self.cache["entities"]})
        for i in range(0, len(todo), 25):
            batch = todo[i:i + 25]
            data = self.get("https://www.wikidata.org/w/api.php",
                            {"action": "wbgetentities", "ids": "|".join(batch),
                             "props": "labels|descriptions|claims", "languages": "fr|en"})
            if data is None:
                continue
            for q, ent in data.get("entities", {}).items():
                claims = ent.get("claims", {})

                def ids(p):
                    out = []
                    for c in claims.get(p, []):
                        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                        if isinstance(v, dict) and "id" in v:
                            out.append(v["id"])
                    return out

                def year(p):
                    for c in claims.get(p, []):
                        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                        m = re.match(r"^([+-]\d+)-", v.get("time", "")) if isinstance(v, dict) else None
                        if m:
                            return int(m.group(1))
                    return None

                lab = ent.get("labels", {})
                desc = ent.get("descriptions", {})
                self.cache["entities"][q] = {
                    "label": (lab.get("fr") or lab.get("en") or {}).get("value", ""),
                    "description": (desc.get("fr") or desc.get("en") or {}).get("value", ""),
                    "p31": ids("P31"), "coords": "P625" in claims,
                    "born": year("P569"), "died": year("P570"),
                    "founded": year("P571"), "dissolved": year("P576")}
        return {q: self.cache["entities"].get(q) for q in qids if q}


# Checks 

def type_check(info, coarse):
    if not info:
        return ""
    human = bool(set(info["p31"]) & HUMAN)
    if coarse == "Person":
        return "ok" if human else "not a human"
    if coarse == "Location":
        return "ok" if info["coords"] else ("is a human" if human else "no coordinates")
    if coarse == "Organization":
        return "is a human" if human else "ok"
    return "-"


def date_check(info, pub_year):
    if not info or not pub_year:
        return ""
    if info.get("born") and info["born"] > pub_year:
        return "born after publication"
    if info.get("founded") and info["founded"] > pub_year:
        return "founded after publication"
    return "ok"


def pub_year(doc):
    m = re.match(r"^(\d{4})-", doc)
    return int(m.group(1)) if m else None


# Main

def main():
    p = argparse.ArgumentParser(description="impresso NEL on the NER outputs + comparison")
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    p.add_argument("--groups", nargs="+", help="Only these sub-folders, e.g. --groups HRG")
    p.add_argument("--limit", type=int, help="Only the first N documents per sub-folder (quick test)")
    p.add_argument("--all-docs", action="store_true",
                   help="Link every document a method has (default: only documents every selected "
                        "system method has output for, so all are compared on the same pages)")
    p.add_argument("--rawtext-dir", default=str(RAWTEXT_DIR))
    p.add_argument("--gold-dir", default=str(GOLD_DIR))
    p.add_argument("--out-dir", default=str(NEL_DIR / "nel_impresso"))
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--beams", type=int, default=5, help="Beam size = candidates kept (default 5)")
    p.add_argument("--context-words", type=int, default=15, help="Words of context on each side (default 15)")
    p.add_argument("--max-input-tokens", type=int, default=128)
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Treat links below this model confidence (0-1) as NIL (default 0 = keep all)")
    p.add_argument("--top1-only", action="store_true",
                   help="Only try the top candidate's Wikipedia page (then Wikidata search), not candidates 2-5")
    p.add_argument("--no-search", action="store_true",
                   help="Don't fall back to a Wikidata text search when no generated Wikipedia page exists "
                        "(stricter: fewer, but more reliable, links)")
    p.add_argument("--skip-coarse", nargs="*", default=["Time"],
                   help="Coarse types not sent to the linker (default: Time)")
    p.add_argument("--contact", help="E-mail or URL added to the User-Agent sent to Wikipedia/Wikidata "
                                     "(Wikimedia asks for one; optional)")
    p.add_argument("--compare-only", action="store_true",
                   help="Don't load the model or call the APIs; rebuild the tables from the caches")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(args.methods))
    skip = {s.capitalize() for s in args.skip_coarse}

    # Documents and entities
    texts = {}  # (group, doc) -> path
    for group, path in list_documents(args.rawtext_dir, args.groups):
        texts[(group, doc_key(path.stem))] = path
    if not texts:
        sys.exit(f"No .md files in {args.rawtext_dir}")
    groups = sorted({g for g, _ in texts})

    found = {}  # method -> {group: {doc: path}}
    for m in methods:
        if m == GOLD:
            gp = index_folder(Path(args.gold_dir), "_Human")
            idx = {g: dict(gp) for g in groups} if gp else {}
        else:
            folder, suffix = METHODS[m]
            idx = index_method(NER_DIR / folder, suffix)
        idx = {g: {d: path for d, path in v.items() if (g, d) in texts} for g, v in idx.items() if g in groups}
        idx = {g: v for g, v in idx.items() if v}
        if not idx:
            print(f"  {m}: no output found -- left out")
            continue
        found[m] = idx
        print(f"  {m}: " + ", ".join(f"{group_label(g)} {len(v)} file(s)" for g, v in sorted(idx.items())))
    systems = [m for m in found if m != GOLD]
    if not systems and GOLD not in found:
        sys.exit("Nothing to link.")

    plan = {}  # group -> sorted docs
    for g in groups:
        sets = [set(found[m].get(g, {})) for m in systems] or [set(found.get(GOLD, {}).get(g, {}))]
        docs = set.union(*sets) if args.all_docs else set.intersection(*sets)
        docs = sorted(docs)
        if args.limit is not None:
            docs = docs[:args.limit]
        plan[g] = docs
        print(f"  {group_label(g)}: {len(docs)} document(s) to link")

    entities = {}  # (method, group, doc) -> {norm: info}
    for m, idx in found.items():
        for g in groups:
            for d in plan[g]:
                if d in idx.get(g, {}):
                    entities[(m, g, d)] = read_entities(idx[g][d])

    # One linking unit per (group, doc, normalized text), with a canonical input.
    units = {}  # (group, doc, norm) -> {"input", "located", "coarse"}
    order = [GOLD] + systems if GOLD in found else systems
    doc_text = {}
    for g in groups:
        for d in plan[g]:
            doc_text[(g, d)] = read_document(texts[(g, d)])
    for m in order:
        for (mm, g, d), ents in entities.items():
            if mm != m:
                continue
            for n, e in ents.items():
                if (g, d, n) in units:
                    continue
                if e["coarse"] and e["coarse"] <= skip:
                    continue
                span = find_span(doc_text[(g, d)], e["display"], e["start"])
                inp, located = model_input(doc_text[(g, d)], span, e["display"], args.context_words)
                units[(g, d, n)] = {"input": inp, "located": located}
    print(f"{len(units)} distinct entity text(s) to link "
          f"({sum(not u['located'] for u in units.values())} not found in the page text: linked without context)")

    cache_path = out_dir / "_nel_model_cache.json"
    model_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def ckey(inp):
        return f"{args.model}|{args.beams}|{args.max_input_tokens}|{inp}"

    todo = sorted({u["input"] for u in units.values() if ckey(u["input"]) not in model_cache})
    revision = model_cache.get("__revision__")
    if todo and not args.compare_only:
        nel = ImpressoNEL(args.model, resolve_device(args.device), args.beams, args.max_input_tokens)
        revision = nel.revision
        model_cache["__revision__"] = revision
        t0 = time.time()
        try:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                for inp, (cands, confs) in zip(batch, nel.predict(batch)):
                    model_cache[ckey(inp)] = {"candidates": cands, "confidences": confs}
                done = i + len(batch)
                rate = (time.time() - t0) / done
                print(f"  linked {done}/{len(todo)}  (~{rate * (len(todo) - done) / 60:.1f} min left)", end="\r")
                if (i // args.batch_size) % 20 == 0:
                    cache_path.write_text(json.dumps(model_cache, ensure_ascii=False), encoding="utf-8")
        finally:
            cache_path.write_text(json.dumps(model_cache, ensure_ascii=False), encoding="utf-8")
        print()
        if nel.truncated:
            print(f"  Note: {nel.truncated} input(s) were longer than --max-input-tokens and were cut")
    elif todo:
        print(f"  --compare-only: {len(todo)} entity text(s) have no cached link and count as 'not run'")

    # Resolve to Wikidata 
    wiki = Wiki(out_dir / "_wiki_cache.json", args.contact, offline=args.compare_only)
    preds = {k: model_cache.get(ckey(u["input"])) for k, u in units.items()}

    def known(t, lang):
        return f"{lang}|{t}" in wiki.cache["titles"]

    # Pass 1: every unit's top candidate. Pass 2: candidates 2-5, only where the top one has no page.
    top = {split_prediction(pr["candidates"][0]) for pr in preds.values() if pr}
    if not args.compare_only:
        print(f"Looking up {len(top)} Wikipedia page(s) (top candidates) ...")
    try:
        wiki.titles_to_qids(top)
        if not args.top1_only:
            more = set()
            for pr in preds.values():
                if pr and known(*split_prediction(pr["candidates"][0])) \
                        and not wiki.cache["titles"].get("{1}|{0}".format(*split_prediction(pr["candidates"][0]))):
                    more |= {split_prediction(c) for c in pr["candidates"][1:]}
            more = {x for x in more if not known(*x)}
            if more and not args.compare_only and not wiki.blocked:
                print(f"\nLooking up {len(more)} more page(s) for entities whose top candidate has no page ...")
            wiki.titles_to_qids(more)
        print()
        links = {}  # unit -> dict
        for k, pr in preds.items():
            if not pr:
                links[k] = {"status": "not run"}
                continue
            res = {"status": "NIL", "qid": "", "prediction": pr["candidates"][0],
                   "candidates": " || ".join(pr["candidates"]), "rank": "", "resolved_by": "",
                   "confidence": pr["confidences"][0]}
            tries = pr["candidates"][:1] if args.top1_only else pr["candidates"]
            pending = False
            for r, c in enumerate(tries, 1):
                t, lang = split_prediction(c)
                if not known(t, lang):
                    pending = True  # lookup failed or not done yet: don't call it NIL
                    break
                q = wiki.cache["titles"].get(f"{lang}|{t}")
                if q:
                    res.update(qid=q, prediction=c, rank=r, resolved_by="wikipedia",
                               confidence=pr["confidences"][r - 1])
                    break
            if not res["qid"] and not pending and not args.no_search:
                t, lang = split_prediction(pr["candidates"][0])
                if t:
                    q = wiki.search(t, lang)
                    if q:
                        res.update(qid=q, rank=1, resolved_by="wikidata search")
                    elif f"{lang}|{t}" not in wiki.cache["search"]:
                        pending = True
            if res["qid"] and res["confidence"] < args.min_confidence:
                res.update(qid="", resolved_by=f"below --min-confidence {args.min_confidence}")
            if res["qid"]:
                res["status"] = "linked"
            elif pending:
                res["status"] = "lookup pending"
            links[k] = res
        info = wiki.entity_info({x.get("qid") for x in links.values() if x.get("qid")})
    finally:
        wiki.save()
    if wiki.calls:
        print(f"  {wiki.calls} API call(s) made")

    # Per-method rows 
    rows = []
    for (m, g, d), ents in sorted(entities.items(), key=lambda x: (order.index(x[0][0]), x[0][1], x[0][2])):
        for n, e in sorted(ents.items()):
            coarse = "|".join(sorted(e["coarse"]))
            c1 = sorted(e["coarse"])[0] if e["coarse"] else ""
            base = {"method": m, "group": g, "document": d, "entity": e["display"], "norm": n,
                    "fine_type": "|".join(sorted(e["fine"])), "coarse_type": coarse,
                    "human_qid": e["human_qid"] or ""}
            if (g, d, n) not in units:
                rows.append({**base, "status": f"skipped ({coarse})"})
                continue
            lk = links[(g, d, n)]
            it = info.get(lk.get("qid")) if lk.get("qid") else None
            rows.append({**base, "status": lk["status"], "qid": lk.get("qid", ""),
                         "label": it["label"] if it else "", "description": it["description"] if it else "",
                         "wikipedia": lk.get("prediction", ""), "candidate_rank": lk.get("rank", ""),
                         "resolved_by": lk.get("resolved_by", ""), "confidence": lk.get("confidence", ""),
                         "type_check": type_check(it, c1), "date_check": date_check(it, pub_year(d)),
                         "born": (it or {}).get("born") or "", "died": (it or {}).get("died") or "",
                         "located_in_text": "yes" if units[(g, d, n)]["located"] else "no",
                         "candidates": lk.get("candidates", ""), "model_input": units[(g, d, n)]["input"]})
    fields = ["method", "group", "document", "entity", "fine_type", "coarse_type", "status", "qid", "label",
              "description", "wikipedia", "candidate_rank", "resolved_by", "confidence", "type_check",
              "date_check", "born", "died", "human_qid", "located_in_text", "candidates", "model_input"]
    with open(out_dir / "entity_links.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    present = [m for m in order if any(r["method"] == m for r in rows)]
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["group"])].append(r)

    # Data matrices
    ent_rows, qid_rows = {}, {}
    for r in rows:
        k = (r["group"], r["document"], r["norm"])
        er = ent_rows.setdefault(k, {"group": r["group"], "document": r["document"], "entity": r["entity"],
                                     "coarse_type": r["coarse_type"]})
        er[r["method"]] = r.get("qid") or ("NIL" if r["status"] == "NIL" else r["status"])
        if r.get("qid"):
            er["label"] = r["label"]
            qk = (r["group"], r["document"], r["qid"])
            qr = qid_rows.setdefault(qk, {"group": r["group"], "document": r["document"], "qid": r["qid"],
                                          "label": r["label"], "description": r["description"]})
            qr[r["method"]] = " | ".join(sorted(set(filter(None, (qr.get(r["method"], "").split(" | ")
                                                                  + [r["entity"]])))))
    for qr in qid_rows.values():
        qr["methods"] = sum(1 for m in present if qr.get(m))
    for er in ent_rows.values():
        er["methods_finding_it"] = sum(1 for m in present if er.get(m))
    with open(out_dir / "entity_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "document", "entity", "coarse_type", "label", *present,
                                          "methods_finding_it"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(ent_rows.values(), key=lambda x: (x["group"], x["document"], x["entity"].casefold())))
    with open(out_dir / "qid_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "document", "qid", "label", "description", *present,
                                          "methods"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(qid_rows.values(), key=lambda x: (x["group"], x["document"], -x["methods"],
                                                             x["label"].casefold())))

    # Summary 
    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "    -"

    width = max(len(m) for m in present) + 2
    section_no = [0]

    def section(title): 
        section_no[0] += 1
        return f"{section_no[0]}. {title}"

    L = ["impresso NEL on the NER outputs", "",
         f"Model {args.model}" + (f" @ {revision}" if revision else "") + f"; beams {args.beams}, "
         f"context {args.context_words} words each side, min confidence {args.min_confidence}, "
         f"{'top candidate only' if args.top1_only else 'candidates tried in rank order'}",
         "Documents: " + ", ".join(f"{group_label(g)} {len(plan[g])}" for g in groups)
         + ("" if args.all_docs else ""),
         "Counts are distinct entity texts per document, summed over documents. "
         f"Skipped (not linked): {', '.join(sorted(skip)) or 'none'}.", ""]

    pend = sum(1 for x in links.values() if x["status"] in ("lookup pending", "not run"))
    if pend:
        L += [f"!! {pend} of {len(links)} entity text(s) are not finished ('lookup pending' / 'not run') and are",
              "!! left out of every count below. Re-run the same command to complete them.", ""]
    L += [section("How many entities could be linked to Wikidata?"), ""]
    head = "".ljust(width) + "".join(h.rjust(11) for h in ["entities", "linked", "NIL", "% linked"]) \
        + "".join(f"{c[:4]} %".rjust(10) for c in LINKABLE) + "".join(
            h.rjust(13) for h in ["via search", "distinct QID", "mean conf", "type issue", "date issue"])
    for g in groups:
        L += [f"  {group_label(g)}", "  " + head]
        for m in present:
            rs = [r for r in by[(m, g)] if r["status"] in ("linked", "NIL")]
            linked = [r for r in rs if r["status"] == "linked"]
            line = m.ljust(width) + "".join(str(x).rjust(11) for x in
                                            [len(rs), len(linked), sum(r["status"] == "NIL" for r in rs)])
            line += pct(len(linked), len(rs)).rjust(11)
            for c in LINKABLE:
                sub = [r for r in rs if c in r["coarse_type"].split("|")]
                line += pct(sum(r["status"] == "linked" for r in sub), len(sub)).rjust(10)
            confs = [float(r["confidence"]) for r in linked if r["confidence"] != ""]
            line += str(sum(r["resolved_by"] == "wikidata search" for r in linked)).rjust(13)
            line += str(len({(r["document"], r["qid"]) for r in linked})).rjust(13)
            line += (f"{sum(confs) / len(confs):.3f}" if confs else "-").rjust(13)
            line += str(sum(r["type_check"] not in ("ok", "-", "") for r in linked)).rjust(13)
            line += str(sum(r["date_check"] not in ("ok", "") for r in linked)).rjust(13)
            L.append("  " + line)
        L.append("")
    L += ["  via search = no generated Wikipedia page existed; the item came from a Wikidata text search on",
          "  the top title (less reliable; --no-search turns it off).",
          "  type issue = Person not linked to a human / Location without coordinates / Organization linked to",
          "  a human; date issue = person born or organization founded after the newspaper's date. Both are",
          "  likely wrong links (see type_check / date_check in entity_links.csv).", ""]

    # Wikidata items reached, per document
    reach = defaultdict(set)  # (method, group, doc) -> qids
    for r in rows:
        if r.get("qid"):
            reach[(r["method"], r["group"], r["document"])].add(r["qid"])
    sys_present = [m for m in present if m != GOLD]

    if GOLD in present:
        gold_docs = {(g, d) for (m, g, d) in entities if m == GOLD}
        L += [section("Do the methods reach the same Wikidata items as the gold entities? (gold documents only)"),
              ""]
        head2 = "".ljust(width) + "".join(h.rjust(11) for h in ["gold QIDs", "found", "extra", "missed",
                                                                 "precision", "recall", "F1"])
        for g in groups:
            L += [f"  {group_label(g)}", "  " + head2]
            for m in sys_present:
                tp = fp = fn = 0
                for (gg, d) in gold_docs:
                    if gg != g or (m, g, d) not in entities:
                        continue
                    gq, mq = reach[(GOLD, g, d)], reach[(m, g, d)]
                    tp, fp, fn = tp + len(gq & mq), fp + len(mq - gq), fn + len(gq - mq)
                pr = tp / (tp + fp) if tp + fp else None
                rc = tp / (tp + fn) if tp + fn else None
                f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
                L.append("  " + m.ljust(width) + "".join(str(x).rjust(11) for x in [tp + fn, tp, fp, fn])
                         + "".join((f"{x:.3f}" if x is not None else "-").rjust(11) for x in [pr, rc, f1]))
            L.append("")

        human = [r for r in rows if r["method"] == GOLD and r["human_qid"] and r["status"] in ("linked", "NIL")]
        if human:
            L += [section("Linking accuracy against the human-checked links (gold 'wikidata' column)"), ""]
            for g in groups:
                hs = [r for r in human if r["group"] == g]
                if not hs:
                    continue
                inkb = [r for r in hs if r["human_qid"] != "NIL"]
                nil = [r for r in hs if r["human_qid"] == "NIL"]
                ok = [r for r in hs if (r["qid"] or "NIL") == r["human_qid"]]
                L.append(f"  {group_label(g)}  gold entities with a human link: {len(hs)}; model correct "
                         f"{len(ok)} ({pct(len(ok), len(hs)).strip()})")
                L.append(f"        in Wikidata: {len(inkb)}, linked to the right item "
                         f"{sum(r['qid'] == r['human_qid'] for r in inkb)} ({pct(sum(r['qid'] == r['human_qid'] for r in inkb), len(inkb)).strip()}), "
                         f"wrong item {sum(bool(r['qid']) and r['qid'] != r['human_qid'] for r in inkb)}, "
                         f"left NIL {sum(not r['qid'] for r in inkb)}")
                L.append(f"        not in Wikidata (NIL): {len(nil)}, correctly NIL {sum(not r['qid'] for r in nil)}, "
                         f"wrongly linked {sum(bool(r['qid']) for r in nil)}")
                for c in LINKABLE:
                    sub = [r for r in hs if c in r["coarse_type"].split("|")]
                    if sub:
                        good = sum((r["qid"] or "NIL") == r["human_qid"] for r in sub)
                        L.append(f"        {c:<13} {good}/{len(sub)} correct")
            L += ["", "   End-to-end (NER + NEL, as in HIPE): each method's Wikidata items vs the human links", ""]
            human_q = defaultdict(set)
            for r in human:
                if r["human_qid"] != "NIL":
                    human_q[(r["group"], r["document"])].add(r["human_qid"])
            for g in groups:
                L += [f"  {group_label(g)}", "  " + "".ljust(width) + "".join(
                    h.rjust(11) for h in ["right", "wrong", "missed", "precision", "recall", "F1"])]
                for m in present:
                    tp = fp = fn = 0
                    for (gg, d), hq in human_q.items():
                        if gg != g or (m, g, d) not in entities:
                            continue
                        mq = reach[(m, g, d)]
                        tp, fp, fn = tp + len(hq & mq), fp + len(mq - hq), fn + len(hq - mq)
                    pr = tp / (tp + fp) if tp + fp else None
                    rc = tp / (tp + fn) if tp + fn else None
                    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
                    L.append("  " + m.ljust(width) + "".join(str(x).rjust(11) for x in [tp, fp, fn])
                             + "".join((f"{x:.3f}" if x is not None else "-").rjust(11) for x in [pr, rc, f1]))
                L.append("")

    if len(sys_present) > 1:
        L += [section("Which Wikidata items does each method reach that the others don't?"), ""]
        for g in groups:
            docs = [d for d in plan[g] if all((m, g, d) in entities for m in sys_present)]
            only, all_n, union_n = Counter(), 0, 0
            for d in docs:
                sets = {m: reach[(m, g, d)] for m in sys_present}
                union = set().union(*sets.values())
                union_n += len(union)
                all_n += len(set.intersection(*sets.values()))
                for m in sys_present:
                    others = set().union(*(sets[o] for o in sys_present if o != m))
                    only[m] += len(sets[m] - others)
            L.append(f"  {group_label(g)}  {union_n} item(s) reached by at least one method; {all_n} by all "
                     f"{len(sys_present)} ({pct(all_n, union_n).strip()})")
            L.append("        only by: " + ", ".join(f"{m} {only[m]}" for m in sys_present))
        L += ["  (qid_matrix.csv shows, for every item, which methods reached it and through which surface;",
              "   entity_matrix.csv shows, for every entity text, which methods found it and what it linked to)", ""]

    # Most frequent NIL entities and unlinked by type
    nils, shown = Counter(), {}
    for r in rows:
        if r["status"] == "NIL":
            nils[r["norm"]] += 1
            shown.setdefault(r["norm"], (r["entity"], r["coarse_type"]))
    if nils:
        L += [section("Entities most often left NIL"), ""]
        for nm, n in nils.most_common(25):
            L.append(f"  {n:4d}  {shown[nm][0]}  ({shown[nm][1]})")
        L.append("")

    (out_dir / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "_nel_settings.json").write_text(json.dumps({
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "model": args.model,
        "model_revision": revision, "beams": args.beams, "context_words": args.context_words,
        "max_input_tokens": args.max_input_tokens, "min_confidence": args.min_confidence,
        "top1_only": args.top1_only, "no_search": args.no_search, "skip_coarse": sorted(skip), "methods": present,
        "documents": {g: plan[g] for g in groups}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"Tables in {out_dir}")


if __name__ == "__main__":
    main()
