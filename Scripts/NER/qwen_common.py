"""
Qwen Support Script
Code developed with the assistance of Claude Opus 5/5.5

"""

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ner_mistral_broad import locate, parse_response
from ner_utils import NER_DIR, RAWTEXT_DIR, list_documents, read_document, write_entities


class QwenServer:
    def __init__(self, base_url, model, api_key=None, verify=True, max_tokens=2048, temperature=0.0,
                 system_prompt="", schema=None, thinking=False, timeout=900):
        self.base_url = base_url.rstrip("/")
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.system_prompt, self.schema, self.thinking = system_prompt, schema, thinking
        self.timeout, self.verify = timeout, verify
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.use_schema = schema is not None
        self._local = threading.local()
        self._lock = threading.Lock()
        if not verify:
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                pass

    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
            s.headers.update(self.headers)
            s.verify = self.verify
        return s

    def list_models(self):
        try:
            r = self._session().get(f"{self.base_url}/models", timeout=30, verify=self.verify)
        except requests.RequestException as exc:
            sys.exit(f"Can't reach the server at {self.base_url} ({exc}).\n"
                     "Check the address and port, that vLLM has finished loading (instance logs), and add "
                     "--insecure if it uses Vast's self-signed HTTPS certificate.")
        if r.status_code in (401, 403):
            sys.exit(f"The server refused access ({r.status_code}). Pass the instance's OPEN_BUTTON_TOKEN "
                     "with --api-key (or set the OPEN_BUTTON_TOKEN environment variable).")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def chat(self, user):
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": user}],
                   "chat_template_kwargs": {"enable_thinking": self.thinking}}
        last = None
        for attempt in range(1, 5):
            if self.use_schema:
                payload["response_format"] = self.schema
            else:
                payload.pop("response_format", None)
            try:
                # verify is passed per request: a REQUESTS_CA_BUNDLE environment variable would
                # otherwise silently override the session setting
                r = self._session().post(f"{self.base_url}/chat/completions", json=payload,
                                         timeout=self.timeout, verify=self.verify)
                if r.status_code == 400 and self.use_schema and "response_format" in r.text:
                    with self._lock:
                        if self.use_schema:
                            print("    (server rejected structured output; falling back to plain JSON)")
                        self.use_schema = False
                    continue
                if r.status_code in (401, 403):
                    raise RuntimeError(f"access refused ({r.status_code}) -- check --api-key")
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return msg.get("content") or ""
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = exc
                time.sleep(3 * attempt)
        raise RuntimeError(f"request failed after retries: {last}")


def pick_model(server, requested):
    available = server.list_models()
    if requested:
        if available and requested not in available:
            sys.exit(f"The server serves {', '.join(available)}, not '{requested}'. Use one of those with --model.")
        return requested
    qwen = [m for m in available if "qwen" in m.lower()]
    if len(qwen) == 1 or (qwen and len(available) == len(qwen)):
        return qwen[0]
    if len(available) == 1:
        return available[0]
    sys.exit(f"Several models are served ({', '.join(available) or 'none'}); choose one with --model.")


def make_parser(description, default_out):
    import argparse
    p = argparse.ArgumentParser(description=description)
    g = p.add_argument_group("server")
    g.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM server, e.g. https://<instance ip>:<port>/v1 (default http://localhost:8000/v1)")
    g.add_argument("--api-key", default=os.environ.get("OPEN_BUTTON_TOKEN") or os.environ.get("VLLM_API_KEY"),
                   help="Bearer token: the Vast instance's OPEN_BUTTON_TOKEN (or set that environment variable)")
    g.add_argument("--insecure", action="store_true",
                   help="Accept a self-signed HTTPS certificate (needed for Vast's direct HTTPS port)")
    g.add_argument("--model", help="Model name as the server lists it (default: the Qwen model it serves)")
    g.add_argument("--workers", type=int, default=16,
                   help="Documents processed in parallel (default 16; keep at or below vLLM's --max-num-seqs)")
    g.add_argument("--thinking", action="store_true",
                   help="Allow Qwen's thinking mode (default: off -- much slower, can overrun --max-tokens)")
    g.add_argument("--chunk-chars", type=int, default=2500, help="Characters of text per request (default 2500)")
    g.add_argument("--max-tokens", type=int, default=2048, help="Max answer tokens per request (default 2048)")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default 0, as in the Mistral runs)")
    io = p.add_argument_group("files")
    io.add_argument("--rawtext-dir", default=str(RAWTEXT_DIR))
    io.add_argument("--out-dir", default=str(NER_DIR / default_out))
    io.add_argument("--groups", nargs="+", help="Only these rawtext sub-folders, e.g. --groups HRG")
    io.add_argument("--limit", type=int, help="Only the first N documents of each sub-folder (quick test)")
    io.add_argument("--overwrite", action="store_true", help="Redo documents that already have output")
    return p


def make_server(args, system_prompt, schema):
    server = QwenServer(args.base_url, None, api_key=args.api_key, verify=not args.insecure,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        system_prompt=system_prompt, schema=schema, thinking=args.thinking)
    server.model = pick_model(server, args.model)
    return server


def select_docs(args):
    docs = list_documents(args.rawtext_dir, args.groups)
    if args.limit is not None:
        kept, per_group = [], {}
        for group, path in docs:
            per_group[group] = per_group.get(group, 0) + 1
            if per_group[group] <= args.limit:
                kept.append((group, path))
        docs = kept
    return docs


def run_parallel(args, docs, suffix, fields, process, log_name, log_fields, label, server):
    """process(group, path, text) -> (entities, stats) for one document; raise to
    mark it failed (nothing written, so a re-run retries it). Documents run in
    parallel; finished ones are skipped unless --overwrite."""
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], 0
    for group, path in docs:
        out_dir = out_root / group if group else out_root
        out_path = out_dir / f"{path.stem}_{suffix}.csv"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((group, path, out_dir, out_path))

    print(f"{label} | model {server.model} | thinking {'on' if server.thinking else 'off'} | "
          f"{args.workers} parallel document(s)")
    print(f"{len(docs)} document(s): {skipped} already done, {len(jobs)} to process -> {out_root}")

    log_path = out_root / log_name
    lock = threading.Lock()
    done = failed = 0
    t_start = time.time()

    def work(job):
        group, path, out_dir, out_path = job
        text = read_document(path)
        t0 = time.time()
        entities, stats = process(group, path, text)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_entities(out_path, text, entities, fields)
        stats["seconds"] = round(time.time() - t0, 1)
        return len(entities), stats

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            group, path, _, _ = futures[fut]
            name = f"{group}/{path.stem}" if group else path.stem
            try:
                n, stats = fut.result()
            except Exception as exc:
                failed += 1
                print(f"  [{i}/{len(jobs)}] {name}: FAILED -- {type(exc).__name__}: {exc}")
                continue
            done += 1
            extra = (f"; dropped {stats.get('invalid_type', 0)} invalid type, {stats.get('not_in_text', 0)} "
                     f"not in text" if stats.get("invalid_type") or stats.get("not_in_text") else "")
            print(f"  [{i}/{len(jobs)}] {name}: {n} entities ({stats['seconds']}s){extra}")
            with lock:
                new = not log_path.exists()
                with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow({"group": group, "document": path.stem, "model": server.model, **stats})

    minutes = (time.time() - t_start) / 60
    print(f"\nFinished in {minutes:.1f} min: {done} written, {skipped} skipped (already done), {failed} failed.")
    if failed:
        print("Re-run the same command to retry the failed documents.")
    return failed


def extract_chunks(server, text, chunk_chars, user_builder, allowed, normalise_type, make_entity, chunk_spans):
    """Send each chunk, validate the answer, return (entities, stats). Shared by all three scripts."""
    st = {"chunks": 0, "returned": 0, "invalid_type": 0, "not_in_text": 0, "salvaged_chunks": 0}
    entities = []
    for s, e in chunk_spans(text, chunk_chars):
        chunk = text[s:e]
        st["chunks"] += 1
        parsed, salvaged = parse_response(server.chat(user_builder(chunk, s, e)))
        if parsed is None:
            raise RuntimeError(f"unreadable answer for characters {s}-{e}")
        st["salvaged_chunks"] += int(salvaged)
        st["returned"] += len(parsed)
        for surface, raw_type in parsed:
            t = normalise_type(raw_type)
            if t is None or t not in allowed:
                st["invalid_type"] += 1
                continue
            loc = locate(surface, chunk)  # always checked against the plain text
            if loc is None:
                st["not_in_text"] += 1
                continue
            a, b = s + loc[0], s + loc[1]
            entities.append(make_entity(text[a:b], t, raw_type, a, b))
    return entities, st
