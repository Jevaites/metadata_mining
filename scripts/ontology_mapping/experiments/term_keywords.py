#!/usr/bin/env python3
"""
Idea B: describe every ontology term the way the samples are described. An LLM imagines a few
microbial samples that would be annotated with the term and writes, for each, the *same fields the
production prompt extracts from sample metadata* (5-8 keywords in curly brackets + a sub-biome).
Embedding those gives term vectors in the "genre" of the sample keyword embeddings, instead of a
bare term name. For terms with Metalog training samples we already have the real thing (their
centroid, used by `prototype`), so this matters for zero-shot mapping and for unseen terms.

Steps (run the API steps on your Mac; nothing here is sent anywhere by the other steps):
  generate  --dry_run    token and cost estimate, no API call
  generate  --max_terms 50   pilot: real calls for 50 terms, prints the measured cost per term
  generate               all terms, synchronous, resumable (raw responses in --raw)
  batch_submit [--dry_run]   the same through the Batch API (50 % cheaper, results within 24 h)
  batch_collect              check the batches, collect the finished ones (rerun until all collected)
  build                  raw responses -> variants TSV for term_text_variants.py:
                           llm_kw                 the imagined keyword lists (average of the examples)
                           llm_kw_sb              keyword block + sub-biome block, like the samples
                           label_llm_kw_parents   label + imagined keywords + parent labels
Then embed and evaluate with term_text_variants.py (see experiments/README.md).

Model = the production sample keywords: gpt-5.1, chat completions, no reasoning effort set (gpt-5.1's
default is none), temperature 1.0, top_p 0.75. Two deliberate differences from production:
  * no frequency / presence penalties by default (--sampling production restores 0.25 / 1.5). With
    several terms per request the JSON repeats the same keys dozens of times; the penalties then push
    the model to mangle keys ("--biome", invented "__comment__" keys), drop terms, or run away into
    whitespace until max_completion_tokens (seen in the first pilot: 0/10 terms, 6,000 tokens).
  * Structured Outputs (a strict JSON schema whose required keys are exactly the request's term ids),
    so every response has one entry per term with exactly the two fields.
Prompt: experiments/term_keywords_prompt.txt (the production field definitions, rephrased for a term).

cd scripts/ontology_mapping; X=~/MicrobeAtlasProject/ontology_mapping/experiments/term_text
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl --dry_run
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl --max_terms 50
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl
python3 experiments/term_keywords.py build --raw $X/term_keywords_raw.jsonl --output $X/llm_variants.tsv.gz

Batch API instead of the synchronous `generate` (terms already in --raw are skipped):
python3 experiments/term_keywords.py batch_submit --raw $X/term_keywords_raw_v2.jsonl --dry_run
python3 experiments/term_keywords.py batch_submit --raw $X/term_keywords_raw_v2.jsonl
python3 experiments/term_keywords.py batch_collect --raw $X/term_keywords_raw_v2.jsonl   # rerun until all collected
  State in <raw>.batch.json (batch ids, which terms each request holds, collected or not). A second
  batch_submit (or generate) refuses to run while a batch is uncollected, so nothing is paid twice.
  Failed requests, unparsable responses and batches that fail / expire / are cancelled simply leave
  their terms missing: another batch_submit (or generate) retries only those.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # scripts/ontology_mapping
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # scripts/
from common import load_terms, path  # noqa: E402
from _setup import DEFAULTS  # noqa: E402

SAMPLINGS = {"no_penalties": {"temperature": 1.0, "top_p": 0.75},
             "production": {"temperature": 1.0, "top_p": 0.75, "frequency_penalty": 0.25, "presence_penalty": 1.5},
             "none": {}}
SAMPLING_KEYS = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
PROMPT = os.path.join(HERE, "term_keywords_prompt.txt")
SHOWCASE = ["ENVO_00002003", "ENVO_2100002", "UBERON_0001988", "ENVO_01001002", "ENVO_00002261"]  # printed by build


# ----------------------------------------------------------------------------- requests
def term_block(t, label_of):
    """The user-message text for one term."""
    parents = [label_of[p] for p in t.parents.split("||") if p in label_of]
    lines = [f"term-id: {t.term_id}", f"label: {t.label}"]
    if t.synonyms:
        lines.append("synonyms: " + "; ".join(s for s in t.synonyms.split("||") if s))
    if t.definition:
        lines.append(f"definition: {t.definition}")
    if parents:
        lines.append("parent terms: " + "; ".join(parents))
    return "\n".join(lines)


def requests_for(terms, args):
    """-> list of (request_id, [term_ids], chat-completions body) for terms not done yet."""
    system = open(PROMPT).read().replace("{n_examples}", str(args.n_examples))
    label_of = dict(zip(terms["term_id"], terms["label"]))
    done = done_terms(args.raw)
    todo = terms[~terms["term_id"].isin(done)]
    if args.term_set == "metalog":  # only the terms Metalog uses as labels (cheap; see the caveat in --help)
        from common import SLOTS, read_tsv
        labels = set(read_tsv(args.samples)[SLOTS].to_numpy().ravel()) - {""}
        todo = todo[todo["term_id"].isin(labels | set(SHOWCASE))]
    # showcase terms first, then a fixed random order, so a --max_terms pilot is a random sample
    rank = {t: i for i, t in enumerate(SHOWCASE)}
    todo = todo.assign(_o=[rank.get(t, len(rank) + r) for t, r in
                           zip(todo["term_id"], np.random.default_rng(0).permutation(len(todo)))])
    todo = todo.sort_values("_o")
    if args.max_terms:
        todo = todo.head(max(0, args.max_terms - len(done)))
    out = []
    for start in range(0, len(todo), args.terms_per_request):
        chunk = todo.iloc[start:start + args.terms_per_request]
        user = "\n\n".join(term_block(t, label_of) for t in chunk.itertuples())
        body = {"model": args.model, "response_format": schema(list(chunk["term_id"])),
                "max_completion_tokens": args.max_completion_tokens or 400 * len(chunk),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body.update(SAMPLINGS[sampling_for(args)])
        if args.reasoning_effort:
            body["reasoning_effort"] = args.reasoning_effort
        out.append((f"terms-{chunk['term_id'].iloc[0]}-{len(chunk)}", list(chunk["term_id"]), body))
    return out, len(done)


PRICES = {"gpt-5.1": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.4)}  # $ / 1M in, out
REJECTS_SAMPLING = ("gpt-5-mini", "gpt-5-nano")  # they reject top_p / penalties (include-list-and-model-choice.md)


def sampling_for(args):
    if args.sampling == "auto":
        return "none" if args.model.startswith(REJECTS_SAMPLING) else "no_penalties"
    return args.sampling


def prices(args):
    base = PRICES.get(args.model, PRICES["gpt-5.1"])
    return args.price_in if args.price_in is not None else base[0], args.price_out if args.price_out is not None else base[1]


def schema(term_ids):
    """Strict Structured Outputs schema: exactly these term ids, each a list of imagined samples."""
    sample = {"type": "object", "additionalProperties": False, "required": ["keywords", "sub-biome"],
              "properties": {"keywords": {"type": "string"}, "sub-biome": {"type": "string"}}}
    return {"type": "json_schema", "json_schema": {"name": "term_samples", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": term_ids,
        "properties": {t: {"type": "array", "items": sample} for t in term_ids}}}}


def done_terms(raw):
    done = set()
    if raw and os.path.exists(path(raw)):
        for line in open(path(raw)):
            record = json.loads(line)
            done.update(t for t, s in record["parsed"].items() if s)
    return done


def parse(content, term_ids, n_examples):
    """{term_id: [(keywords, sub_biome), ...]} from one response (missing/garbled terms are left out,
    so a rerun retries them)."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    if "terms" not in data:  # structured-output format: {term_id: [samples]}
        data = {"terms": [{"term-id": k, "samples": v} for k, v in data.items() if isinstance(v, list)]}
    entries = data.get("terms", [])
    by_id = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        tid = str(e.get("term-id", "")).strip().replace(":", "_")
        if tid not in term_ids and i < len(term_ids):
            tid = term_ids[i]  # fall back to the input order
        samples = [(str(s.get("keywords", "")).strip(), str(s.get("sub-biome", "")).strip())
                   for s in e.get("samples", []) if isinstance(s, dict) and s.get("keywords")]
        if tid in term_ids and samples:
            by_id[tid] = samples[:n_examples]
    return by_id


# ----------------------------------------------------------------------------- cost
def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def estimate(reqs, args):
    n_in = sum(count_tokens(m["content"]) for _, _, b in reqs for m in b["messages"])
    n_terms = sum(len(ids) for _, ids, _ in reqs)
    n_out = n_terms * (args.n_examples * args.out_tokens_per_example + 20) + len(reqs) * 10
    price_in, price_out = prices(args)
    cost = n_in / 1e6 * price_in + n_out / 1e6 * price_out
    print(f"{len(reqs)} requests for {n_terms} terms: ~{n_in:,} input tokens, ~{n_out:,} output tokens "
          f"(estimated at {args.out_tokens_per_example} per imagined sample; the pilot measures it)")
    if args.reasoning_effort:
        print(f"  NOT included: reasoning tokens (reasoning_effort={args.reasoning_effort}); run a pilot to measure them")
    print(f"estimated cost ${cost:.2f} at ${price_in}/${price_out} per 1M in/out "
          f"(Batch API about half: ${cost / 2:.2f})")


def measured(raw, args):
    rows = [json.loads(line) for line in open(path(raw))] if os.path.exists(path(raw)) else []
    rows = [r for r in rows if r.get("usage")]
    if not rows:
        return
    price_in, price_out = prices(args)
    n_terms = sum(sum(1 for s in r["parsed"].values() if s) for r in rows)
    n_in = sum(r["usage"]["prompt_tokens"] for r in rows)
    n_out = sum(r["usage"]["completion_tokens"] for r in rows)
    cost = sum((r["usage"]["prompt_tokens"] / 1e6 * price_in + r["usage"]["completion_tokens"] / 1e6 * price_out)
               * (0.5 if r.get("api") == "batch" else 1.0) for r in rows)
    total = len(load_terms(args.ontology_terms))
    extrapolated = f" -> about ${cost / n_terms * total:.2f} for all {total} terms at this mix" if n_terms else ""
    print(f"measured so far: {n_terms} terms parsed, {n_in:,} in / {n_out:,} out tokens = ${cost:.2f} "
          f"(Batch API requests counted at half price){extrapolated}")


# ----------------------------------------------------------------------------- steps
def client_for(args):
    from openai import OpenAI
    return OpenAI(api_key=open(path(args.api_key_path)).read().strip(), base_url=args.base_url, max_retries=8)


def generate(args):
    terms = load_terms(args.ontology_terms)
    if pending_batches(args) and not args.dry_run:
        sys.exit("batch(es) still pending for this --raw: run batch_collect first, or their terms are paid twice")
    reqs, n_done = requests_for(terms, args)
    print(f"{len(terms)} terms, {n_done} already done")
    if args.dry_run:
        estimate(reqs, args)
        return
    if not reqs:
        measured(args.raw, args)
        return
    client, lock = client_for(args), threading.Lock()
    os.makedirs(os.path.dirname(path(args.raw)) or ".", exist_ok=True)
    drop = set()  # sampling parameters the model rejects (dropped once for all requests)

    def call(req):
        rid, ids, body = req
        for _ in range(3):
            try:
                r = client.chat.completions.create(**{k: v for k, v in body.items() if k not in drop})
                break
            except Exception as e:  # e.g. "Unsupported parameter: 'top_p'"
                bad = [k for k in SAMPLING_KEYS if k in str(e) and k not in drop]
                if not bad:
                    raise
                with lock:
                    drop.update(bad)
                print(f"  model rejected {bad}: retrying without (recorded in the raw file)", flush=True)
        parsed = parse(r.choices[0].message.content, ids, args.n_examples)
        record = {"request": rid, "term_ids": ids, "model": r.model, "api": "sync", "sampling": sampling_for(args),
                  "reasoning_effort": args.reasoning_effort,
                  "dropped_params": sorted(drop),
                  "usage": {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
                  "parsed": {t: parsed.get(t, []) for t in ids}, "content": r.choices[0].message.content}
        with lock, open(path(args.raw), "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return len(parsed), len(ids)

    t0, ok, n = time.time(), 0, 0
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, (good, total) in enumerate(pool.map(call, reqs), 1):
            ok, n = ok + good, n + total
            if i % 20 == 0 or i == len(reqs):
                print(f"  {i}/{len(reqs)} requests, {ok}/{n} terms parsed ({time.time() - t0:.0f}s)", flush=True)
    measured(args.raw, args)
    if ok < n:
        print(f"{n - ok} terms could not be parsed; rerun the same command to retry them")


def batch_state(args):
    """<raw>.batch.json: {"batches": [{"batch_id", "requests": {custom_id: [term_ids]}, "collected"}]}"""
    f = path(args.raw) + ".batch.json"
    if not os.path.exists(f):
        return f, {"batches": []}
    state = json.load(open(f))
    if "batch_id" in state:  # single-batch format
        state = {"batches": [{"batch_id": state["batch_id"], "requests": state["requests"], "collected": False}]}
    return f, state


def pending_batches(args):
    return [b for b in batch_state(args)[1]["batches"] if not b["collected"]]


def batch_submit(args):
    terms = load_terms(args.ontology_terms)
    pending = pending_batches(args)
    if pending:
        sys.exit(f"{len(pending)} batch(es) not collected yet ({', '.join(b['batch_id'] for b in pending)}): run "
                 f"batch_collect first (a failed / expired / cancelled batch counts as collected once seen)")
    reqs, n_done = requests_for(terms, args)
    print(f"{len(terms)} terms, {n_done} already done")
    if not reqs:
        return
    estimate(reqs, args)
    parts = [reqs[i:i + args.batch_requests] for i in range(0, len(reqs), args.batch_requests)]
    print(f"-> {len(parts)} batch(es) of up to {args.batch_requests} requests")
    if args.dry_run:
        return
    client, (state_file, state) = client_for(args), batch_state(args)
    os.makedirs(os.path.dirname(path(args.raw)) or ".", exist_ok=True)
    for k, part in enumerate(parts):
        batch_input = f"{path(args.raw)}.batch_input_{len(state['batches'])}.jsonl"
        with open(batch_input, "w") as handle:
            for rid, _, body in part:
                handle.write(json.dumps({"custom_id": rid, "method": "POST", "url": "/v1/chat/completions",
                                         "body": body}) + "\n")
        upload = client.files.create(file=open(batch_input, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                      completion_window="24h", metadata={"job": "term_keywords"})
        state["batches"].append({"batch_id": batch.id, "requests": {rid: ids for rid, ids, _ in part},
                                 "sampling": sampling_for(args), "reasoning_effort": args.reasoning_effort,
                                 "collected": False})
        json.dump(state, open(state_file, "w"))  # saved after every batch, so an interrupted submit is not lost
        print(f"submitted batch {batch.id} ({len(part)} requests, {os.path.getsize(batch_input) / 1e6:.1f} MB)")
    print("run batch_collect with the same --raw to check on them and collect the results")


def batch_records(client, file_id, b, args):
    """Raw-file records from a batch output (or error) file; failed requests get empty `parsed`."""
    if not file_id:
        return []
    records = []
    for line in client.files.content(file_id).text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = b["requests"].get(r.get("custom_id"))
        if ids is None:
            continue
        response = r.get("response") or {}
        body = response.get("body") or {} if response.get("status_code") == 200 else {}
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        parsed = parse(content, ids, args.n_examples)
        u = body.get("usage") or {}
        records.append({"request": r["custom_id"], "term_ids": ids, "model": body.get("model"), "api": "batch",
                        "batch_id": b["batch_id"], "sampling": b.get("sampling"),
                        "reasoning_effort": b.get("reasoning_effort"), "finish_reason": choice.get("finish_reason"),
                        "error": r.get("error") or (None if body else response.get("body")),
                        "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                                  "completion_tokens": u.get("completion_tokens", 0)},
                        "parsed": {t: parsed.get(t, []) for t in ids}, "content": content})
    return records


def batch_collect(args):
    state_file, state = batch_state(args)
    if not state["batches"]:
        sys.exit(f"no batch recorded for {args.raw}: run batch_submit first")
    client, waiting = client_for(args), 0
    for b in state["batches"]:
        if b["collected"]:
            continue
        batch = client.batches.retrieve(b["batch_id"])
        c = batch.request_counts
        counts = f"{c.completed} done, {c.failed} failed of {c.total}" if c else ""
        print(f"batch {batch.id}: {batch.status} {counts}")
        if batch.status in ("validating", "in_progress", "finalizing", "cancelling"):
            waiting += 1
            continue
        if batch.status == "failed":  # rejected as a whole (e.g. enqueued-token limit): nothing was run
            for e in (batch.errors.data if batch.errors and batch.errors.data else []):
                print(f"   error: {e.code}: {e.message}")
        records = batch_records(client, batch.output_file_id, b, args) + \
            batch_records(client, batch.error_file_id, b, args)
        with open(path(args.raw), "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        ok = sum(1 for r in records for s in r["parsed"].values() if s)
        n = sum(len(ids) for ids in b["requests"].values())
        print(f"   collected {len(records)} responses: {ok}/{n} terms parsed")
        b["collected"] = True
        json.dump(state, open(state_file, "w"))
    measured(args.raw, args)
    if waiting:
        print(f"{waiting} batch(es) still running: rerun batch_collect later")
    else:
        missing = len(load_terms(args.ontology_terms)) - len(done_terms(args.raw))
        print(f"all batches collected; {missing} terms still missing. Retry them with batch_submit (another "
              f"batch) or generate (synchronous), same --raw")


def build(args):
    from embed_subbiomes_keywords import clean_text  # the cleaning applied to the sample texts
    terms = load_terms(args.ontology_terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    examples = {}
    for line in open(path(args.raw)):
        for t, samples in json.loads(line)["parsed"].items():
            if samples:
                examples[t] = samples  # the latest successful response wins
    rows, missing = [], 0
    for t in terms.itertuples():
        parents = [label_of[p] for p in t.parents.split("||") if p in label_of]
        samples = examples.get(t.term_id)
        if not samples:  # no usable response: the plain label stands in, so every term has a vector
            missing += 1
            samples = [(t.label, t.label)]
        for kw, sb in samples:
            kw_text = clean_text(kw, True)
            sb_text = clean_text(sb, False) or kw_text
            x = args.suffix
            rows += [(t.term_id, f"llm_kw{x}", kw_text, "both"),
                     (t.term_id, f"llm_kw_sb{x}", kw_text, "kw"), (t.term_id, f"llm_kw_sb{x}", sb_text, "sb"),
                     (t.term_id, f"label_llm_kw_parents{x}", " ".join([t.label, kw_text] + parents), "both")]
    out = pd.DataFrame(rows, columns=["term_id", "variant", "text", "block"])
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    out.to_csv(path(args.output), sep="\t", index=False)
    print(f"{len(terms) - missing} terms with LLM examples, {missing} fall back to their label")
    print(out.groupby(["variant", "block"])["text"].agg(rows="size", distinct="nunique",
                                                        median_chars=lambda s: int(s.str.len().median())))
    for tid in SHOWCASE:
        if tid in examples:
            print(f"\n{tid} {label_of[tid]}:")
            for kw, sb in examples[tid]:
                print(f"   {kw}   | sub-biome: {sb}")
    print(f"wrote {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    for name in ["generate", "batch_submit", "batch_collect", "build"]:
        s = sub.add_parser(name)
        s.add_argument("--ontology_terms", default=DEFAULTS["ontology_terms"])
        s.add_argument("--raw", required=True, help="JSONL of raw responses (resumable)")
        s.add_argument("--n_examples", type=int, default=3, help="Imagined samples per term")
        s.add_argument("--api_key_path", default="~/MicrobeAtlasProject/my_api_key")
        s.add_argument("--base_url", default=None)
        s.add_argument("--model", default="gpt-5.1", help="e.g. gpt-5.1 (production sample keywords) or gpt-5-mini")
        s.add_argument("--price_in", type=float, default=None, help="$ per 1M input tokens (default: by --model)")
        s.add_argument("--price_out", type=float, default=None, help="$ per 1M output tokens (default: by --model)")
        if name in ("generate", "batch_submit"):
            s.add_argument("--terms_per_request", type=int, default=10)
            s.add_argument("--max_terms", type=int, default=None, help="Stop after this many terms (pilot)")
            s.add_argument("--max_completion_tokens", type=int, default=None, help="Default: 400 per term in the request")
            s.add_argument("--sampling", choices=["auto"] + list(SAMPLINGS), default="auto",
                           help="auto: no_penalties for gpt-5.1, none for gpt-5-mini/nano (they reject them); "
                                "no_penalties: temperature 1.0, top_p 0.75; production: + frequency 0.25, presence 1.5 "
                                "(breaks multi-term JSON, see above); none: send nothing")
            s.add_argument("--reasoning_effort", default=None, choices=["none", "minimal", "low", "medium", "high"],
                           help="not sent by default (gpt-5.1: none; gpt-5-mini: medium). The other chat used low for mini")
            s.add_argument("--out_tokens_per_example", type=int, default=75, help="For the dry-run estimate only")
            s.add_argument("--term_set", choices=["all", "metalog"], default="all",
                           help="metalog: only terms used as Metalog labels (~5 %% of the cost). Other terms then keep "
                                "their plain label, which favours the LLM terms in open-vocabulary retrieval")
            s.add_argument("--samples", default=DEFAULTS["samples"], help="training set (for --term_set metalog)")
        if name == "generate":
            s.add_argument("--workers", type=int, default=8)
        if name in ("generate", "batch_submit"):
            s.add_argument("--dry_run", action="store_true", help="token and cost estimate only, nothing is sent")
        if name == "batch_submit":
            s.add_argument("--batch_requests", type=int, default=2000,
                           help="requests per batch (default: everything in one). If a batch fails with an "
                                "enqueued-token limit, collect it and resubmit with e.g. 300")
        if name == "build":
            s.add_argument("--output", required=True)
            s.add_argument("--suffix", default="", help="appended to the variant names, e.g. _mini, to compare models")
    args = p.parse_args()
    {"generate": generate, "batch_submit": batch_submit, "batch_collect": batch_collect, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
