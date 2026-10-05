#!/usr/bin/env python3
"""
Re-run the keyword extraction on a dev set with a new system prompt.

Built to validate openai_system_better_prompt_v2_category_only.txt against the
current output. Three things make the comparison meaningful:

  paired      the same samples that already have keywords, so old and new are
              compared on identical inputs rather than against a corpus average;
  linked      only Metalog-linked samples, so the mapping CV can be run on the
              same dev set afterwards without re-extracting;
  spread      at most --per_study samples per study, so a couple of large
              studies cannot dominate the identity statistics.

One sample per request: the project paper found biome accuracy falls from 78.8%
to 72.2% when ~15 samples are batched into one call.

Metadata comes from sample.info.gz and is cleaned with the same rules as
clean_and_envo_translate.py (drop experiment*/run* lines and missing values,
translate ENVO/UBERON/FOODON/PO codes to labels), so the model sees what the
production pipeline shows it.

Outputs, in the same `sample_id<TAB>value` format as the existing files so every
downstream script works on them unchanged:
    GPT_keywords_v2.txt  GPT_sub_biomes_v2.txt  GPT_biomes_v2.txt  GPT_geo_texts_v2.txt
Resumable: rerun the same command and it picks up where it stopped.

    python3 scripts/embeddings/analyses/extract_keywords_v2.py --n 2000            # see --help for the rest
"""
import argparse, glob, gzip, os, pickle, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

MISSING = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "unspecified",
           "not applicable", "not collected", "not provided"}
CODE = re.compile(r"\[?\b(ENVO|UBERON|FOODON|PO)[:_](\d{7,8})\b\]?", re.IGNORECASE)
FIELDS = ["biomes", "geo_texts", "keywords", "sub_biomes"]     # order of the ___ fields after the id


def clean_record(lines, onto):
    """Same rules as clean_and_envo_translate.clean_file, on an in-memory record."""
    kept = []
    for line in lines:
        key, has_value, value = line.partition("=")
        if key.lower().startswith(("experiment", "run")) or (has_value and value.strip().lower() in MISSING):
            continue
        if has_value:
            value = CODE.sub(lambda m: f"'{onto[k]}'" if (k := f"{m[1].upper()}_{m[2]}") in onto else m[0], value)
            kept.append(f"{key}={value}")
        else:
            kept.append(key)
    return "\n".join(kept)


def choose(root, n, per_study, seed):
    """Metalog-linked samples that already have keywords, spread over studies."""
    study = {}
    with gzip.open(f"{root}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        i_s, i_st = h.index("sample_id"), h.index("study_code")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(i_s, i_st):
                study[f[i_s]] = f[i_st]
    have = set()
    for line in open(f"{root}/sidequest/latest/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if raw.strip() and sid in study:
            have.add(sid)
    by = defaultdict(list)
    for s in sorted(have):
        by[study[s]].append(s)
    rng = random.Random(seed)
    pool = []
    for st in sorted(by):
        v = by[st]
        rng.shuffle(v)
        pool += [(s, st) for s in v[:per_study]]
    rng.shuffle(pool)
    return pool[:n], study


def fetch_metadata(root, wanted, cache):
    """One pass over sample.info.gz for the chosen records; cached for reruns."""
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            got = pickle.load(fh)
        if wanted <= set(got):
            print(f"  metadata cache hit ({len(got):,} records)", flush=True)
            return got
    print(f"  scanning sample.info.gz for {len(wanted):,} records "
          f"(one pass, a few minutes)...", flush=True)
    onto = pickle.load(open(f"{root}/ontologies_dict.pkl", "rb"))
    got, sid, buf, seen, collecting = {}, None, [], 0, False
    with gzip.open(f"{root}/sample.info.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line[:1] == ">":                      # only records we want are buffered:
                if collecting:                       # 99.9% of lines cost one comparison
                    got[sid] = clean_record(buf, onto)
                sid = line[1:].strip()
                collecting = sid in wanted
                buf = []
                seen += 1
                if seen % 500_000 == 0:
                    print(f"    {seen:,} records scanned, {len(got):,} collected", flush=True)
            elif collecting:
                line = line.rstrip("\n")
                if line:
                    buf.append(line)
            if len(got) == len(wanted):
                break
    if collecting and sid not in got:
        got[sid] = clean_record(buf, onto)
    with open(cache, "wb") as fh:
        pickle.dump(got, fh)
    print(f"  collected {len(got):,} of {len(wanted):,} requested", flush=True)
    return got


def parse_json(text):
    """Production JSON object; keys are hyphenated in the prompt, be tolerant."""
    import json
    try:
        d = json.loads(text)
    except Exception:
        return None
    alt = {"biomes": ("biome-label", "biome_label", "biome"),
           "geo_texts": ("geo-location", "geo_location", "geo"),
           "keywords": ("keywords",),
           "sub_biomes": ("sub-biome", "sub_biome", "subbiome")}
    out = {}
    for field, keys in alt.items():
        v = next((d[k] for k in keys if k in d), None)
        if v is None:
            return None
        out[field] = " ".join(str(v).split())
    return out


def parse(text):
    """'id___biome___geo___{kw}___subbiome' -> dict, tolerant of stray whitespace."""
    parts = [p.strip() for p in text.strip().split("___")]
    if len(parts) < 5:
        return None
    sid, rest = parts[0], parts[1:5]
    out = {"sample_id": sid.split()[-1]}
    for name, val in zip(FIELDS, rest):
        out[name] = " ".join(val.split())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--prompt", default=None,
                   help="path to the system prompt; default = the repo's batch v2 prompt")
    p.add_argument("--output_format", choices=["json", "inline"], default="json",
                   help="json reproduces the production run; inline is the ___ format")
    p.add_argument("--api_key", default="my_api_key", help="relative to --root")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--per_study", type=int, default=8)
    p.add_argument("--model", default="gpt-5.1")   # what the production run used
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.75)
    p.add_argument("--frequency_penalty", type=float, default=0.25)
    p.add_argument("--presence_penalty", type=float, default=1.5)
    p.add_argument("--max_tokens", type=int, default=4096,
               help="gpt-5 spends this on reasoning first, so keep it generous")
    p.add_argument("--reasoning_effort", default=None,
               choices=[None, "minimal", "low", "medium", "high"],
               help="gpt-5 only; lower is cheaper and faster for an extraction task")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rpm", type=float, default=450, help="max requests per minute")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="v2", help="output suffix: GPT_keywords_<tag>.txt")
    p.add_argument("--out_dir", default=None, help="default: <root>/sidequest/latest")
    p.add_argument("--dry_run", action="store_true", help="select, clean and print - no API calls")
    p.add_argument("--price_in", type=float, default=None,
                   help="USD per 1M input tokens; enables the cost line")
    p.add_argument("--price_out", type=float, default=None, help="USD per 1M output tokens")
    p.add_argument("--price_cached", type=float, default=None,
                   help="USD per 1M cached input tokens (default: price_in / 10)")
    a = p.parse_args()

    root = os.path.expanduser(a.root)
    out_dir = a.out_dir or f"{root}/sidequest/latest"
    os.makedirs(out_dir, exist_ok=True)
    paths = {f: f"{out_dir}/GPT_{f}_{a.tag}.txt" for f in FIELDS}

    chosen, study = choose(root, a.n, a.per_study, a.seed)
    print(f"selected {len(chosen):,} samples from {len({s for _, s in chosen})} studies", flush=True)
    meta = fetch_metadata(root, {s for s, _ in chosen}, f"{out_dir}/.metadata_devset.pkl")
    chosen = [(s, st) for s, st in chosen if meta.get(s)]
    print(f"  {len(chosen):,} have metadata", flush=True)

    done = set()
    if os.path.exists(paths["keywords"]):
        done = {l.split("\t", 1)[0] for l in open(paths["keywords"], encoding="utf-8") if l.strip()}
        print(f"  {len(done):,} already done, resuming", flush=True)
    todo = [(s, st) for s, st in chosen if s not in done]
    here = os.path.dirname(os.path.abspath(__file__))
    prompt_path = (os.path.expanduser(a.prompt) if a.prompt else
                   os.path.join(here, "..", "source_data",
                                "openai_system_better_prompt_batch_v2_category_only.txt"))
    if a.prompt and not os.path.isabs(prompt_path) and not os.path.exists(prompt_path):
        prompt_path = f"{root}/{a.prompt}"
    system_prompt = open(prompt_path, encoding="utf-8").read()
    print(f"  prompt: {os.path.normpath(prompt_path)} ({len(system_prompt):,} chars)", flush=True)

    if a.dry_run:
        print(f"\nDRY RUN - {len(todo):,} samples would be sent. First request:\n")
        s, _ = todo[0]
        print("-" * 70 + "\nSYSTEM PROMPT (first 400 chars)\n" + "-" * 70)
        print(system_prompt[:400] + " ...")
        print("-" * 70 + f"\nUSER CONTENT for {s}\n" + "-" * 70)
        print(f"Sample ID: {s}, Metadata: {meta[s][:1200]}")
        lens = [len(meta[s]) for s, _ in todo]
        print("-" * 70)
        print(f"\ncleaned metadata length: median {sorted(lens)[len(lens)//2]:,} chars, "
              f"max {max(lens):,}")
        print(f"rough input tokens: {sum(lens)//4 + len(todo)*len(system_prompt)//4:,}")
        return

    from openai import OpenAI
    client = OpenAI(api_key=open(f"{root}/{a.api_key}").read().strip(), max_retries=5)
    lock, gate = threading.Lock(), threading.Semaphore(1)
    stamps, files = [], {f: open(paths[f], "a", encoding="utf-8") for f in FIELDS}
    raw_log = open(f"{out_dir}/raw_responses_{a.tag}.txt", "a", encoding="utf-8")
    ok = fail = 0
    usage = {"in": 0, "cached": 0, "out": 0, "reasoning": 0}

    def throttle():
        with gate:
            now = time.time()
            stamps[:] = [t for t in stamps if now - t < 60]
            if len(stamps) >= a.rpm:
                time.sleep(60 - (now - stamps[0]) + 0.1)
            stamps.append(time.time())

    SAMPLING = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
    dropped = set()          # params this model rejects; learned once, shared by all workers

    def build(sid):
        content = f"Sample ID: {sid}, Metadata: {meta[sid]}"      # production wording
        prompt = (system_prompt.replace("microbial metagenomic samples", "microbial metagenomic sample")
                               .replace("from their metadata texts", "from its metadata text"))
        body = dict(model=a.model,
                    messages=[{"role": "system", "content": prompt},
                              {"role": "user", "content": content}])
        for k, v in zip(SAMPLING, (a.temperature, a.top_p, a.frequency_penalty, a.presence_penalty)):
            if k not in dropped:
                body[k] = v
        body["max_tokens" if a.model.startswith("gpt-3.5") else "max_completion_tokens"] = a.max_tokens
        if a.output_format == "json":
            body["response_format"] = {"type": "json_object"}
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
        return body

    def unsupported_param(exc):
        """Name of the sampling param a 400 unsupported_parameter names, else None."""
        for src in (getattr(exc, "body", None), getattr(exc, "response", None)):
            try:
                d = src if isinstance(src, dict) else src.json()
                p = d.get("error", {}).get("param")
                if d.get("error", {}).get("code") == "unsupported_parameter" and p in SAMPLING:
                    return p
            except Exception:
                pass
        m = re.search(r"Unsupported parameter: '(\w+)'", str(exc))     # fall back to the text
        return m[1] if m and m[1] in SAMPLING else None

    def one(item):
        nonlocal ok, fail
        sid, _ = item
        text = None
        for _attempt in range(len(SAMPLING) + 1):
            try:
                throttle()
                r = client.chat.completions.create(**build(sid))
                text = r.choices[0].message.content or ""
                if (u := getattr(r, "usage", None)):
                    with lock:
                        usage["in"] += u.prompt_tokens or 0
                        usage["out"] += u.completion_tokens or 0
                        usage["cached"] += getattr(
                            getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                        usage["reasoning"] += getattr(
                            getattr(u, "completion_tokens_details", None),
                            "reasoning_tokens", 0) or 0
                break
            except Exception as exc:
                p = unsupported_param(exc)
                if p and p not in dropped:
                    with lock:
                        if p not in dropped:
                            dropped.add(p)
                            print(f"    note: {a.model} rejects {p!r} - dropping it "
                                  f"and retrying", flush=True)
                    continue                                   # same sample, one param lighter
                if p:
                    continue                                   # another worker just dropped it
                with lock:
                    fail += 1
                    raw_log.write(f"### {sid}\tERROR\t{exc}\n")
                return
        if text is None:
            with lock:
                fail += 1
                raw_log.write(f"### {sid}\tERROR\tgave up after dropping {sorted(dropped)}\n")
            return
        rec = (parse_json(text) if a.output_format == "json" else parse(text)) \
            if text and text.strip() else None
        with lock:
            raw_log.write(f"### {sid}\n{text}\n")
            if rec is None:
                fail += 1
            else:
                for f in FIELDS:
                    files[f].write(f"{sid}\t{rec[f]}\n")
                    files[f].flush()
                ok += 1
            if (ok + fail) % 50 == 0:
                print(f"    {ok + fail:,}/{len(todo):,}  ok {ok:,}  failed {fail:,}", flush=True)

    print(f"\nsending {len(todo):,} requests with {a.model} "
          f"({a.workers} workers, {a.rpm:.0f} rpm)...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo))
    for f in files.values():
        f.close()
    raw_log.close()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {ok:,} parsed, {fail:,} failed", flush=True)
    if dropped:
        print(f"  sampling params dropped ({a.model} rejects them): {', '.join(sorted(dropped))}",
              flush=True)
    if usage["in"] or usage["out"]:
        n_req = max(ok + fail, 1)
        print(f"  tokens: {usage['in']:,} in ({usage['cached']:,} cached), "
              f"{usage['out']:,} out ({usage['reasoning']:,} reasoning)")
        print(f"  per sample: {usage['in'] / n_req:,.0f} in, {usage['out'] / n_req:,.0f} out")
        if a.price_in is not None and a.price_out is not None:
            pc = a.price_cached if a.price_cached is not None else a.price_in / 10
            fresh = usage["in"] - usage["cached"]
            cost = (fresh * a.price_in + usage["cached"] * pc
                    + usage["out"] * a.price_out) / 1e6
            print(f"  cost: ${cost:.4f} for {n_req:,} samples "
                  f"= ${cost / n_req * 1000:.3f} per 1,000 samples "
                  f"-> ${cost / n_req * 3e6:,.0f} for 3M")
    for f in FIELDS:
        print(f"  {paths[f]}")

    report(root, paths["keywords"], f"{root}/sidequest/latest/GPT_keywords.txt")


def report(root, new_path, old_path):
    """Paired identity-marker comparison on the samples that were re-extracted."""
    GEO = re.compile(r"\b(National Park|Basin|Bay|Sea|Ocean|River|Lake|Island|University|Institute|"
                     r"Project|Consortium|Illumina|MiSeq|HiSeq|NovaSeq|PacBio|Nanopore|Cohort|Trial)\b")
    COUNTRY = re.compile(r"\b(China|USA|Japan|Germany|France|Canada|Sweden|Spain|Brazil|India|Italy|"
                         r"Korea|Australia|Netherlands|Denmark|Norway|Finland|Switzerland|Austria|"
                         r"Belgium|Mexico|Russia|Poland|Portugal)\b")
    GENERIC = re.compile(r"\b(metagenome|microbiome|microbial community)\b", re.I)
    new = {l.split("\t", 1)[0]: l.split("\t", 1)[1].strip()
           for l in open(new_path, encoding="utf-8") if "\t" in l}
    old = {}
    for l in open(old_path, encoding="utf-8", errors="replace"):
        sid, _, v = l.rstrip("\n").partition("\t")
        if sid in new:
            old[sid] = v
    both = sorted(set(new) & set(old))
    if not both:
        return
    print(f"\n=== identity markers, paired on {len(both):,} samples ===")
    print(f"  {'marker':<26} {'old':>8} {'new':>8}")
    for name, rx in [("country name", COUNTRY), ("place/project/instrument", GEO),
                     ("metagenome / microbiome", GENERIC)]:
        o = sum(1 for s in both if rx.search(old[s])) / len(both)
        n = sum(1 for s in both if rx.search(new[s])) / len(both)
        print(f"  {name:<26} {o:>7.1%} {n:>7.1%}")
    ow = sum(len(old[s].split()) for s in both) / len(both)
    nw = sum(len(new[s].split()) for s in both) / len(both)
    print(f"  {'mean words per sample':<26} {ow:>8.1f} {nw:>8.1f}")


if __name__ == "__main__":
    main()
