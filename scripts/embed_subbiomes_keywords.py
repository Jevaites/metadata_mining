#!/usr/bin/env python3
"""
Embed the per-sample sub-biome / keyword free text from MicrobeAtlas.

Input (~/MicrobeAtlasProject/sidequest/latest), tab-separated, no header:
    sample_id<TAB>text                  # GPT_sub_biomes.txt
    sample_id<TAB>{kw one, kw two}      # GPT_keywords.txt

Idea: many samples share the exact same text (2428 distinct sub-biome strings
for 9894 samples in the perbiome2000 subset), so embed each *distinct* text
once and broadcast its vector to every sample that has that text.

Outputs per target, tagged by model + dim + subset so runs never collide:
    GPT_{target}_unique_embeddings__{tag}.h5   texts, embeddings      (per distinct text)
    GPT_{target}_embeddings__{tag}.h5          sample_ids, texts, embeddings  (per sample)

Both files ARE the resume state: re-running the same command reads back what
is already in them and only does the missing work. No side-car state files.

    # cheap trial, no API calls
    python3 scripts/embed_subbiomes_keywords.py --n_per_biome 2000 --dry_run

    # for real
    python3 scripts/embed_subbiomes_keywords.py --n_per_biome 2000 --yes

    # same subset, different model/dim
    python3 scripts/embed_subbiomes_keywords.py --n_per_biome 2000 \
        --model text-embedding-3-large --embedding_dim 1024 --yes

    # everything (~3.4M samples)
    python3 scripts/embed_subbiomes_keywords.py --full --yes
"""

import argparse
import json
import os
import random
import shutil
import time

import h5py
import numpy as np
from openai import OpenAI

# text-embedding-3-* accept a `dimensions` argument that truncates (and
# renormalises) the vector; anything else must run at its native size.
NATIVE_DIM = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
PRICE_PER_1M_TOKENS = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
NA_VALUES = {"", "nan", "none", "na", "n/a"}
MAX_BATCH = 2048  # OpenAI's hard cap on inputs per embeddings.create call
DEFAULT_INPUT_DIR = os.path.expanduser("~/MicrobeAtlasProject/sidequest/latest")


# --------------------------------------------------------------------------
# reading the source text files
# --------------------------------------------------------------------------

def clean_text(text, is_keywords):
    """sub_biomes: plain strip. keywords: '{a, b, c}' -> 'a b c'."""
    text = text.strip()
    if is_keywords:
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        text = " ".join(text.replace(",", " ").split())
    return text


def iter_samples(path, keep_ids, is_keywords):
    """Stream (sample_id, cleaned_text), skipping malformed / empty / NA lines."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            sid, _, raw = line.rstrip("\n").partition("\t")
            if not sid or not raw:
                continue
            if keep_ids is not None and sid not in keep_ids:
                continue
            text = clean_text(raw, is_keywords)
            if text.lower() in NA_VALUES:
                continue
            yield sid, text


# --------------------------------------------------------------------------
# which samples to embed
# --------------------------------------------------------------------------

def pick_subset(args):
    """-> (set of sample ids, or None meaning 'all'; short tag for filenames)."""
    if args.sample_ids_file:
        path = os.path.expanduser(args.sample_ids_file)
        ids = {line.strip() for line in open(path, encoding="utf-8") if line.strip()}
        print(f"Loaded {len(ids)} sample IDs from {path}")
        return ids, "ids-" + os.path.splitext(os.path.basename(path))[0]

    if args.n_per_biome:
        groups = {}
        for sid, biome in iter_samples(os.path.join(args.input_dir, args.biomes_file), None, False):
            groups.setdefault(biome, []).append(sid)
        rng = random.Random(args.seed)
        ids = set()
        for biome in sorted(groups):
            pool = groups[biome]
            chosen = pool if len(pool) <= args.n_per_biome else rng.sample(pool, args.n_per_biome)
            ids.update(chosen)
            print(f"  biome '{biome}': {len(chosen)} / {len(pool)} selected")
        return ids, f"perbiome{args.n_per_biome}_seed{args.seed}"

    if args.n:
        all_ids = [sid for sid, _ in iter_samples(
            os.path.join(args.input_dir, args.sub_biomes_file), None, False)]
        n = min(args.n, len(all_ids))
        return set(random.Random(args.seed).sample(all_ids, n)), f"n{n}_seed{args.seed}"

    if not args.full:
        raise SystemExit("Pick --n / --n_per_biome / --sample_ids_file, or pass --full.")
    return None, "full"


def estimate_tokens(texts, model):
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return sum(len(enc.encode(t)) for t in texts), "tiktoken"
    except Exception:
        return sum(len(t) for t in texts) // 4, "~4 chars/token"


# --------------------------------------------------------------------------
# HDF5: append-only, and the file is its own resume state
# --------------------------------------------------------------------------

def h5_append(path, **columns):
    """Append equal-length columns, creating growable datasets on first write."""
    with h5py.File(path, "a") as f:
        for name, data in columns.items():
            if isinstance(data, np.ndarray):
                dtype, extra = data.dtype, data.shape[1:]
            else:
                dtype, extra = h5py.string_dtype("utf-8"), ()
            if name not in f:
                f.create_dataset(name, data=data, dtype=dtype, maxshape=(None,) + extra)
            else:
                dset = f[name]
                dset.resize(dset.shape[0] + len(data), axis=0)
                dset[-len(data):] = data


def h5_strings(path, column):
    """Read a string column back, or [] if the file does not exist yet."""
    if not os.path.exists(path):
        return []
    with h5py.File(path, "r") as f:
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in f[column][:]]


def h5_align(path):
    """Cut every column back to the shortest one.

    h5_append resizes and writes each column in turn, so a crash in the middle
    of a call (a full disk, a kill) leaves them at different lengths - and the
    longer ones point past the end of the file. Since resume trusts the row
    count, an unrepaired file would silently misalign sample_ids against
    embeddings from that point on. Dropping the incomplete tail is always safe:
    those rows get rewritten from the source."""
    if not os.path.exists(path):
        return
    with h5py.File(path, "a") as f:
        lengths = {name: f[name].shape[0] for name in f}
        if len(set(lengths.values())) < 2:
            return
        shortest = min(lengths.values())
        print(f"  repairing {os.path.basename(path)}: ragged columns {lengths} -> {shortest} rows")
        for name in f:
            f[name].resize(shortest, axis=0)


def check_disk_space(out_dir, needed_bytes):
    free = shutil.disk_usage(out_dir).free
    print(f"Disk: {needed_bytes / 1024 ** 3:.1f} GB needed, {free / 1024 ** 3:.1f} GB free on {out_dir}")
    if free < needed_bytes * 1.05:
        raise SystemExit(
            "Not enough free space. Free some up, or pass --unique_only to skip the "
            "per-sample file (it is the big one; you can join the compact table to "
            "the source text file yourself later)."
        )


# --------------------------------------------------------------------------
# stage 1: embed each distinct text once
# --------------------------------------------------------------------------

def embed_unique(label, texts, out_path, client, model, dim, batch_size):
    h5_align(out_path)
    done = set(h5_strings(out_path, "texts"))
    todo = [t for t in texts if t not in done]
    print(f"[{label}] {len(texts)} distinct texts: {len(done)} already embedded, {len(todo)} to do")

    kwargs = {"model": model}
    if dim != NATIVE_DIM[model]:
        kwargs["dimensions"] = dim

    start_time = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        t0 = time.time()
        response = client.embeddings.create(input=batch, **kwargs)
        vectors = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        assert vectors.shape == (len(batch), dim), f"got {vectors.shape}, expected dim {dim}"
        h5_append(out_path, texts=batch, embeddings=vectors)
        print(f"  [{label}] {i + len(batch)}/{len(todo)} texts ({time.time() - t0:.1f}s)")
    print(f"[{label}] unique texts done in {(time.time() - start_time) / 60:.1f} min -> {out_path}")


# --------------------------------------------------------------------------
# stage 2: broadcast each distinct text's vector to every sample sharing it
# --------------------------------------------------------------------------

def materialize(label, samples, text_to_vector, out_path, chunk=20_000):
    h5_align(out_path)
    done = set(h5_strings(out_path, "sample_ids"))
    if done:
        print(f"  [{label}] resuming: {len(done)} samples already written")

    ids, texts, vectors = [], [], []
    n_written = n_missing = 0

    def flush():
        nonlocal n_written
        if ids:
            h5_append(out_path, sample_ids=ids, texts=texts,
                      embeddings=np.asarray(vectors, dtype=np.float32))
            n_written += len(ids)
            print(f"  [{label}] {n_written} samples written")
            ids.clear()
            texts.clear()
            vectors.clear()

    for sid, text in samples:
        if sid in done:
            continue
        vector = text_to_vector.get(text)
        if vector is None:  # only possible if a previous run died mid-batch
            n_missing += 1
            continue
        ids.append(sid), texts.append(text), vectors.append(vector)
        if len(ids) >= chunk:
            flush()
    flush()

    if n_missing:
        print(f"  [{label}] WARNING {n_missing} samples had no embedded text - re-run to fill them in")
    print(f"[{label}] per-sample file done: {n_written} rows -> {out_path}")


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", default=None, help="Default: <input_dir>/embeddings")
    p.add_argument("--sub_biomes_file", default="GPT_sub_biomes.txt")
    p.add_argument("--keywords_file", default="GPT_keywords.txt")
    p.add_argument("--biomes_file", default="GPT_biomes.txt", help="Only used by --n_per_biome.")
    p.add_argument("--targets", nargs="+", choices=["sub_biomes", "keywords"],
                   default=["sub_biomes", "keywords"])
    p.add_argument("--api_key_path", default=os.path.expanduser("~/MicrobeAtlasProject/my_api_key_embeddings"))

    p.add_argument("--model", default="text-embedding-3-small", choices=sorted(NATIVE_DIM))
    p.add_argument("--embedding_dim", type=int, default=None, help="Default: the model's native size.")
    p.add_argument("--batch_size", type=int, default=MAX_BATCH)

    p.add_argument("--n", type=int, help="Random subset of N samples.")
    p.add_argument("--n_per_biome", type=int, help="Up to N samples per top-level biome.")
    p.add_argument("--sample_ids_file", help="Explicit sample IDs, one per line (reuse an earlier subset).")
    p.add_argument("--full", action="store_true", help="Every sample. Required if no subset flag is given.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--unique_only", action="store_true",
                   help="Skip the per-sample file (same size as before dedup; dedup saves API cost, not disk).")
    p.add_argument("--dry_run", action="store_true", help="Estimate tokens/cost, call no API.")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return p.parse_args()


def main():
    args = parse_args()
    dim = args.embedding_dim or NATIVE_DIM[args.model]
    if dim > NATIVE_DIM[args.model]:
        raise SystemExit(f"--embedding_dim {dim} > {args.model}'s native {NATIVE_DIM[args.model]}")

    args.input_dir = os.path.expanduser(args.input_dir)
    out_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(args.input_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    source = {"sub_biomes": (os.path.join(args.input_dir, args.sub_biomes_file), False),
              "keywords": (os.path.join(args.input_dir, args.keywords_file), True)}

    keep_ids, subset_tag = pick_subset(args)
    tag = f"{args.model}__dim{dim}__{subset_tag}"
    print(f"\nRun tag: {tag}")

    if keep_ids is not None:  # let a later run reuse this exact subset
        ids_path = os.path.join(out_dir, f"subset_ids__{subset_tag}.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(keep_ids)) + "\n")
        print(f"Subset IDs: {ids_path}")
    with open(os.path.join(out_dir, f"run_manifest__{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "embedding_dim": dim, "tag": tag,
                   "subset_size": len(keep_ids) if keep_ids else None,
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)

    # Read each source once: keep the sample list (subsets) or re-read lazily (full run),
    # and collect distinct texts in first-seen order.
    samples_of, unique_of, n_samples_of, total_tokens = {}, {}, {}, 0
    for target in args.targets:
        path, is_keywords = source[target]
        print(f"\n[{target}] scanning {path}")
        if keep_ids is None:
            samples_of[target] = lambda p=path, k=is_keywords: iter_samples(p, None, k)
        else:
            rows = list(iter_samples(path, keep_ids, is_keywords))
            samples_of[target] = lambda rows=rows: iter(rows)
            print(f"  {len(keep_ids) - len(rows)} of the {len(keep_ids)} requested IDs have no usable text here")

        seen, n_samples = {}, 0          # dict preserves first-seen order
        for _, text in samples_of[target]():
            seen[text] = None
            n_samples += 1
        unique = unique_of[target] = list(seen)
        n_samples_of[target] = n_samples
        print(f"  {n_samples} samples -> {len(unique)} distinct texts ({n_samples / max(len(unique), 1):.1f}x dedup)")

        tokens, how = estimate_tokens(unique, args.model)
        total_tokens += tokens
        print(f"  ~{tokens:,} tokens over the distinct texts ({how})")

    cost = total_tokens / 1e6 * PRICE_PER_1M_TOKENS[args.model]
    print(f"\nEstimated {total_tokens:,} tokens = ${cost:.4f} with {args.model}")

    # vectors + the text column, per target, minus whatever is already written
    needed = 0
    for target in args.targets:
        needed += len(unique_of[target]) * dim * 4 + sum(len(t) for t in unique_of[target])
        if not args.unique_only:
            needed += n_samples_of[target] * (dim * 4 + 12)
        for path in (os.path.join(out_dir, f"GPT_{target}_unique_embeddings__{tag}.h5"),
                     os.path.join(out_dir, f"GPT_{target}_embeddings__{tag}.h5")):
            if os.path.exists(path):
                needed -= os.path.getsize(path)
    check_disk_space(out_dir, max(needed, 0))

    if args.dry_run:
        print("--dry_run: stopping before any API call.")
        return
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return

    client = OpenAI(api_key=open(os.path.expanduser(args.api_key_path)).read().strip(), max_retries=8)

    for target in args.targets:
        unique_path = os.path.join(out_dir, f"GPT_{target}_unique_embeddings__{tag}.h5")
        embed_unique(target, unique_of[target], unique_path, client, args.model, dim,
                     min(args.batch_size, MAX_BATCH))
        if args.unique_only:
            continue
        with h5py.File(unique_path, "r") as f:
            text_to_vector = dict(zip(
                (t.decode("utf-8") if isinstance(t, bytes) else t for t in f["texts"][:]),
                f["embeddings"][:],
            ))
        materialize(target, samples_of[target](), text_to_vector,
                    os.path.join(out_dir, f"GPT_{target}_embeddings__{tag}.h5"))
        del text_to_vector

    print("\nDone.")


if __name__ == "__main__":
    main()
