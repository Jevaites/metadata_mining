#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute OpenAI embeddings for the per-sample sub-biome and keyword free-text
fields exported from MicrobeAtlas (see ~/MicrobeAtlasProject/sidequest/latest).

Input format (tab-separated, one sample per line, no header):
    sample_id<TAB>text
Keyword files additionally wrap the keyword list in braces:
    sample_id<TAB>{keyword one, keyword two, ...}

Deduplication
-------------
Many samples share the exact same sub-biome/keyword text (e.g. "human gut"
appears for thousands of samples), and text-embedding-3-small is
deterministic, so embedding the same string twice wastes money and time.
This script always embeds each *distinct* text exactly once, then assigns
that vector to every sample that has that text. On the current
sidequest/latest data this cuts sub-biome API calls ~108x (3.44M samples ->
31.7k unique texts) and keyword API calls ~2.3x (3.44M -> ~1.52M unique
cleaned texts).

Two output files per target:
  - GPT_{target}_unique_embeddings__{tag}.h5: the compact table that was
    actually sent to the API - "texts" and "embeddings" only, one row per
    *distinct* text. Always written. Small (tens of MB for sub-biomes,
    single-digit GB for keywords at full scale).
  - GPT_{target}_embeddings__{tag}.h5: one row per *sample* -
    "sample_ids"/"texts"/"embeddings" - matching the schema used by the
    existing embeddings/GPT_*_embeddings*.h5 files in this project, so
    scripts/production/align_and_average_embeddings.py and friends can use
    it unchanged. Written by default; pass --no_materialize_full to skip it
    if you only need the compact table (e.g. at full scale, where this file
    is exactly as large as before deduplication - dedup saves API cost/time,
    not this file's size, since it still has one row per sample).

Alongside each run a JSON manifest and (for subsets) a plain-text sample-ID
list are written, so a later run can reuse the exact same subset or be
compared apples-to-apples.

Designed to make small, cheap trial runs easy: --n / --n_per_biome / --frac /
--sample_ids_file all let you embed a subset instead of every sample, and
output/state/manifest files are tagged by model + subset so trial runs never
clobber each other or a full run. Embedding every sample requires the
explicit --full flag, as a guard against accidentally kicking off a huge,
expensive run. --dry_run estimates tokens/cost (exactly, from the unique
texts) and writes the subset manifest without calling the API at all.

Progress is resumable at both stages: if interrupted (Ctrl-C, network error,
laptop sleep), re-running the same command skips unique texts already
embedded and samples already materialized.

Examples
--------
# Cheap trial: 2,000 random samples, both targets, default model, no API calls
python scripts/embed_subbiomes_keywords.py --n 2000 --dry_run

# Same, but actually call the API
python scripts/embed_subbiomes_keywords.py --n 2000 --yes

# 200 samples per top-level biome (stratified), sub-biomes only
python scripts/embed_subbiomes_keywords.py --n_per_biome 200 --targets sub_biomes --yes

# Reuse the exact same subset to try a different model
python scripts/embed_subbiomes_keywords.py \
    --sample_ids_file ~/MicrobeAtlasProject/sidequest/latest/embeddings/subset_ids__n2000_seed42.txt \
    --model text-embedding-3-large --embedding_dim 3072 --yes

# Same subset again, truncated to 1024 dims via the API's native `dimensions`
# parameter (only text-embedding-3-* models support this; --embedding_dim
# must equal the model's native size for any other model, e.g. ada-002)
python scripts/embed_subbiomes_keywords.py \
    --sample_ids_file ~/MicrobeAtlasProject/sidequest/latest/embeddings/subset_ids__n2000_seed42.txt \
    --model text-embedding-3-large --embedding_dim 1024 --yes

# Full run (all ~3.4M samples) - resumable if interrupted
python scripts/embed_subbiomes_keywords.py --full --yes

# Full run, keywords only, skip the ~21GB per-sample file and keep just the
# ~9GB compact unique-text table (join it to sample_id->text yourself later)
python scripts/embed_subbiomes_keywords.py --full --targets keywords --no_materialize_full --yes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import h5py
    import numpy as np
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for embed_subbiomes_keywords.py. "
        "Install with: pip install h5py numpy openai"
    ) from exc

try:
    # Reuse the shared helpers already used by the ontology-mapping scripts
    # in this repo, when this script is run from inside scripts/.
    from ontology_mapping_utils import load_api_key, write_json
except ImportError:
    def load_api_key(path: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    def write_json(path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)


# Rough public per-1M-token pricing for OpenAI embedding models, USD.
# Verify at https://openai.com/api/pricing before trusting a big estimate -
# override with --price_per_million_tokens if this table is stale.
KNOWN_PRICING_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Native (max) output dimensionality per model. The OpenAI API's "dimensions"
# parameter (Matryoshka-style truncation) is only accepted for the
# text-embedding-3-* family - passing it to text-embedding-ada-002 is a 400
# error, so ada-002 must always be run at its native size.
MODEL_NATIVE_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
MODELS_SUPPORTING_DIMENSIONS = {"text-embedding-3-small", "text-embedding-3-large"}

NA_VALUES = {"nan", "none", "na", "n/a", ""}

# OpenAI enforces at most 2048 inputs per embeddings.create call.
MAX_API_BATCH_SIZE = 2048

# Above this many unique texts, warn before loading them all into memory to
# materialize the per-sample file (rough size: n * (embedding_dim*4 + ~250
# bytes of Python/dict overhead per entry)).
MATERIALIZE_WARN_THRESHOLD = 300_000

SampleSourceFactory = Callable[[], Iterator[Tuple[str, str]]]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed GPT sub-biome / keyword text per sample with an OpenAI embedding model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input_dir",
        default=os.path.join(os.path.expanduser("~"), "MicrobeAtlasProject/sidequest/latest"),
        help="Directory containing GPT_sub_biomes.txt / GPT_keywords.txt / GPT_biomes.txt.",
    )
    parser.add_argument("--sub_biomes_file", default="GPT_sub_biomes.txt")
    parser.add_argument("--keywords_file", default="GPT_keywords.txt")
    parser.add_argument(
        "--biomes_file", default="GPT_biomes.txt",
        help="Used only for --n_per_biome stratified sampling.",
    )
    parser.add_argument(
        "--targets", nargs="+", choices=["sub_biomes", "keywords"],
        default=["sub_biomes", "keywords"],
    )

    parser.add_argument("--output_dir", default=None, help="Default: <input_dir>/embeddings")
    parser.add_argument(
        "--api_key_path",
        default=os.path.join(os.path.expanduser("~"), "MicrobeAtlasProject/my_api_key_embeddings"),
    )

    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument(
        "--embedding_dim", type=int, default=None,
        help="Defaults to --model's native output size (1536 for "
             "text-embedding-3-small/ada-002, 3072 for text-embedding-3-large). "
             "Set explicitly to truncate via the API's `dimensions` parameter "
             "(text-embedding-3-* models only).",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1000,
        help=f"Distinct texts per API call (capped at {MAX_API_BATCH_SIZE}).",
    )
    parser.add_argument("--max_requests_per_round", type=int, default=100)
    parser.add_argument(
        "--wait_time", type=float, default=20.0,
        help="Seconds to sleep every --max_requests_per_round calls (basic rate-limit pacing).",
    )
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff", type=float, default=5.0, help="Seconds, multiplied by attempt number.")

    # Subset selection. Precedence: sample_ids_file > n_per_biome > n/frac > --full.
    parser.add_argument("--n", type=int, default=None, help="Embed a random subset of N samples.")
    parser.add_argument("--frac", type=float, default=None, help="Embed a random fraction of samples (0-1).")
    parser.add_argument(
        "--n_per_biome", type=int, default=None,
        help="Stratified subset: up to N samples per top-level biome label (from --biomes_file).",
    )
    parser.add_argument(
        "--sample_ids_file", default=None,
        help="Explicit list of sample IDs to embed (one per line). Overrides --n/--frac/--n_per_biome.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Embed every sample. Required as an explicit confirmation if no subset flag is given.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--keyword_sep", default=" ",
        help="Separator used to join keywords after stripping braces/commas.",
    )
    parser.add_argument(
        "--keep_keyword_commas", action="store_true",
        help="Do not strip commas out of keyword text. Default matches the legacy pipeline "
        "(scripts/production/make_embeddings.py): commas are replaced by --keyword_sep.",
    )

    parser.add_argument(
        "--materialize_full", dest="materialize_full", action="store_true", default=True,
        help="Also write a one-row-per-sample H5 (sample_ids/texts/embeddings), duplicating "
        "embeddings across samples that share a text. This is what existing scripts (e.g. "
        "align_and_average_embeddings.py) expect. Default: on.",
    )
    parser.add_argument(
        "--no_materialize_full", dest="materialize_full", action="store_false",
        help="Skip the per-sample file; keep only the compact unique-text table. Much less disk "
        "(and, for a big unique-text count, much less RAM at write time) - join it to "
        "sample_id->text (straight from GPT_*.txt) yourself when you need a specific sample's vector.",
    )

    parser.add_argument(
        "--run_tag", default=None,
        help="Custom suffix for output/state/manifest filenames. Default: auto-generated from model + subset.",
    )
    parser.add_argument(
        "--price_per_million_tokens", type=float, default=None,
        help="Override the built-in price table for cost estimates.",
    )

    parser.add_argument(
        "--dry_run", action="store_true",
        help="Estimate tokens/cost and write the subset manifest, but do not call the API.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reading / cleaning the source text files
# ---------------------------------------------------------------------------

def count_lines(path: str) -> int:
    try:
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True, check=True)
        return int(result.stdout.split()[0])
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)


def load_all_sample_ids(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            sid = line.split("\t", 1)[0].strip()
            if sid:
                ids.append(sid)
    return ids


def load_biome_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            sid, biome = parts[0].strip(), parts[1].strip()
            if sid and biome:
                labels[sid] = biome
    return labels


def stratified_subset(biome_labels: Dict[str, str], n_per_biome: int, seed: int) -> Set[str]:
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = {}
    for sid, biome in biome_labels.items():
        groups.setdefault(biome, []).append(sid)

    selected: Set[str] = set()
    for biome in sorted(groups):
        ids = groups[biome]
        chosen = ids if len(ids) <= n_per_biome else rng.sample(ids, n_per_biome)
        selected.update(chosen)
        print(f"  biome '{biome}': {len(chosen)} / {len(ids)} selected")
    return selected


def clean_keyword_text(text: str, sep: str, strip_commas: bool) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if strip_commas:
        text = text.replace(",", sep)
    return " ".join(text.split())


def iter_samples(
    path: str,
    target_ids: Optional[Set[str]],
    is_keywords: bool,
    keyword_sep: str,
    strip_commas: bool,
) -> Iterator[Tuple[str, str]]:
    """Stream (sample_id, text) pairs from a sample_id<TAB>text file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sid, text = parts[0].strip(), parts[1].strip()
            if not sid or not text:
                continue
            if target_ids is not None and sid not in target_ids:
                continue
            if is_keywords:
                text = clean_keyword_text(text, keyword_sep, strip_commas)
            if text.lower() in NA_VALUES:
                continue
            yield sid, text


# ---------------------------------------------------------------------------
# Subset resolution + manifest
# ---------------------------------------------------------------------------

def resolve_target_ids(args: argparse.Namespace) -> Tuple[Optional[Set[str]], str]:
    """Returns (target_ids, subset_tag). target_ids is None for a full run."""
    if args.sample_ids_file:
        ids_path = os.path.abspath(os.path.expanduser(args.sample_ids_file))
        with open(ids_path, "r", encoding="utf-8") as handle:
            ids = {line.strip() for line in handle if line.strip()}
        tag = f"ids-{os.path.splitext(os.path.basename(ids_path))[0]}"
        print(f"Loaded {len(ids)} sample IDs from {ids_path}")
        return ids, tag

    if args.n_per_biome is not None:
        biomes_path = os.path.join(args.input_dir, args.biomes_file)
        print(f"Loading biome labels from {biomes_path} for stratified sampling...")
        biome_labels = load_biome_labels(biomes_path)
        ids = stratified_subset(biome_labels, args.n_per_biome, args.seed)
        tag = f"perbiome{args.n_per_biome}_seed{args.seed}"
        return ids, tag

    if args.n is not None or args.frac is not None:
        ref_path = os.path.join(args.input_dir, args.sub_biomes_file)
        print(f"Loading all sample IDs from {ref_path} to draw a random subset...")
        all_ids = load_all_sample_ids(ref_path)
        if args.n is not None:
            n = min(args.n, len(all_ids))
            tag = f"n{n}_seed{args.seed}"
        else:
            n = max(1, round(len(all_ids) * args.frac))
            tag = f"frac{args.frac}_seed{args.seed}"
        rng = random.Random(args.seed)
        ids = set(rng.sample(all_ids, n))
        return ids, tag

    if not args.full:
        raise SystemExit(
            "No subset selected and --full not passed. Pick one of --n / --frac / "
            "--n_per_biome / --sample_ids_file for a cheap trial run, or pass --full "
            "to explicitly embed every sample."
        )
    return None, "full"


def write_subset_manifest(
    output_dir: str,
    tag: str,
    target_ids: Optional[Set[str]],
    args_dict: dict,
) -> Tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"run_manifest__{tag}.json")

    ids_path = None
    if target_ids is not None:
        ids_path = os.path.join(output_dir, f"subset_ids__{tag}.txt")
        with open(ids_path, "w", encoding="utf-8") as handle:
            for sid in sorted(target_ids):
                handle.write(sid + "\n")

    manifest = dict(args_dict)
    manifest["subset_size"] = len(target_ids) if target_ids is not None else None
    manifest["subset_ids_file"] = ids_path
    manifest["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(manifest_path, manifest)
    return manifest_path, ids_path


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def discover_unique_texts(samples: Iterable[Tuple[str, str]]) -> Tuple[List[str], int]:
    """One pass over (sample_id, text) pairs -> (distinct texts in first-seen order, n_samples_seen)."""
    seen: Dict[str, None] = {}
    n_seen = 0
    for _, text in samples:
        n_seen += 1
        if text not in seen:
            seen[text] = None
    return list(seen.keys()), n_seen


def estimate_dict_memory_bytes(n_rows: int, embedding_dim: int) -> int:
    # float32 vectors plus a generous per-entry allowance for Python/dict/numpy overhead.
    return int(n_rows * (embedding_dim * 4 + 250))


# ---------------------------------------------------------------------------
# Cost / token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(texts: Sequence[str], model: str) -> Tuple[int, str]:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        total = sum(len(enc.encode(t)) for t in texts)
        return total, "tiktoken (exact)"
    except Exception:
        # OpenAI's own rule of thumb for English text when tiktoken/its
        # vocab file isn't reachable (e.g. restricted network).
        total = sum(max(1, len(t)) for t in texts) // 4
        return total, "heuristic (~4 chars/token, tiktoken unavailable)"


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _append_datasets(path: str, columns: "Dict[str, Tuple[Sequence, str, Tuple[int, ...]]]") -> None:
    mode = "r+" if os.path.exists(path) else "w"
    with h5py.File(path, mode) as handle:
        n_new = None
        for name, (data, dtype, extra_shape) in columns.items():
            if n_new is None:
                n_new = len(data)
            if name not in handle:
                handle.create_dataset(name, data=data, maxshape=(None,) + extra_shape, dtype=dtype)
            else:
                dset = handle[name]
                old_n = dset.shape[0]
                dset.resize(old_n + n_new, axis=0)
                dset[old_n:] = data


def append_to_h5(path: str, ids: List[str], texts: List[str], embeddings: "np.ndarray", embedding_dim: int) -> None:
    """One row per sample: sample_ids / texts / embeddings (matches the existing pipeline's schema)."""
    dt = h5py.string_dtype(encoding="utf-8")
    _append_datasets(path, {
        "sample_ids": (ids, dt, ()),
        "texts": (texts, dt, ()),
        "embeddings": (embeddings, "f4", (embedding_dim,)),
    })


def append_unique_to_h5(path: str, texts: List[str], embeddings: "np.ndarray", embedding_dim: int) -> None:
    """One row per distinct text: texts / embeddings only."""
    dt = h5py.string_dtype(encoding="utf-8")
    _append_datasets(path, {
        "texts": (texts, dt, ()),
        "embeddings": (embeddings, "f4", (embedding_dim,)),
    })


def load_done_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def load_unique_text_to_vector(compact_path: str) -> Dict[str, "np.ndarray"]:
    with h5py.File(compact_path, "r") as handle:
        texts = handle["texts"][:]
        embeddings = handle["embeddings"][:]
    result: Dict[str, "np.ndarray"] = {}
    for i, raw in enumerate(texts):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        result[text] = embeddings[i]
    return result


# ---------------------------------------------------------------------------
# Stage 1: embed each distinct text once
# ---------------------------------------------------------------------------

def call_embeddings_with_retry(
    client: "OpenAI",
    model: str,
    embedding_dim: int,
    texts: List[str],
    max_retries: int,
    retry_backoff: float,
) -> Optional["np.ndarray"]:
    last_exc: Optional[Exception] = None
    create_kwargs: Dict[str, object] = {"input": texts, "model": model}
    if model in MODELS_SUPPORTING_DIMENSIONS and embedding_dim != MODEL_NATIVE_DIMS.get(model):
        create_kwargs["dimensions"] = embedding_dim

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(**create_kwargs)
            vectors = [item.embedding for item in response.data]
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.shape[1] != embedding_dim:
                raise ValueError(
                    f"Expected embedding_dim={embedding_dim}, got {arr.shape[1]}. "
                    "Check --model/--embedding_dim."
                )
            return arr
        except Exception as exc:  # noqa: BLE001 - we want to retry almost anything transient
            last_exc = exc
            sleep_s = retry_backoff * attempt
            print(f"    request failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {sleep_s:.0f}s...")
            time.sleep(sleep_s)
    print(f"    giving up on this batch of {len(texts)} texts after {max_retries} attempts: {last_exc}")
    return None


def embed_unique_texts(
    label: str,
    unique_texts: List[str],
    compact_path: str,
    done_texts_path: str,
    failed_texts_path: str,
    client: "OpenAI",
    model: str,
    embedding_dim: int,
    batch_size: int,
    max_requests_per_round: int,
    wait_time: float,
    max_retries: int,
    retry_backoff: float,
) -> None:
    done_texts = load_done_ids(done_texts_path)
    if done_texts:
        print(f"  resuming '{label}' unique-text embedding: {len(done_texts)} texts already embedded")

    buffer_texts: List[str] = []
    state = {"n_new": 0, "request_count": 0}
    start_time = time.time()

    with open(done_texts_path, "a", encoding="utf-8") as done_handle, \
            open(failed_texts_path, "a", encoding="utf-8") as failed_handle:

        def flush() -> None:
            if not buffer_texts:
                return
            t0 = time.time()
            arr = call_embeddings_with_retry(client, model, embedding_dim, buffer_texts, max_retries, retry_backoff)
            state["request_count"] += 1
            if arr is None:
                for text in buffer_texts:
                    failed_handle.write(text + "\n")
                failed_handle.flush()
            else:
                append_unique_to_h5(compact_path, buffer_texts, arr, embedding_dim)
                for text in buffer_texts:
                    done_handle.write(text + "\n")
                done_handle.flush()
                state["n_new"] += len(buffer_texts)
            elapsed = time.time() - t0
            print(
                f"  [{label}] unique-text batch {state['request_count']}: {len(buffer_texts)} texts in "
                f"{elapsed:.1f}s ({state['n_new']} new unique texts embedded this run)"
            )
            if state["request_count"] % max_requests_per_round == 0:
                print(f"  [{label}] pausing {wait_time:.0f}s after {state['request_count']} requests...")
                time.sleep(wait_time)

        for text in unique_texts:
            if text in done_texts:
                continue
            buffer_texts.append(text)
            if len(buffer_texts) >= batch_size:
                flush()
                buffer_texts = []
        flush()

    elapsed_total = time.time() - start_time
    print(f"[{label}] unique-text embedding done: {state['n_new']} newly embedded in {elapsed_total / 60:.1f} min")


# ---------------------------------------------------------------------------
# Stage 2 (optional): materialize one row per sample from the compact table
# ---------------------------------------------------------------------------

def materialize_full(
    label: str,
    sample_source_factory: SampleSourceFactory,
    text_to_vector: Dict[str, "np.ndarray"],
    output_path: str,
    done_ids_path: str,
    skipped_path: str,
    embedding_dim: int,
    chunk_size: int = 20_000,
) -> None:
    done_ids = load_done_ids(done_ids_path)
    if done_ids:
        print(f"  resuming '{label}' materialize: {len(done_ids)} samples already written")

    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    buffer_vecs: List["np.ndarray"] = []
    n_written = 0
    n_skipped = 0
    start_time = time.time()

    with open(done_ids_path, "a", encoding="utf-8") as done_handle, \
            open(skipped_path, "a", encoding="utf-8") as skipped_handle:

        def flush() -> None:
            nonlocal n_written
            if not buffer_ids:
                return
            arr = np.asarray(buffer_vecs, dtype=np.float32)
            append_to_h5(output_path, buffer_ids, buffer_texts, arr, embedding_dim)
            for sid in buffer_ids:
                done_handle.write(sid + "\n")
            done_handle.flush()
            n_written += len(buffer_ids)
            print(f"  [{label}] materialize: {n_written} samples written so far")

        for sid, text in sample_source_factory():
            if sid in done_ids:
                continue
            vec = text_to_vector.get(text)
            if vec is None:
                skipped_handle.write(f"{sid}\t{text}\n")
                skipped_handle.flush()
                n_skipped += 1
                continue
            buffer_ids.append(sid)
            buffer_texts.append(text)
            buffer_vecs.append(vec)
            if len(buffer_ids) >= chunk_size:
                flush()
                buffer_ids, buffer_texts, buffer_vecs = [], [], []
        flush()

    elapsed_total = time.time() - start_time
    if n_skipped:
        print(f"  [{label}] materialize: {n_skipped} samples skipped (their text failed to embed) -> {skipped_path}")
    print(
        f"[{label}] materialize done: {n_written} samples written to {output_path} "
        f"in {elapsed_total / 60:.1f} min"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    native_dim = MODEL_NATIVE_DIMS.get(args.model)
    explicit_dim_given = args.embedding_dim is not None

    if not explicit_dim_given:
        args.embedding_dim = native_dim if native_dim is not None else 1536
        if native_dim is not None:
            print(f"--embedding_dim not set; using {args.model}'s native size ({native_dim}).")

    if native_dim is None:
        print(
            f"warning: unrecognized --model {args.model!r}; assuming it does not "
            "support the `dimensions` parameter and will always return its native "
            "size. If --embedding_dim doesn't match, batches will fail with a "
            "shape-mismatch error."
        )
    elif args.model not in MODELS_SUPPORTING_DIMENSIONS:
        if args.embedding_dim != native_dim:
            raise SystemExit(
                f"--model {args.model} does not support the API's `dimensions` "
                f"truncation parameter, so --embedding_dim must be its native "
                f"size ({native_dim}), not {args.embedding_dim}."
            )
    elif args.embedding_dim > native_dim:
        raise SystemExit(
            f"--embedding_dim {args.embedding_dim} exceeds {args.model}'s native "
            f"size ({native_dim}); `dimensions` can only truncate, not expand."
        )

    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = (
        os.path.abspath(os.path.expanduser(args.output_dir))
        if args.output_dir
        else os.path.join(input_dir, "embeddings")
    )
    os.makedirs(output_dir, exist_ok=True)

    target_paths = {
        "sub_biomes": (os.path.join(input_dir, args.sub_biomes_file), False),
        "keywords": (os.path.join(input_dir, args.keywords_file), True),
    }
    for target in args.targets:
        path, _ = target_paths[target]
        if not os.path.exists(path):
            raise SystemExit(f"Input file not found for target '{target}': {path}")

    strip_commas = not args.keep_keyword_commas
    batch_size = min(args.batch_size, MAX_API_BATCH_SIZE)
    if args.batch_size > MAX_API_BATCH_SIZE:
        print(f"Note: --batch_size capped at {MAX_API_BATCH_SIZE} (OpenAI's per-request limit).")

    target_ids, subset_tag = resolve_target_ids(args)
    model_tag = args.run_tag or f"{args.model}__{subset_tag}"

    manifest_path, ids_path = write_subset_manifest(output_dir, subset_tag, target_ids, vars(args))
    print(f"\nRun tag:        {model_tag}")
    print(f"Run manifest:   {manifest_path}")
    if ids_path:
        print(f"Subset ID list: {ids_path}")

    price_per_million = args.price_per_million_tokens
    if price_per_million is None:
        price_per_million = KNOWN_PRICING_PER_MILLION_TOKENS.get(args.model)

    # ---- build a re-runnable sample source per target, then dedup + estimate ----
    source_factories: Dict[str, SampleSourceFactory] = {}
    total_expected: Dict[str, int] = {}
    unique_texts_by_target: Dict[str, List[str]] = {}
    total_tokens_estimate = 0

    for target in args.targets:
        path, is_keywords = target_paths[target]
        print(f"\n[{target}] scanning {path} ...")

        if target_ids is not None:
            samples = list(iter_samples(path, target_ids, is_keywords, args.keyword_sep, strip_commas))
            missing = len(target_ids) - len(samples)
            if missing:
                print(f"  note: {missing} requested sample IDs had no usable '{target}' text and will be skipped")
            source_factories[target] = (lambda samples=samples: iter(samples))
            total_expected[target] = len(samples)
        else:
            source_factories[target] = (
                lambda path=path, is_keywords=is_keywords: iter_samples(
                    path, None, is_keywords, args.keyword_sep, strip_commas
                )
            )
            total_expected[target] = count_lines(path)

        unique_texts, n_seen = discover_unique_texts(source_factories[target]())
        unique_texts_by_target[target] = unique_texts
        dedup_factor = (n_seen / len(unique_texts)) if unique_texts else 0.0
        print(f"  {n_seen} samples -> {len(unique_texts)} distinct texts ({dedup_factor:.1f}x deduplication)")

        tokens, method = estimate_tokens(unique_texts, args.model)
        print(f"  ~{tokens:,} tokens across distinct texts ({method})")
        total_tokens_estimate += tokens

    print()
    print(f"Estimated total tokens (distinct texts only): {total_tokens_estimate:,}")
    if price_per_million is not None:
        cost = total_tokens_estimate / 1_000_000 * price_per_million
        print(f"Estimated cost @ ${price_per_million:.4f} / 1M tokens: ${cost:.4f}")
    else:
        print(f"No known price for model '{args.model}'; pass --price_per_million_tokens to estimate cost.")

    if args.materialize_full:
        print("\n--materialize_full is on: a one-row-per-sample file will also be written for each target")
        print("(same disk footprint as without deduplication - only the API cost/time is reduced).")

    if args.dry_run:
        print("\n--dry_run set: not calling the API. The manifest/subset files above are ready for a real run")
        if ids_path:
            print(f"(re-run the same command without --dry_run, or add --sample_ids_file {ids_path} to a new "
                  "command to reuse this exact subset).")
        else:
            print("(re-run the same command without --dry_run to start the full embedding run).")
        return

    if not args.yes:
        answer = input("\nProceed with embedding? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    api_key = load_api_key(os.path.abspath(os.path.expanduser(args.api_key_path)))
    client = OpenAI(api_key=api_key)

    for target in args.targets:
        unique_texts = unique_texts_by_target[target]
        compact_path = os.path.join(output_dir, f"GPT_{target}_unique_embeddings__{model_tag}.h5")
        done_texts_path = os.path.join(output_dir, f".done_texts__{target}__{model_tag}.txt")
        failed_texts_path = os.path.join(output_dir, f"failed_texts__{target}__{model_tag}.tsv")

        print(f"\n=== {target}: embedding {len(unique_texts)} distinct texts -> {compact_path} ===")
        embed_unique_texts(
            label=target,
            unique_texts=unique_texts,
            compact_path=compact_path,
            done_texts_path=done_texts_path,
            failed_texts_path=failed_texts_path,
            client=client,
            model=args.model,
            embedding_dim=args.embedding_dim,
            batch_size=batch_size,
            max_requests_per_round=args.max_requests_per_round,
            wait_time=args.wait_time,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )

        if not args.materialize_full:
            continue
        if not os.path.exists(compact_path):
            print(f"  [{target}] no unique texts were successfully embedded; skipping materialize step")
            continue

        with h5py.File(compact_path, "r") as handle:
            n_unique_embedded = handle["embeddings"].shape[0]

        if n_unique_embedded > MATERIALIZE_WARN_THRESHOLD:
            est_mb = estimate_dict_memory_bytes(n_unique_embedded, args.embedding_dim) / (1024 ** 2)
            print(
                f"  [{target}] loading {n_unique_embedded:,} unique embeddings into memory to materialize "
                f"the per-sample file (~{est_mb:,.0f} MB) - pass --no_materialize_full to skip this."
            )
        text_to_vector = load_unique_text_to_vector(compact_path)

        output_path = os.path.join(output_dir, f"GPT_{target}_embeddings__{model_tag}.h5")
        done_ids_path = os.path.join(output_dir, f".done_ids__{target}__{model_tag}.txt")
        skipped_path = os.path.join(output_dir, f"skipped__{target}__{model_tag}.tsv")

        print(f"=== {target}: materializing per-sample file -> {output_path} ===")
        materialize_full(
            label=target,
            sample_source_factory=source_factories[target],
            text_to_vector=text_to_vector,
            output_path=output_path,
            done_ids_path=done_ids_path,
            skipped_path=skipped_path,
            embedding_dim=args.embedding_dim,
        )
        del text_to_vector

    print("\nAll targets complete.")


if __name__ == "__main__":
    main()
