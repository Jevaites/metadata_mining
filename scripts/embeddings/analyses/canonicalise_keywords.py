#!/usr/bin/env python3
"""
Replace each free-text keyword with its nearest ontology term label.

Why: arms_knn.py showed that stripping every explicit identity token moved study
fingerprinting only 0.935 -> 0.842, while switching to three-word sub-biome text
moved it to 0.365 (0.430 on tie-free samples). So much of the study signal lives
in the richness and phrasing of the text, not in identity words. Mapping each
keyword onto a controlled vocabulary destroys phrasing while keeping meaning,
which isolates that effect - and is the ontology-mapping goal anyway.

TWO THINGS THIS SCRIPT GETS RIGHT, both learned the hard way (the first version
produced 'stool -> latrine' and 'Homo sapiens -> human house'):

  vocabulary   match against the SLOT VOCABULARY - the ~414 terms that actually
               appear as Metalog biome/feature/material labels - not all 18,849
               ENVO/Uberon terms. term-text-variants.md measured open retrieval
               over everything at top-1 0.033/0.128/0.339: it does not work.
  term text    match against the PLAIN LABEL by default, not 'label; synonyms'.
               Same doc: synonyms pull 'fecal material' away from gut keyword
               lists, and open retrieval on material goes 0.095 -> 0.339 with
               plain labels.

Keywords whose nearest term is below --min_sim are KEPT AS WRITTEN. That is
deliberate: hosts, sexes, ages, diseases and measurements ('0 m depth',
'-1.45 degrees C') have no term in an environmental slot vocabulary, and they
carry the within-study variation we want to preserve.

ALWAYS look at --review output before trusting any metric computed on the result.

    python3 scripts/embeddings/analyses/canonicalise_keywords.py --tag mini_v3_9k --dry_run
    python3 scripts/embeddings/analyses/canonicalise_keywords.py --tag mini_v3_9k --review 40
"""
import argparse, gzip, os, sys
from collections import Counter

import numpy as np
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # scripts/: for ontology_mapping.common
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import MAX_BATCH, embed_unique, estimate_tokens
from ontology_mapping.common import SLOTS, load_terms


def slot_vocabulary(root, terms):
    """The terms that appear as Metalog slot labels, as (label, term_text) rows."""
    used = set()
    with gzip.open(f"{root}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        ix = [h.index(s) for s in SLOTS]
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(ix):
                used.update(f[i] for i in ix if f[i])
    sub = terms[terms["term_id"].isin(used)].reset_index(drop=True)
    missing = used - set(sub["term_id"])
    print(f"slot vocabulary: {len(sub):,} terms "
          f"({len(used):,} distinct Metalog codes, {len(missing)} not in the term table)")
    return sub


def vectors_for(texts, h5_path, client, model, dim, label):
    """L2-normalised vectors for `texts`, embedding whatever the h5 does not have."""
    import h5py
    if client is not None:
        embed_unique(label, texts, h5_path, client, model, dim, MAX_BATCH)
    with h5py.File(os.path.expanduser(h5_path), "r") as h:
        have = [t.decode() if isinstance(t, bytes) else t for t in h["texts"][:]]
        X = np.asarray(h["embeddings"][:], dtype=np.float32)
    row = {t: i for i, t in enumerate(have)}
    absent = [t for t in texts if t not in row]
    if absent:
        raise SystemExit(f"{len(absent)} {label} texts are not in {h5_path} "
                         f"(e.g. {absent[0]!r}); rerun without --no_embed")
    return normalize(X[[row[t] for t in texts]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--tag", required=True, help="reads GPT_keywords_<tag>.txt")
    p.add_argument("--out_tag", default=None, help="default: <tag>_canon")
    p.add_argument("--term_text", choices=["label", "label_syn"], default="label",
                   help="what the term is matched by; 'label' is better for retrieval")
    p.add_argument("--model", default="text-embedding-3-large")
    p.add_argument("--embedding_dim", type=int, default=1024)
    p.add_argument("--min_sim", type=float, default=0.60,
                   help="below this cosine the keyword is kept as written")
    p.add_argument("--review", type=int, default=25, help="mappings to print for inspection")
    p.add_argument("--no_embed", action="store_true", help="use only vectors already on disk")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    root = os.path.expanduser(a.root)
    lat = f"{root}/sidequest/latest"
    out_tag = a.out_tag or f"{a.tag}_canon"
    suffix = f"{a.model}__dim{a.embedding_dim}"
    kw_h5 = f"{lat}/embeddings/keyword_atoms__{suffix}.h5"
    slot_h5 = f"{root}/ontology_mapping/slot_vocab_{a.term_text}__{suffix}.h5"
    if a.term_text == "label_syn":                       # reuse the full term index
        slot_h5 = f"{root}/ontology_mapping/ontology_terms_unique_embeddings__{suffix}.h5"

    rows = []
    for line in open(f"{lat}/GPT_keywords_{a.tag}.txt", encoding="utf-8", errors="replace"):
        sid, _, v = line.rstrip("\n").partition("\t")
        rows.append((sid, [w.strip() for w in v.strip().strip("{}").split(",") if w.strip()]))
    atoms = sorted({k.lower() for _, ks in rows for k in ks})
    print(f"{len(rows):,} samples, {sum(len(k) for _, k in rows):,} keyword instances, "
          f"{len(atoms):,} distinct keywords")

    terms = load_terms(f"{root}/ontology_terms.tsv.gz")
    vocab = slot_vocabulary(root, terms)
    labels = list(vocab["label"])
    match_on = labels if a.term_text == "label" else list(vocab["text"])

    if a.dry_run:
        n1, _ = estimate_tokens(atoms, a.model)
        n2, _ = estimate_tokens(match_on, a.model)
        print(f"~{n1:,} tokens for the keywords, ~{n2:,} for the vocabulary")
        print("first 10 vocabulary labels:", labels[:10])
        return

    client = None
    if not a.no_embed:
        from openai import OpenAI
        client = OpenAI(api_key=open(f"{root}/my_api_key_embeddings").read().strip())
    K = vectors_for(atoms, kw_h5, client, a.model, a.embedding_dim, "keyword atoms")
    T = vectors_for(match_on, slot_h5, client, a.model, a.embedding_dim, "slot vocabulary")

    best = {}
    for s in range(0, len(K), 1024):
        S = K[s:s + 1024] @ T.T
        j = S.argmax(1)
        for r, (jj, atom) in enumerate(zip(j, atoms[s:s + 1024])):
            best[atom] = (labels[jj], float(S[r, jj]))

    sims = np.array([s for _, s in best.values()])
    print(f"cosine to nearest slot term: p10 {np.quantile(sims,.1):.3f} "
          f"median {np.median(sims):.3f} p90 {np.quantile(sims,.9):.3f}")
    print(f"distinct keywords at or above --min_sim {a.min_sim}: "
          f"{(sims >= a.min_sim).sum():,} of {len(sims):,} "
          f"({100*(sims >= a.min_sim).mean():.1f}%)")

    kept = Counter()
    with open(f"{lat}/GPT_keywords_{out_tag}.txt", "w", encoding="utf-8") as out:
        for sid, kws in rows:
            canon = []
            for k in kws:
                lab, sim = best[k.lower()]
                v = lab if sim >= a.min_sim else k
                kept["mapped" if sim >= a.min_sim else "kept as written"] += 1
                if v not in canon:
                    canon.append(v)
            out.write(f"{sid}\t{{{', '.join(canon)}}}\n")
    tot = sum(kept.values())
    for k, v in kept.most_common():
        print(f"  {k}: {v:,} ({100*v/tot:.1f}%)")
    print(f"wrote {lat}/GPT_keywords_{out_tag}.txt")

    if a.review:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(atoms), min(a.review, len(atoms)), replace=False)
        print(f"\nREVIEW - {len(pick)} random keywords. Check these before trusting any metric.")
        for i in sorted(pick):
            lab, sim = best[atoms[i]]
            mark = "  " if sim >= a.min_sim else "..."   # ... = kept as written
            print(f"  {mark} {atoms[i][:34]:<34} {sim:.3f}  {lab}")


if __name__ == "__main__":
    main()
