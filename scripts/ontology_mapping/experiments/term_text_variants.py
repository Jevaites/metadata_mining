#!/usr/bin/env python3
"""
Does richer term text make better term embeddings? Tests what goes into the text that represents
an ontology term before it is embedded (today: common.term_text = 'label; synonym; synonym').

Variants (one text per term and variant):
  label                 label
  label_exact           label; EXACT synonyms only (needs --obo: the scope is not in the term table)
  label_syn             label; all synonyms              <- current (common.term_text)
  label_syn_def         label; all synonyms. definition
  label_syn_parents     label; all synonyms. Is a: parent label; parent label
  label_syn_def_parents label; all synonyms. definition. Is a: parent labels

Three steps:
  build     write the variant texts (TSV: term_id, variant, text)
  embed     embed every distinct text into one .h5 (resumable; the current term .h5 is copied
            in first, so label_syn costs nothing). Needs the OpenAI API: run it on your Mac.
  evaluate  for each variant, the term-vector methods of 5_evaluate.py (retrieval,
            retrieval_open, retrieval_prior, prototype, prototype_open with an unseen-term bonus,
            label_reg), plus zero-shot retrieval with the sub-biome alone as the query, on
            fold seeds 0-4; writes JSON + a summary table.

cd scripts/ontology_mapping
P=~/MicrobeAtlasProject; X=$P/ontology_mapping/experiments/term_text
python experiments/term_text_variants.py build --obo ENVO=envo.obo UBERON=uberon.obo --output $X/variants.tsv.gz
python experiments/term_text_variants.py embed --variants $X/variants.tsv.gz --output $X/term_variants.h5 --dry_run
python experiments/term_text_variants.py embed --variants $X/variants.tsv.gz --output $X/term_variants.h5
python experiments/term_text_variants.py evaluate --variants $X/variants.tsv.gz --vectors $X/term_variants.h5 --output $X/results.json
"""
import argparse
import importlib
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # scripts/ontology_mapping
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # scripts/
from common import SLOTS, load_npz, load_terms, path, read_tsv, select_samples, study_folds  # noqa: E402
from _setup import DEFAULTS  # noqa: E402

from methods import prototype_model  # noqa: E402
VARIANTS = ["label", "label_exact", "label_syn", "label_syn_def", "label_syn_parents", "label_syn_def_parents"]
SYNONYM = re.compile(r'^synonym: "((?:[^"\\]|\\.)*)" (EXACT|NARROW|BROAD|RELATED)')


def exact_synonyms(sources):
    """{term_id: [EXACT synonyms]} from OBO files given as PREFIX=path_or_url."""
    sys.path.insert(0, os.path.dirname(HERE))
    read_obo = importlib.import_module("1_build_term_index").read_obo
    exact, term = {}, None
    for pair in sources:
        _, source = pair.split("=", 1)
        for line in read_obo(source):
            if line.startswith("["):
                term = None
            elif line.startswith("id: "):
                term = line[4:].strip().replace(":", "_")
                exact.setdefault(term, [])
            elif term and (m := SYNONYM.match(line)) and m.group(2) == "EXACT":
                exact[term].append(m.group(1).replace('\\"', '"'))
    return exact


def build(args):
    terms = load_terms(args.ontology_terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    exact = exact_synonyms(args.obo) if args.obo else None
    rows = []
    for t in terms.itertuples():
        syn = [s for s in t.synonyms.split("||") if s]
        parents = [label_of[p] for p in t.parents.split("||") if p in label_of]
        # label_syn stays byte-identical to common.term_text (so its vectors are reused, trailing
        # "; " included for terms without synonyms); the other variants start from a clean base
        base = "; ".join([t.label] + syn)
        definition = f"{t.definition.strip().rstrip('.')}." if t.definition.strip() else ""
        is_a = f"Is a: {'; '.join(parents)}." if parents else ""
        texts = {
            "label": t.label,
            "label_syn": t.text,
            "label_syn_def": " ".join(filter(None, [base + ".", definition])),
            "label_syn_parents": " ".join(filter(None, [base + ".", is_a])),
            "label_syn_def_parents": " ".join(filter(None, [base + ".", definition, is_a])),
        }
        if exact is not None:
            texts["label_exact"] = "; ".join(dict.fromkeys([t.label] + exact.get(t.term_id, [])))  # OBO repeats some
        rows += [(t.term_id, v, x) for v, x in texts.items()]
    out = pd.DataFrame(rows, columns=["term_id", "variant", "text"])
    if exact is not None:
        missing = sum(t not in exact for t in terms["term_id"])
        print(f"exact synonyms: {missing} of {len(terms)} terms not found in the OBO files (their label is used alone)")
        n_all = terms.set_index("term_id")["synonyms"].str.count(r"\|\|").add(1).where(terms.set_index("term_id")["synonyms"] != "", 0)
        n_exact = pd.Series({t: len(v) for t, v in exact.items()})
        print(f"synonyms per term: all {n_all.mean():.2f}, EXACT {n_exact.reindex(n_all.index).fillna(0).mean():.2f}")
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    out.to_csv(path(args.output), sep="\t", index=False)
    print(out.groupby("variant")["text"].agg(n="size", distinct="nunique", median_chars=lambda s: int(s.str.len().median())))
    print(f"wrote {args.output}")


def embed(args):
    from embed_subbiomes_keywords import PRICE_PER_1M_TOKENS, embed_unique, estimate_tokens
    texts = list(dict.fromkeys(read_tsv(args.variants)["text"]))
    out = path(args.output)
    seed = path(args.term_vectors) if args.term_vectors else None  # the current label_syn texts are embedded there
    start_from = out if os.path.exists(out) else seed if seed and os.path.exists(seed) else None
    done = set()
    if start_from:
        import h5py
        with h5py.File(start_from, "r") as handle:
            done = {t.decode() if isinstance(t, bytes) else t for t in handle["texts"][:]}
    todo = [t for t in texts if t not in done]
    n_tokens, how = estimate_tokens(todo, args.model)
    print(f"{len(texts)} distinct texts, {len(texts) - len(todo)} already embedded, {len(todo)} to embed: "
          f"{n_tokens:,} tokens ({how}) = ${n_tokens / 1e6 * PRICE_PER_1M_TOKENS[args.model]:.2f}")
    if args.dry_run or not todo:
        return
    if start_from == seed:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.copy(seed, out)
    from openai import OpenAI
    client = OpenAI(api_key=open(path(args.api_key_path)).read().strip(), max_retries=8)
    # 512 per call: definitions make some texts long, and a call is limited in total tokens
    embed_unique("term_variants", texts, out, client, args.model, args.embedding_dim, 512)


def evaluate_variants(args):
    """For every fold seed, fold and slot, score every variant on the same split. label_reg's ridge
    (as methods.label_regression: sklearn Ridge, alpha 10, with intercept) is solved in closed
    form once per split and reused for every variant, since only the targets change."""
    import h5py
    terms = load_terms(args.ontology_terms)
    term_ids = terms["term_id"].to_numpy()
    row = {t: i for i, t in enumerate(term_ids)}
    variants = pd.concat([read_tsv(v) for v in args.variants], ignore_index=True)
    variants["block"] = variants["block"].fillna("both").replace("", "both") if "block" in variants else "both"
    with h5py.File(path(args.vectors), "r") as handle:
        row_of = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        vectors = handle["embeddings"][:]
    names = [v for v in VARIANTS if v in set(variants["variant"])]
    names += [v for v in dict.fromkeys(variants["variant"]) if v not in names]
    if args.only:
        names = [v for v in names if v in args.only]
    T_of, TM_of = {}, {}  # one-block term vectors (label_reg, sub-biome query) / two-block term side
    for variant in names:
        v = variants[variants.variant == variant]
        missing = sorted(set(v.loc[~v["text"].isin(row_of), "term_id"]) | (set(term_ids) - set(v["term_id"])))
        if missing:
            print(f"{variant}: {len(missing)} terms without an embedded text, skipped (run the embed step)")
            continue
        blocks = {}
        for block, g in v.groupby("block"):  # several texts per term (LLM examples) are averaged
            vec = pd.DataFrame(normalize(vectors[[row_of[t] for t in g["text"]]]))
            mean = vec.groupby(g["term_id"].to_numpy()).mean().reindex(term_ids)
            blocks[block] = normalize(mean.to_numpy())
        if "both" in blocks:
            T_of[variant] = blocks["both"]
            TM_of[variant] = np.hstack([blocks["both"]] * 2) / np.sqrt(2)
        else:  # separate keyword and sub-biome blocks, like the sample side [kw, sb]
            T_of[variant] = normalize(blocks["kw"] + blocks["sb"])
            TM_of[variant] = np.hstack([blocks["kw"], blocks["sb"]]) / np.sqrt(2)
    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kr), set(sr)])
    SB = B[[sr[s] for s in samples["sample_id"]]]
    X = np.hstack([K[[kr[s] for s in samples["sample_id"]]], SB]) / np.sqrt(2)
    print(f"{len(samples)} samples; variants: {', '.join(T_of)}", flush=True)
    bonus = args.bonus
    hits = {v: {slot: {} for slot in SLOTS} for v in T_of}
    for seed in range(args.fold_seeds):
        for tr, te in study_folds(samples["study_code"], 5, seed):
            for slot in SLOTS:
                y = samples[slot].to_numpy()
                a, b = tr[y[tr] != ""], te[y[te] != ""]
                closed = np.where(np.isin(term_ids, y[a]))[0]
                all_rows = np.arange(len(term_ids))
                unseen = ~np.isin(y[b], y[a])
                x_mean = X[a].mean(0)
                Xc = X[a] - x_mean
                ridge = np.linalg.solve(Xc.T @ Xc + 10.0 * np.eye(X.shape[1]), Xc.T)  # (dim x n_train)
                for variant, T in T_of.items():
                    TM = TM_of[variant]  # term side, weighted as the two sample blocks

                    def proto(vocab_rows, alpha, bonus=0.0):
                        mean, P, bias = prototype_model(X[a], y[a], TM[vocab_rows], term_ids[vocab_rows],
                                                                 alpha, 0.1, bonus)
                        return term_ids[vocab_rows][(normalize(X[b] - mean) @ P.T + bias).argmax(1)]
                    Y = T[[row[v] for v in y[a]]]
                    W = ridge @ (Y - Y.mean(0))
                    reg = (X[b] - x_mean) @ W + Y.mean(0)
                    pred = {
                        "retrieval": term_ids[closed][(X[b] @ TM[closed].T).argmax(1)],
                        "retrieval_sub_biome_query": term_ids[closed][(SB[b] @ T[closed].T).argmax(1)],
                        "retrieval_open": term_ids[(X[b] @ TM.T).argmax(1)],
                        "retrieval_prior": proto(closed, 0.0),
                        "prototype": proto(closed, 0.5),
                        f"prototype_open_bonus{bonus}": proto(all_rows, 0.5, bonus),
                        "label_reg": term_ids[closed][(normalize(reg) @ T[closed].T).argmax(1)],
                    }
                    for m, p in pred.items():
                        h = p == y[b]
                        hits[variant][slot].setdefault(m, []).append((h, y[b]))
                        if m in ("retrieval_open", f"prototype_open_bonus{bonus}"):
                            hits[variant][slot].setdefault(m + " | unseen labels", []).append((h[unseen], y[b][unseen]))
        print(f"fold seed {seed} done", flush=True)

    def summary(h, macro):
        """pooled over the 5 folds of a seed (each seed covers every sample once), mean over seeds;
        macro = mean of the per-label accuracies (every label counts once, however frequent)."""
        per_seed = []
        for i in range(args.fold_seeds):
            hit = np.concatenate([x for x, _ in h[i * 5:(i + 1) * 5]])
            gold = np.concatenate([g for _, g in h[i * 5:(i + 1) * 5]])
            per_seed.append(pd.Series(hit).groupby(gold).mean().mean() if macro else hit.mean())
        return round(float(np.mean(per_seed)), 4)
    results = {v: {slot: {**{m: summary(h, False) for m, h in by_slot.items()},
                          **{f"{m} | macro": summary(h, True) for m, h in by_slot.items() if "unseen" not in m}}
                   for slot, by_slot in by_variant.items()} for v, by_variant in hits.items()}
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    with open(path(args.output), "w") as handle:
        json.dump(results, handle, indent=1)
    methods = list(next(iter(results.values()))["biome"])
    print(f"\ntop-1, mean over fold seeds 0-{args.fold_seeds - 1}; biome / feature / material")
    for m in methods:
        print(f"\n{m}")
        for v, r in results.items():
            print(f"  {v:22s} " + " / ".join(f"{r[s][m]:.3f}" for s in SLOTS))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ontology_terms", default=DEFAULTS["ontology_terms"])
    b.add_argument("--obo", nargs="*", default=[], help="PREFIX=path_or_url, the OBO files of step 1 (for EXACT synonyms)")
    b.add_argument("--output", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--variants", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--term_vectors", default=DEFAULTS["term_vectors"], help="current term .h5, copied in first")
    e.add_argument("--model", default="text-embedding-3-large")
    e.add_argument("--embedding_dim", type=int, default=1024)
    e.add_argument("--api_key_path", default="~/MicrobeAtlasProject/my_api_key_embeddings")
    e.add_argument("--dry_run", action="store_true")
    v = sub.add_parser("evaluate")
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        v.add_argument(f"--{name}", default=DEFAULTS[name])
    v.add_argument("--variants", nargs="+", required=True, help="one or more variants TSVs (build / term_keywords.py build)")
    v.add_argument("--only", nargs="*", default=None, help="evaluate only these variants")
    v.add_argument("--vectors", required=True)
    v.add_argument("--fold_seeds", type=int, default=5)
    v.add_argument("--bonus", type=float, default=0.5, help="unseen-term bonus for prototype_open")
    v.add_argument("--output", required=True)
    args = p.parse_args()
    {"build": build, "embed": embed, "evaluate": evaluate_variants}[args.step](args)


if __name__ == "__main__":
    main()
