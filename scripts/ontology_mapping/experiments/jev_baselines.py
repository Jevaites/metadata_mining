#!/usr/bin/env python3
"""
Fair baselines for the Jev whole-vocabulary Choice (README 7d, claude/jev-pilot-results.md).

The pilot's prototype ranked the labels of its *fold's* training set; Jev chose among the slot labels used by any
*other study*. To compare like with like, this scores three of our own models over exactly Jev's vocabulary, trained
on every other study (leave-one-study-out, the same rule that defines Jev's vocabulary), on the same pilot samples:
  prototype         alpha 0.5, beta 0.1 (the pipeline's model, methods.prototype_model, here leave-one-study-out)
  retrieval_prior   alpha 0, beta 0.1: centred term vectors + log label frequency (no training centroids)
  retrieval         alpha 0, beta 0: centred term vectors only (zero-shot, like Jev: no label frequencies)
It reports top-1 exact, recall@3 / @5 / @10, and the candidate recall when each source (these three and Jev) adds
its 3 best terms that are not already among the pilot's candidates (top-5 + broader terms), so that every source
adds the same number of terms. The pilot's own (fold-trained) prototype is reported from the pilot file.

cd scripts/ontology_mapping; X=~/MicrobeAtlasProject/ontology_mapping/experiments
python3 experiments/jev_baselines.py --pilot $X/rerank/pilot_k5anc.jsonl --inputs $X/jev/inputs.json --flat $X/jev/flat.jsonl \
    --label_map ~/MicrobeAtlasProject/metalog/clean/previous_2026-10-01/biome_label_map.tsv --output $X/jev/baselines.json
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, path  # noqa: E402
import jev_pilot as jp  # noqa: E402

MODELS = {"prototype": (0.5, 0.1), "retrieval_prior": (0.0, 0.1), "retrieval": (0.0, 0.0)}


def load_training(args):
    """The training samples (training set + label map, those with both embeddings), their vectors and the term
    vectors. -> namespace(S, X, studies, vec(ids) -> vectors, terms, trow, TM)."""
    from common import load_npz, load_terms, read_tsv, select_samples
    import hierarchical_backoff as hb
    terms = load_terms(args.ontology_terms)
    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    S = select_samples(args.samples, [set(kr), set(sr)])
    if args.label_map:
        lm = read_tsv(args.label_map)
        for slot in SLOTS:
            m = {r.from_id.strip(): r.to_id.strip() for r in lm.itertuples() if r.slot.strip() in (slot, "*")}
            S[slot] = S[slot].map(lambda t: m.get(t, t))
    vec = lambda ids: (np.hstack([K[[kr[s] for s in ids]], B[[sr[s] for s in ids]]]) / np.sqrt(2)).astype(np.float32)
    return SimpleNamespace(S=S, X=vec(S["sample_id"]), studies=S["study_code"].to_numpy(), vec=vec, terms=terms,
                           trow={t: i for i, t in enumerate(terms["term_id"])},
                           TM=hb.term_matrix(terms, args.term_vectors).astype(np.float32))


def loso_rankings(T, slot, ps, ctx, top=30):
    """Rank Jev's vocabulary for each pilot sample of `slot` with the three models of MODELS, trained on every other
    study. -> {model: {sample_id: [term, ...] best first}}.

    methods.prototype_model, leave-one-study-out by subtraction: per (label, study) sums of the centred, normalised
    training samples, computed once; a group's sums = all - its own study. The centring mean is the slot's overall
    training mean (leaving one study out of 539 moves it negligibly)."""
    from sklearn.preprocessing import normalize
    X, studies, y = T.X, T.studies, T.S[slot].to_numpy()
    ranked = {name: {} for name in MODELS}
    lab = y != ""
    mean = X[lab].mean(axis=0)
    Z = normalize(X[lab] - mean)
    ylab, slab = y[lab], studies[lab]
    labels = sorted(set(ylab))
    li = {t: i for i, t in enumerate(labels)}
    pairs = {}
    for i, (t, st) in enumerate(zip(ylab, slab)):
        pairs.setdefault((li[t], st), []).append(i)
    total = np.zeros((len(labels), X.shape[1]), np.float32)
    count = np.zeros(len(labels))
    own = {}
    for (l, st), rows in pairs.items():
        v = Z[rows].sum(axis=0)
        total[l] += v
        count[l] += len(rows)
        own.setdefault(st, []).append((l, v, len(rows)))
    a, b = MODELS["prototype"]
    for study in sorted({o["study"] for o in ps}):
        group = [o for o in ps if o["study"] == study]
        vocab = ctx.vocab(slot, study)[0]
        sums, n_lab = total.copy(), count.copy()
        for l, v, n in own.get(study, []):
            sums[l] -= v
            n_lab[l] -= n
        rows = [li[v] for v in vocab]
        n = n_lab[rows]
        P0 = normalize(T.TM[[T.trow[v] for v in vocab]] - T.TM[[T.trow[v] for v in vocab]].mean(axis=0))  # alpha 0
        P = P0.copy()
        seen = n > 0
        P[seen] = normalize(a * normalize(sums[rows][seen]) + (1 - a) * P0[seen])
        bias = b * np.log((n + 0.5) / (n + 0.5).sum())
        Xte = normalize(T.vec([o["sample_id"] for o in group]) - mean)
        scores = {"prototype": Xte @ P.T + bias, "retrieval_prior": Xte @ P0.T + bias, "retrieval": Xte @ P0.T}
        for name, sc in scores.items():
            for o, row in zip(group, sc):
                ranked[name][o["sample_id"]] = [vocab[i] for i in np.argsort(-row)[:top]]
    return ranked


def main():
    from _setup import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--flat", required=True, help="jev_pilot.py run --arm flat output")
    p.add_argument("--label_map", default=None)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes", "term_vectors"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--add", type=int, default=3, help="new terms each source adds to the candidates")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    t0 = time.time()
    pilot = jp.read_jsonl(args.pilot)
    flat = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(args.flat)}
    ctx = jp.Ctx(json.load(open(path(args.inputs))), SimpleNamespace(min_studies=1))
    T = load_training(args)
    print(f"{len(T.S)} training samples, {len(pilot)} pilot samples ({time.time() - t0:.0f}s)", flush=True)
    report = {}
    for slot in SLOTS:
        ps = [o for o in pilot if o["slot"] == slot and (slot, o["sample_id"]) in flat]
        ranked = loso_rankings(T, slot, ps, ctx)
        for o in ps:  # Jev: its vocabulary Choice, best first
            P = flat[(slot, o["sample_id"])]["vocab"]
            ranked.setdefault("jev_vocab", {})[o["sample_id"]] = sorted(P, key=lambda t: -P[t])[:30]
        gold = {o["sample_id"]: o["gold"] for o in ps}
        cands = {o["sample_id"]: [c["term_id"] for c in o["candidates"]] for o in ps}
        res = {"n": len(ps), "gold in Jev's vocabulary": round(float(np.mean([g in ctx.vocab(slot, o["study"])[0] for o, g in zip(ps, gold.values())])), 3),
               "pilot prototype (fold-trained) recall@3 / @5 / candidates": [
                   round(float(np.mean([gold[s] in cands[s][:n] for s in gold])), 3) for n in (3, 5)] +
                   [round(float(np.mean([gold[s] in cands[s] for s in gold])), 3)]}
        for name, r in ranked.items():
            res[name] = {"exact": round(float(np.mean([r[s][0] == gold[s] for s in gold])), 3),
                         "recall@3 / @5 / @10": [round(float(np.mean([gold[s] in r[s][:n] for s in gold])), 3) for n in (3, 5, 10)],
                         f"candidates + {args.add} new terms: recall": round(float(np.mean(
                             [gold[s] in cands[s] or gold[s] in [t for t in r[s] if t not in cands[s]][:args.add] for s in gold])), 3)}
        st = {o["sample_id"]: o["study"] for o in ps}
        ids = list(gold)
        hit = {name: [gold[s] in cands[s] or gold[s] in [t for t in r[s] if t not in cands[s]][:args.add] for s in ids]
               for name, r in ranked.items()}
        for name in MODELS:  # paired, study bootstrap: Jev's addition vs this source's
            res[f"jev_vocab vs {name}: recall difference (95% CI)"] = jp.boot_diff(hit["jev_vocab"], hit[name], [st[s] for s in ids])
        report[slot] = res
        print(f"\n== {slot} (n={len(ps)}, {time.time() - t0:.0f}s)")
        for k, v in res.items():
            print(f"  {k:55} {v}")
    if args.output:
        json.dump(report, open(path(args.output), "w"), indent=1)


if __name__ == "__main__":
    main()
