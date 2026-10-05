#!/usr/bin/env python3
"""
Draft a label map for the biome slot, for review by a curator.

MIxS asks for an ENVO biome (ENVO:00000428 or below) in env_broad_scale, but ~45 % of Metalog's
biome labels are other kinds of terms: environmental systems (animal-associated environment),
features (lentic water body, hospital), materials (soil, lake sediment), processes. This script
proposes, for each such term, the ENVO biome to use instead, with the evidence behind it:

  knn       samples from OTHER studies with a real biome label vote for the biome of the term's
            samples (k nearest neighbours by keyword + sub-biome embedding, one vote per study, as
            knn_study): "what did other curators call samples like these?"
  model     a prototype model trained only on real-biome labels, averaged over the term's samples
  terms     the ENVO biome terms (any, not only the ones Metalog uses) whose label embedding is
            closest to the term's label

Proposal: knn and model agree -> that biome, or the closest ENVO biome term by label when it lies
under it with label cosine >= --min_cos (more specific, same branch); otherwise a 2-of-3 vote of
knn, model and closest term; if knn and model are in one branch, the broader of the two; otherwise
the better-supported of the two (vote share vs mean probability). Anything but agreement is
`review`. Kept as conventions (not remapped):
animal-associated environment, plant-associated environment (host-associated samples have no
ENVO biome). The slot root 'biome' itself gets an empty to_id: it says nothing and is dropped.

Output (TSV): the first four columns are the label-map format of 2b_clean_metalog.py --label_map
(slot, from_id, to_id, reason); the other columns are evidence for the reviewer. Edit to_id (or set
it equal to from_id to keep a term) and fill `review_decision` / `review_notes`.

cd scripts/ontology_mapping
python3 experiments/draft_biome_label_map.py --output ~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import importlib  # noqa: E402
from common import ancestor_sets, load_npz, load_terms, path, select_samples  # noqa: E402
from _setup import DEFAULTS  # noqa: E402

evaluate = importlib.import_module("5_evaluate")
hb = importlib.import_module("hierarchical_backoff")
BIOME_ROOT = "ENVO_00000428"
KEEP = {"ENVO_01001002": "animal-associated environment", "ENVO_01001001": "plant-associated environment"}
# best-effort corrections where the automatic evidence is clearly wrong (still to be reviewed)
MANUAL = {"ENVO_00005797": ("ENVO_01000252", "lake bottom mud: a freshwater lake material, knn votes came from marine studies"),
          "ENVO_00003861": ("ENVO_01000219", "industrial building: no climate evidence; samples are soil at an industrial site")}
KIND_ROOTS = {"ENVO_00010483": "material", "ENVO_01000254": "environmental system",
              "ENVO_01000813": "astronomical body part (feature)", "ENVO_00000070": "human construction",
              "ENVO_02500000": "process", "BFO_0000015": "process"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--term_vectors", default="~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5",
                   help="h5 with the plain-label and label_syn term texts (term_text_variants.py embed)")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--min_cos", type=float, default=0.6, help="label cosine needed to prefer a more specific biome term")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    terms = load_terms(args.ontology_terms)
    ids = terms["term_id"].to_numpy()
    label = dict(zip(ids, terms["label"]))
    parents = {t: set(x.split("||")) for t, x in zip(ids, terms["parents"]) if x}
    anc = ancestor_sets(parents)
    is_biome = lambda t: t != BIOME_ROOT and BIOME_ROOT in anc.get(t, ())

    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    S = select_samples(args.samples, [set(kr), set(sr)], max_per_study=0)
    S = S[S["biome"] != ""].reset_index(drop=True)
    X = np.hstack([K[[kr[s] for s in S["sample_id"]]], B[[sr[s] for s in S["sample_id"]]]]) / np.sqrt(2)
    y, study = S["biome"].to_numpy(), S["study_code"].to_numpy()
    good = np.array([is_biome(t) or t in KEEP for t in y])  # labels that stay as they are
    targets = sorted(set(y[~good]) - {BIOME_ROOT})
    print(f"{len(S)} samples with a biome label, {good.mean():.1%} already a biome or a kept convention; "
          f"{len(targets)} other terms to review, plus the root 'biome' ({(y == BIOME_ROOT).sum()} samples)")

    # model trained on real biome labels only
    vocab = np.unique(y[good])
    TM = hb.term_matrix(terms, args.term_vectors)
    trow = {t: i for i, t in enumerate(ids)}
    mean, P, bias = evaluate.prototype_model(X[good], y[good], TM[[trow[v] for v in vocab]], vocab, 0.5, 0.1)
    Sg = normalize(X[good] - mean) @ P.T + bias
    T = hb.fit_temperature([(Sg, np.searchsorted(vocab, y[good]))])

    # plain-label vectors of all ENVO biome terms (also the ones Metalog never uses)
    import h5py
    with h5py.File(path(args.term_vectors), "r") as handle:
        row = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        V = handle["embeddings"][:]
    biome_terms = [t for t in ids if is_biome(t) and label[t] in row]
    BV = normalize(V[[row[label[t]] for t in biome_terms]])

    good_idx = np.where(good)[0]
    rows = []
    for t in [BIOME_ROOT] + targets:
        m = np.where(y == t)[0]
        studies = Counter(study[m])
        kind = ", ".join(sorted({v for r, v in KIND_ROOTS.items() if r in anc.get(t, ())})) or "other"
        rec = {"slot": "biome", "from_id": t, "from_label": label.get(t, t), "n_samples": len(m),
               "n_studies": len(studies), "studies": "; ".join(f"{s} ({c})" for s, c in studies.most_common(4)),
               "kind": kind}
        if t == BIOME_ROOT:
            rec.update(to_id="", to_label="", reason="slot root: uninformative, dropped", status="drop")
            rows.append(rec)
            continue
        # knn over other studies' real-biome samples, one vote per study
        votes = Counter()
        cand = good_idx
        for start in range(0, len(m), 500):
            q = m[start:start + 500]
            sim = X[q] @ X[cand].T
            for i, s in enumerate(q):
                ok = study[cand] != study[s]
                sims = np.where(ok, sim[i], -np.inf)
                nn = cand[np.argpartition(-sims, args.k)[:args.k]]
                per = Counter(study[nn])  # one vote per study, then each sample counts once
                for j in nn:
                    votes[y[j]] += 1 / per[study[j]] / len(per) / len(m)
        knn = votes.most_common(2)
        # model
        pr = hb.softmax(normalize(X[m] - mean) @ P.T + bias, T).mean(0)
        top = np.argsort(-pr)[:2]
        model = [(vocab[i], float(pr[i])) for i in top]
        # nearest ENVO biome terms by label
        tv = normalize(V[[row[label[t]]]]) if label.get(t) in row else None
        cos = None if tv is None else BV @ tv[0]
        near = [] if tv is None else [biome_terms[i] for i in np.argsort(-cos)[:3]]
        near_cos = [] if tv is None else [float(cos[i]) for i in np.argsort(-cos)[:3]]
        # proposal: knn and model agree -> that biome, or the closest ENVO biome term by label when
        # it lies under it and its label is close (cosine >= MIN_COS: more specific, same branch);
        # else a 2-of-3 vote (knn, model, closest term); else the model's choice
        k1, ks = knn[0] if knn else ("", 0)
        m1 = model[0][0]
        n1, c1 = (near[0], near_cos[0]) if near else ("", 0.0)
        if k1 == m1:
            more = n1 and n1 != k1 and k1 in anc.get(n1, ()) and c1 >= args.min_cos
            to, status = (n1, "agree, more specific") if more else (k1, "agree")
        elif n1 in (k1, m1):
            to, status = n1, "review (2 of 3 sources)"
        elif m1 in anc.get(k1, ()) or k1 in anc.get(m1, ()):  # same branch: the broader one is safe
            to, status = (m1 if m1 in anc.get(k1, ()) else k1), "review (broader of knn / model)"
        else:  # the better-supported source
            to, status = (k1 if ks >= model[0][1] else m1), "review"
        if t in MANUAL:
            to, status = MANUAL[t][0], "manual (Claude): " + MANUAL[t][1]
        rec.update(to_id=to, to_label=label.get(to, ""), status=status,
                   reason=f"not an ENVO biome ({kind}); biome used by other studies for similar samples",
                   knn_top=f"{label.get(k1, '')} {ks:.0%}",
                   knn_second=f"{label.get(knn[1][0], '')} {knn[1][1]:.0%}" if len(knn) > 1 else "",
                   model_top=f"{label[model[0][0]]} {model[0][1]:.0%}", model_second=f"{label[model[1][0]]} {model[1][1]:.0%}",
                   nearest_biome_terms="; ".join(f"{label[n]} {c:.2f}" for n, c in zip(near, near_cos)),
                   feature_slot_top="; ".join(f"{label.get(v, v) or '(empty)'} {c / len(m):.0%}" for v, c in Counter(S['feature'].to_numpy()[m]).most_common(2)),
                   material_slot_top="; ".join(f"{label.get(v, v) or '(empty)'} {c / len(m):.0%}" for v, c in Counter(S['material'].to_numpy()[m]).most_common(2)),
                   example_text=str(S["text"].iloc[m[0]])[:160].replace("\t", " ").replace("\n", " "))
        rows.append(rec)
    cols = ["slot", "from_id", "to_id", "reason", "from_label", "to_label", "status", "n_samples", "n_studies",
            "kind", "knn_top", "knn_second", "model_top", "model_second", "nearest_biome_terms",
            "feature_slot_top", "material_slot_top", "studies", "example_text"]
    out = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    out["review_decision"] = ""
    out["review_notes"] = ""
    out = pd.concat([out.iloc[:1], out.iloc[1:].sort_values("n_samples", ascending=False)])
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    out.to_csv(path(args.output), sep="\t", index=False)
    print(out["status"].value_counts().to_string())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
