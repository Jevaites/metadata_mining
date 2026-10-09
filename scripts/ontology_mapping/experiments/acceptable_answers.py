#!/usr/bin/env python3
"""
Acceptable answers: how many top-1 "errors" are a defensible alternative ontology term, and how to score them
(project doc claude/label-equivalence-and-acceptable-answers.md; experiments README §15).

1. Curator disagreement. "Near-duplicates" are pairs of samples from different studies whose sample vectors
   [keywords, sub-biome] / sqrt(2) have cosine >= --dup_threshold. Where the two labels of a slot differ, the label
   pair is recorded with its signals: is_a relation, direct (non-is_a) ENVO relation, term-embedding cosine, number
   of sample pairs and of distinct study pairs. Each pair gets a kind:
     equivalent   two names for one thing          rhizosphere / rhizosphere environment
     broader      one term is an is_a ancestor     marine biome / ocean biome
     facet        two true facets of one sample    ocean / marine photic zone, river / river water
     different    different facts                  surface water / deep chlorophyll maximum layer
   from --pair_kinds (a reviewed or drafted TSV) when the pair is listed there, else an automatic guess
   (is_a related -> broader, automatic synonym -> equivalent, direct ENVO relation -> facet, else different).
   -> label_pairs.tsv, composition.tsv
2. Automatic synonym candidates among the labels in use: term-embedding cosine >= --syn_cosine, or the same label
   once generic head words (environment, material, biome, zone, layer) are removed.  -> synonym_candidates.tsv
3. Acceptance rules applied to the cross-validated top-1 of --method (5_evaluate.py predictions), each alone and
   combined per sample (logical OR, never summed):
     exact       top-1 == gold
     broader     top-1 is an is_a ancestor of gold (any depth)
     narrower    top-1 is an is_a descendant of gold (any depth)
     synonyms    {gold, top-1} is an automatic synonym pair (step 2)
     co-labels   {gold, top-1} was used for near-duplicates in >= --min_study_pairs distinct study pairs, counting
                 only pairs of training studies of the sample's fold (so the test study never vouches for itself)
     facets      gold and top-1 are directly linked by a non-is_a ENVO relation (either direction) and not is_a related
   Metrics: micro (share of samples), macro (mean over gold labels), macro over learnable labels (used by >= 2
   studies, so they can be in training when tested).  -> rule_scores.tsv, overlap.tsv
4. Graded credit for broader answers: exact; depth (0.5 ** is_a steps); information ratio IC(answer) / IC(gold), with
   IC(t) = -log(share of the slot's labelled samples labelled t or below t); full credit; hierarchical F (overlap of
   the ancestor-or-self sets).  -> graded_credit.tsv

Example (the defaults are the pipeline's paths; ~1-2 min):
python experiments/acceptable_answers.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --envo_obo ~/MicrobeAtlasProject/ontologies/envo.obo \
  --pair_kinds experiments/label_pair_kinds_draft.tsv \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/acceptable_answers
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from _setup import DEFAULTS, SLOTS, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds
from common import ancestor_distances, read_tsv, term_ancestors

KINDS = ["equivalent", "broader", "facet", "different"]
GENERIC_HEADS = re.compile(r"\b(environment|material|biome|zone|layer)\b")


# ----------------------------------------------------------------------------- inputs
def envo_relations(obo_path):
    """{term_id: {(relation id, target term_id)}} from the `relationship:` lines of an OBO file (is_a excluded).
    Example line in [Term] ENVO:00000022 (river): 'relationship: RO:0001025 ENVO:...' -> ('RO:0001025', 'ENVO_...')."""
    relations, current = defaultdict(set), None
    with open(path(obo_path)) as handle:
        for line in handle:
            if line.startswith("[Term]"):
                current = None
            elif line.startswith("[Typedef]"):
                current = None  # relation definitions, not terms
            elif line.startswith("id: "):
                current = line[4:].strip().replace(":", "_")
            elif line.startswith("relationship: ") and current:
                relation, target = line.split()[1:3]
                relations[current].add((relation, target.replace(":", "_")))
    return relations


def near_duplicate_pairs(x, studies, threshold, block=2000):
    """(i, j) index arrays of sample pairs from different studies with cosine >= threshold (i < j).
    x is L2-normalised, so x @ x.T is the cosine; computed in row blocks to bound memory."""
    rows, cols = [], []
    for start in range(0, len(x), block):
        r, c = np.nonzero(x[start:start + block] @ x.T >= threshold)
        r += start
        keep = (r < c) & (studies[r] != studies[c])
        rows.append(r[keep])
        cols.append(c[keep])
    return np.concatenate(rows), np.concatenate(cols)


# ----------------------------------------------------------------------------- label-pair signals
def isa_relation(a, b, anc, parents):
    """'a_broader' / 'b_broader' (one is an is_a ancestor of the other), 'siblings' (shared direct parent) or ''."""
    if a in anc.get(b, ()):
        return "a_broader"
    if b in anc.get(a, ()):
        return "b_broader"
    return "siblings" if parents.get(a, set()) & parents.get(b, set()) else ""


def direct_relation(a, b, relations):
    """The non-is_a ENVO relations linking a and b directly, either direction, '|'-joined and sorted (deterministic),
    e.g. 'RO:0001025' (located in); '' if none. Relations are not inherited through ancestors: doing so linked almost
    every pair."""
    found = {relation for subject, target in [(a, b), (b, a)]
             for relation, other in relations.get(subject, ()) if other == target}
    return "|".join(sorted(found))


def synonym_candidates(used, label, vector_of):
    """{(a, b)} automatic synonym pairs among the labels in use (a < b), and a table of them.
    Example: 'rhizosphere' / 'rhizosphere environment' (same head once 'environment' is dropped, cosine 0.91)."""
    norm = {t: GENERIC_HEADS.sub("", label[t].lower()).strip() for t in used}
    pairs, rows = set(), []
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            cosine = float(vector_of[a] @ vector_of[b])
            same_head = norm[a] == norm[b] and norm[a] != ""
            if cosine >= ARGS.syn_cosine or same_head:
                pairs.add((a, b))
                rows.append({"term_a": a, "label_a": label[a], "term_b": b, "label_b": label[b],
                             "term_cos": round(cosine, 3), "same_head": same_head})
    return pairs, pd.DataFrame(rows).sort_values(["term_cos", "term_a", "term_b"], ascending=[False, True, True])


def guess_kind(relation, is_synonym, direct):
    """Automatic kind when a pair is not in --pair_kinds."""
    if relation in ("a_broader", "b_broader"):
        return "broader"
    if is_synonym:
        return "equivalent"
    return "facet" if direct else "different"


# ----------------------------------------------------------------------------- scoring
def macro(hit, gold, labels=None):
    """Mean over gold labels of their per-label accuracy (optionally only over `labels`)."""
    per = pd.Series(np.asarray(hit, dtype=float)).groupby(np.asarray(gold)).mean()
    return per[per.index.isin(labels)].mean() if labels is not None else per.mean()


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "keywords", "sub_biomes", "term_vectors"]:
        ap.add_argument(f"--{name}", default=DEFAULTS[name])
    ap.add_argument("--samples", default="~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz",
                    help="The training set 5_evaluate.py was run on")
    ap.add_argument("--predictions", default="~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz",
                    help="5_evaluate.py predictions.tsv.gz")
    ap.add_argument("--method", default="prototype")
    ap.add_argument("--fold_seed", type=int, default=0, help="The --fold_seed of the 5_evaluate.py run")
    ap.add_argument("--envo_obo", default="~/MicrobeAtlasProject/ontologies/envo.obo")
    ap.add_argument("--pair_kinds", default=None, help="TSV slot, term_a, term_b, kind (e.g. label_pair_kinds_draft.tsv)")
    ap.add_argument("--dup_threshold", type=float, default=0.9)
    ap.add_argument("--syn_cosine", type=float, default=0.88)
    ap.add_argument("--min_study_pairs", type=int, default=2)
    ap.add_argument("--output_dir", required=True)
    ARGS = args = ap.parse_args()
    out_dir = path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- the evaluated samples and their vectors, exactly as in 5_evaluate.py
    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)])
    x = np.hstack([kw[[kw_row[s] for s in samples["sample_id"]]],
                   sb[[sb_row[s] for s in samples["sample_id"]]]]).astype(np.float32) / np.sqrt(2)
    studies = samples["study_code"].to_numpy()
    fold = np.zeros(len(samples), int)
    for f, (_, test) in enumerate(study_folds(studies, 5, args.fold_seed)):
        fold[test] = f
    print(f"{len(samples)} samples, {len(set(studies))} studies", flush=True)

    # --- ontology: terms, is_a closure, direct relations, term vectors of the labels in use
    terms = load_terms(args.ontology_terms)
    label = dict(zip(terms["term_id"], terms["label"]))
    anc, dist = term_ancestors(terms), ancestor_distances(terms)
    parents = {t: set(p.split("||")) - {""} for t, p in zip(terms["term_id"], terms["parents"])}
    relations = envo_relations(args.envo_obo)
    used = sorted(set().union(*[set(samples[s]) for s in SLOTS]) - {""})
    text = dict(zip(terms["term_id"], terms["text"]))
    vector_of = dict(zip(used, load_term_vectors(args.term_vectors, [text[t] for t in used])))
    synonyms, synonym_table = synonym_candidates(used, label, vector_of)
    synonym_table.to_csv(os.path.join(out_dir, "synonym_candidates.tsv"), sep="\t", index=False)
    print(f"{len(synonyms)} automatic synonym candidates among {len(used)} labels in use", flush=True)

    kinds_given = {}
    if args.pair_kinds:
        for r in read_tsv(args.pair_kinds).itertuples():
            kinds_given[(r.slot, *sorted((r.term_a, r.term_b)))] = r.kind

    # --- 1. curator disagreement on near-duplicates
    I, J = near_duplicate_pairs(x, studies, args.dup_threshold)
    print(f"{len(I)} cross-study near-duplicate pairs (cosine >= {args.dup_threshold})", flush=True)
    pair_rows, composition, co_labels, summary = [], [], {}, {"near_duplicate_pairs": int(len(I))}
    for slot in SLOTS:
        g = samples[slot].to_numpy()
        both = (g[I] != "") & (g[J] != "")
        differ = both & (g[I] != g[J])
        summary[f"{slot}/same_label_share"] = round(float((g[I][both] == g[J][both]).mean()), 4)
        n_pairs, study_pairs = Counter(), defaultdict(set)
        for i, j in zip(I[differ], J[differ]):
            key = tuple(sorted((g[i], g[j])))
            n_pairs[key] += 1
            study_pairs[key].add(tuple(sorted((studies[i], studies[j]))))
        for (a, b), n in n_pairs.items():
            relation, direct = isa_relation(a, b, anc, parents), direct_relation(a, b, relations)
            given = kinds_given.get((slot, a, b))
            pair_rows.append({"slot": slot, "term_a": a, "label_a": label[a], "term_b": b, "label_b": label[b],
                              "sample_pairs": n, "study_pairs": len(study_pairs[(a, b)]), "isa_relation": relation,
                              "envo_relation": direct, "term_cos": round(float(vector_of[a] @ vector_of[b]), 3),
                              "auto_synonym": (a, b) in synonyms,
                              "kind": given or guess_kind(relation, (a, b) in synonyms, direct),
                              "kind_source": "pair_kinds" if given else "automatic"})
        # co-labels per test fold: only sample pairs whose two studies are both training studies of that fold
        co_labels[slot] = {}
        for f in range(5):
            train = differ & (fold[I] != f) & (fold[J] != f)
            seen = defaultdict(set)
            for i, j in zip(I[train], J[train]):
                seen[tuple(sorted((g[i], g[j])))].add(tuple(sorted((studies[i], studies[j]))))
            co_labels[slot][f] = {k for k, s in seen.items() if len(s) >= args.min_study_pairs}
    pairs = pd.DataFrame(pair_rows).sort_values(["slot", "sample_pairs", "term_a", "term_b"], ascending=[True, False, True, True])
    pairs.to_csv(os.path.join(out_dir, "label_pairs.tsv"), sep="\t", index=False)
    for slot in SLOTS:
        p = pairs[pairs["slot"] == slot]
        total = p["sample_pairs"].sum()
        row = {"slot": slot, "disagreeing_label_pairs": len(p), "disagreeing_sample_pairs": int(total),
               "from_pair_kinds": int((p["kind_source"] == "pair_kinds").sum())}
        for k in KINDS:
            row[f"{k}_share"] = round(p.loc[p["kind"] == k, "sample_pairs"].sum() / total, 3)
            row[f"{k}_label_pairs"] = int((p["kind"] == k).sum())
        isa = p["isa_relation"].replace({"a_broader": "parent_child", "b_broader": "parent_child", "": "not_linked"})
        for k in ["parent_child", "siblings", "not_linked"]:
            row[f"isa_{k}_sample_pairs"] = int(p.loc[isa == k, "sample_pairs"].sum())
        composition.append(row)
    composition = pd.DataFrame(composition)
    composition.to_csv(os.path.join(out_dir, "composition.tsv"), sep="\t", index=False)
    print("\n" + composition[["slot", "disagreeing_label_pairs"] + [f"{k}_share" for k in KINDS]].to_string(index=False))

    # --- 3. acceptance rules on the CV top-1, and 4. graded credit
    pred = read_tsv(args.predictions)
    pred = pred[(pred["method"] == args.method) & (pred["gold"] != "")]
    fold_of = dict(zip(samples["sample_id"], fold))
    missing = set(pred["sample_id"]) - set(fold_of)
    if missing:
        raise SystemExit(f"{len(missing)} predicted samples are not in --samples: wrong training set for these predictions?")
    scores, overlap, graded = [], [], []
    for slot in SLOTS:
        d = pred[pred["slot"] == slot]
        gold, top1 = d["gold"].to_numpy(), d["pred"].to_numpy()
        n_studies = samples[samples[slot] != ""].groupby(slot)["study_code"].nunique()
        learnable = set(n_studies[n_studies >= 2].index)
        exact = gold == top1
        rule = {"exact": exact,
                "broader": exact | np.array([p in anc.get(g, ()) for g, p in zip(gold, top1)]),
                "narrower": exact | np.array([g in anc.get(p, ()) for g, p in zip(gold, top1)]),
                "synonyms": exact | np.array([tuple(sorted((g, p))) in synonyms for g, p in zip(gold, top1)]),
                "co-labels": exact | np.array([tuple(sorted((g, p))) in co_labels[slot][fold_of[s]]
                                               for g, p, s in zip(gold, top1, d["sample_id"])]),
                "facets": exact | np.array([g != p and isa_relation(g, p, anc, {}) in ("", "siblings")
                                            and direct_relation(g, p, relations) != "" for g, p in zip(gold, top1)])}
        rule["synonyms + co-labels"] = rule["synonyms"] | rule["co-labels"]
        rule["all"] = np.logical_or.reduce([rule[k] for k in ["broader", "narrower", "synonyms", "co-labels", "facets"]])
        for name, hit in rule.items():
            scores.append({"slot": slot, "rule": name, "micro": round(float(hit.mean()), 4),
                           "macro": round(float(macro(hit, gold)), 4),
                           "macro_learnable": round(float(macro(hit, gold, learnable)), 4)})
        singles = ["broader", "narrower", "synonyms", "co-labels", "facets"]
        overlap.append({"slot": slot, **{f"rescued_{k}": int((rule[k] & ~exact).sum()) for k in singles},
                        "rescued_synonyms_and_co-labels": int((rule["synonyms"] & rule["co-labels"] & ~exact).sum()),
                        "micro_gain_sum_of_singles": round(100 * sum(rule[k].mean() - exact.mean() for k in singles), 1),
                        "micro_gain_all": round(100 * (rule["all"].mean() - exact.mean()), 1)})
        # graded credit for broader answers
        counts = Counter()
        for t, n in samples.loc[samples[slot] != "", slot].value_counts().items():
            for a in {t} | anc.get(t, set()):
                counts[a] += n
        total = (samples[slot] != "").sum()
        ic = lambda t: -np.log(max(counts.get(t, 0), 0.5) / total)  # noqa: E731
        credit = defaultdict(list)
        for g, p in zip(gold, top1):
            broader = p in anc.get(g, ())
            credit["exact"].append(float(g == p))
            credit["depth 0.5^k"].append(1.0 if g == p else 0.5 ** dist[g][p] if broader else 0.0)
            credit["IC ratio"].append(1.0 if g == p else (ic(p) / ic(g) if ic(g) > 0 else 0.0) if broader else 0.0)
            credit["full credit"].append(float(g == p or broader))
            A, B = {p} | anc.get(p, set()), {g} | anc.get(g, set())
            credit["hierarchical F"].append(2 * len(A & B) / (len(A) + len(B)))
        for name, c in credit.items():
            graded.append({"slot": slot, "scheme": name, "micro": round(float(np.mean(c)), 4),
                           "macro": round(float(macro(c, gold)), 4),
                           "macro_learnable": round(float(macro(c, gold, learnable)), 4)})
    scores, overlap, graded = pd.DataFrame(scores), pd.DataFrame(overlap), pd.DataFrame(graded)
    for name, table in [("rule_scores", scores), ("overlap", overlap), ("graded_credit", graded)]:
        table.to_csv(os.path.join(out_dir, f"{name}.tsv"), sep="\t", index=False)

    # gains vs exact, in points, the way the findings report them
    base = scores[scores["rule"] == "exact"].set_index("slot")
    view = scores.set_index(["slot", "rule"])
    for col in ["micro", "macro", "macro_learnable"]:
        view[f"d_{col}"] = [round(100 * (v - base.loc[s, col]), 1) for (s, _), v in zip(view.index, view[col])]
    print("\n" + view.to_string())
    print("\n" + overlap.to_string(index=False))
    print("\n" + graded.to_string(index=False))
    summary.update({"composition": composition.to_dict("records"), "rule_scores": scores.to_dict("records"),
                    "overlap": overlap.to_dict("records"), "graded_credit": graded.to_dict("records"),
                    "settings": {k: v for k, v in vars(args).items() if k != "output_dir"}})
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}")


ARGS = None
if __name__ == "__main__":
    main()
