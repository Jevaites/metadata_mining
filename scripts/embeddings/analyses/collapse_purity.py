#!/usr/bin/env python3
"""
The scatter's two arms are pairs of samples that one run gave the SAME sub-biome
string while the other did not. That is a collapse. This asks whether each run's
collapses are correct, using gold_dict as a referee neutral to both runs:

    when a run gives two samples the same string, how often do they share a
    gold label?  (collapse precision)

Compared against the base rate - the chance two random gold samples share a
label - which is what a collapse would score if it carried no information.

    MAP_ROOT=... python3 scripts/embeddings/analyses/collapse_purity.py
"""
import os, pickle, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
OLD, NEW = f"{ROOT}/sidequest", f"{ROOT}/sidequest/latest"
GOLD_FIELD = {"biome": 1, "subbiome": 2}


def stream(path, keep):
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid in keep and raw.strip():
            yield sid, clean_text(raw, False)


def pair_counts(labels):
    """(same-label pairs, total pairs) for one group of samples."""
    n = len(labels)
    return sum(v * (v - 1) // 2 for v in Counter(labels).values()), n * (n - 1) // 2


def precision(texts, labels, all_same=None):
    """Group samples by string; score every within-group pair.

    precision = of the pairs this run merges, how many share a gold label
    recall    = of all same-label pairs that exist, how many does it merge
    Precision alone is unfair: a run that merges almost nothing scores high on
    it. The two together show the actual trade-off."""
    groups = defaultdict(list)
    for sid, t in texts.items():
        if sid in labels:
            groups[t].append(labels[sid])
    same = tot = 0
    sizes = [len(v) for v in groups.values() if len(v) > 1]
    for v in groups.values():
        a, b = pair_counts(v)
        same, tot = same + a, tot + b
    prec = same / tot if tot else None
    rec = same / all_same if all_same else None
    return {"groups": len(groups), "groups_ge2": len(sizes), "samples_in_groups_ge2": sum(sizes),
            "pairs": tot, "same_pairs": same, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec and rec else None,
            "largest_group": max(sizes) if sizes else 0}


def boot(texts, labels, n=600, seed=42):
    """95% CI for precision, resampling the GROUPS (strings), not the samples.

    Resampling samples is wrong here: a sample drawn twice pairs with its own
    copy and is trivially same-label, which inflates precision above the point
    estimate. The group is the natural unit anyway - it is the collapse."""
    import random
    groups = defaultdict(list)
    for sid, t in texts.items():
        if sid in labels:
            groups[t].append(labels[sid])
    pairs = [pair_counts(v) for v in groups.values() if len(v) > 1]
    if not pairs:
        return None
    rng, out = random.Random(seed), []
    for _ in range(n):
        pick = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        tot = sum(b for _, b in pick)
        if tot:
            out.append(sum(a for a, _ in pick) / tot)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def main():
    gold = pickle.load(open(f"{ROOT}/gold_dict.pkl", "rb"))
    print(f"gold_dict: {len(gold):,} samples")

    keep = set(gold)
    old_txt = dict(stream(f"{OLD}/GPT_sub_biomes.txt", keep))
    new_txt = dict(stream(f"{NEW}/GPT_sub_biomes.txt", keep))
    both = set(old_txt) & set(new_txt)
    changed = {s for s in both if old_txt[s] != new_txt[s]}
    print(f"gold samples with a sub-biome in both runs: {len(both):,} "
          f"({len(changed):,} changed text, {len(both) - len(changed):,} same)")

    for subset_name, subset in [("all gold overlap", both), ("changed-text only", changed)]:
        print(f"\n{'=' * 74}\n{subset_name}: {len(subset):,} samples\n{'=' * 74}")
        for level, idx in GOLD_FIELD.items():
            labels = {s: str(gold[s][idx]).strip() for s in subset
                      if len(gold[s]) > idx and str(gold[s][idx]).strip() not in ("", "None", "nan")}
            if len(labels) < 50:
                print(f"  {level}: only {len(labels)} labelled, skipped")
                continue
            vals = list(labels.values())
            base_same, base_tot = pair_counts(vals)
            base = base_same / base_tot
            print(f"\n  gold {level}: {len(labels):,} labelled, {len(set(vals))} classes, "
                  f"base rate {base:.3f}")
            print(f"    {'run':<6} {'grp>=2':>7} {'pairs':>8} {'precision [95% CI]':>22} "
                  f"{'recall':>9} {'F1':>6}")
            for run, txt in [("Dany", old_txt), ("new", new_txt)]:
                sub = {s: txt[s] for s in subset}
                r = precision(sub, labels, base_same)
                if r["precision"] is None:
                    continue
                ci = boot(sub, labels)
                pc = f"{r['precision']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]" if ci else f"{r['precision']:.3f}"
                print(f"    {run:<6} {r['groups_ge2']:>7} {r['pairs']:>8,} {pc:>22} "
                      f"{r['recall']:>9.3f} {r['f1']:>6.3f}")

    # which strings collapse gold samples of different biomes?
    labels = {s: str(gold[s][1]).strip() for s in changed
              if len(gold[s]) > 1 and str(gold[s][1]).strip() not in ("", "None", "nan")}
    for run, txt in [("new", new_txt), ("Dany", old_txt)]:
        groups = defaultdict(list)
        for s in labels:
            groups[txt[s]].append(labels[s])
        impure = sorted(((t, Counter(v)) for t, v in groups.items() if len(set(v)) > 1),
                        key=lambda kv: -sum(kv[1].values()))[:8]
        print(f"\n  {run}: strings covering more than one gold biome")
        for t, c in impure:
            print(f"    {t[:44]:<46} " + ", ".join(f"{k} x{n}" for k, n in c.most_common(5)))


if __name__ == "__main__":
    main()
