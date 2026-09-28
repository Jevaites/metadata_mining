#!/usr/bin/env python3
"""
The two arms of the sub-biomes / changed-text scatter in cosine_agreement.png.

The panel plots 200k random sample PAIRS: x = cosine in Dany's space,
y = cosine in the new space. Two spikes stand out:

    top arm    cos_new  ~ 1, cos_dany anywhere  -> the new run calls these two
                                                   samples identical, Dany did not
    right arm  cos_dany ~ 1, cos_new  anywhere  -> the reverse

Hypothesis: a cosine of exactly 1 means the two samples share a literal text
string (embeddings are deduplicated by text), so each arm is one run's
vocabulary collapsing two samples the other run kept apart. This script tests
that and then asks WHICH strings do the collapsing.

Replays the exact rng sequence of compare_to_previous_embeddings.py (seed 42),
so these are the same 200k points that are in the figure.

    python3 scripts/scatter_arms.py
"""
import json, os, pickle, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_to_previous_embeddings as C

TOL = 1e-4                                  # cosine >= 1 - TOL counts as "1.0"
N_SAMPLES, N_PAIRS, N_QUERIES, K, SEED = 15000, 200_000, 2000, 20, 42
ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
C.OLD, C.NEW, C.GOLD = f"{ROOT}/sidequest", f"{ROOT}/sidequest/latest", f"{ROOT}/gold_dict.pkl"
OUT = f"{C.NEW}/embeddings/vs_previous"


def collision_rate(texts):
    """P(two random samples share a text) - the arm mass we should expect."""
    n, c = len(texts), Counter(texts)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def describe_arm(on_arm, shared, other, free_cos, pi, pj, label, top=12):
    """Group the arm's pairs by the string they share; report what it absorbed.

    shared = texts in the run whose cosine is pinned at 1 (the collapser)
    other  = texts in the run that still separates them
    """
    agg = defaultdict(lambda: {"pairs": 0, "absorbed": set(), "cos": []})
    for idx in np.flatnonzero(on_arm & (shared[pi] == shared[pj])):
        d = agg[shared[pi[idx]]]
        d["pairs"] += 1
        d["absorbed"].add(other[pi[idx]])
        d["absorbed"].add(other[pj[idx]])
        d["cos"].append(free_cos[idx])
    n_by_text = Counter(shared)
    rows = []
    for text, d in sorted(agg.items(), key=lambda kv: -kv[1]["pairs"])[:top]:
        rows.append({"text": text, "pairs": d["pairs"], "samples": n_by_text[text],
                     "distinct_absorbed": len(d["absorbed"]),
                     "median_other_cos": round(float(np.median(d["cos"])), 3),
                     "min_other_cos": round(float(np.min(d["cos"])), 3)})
    print(f"\n  {label} - the strings doing the collapsing:")
    print(f"    {'string':<44} {'pairs':>7} {'samples':>8} {'absorbed':>9} {'med cos':>8} {'min cos':>8}")
    for r in rows:
        print(f"    {r['text'][:43]:<44} {r['pairs']:>7} {r['samples']:>8} "
              f"{r['distinct_absorbed']:>9} {r['median_other_cos']:>8.3f} {r['min_other_cos']:>8.3f}")
    return rows


def worst_cases(on_arm, shared, other, free_cos, pi, pj, label, n=6):
    """Pairs where one run says 'identical' and the other says 'unrelated'."""
    cand = np.flatnonzero(on_arm & (shared[pi] == shared[pj]))
    seen, rows = set(), []
    for idx in cand[np.argsort(free_cos[cand])]:
        key = tuple(sorted((other[pi[idx]], other[pj[idx]])))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"other_cos": round(float(free_cos[idx]), 3),
                     "shared": shared[pi[idx]], "kept_apart": list(key)})
        if len(rows) >= n:
            break
    print(f"\n  {label} - widest disagreements:")
    for r in rows:
        print(f"    other-run cosine {r['other_cos']:+.3f}   shared: {r['shared'][:60]!r}")
        print(f"        vs {r['kept_apart'][0][:66]!r}")
        print(f"        vs {r['kept_apart'][1][:66]!r}")
    return rows


def main():
    rng = np.random.default_rng(SEED)
    gold_ids = set(pickle.load(open(C.GOLD, "rb")))

    # --- replay the sampling of compare_to_previous_embeddings.py, verbatim ---
    same, changed = C.classify("sub_biomes", False)
    print(f"overlap {len(same) + len(changed):,} | identical text {len(same):,}", flush=True)
    chosen = {}
    for arm, ids in {"same text": same, "changed text": changed}.items():
        ids = sorted(ids)
        gold_here = sorted(set(ids) & gold_ids)
        if len(ids) > N_SAMPLES:
            ids = [ids[i] for i in sorted(rng.choice(len(ids), N_SAMPLES, replace=False))]
        chosen[arm] = sorted(set(ids) | set(gold_here))
        print(f"  {arm}: {len(chosen[arm])} samples", flush=True)
    del same, changed

    wanted = {s for ids in chosen.values() for s in ids}
    old_txt = C.texts_for(f"{C.OLD}/GPT_sub_biomes.txt", wanted)
    new_txt = C.texts_for(f"{C.NEW}/GPT_sub_biomes.txt", wanted)
    biome = C.texts_for(f"{C.NEW}/GPT_biomes.txt", wanted)
    old_row = C.rows_for(C.old_embeddings("sub_biomes"), "sample_ids", wanted)
    text_row = C.rows_for(C.new_unique("sub_biomes"), "texts", set(new_txt.values()))
    print(f"loaded {len(old_row)} old rows, {len(text_row)} new text rows", flush=True)

    result = {}
    for arm, ids in chosen.items():                  # order matters for the rng
        ids = [s for s in ids if s in old_row and new_txt.get(s) in text_row]
        old_v = C.fetch(C.old_embeddings("sub_biomes"), [old_row[s] for s in ids])
        new_v = C.fetch(C.new_unique("sub_biomes"), [text_row[new_txt[s]] for s in ids])
        ok = np.isfinite(old_v).all(axis=1) & np.isfinite(new_v).all(axis=1)
        ids = [s for s, good in zip(ids, ok) if good]
        old_v, new_v = C.unit(old_v[ok]), C.unit(new_v[ok])
        pi, pj, a, b, _ = C.geometry(old_v, new_v, rng, N_PAIRS)
        if arm == "same text":                       # burn the same rng draws, then drop it
            C.neighbour_overlap(old_v, new_v, rng, K, N_QUERIES)
            del old_v, new_v, pi, pj, a, b
            continue
        del old_v, new_v
        print(f"\n{'=' * 78}\nchanged text: {len(ids)} samples, {len(a):,} pairs\n{'=' * 78}", flush=True)

        t_new = np.array([new_txt[s] for s in ids], dtype=object)
        t_old = np.array([old_txt[s] for s in ids], dtype=object)
        top, right = b >= 1 - TOL, a >= 1 - TOL
        eq_new, eq_old = t_new[pi] == t_new[pj], t_old[pi] == t_old[pj]

        # --- 1. is the arm really "same string"? ---
        print("\n  arm mass and the duplicate-text hypothesis")
        for name, on, eq, exp in [("top   (cos_new = 1)", top, eq_new, collision_rate(t_new)),
                                  ("right (cos_dany = 1)", right, eq_old, collision_rate(t_old))]:
            print(f"    {name}: {on.sum():>7,} pairs ({on.mean():6.2%})   "
                  f"identical string {np.mean(eq[on]):7.2%}   "
                  f"recall {np.mean(on[eq]):6.2%}   expected from text frequencies {exp:6.2%}")
        print(f"    both arms at once:   {int((top & right).sum()):>7,} pairs "
              f"({(top & right).mean():.2%})")
        print(f"    distinct strings in these {len(ids)} samples: "
              f"new {len(set(t_new))}, Dany {len(set(t_old))}")

        # --- 2. where along the arm do the points sit? ---
        print("\n  distribution of the free coordinate")
        for name, on, free in [("top   arm, cos_dany", top, a), ("right arm, cos_new ", right, b)]:
            q = np.percentile(free[on], [5, 25, 50, 75, 95])
            print(f"    {name}: p5 {q[0]:+.2f}  p25 {q[1]:+.2f}  median {q[2]:+.2f}  "
                  f"p75 {q[3]:+.2f}  p95 {q[4]:+.2f}   below 0.5: {np.mean(free[on] < 0.5):.1%}")
        print(f"    off-arm pairs:       median cos_dany {np.median(a[~top & ~right]):+.2f}  "
              f"median cos_new {np.median(b[~top & ~right]):+.2f}")

        # --- 3. which strings collapse, and what do they absorb? ---
        res_top = describe_arm(top, t_new, t_old, a, pi, pj, "TOP ARM  (new run collapses)")
        res_right = describe_arm(right, t_old, t_new, b, pi, pj, "RIGHT ARM (Dany collapses)")
        worst_top = worst_cases(top, t_new, t_old, a, pi, pj, "TOP ARM")
        worst_right = worst_cases(right, t_old, t_new, b, pi, pj, "RIGHT ARM")

        # --- 4. who lives on each arm? ---
        print("\n  biome mix (new-era GPT_biomes) of the samples involved")
        allb = Counter(biome.get(s, "?") for s in ids)
        for name, on in [("top arm", top), ("right arm", right)]:
            idxs = set(pi[on]).union(pj[on])
            c = Counter(biome.get(ids[i], "?") for i in idxs)
            tot = sum(c.values())
            share = ", ".join(f"{k} {v / tot:.0%}" for k, v in c.most_common(4))
            print(f"    {name:<10} ({tot:>5} samples): {share}")
        tot = sum(allb.values())
        print(f"    {'all':<10} ({tot:>5} samples): "
              + ", ".join(f"{k} {v / tot:.0%}" for k, v in allb.most_common(4)))
        placeholder = {"na", "n/a", "unknown", "none", "not applicable", ""}
        print(f"    placeholder-looking new strings: "
              f"{sum(1 for t in t_new if t.strip().lower() in placeholder)} samples")

        result = {"n_samples": len(ids), "n_pairs": int(len(a)),
                  "distinct_new_texts": len(set(t_new)), "distinct_dany_texts": len(set(t_old)),
                  "top_arm": {"pairs": int(top.sum()), "fraction": float(top.mean()),
                              "identical_string": float(np.mean(eq_new[top])),
                              "expected_from_frequencies": collision_rate(t_new),
                              "free_cos_median": float(np.median(a[top])),
                              "free_cos_below_0.5": float(np.mean(a[top] < 0.5)),
                              "collapsers": res_top, "worst": worst_top},
                  "right_arm": {"pairs": int(right.sum()), "fraction": float(right.mean()),
                                "identical_string": float(np.mean(eq_old[right])),
                                "expected_from_frequencies": collision_rate(t_old),
                                "free_cos_median": float(np.median(b[right])),
                                "free_cos_below_0.5": float(np.mean(b[right] < 0.5)),
                                "collapsers": res_right, "worst": worst_right}}
        np.savez_compressed(f"{OUT}/scatter_arms_points.npz", cos_dany=a, cos_new=b,
                            top=top, right=right, eq_new=eq_new, eq_old=eq_old)

    path = f"{OUT}/scatter_arms.json"
    json.dump(result, open(path, "w", encoding="utf-8"), indent=2, default=str)
    print(f"\nwrote {path}\nwrote {OUT}/scatter_arms_points.npz", flush=True)


if __name__ == "__main__":
    main()
