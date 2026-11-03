#!/usr/bin/env python3
"""
Follow-up analyses of the Jev pilot (README 7d, claude/jev-pilot-results.md, the Claude doc "Jev for ontology
mapping"). No API calls: everything is read from the files jev_pilot.py run / score wrote, the cache and the
pipeline's inputs. Each subcommand prints one block of numbers quoted in the write-up.

  confidence      rerank Choice: exact top-1 of the Choice alone, AUROC of several confidence signals (top
                  probability, margin, the Noul of the pick, max Noul, product) and a calibration table
  tokens          share of input tokens by question kind in the flat request (vocab / rerank / fits / state)
  vs_gpt          Jev rerank - gpt-4.1-mini exact top-1, paired, study bootstrap; how often they pick the same term
  gate            random samples: gain over the prototype when only its q least-confident samples are reranked,
                  accuracy by confidence band, and the selection-effect checks (gate by Jev's confidence instead,
                  "either right", "only one right")
  zero_shot       zero-shot over Jev's vocabulary: uncentred embedding retrieval with plain labels and with
                  label; synonyms (the term-text-variants setup) vs Jev's vocabulary Choice; exact / recall@3
  topdown         top-down walks: exact top-1, gold among the finished beam paths, where the best path leaves
                  gold's branch (decision 1 / 2 / 3+), rank of gold's branch at the first decision
  ontology_size   terms and tokens of one Choice question per term set (slot vocabularies, ENVO branches, all of
                  ENVO / UBERON), by what each option carries
  open_retrieval  gold in the top k of plain embedding retrieval over all of ENVO + UBERON (k = 10 ... 255)
  open_hybrid     retrieve-then-choose results (run_jev_followups.sh): its pick as a fallback for labels no other
                  study used (oracle and a usable rule), and its most frequent errors
  open_detector   the same fallback, triggered by the coverage flag or the prototype's low confidence instead
  open_examples   the open Choice with similar labelled samples: vs a plain vote of the same samples and vs Jev
                  reranking the prototype's candidates; where its gain comes from

cd scripts/ontology_mapping
python3 experiments/jev_analyses.py all          # every block, both sample sets (a few minutes)
python3 experiments/jev_analyses.py gate         # one block
Paths default to ~/MicrobeAtlasProject/ontology_mapping/experiments/{jev,rerank}/ (--jev / --rerank).
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, path  # noqa: E402
import jev_pilot as jp  # noqa: E402

E = "~/MicrobeAtlasProject/ontology_mapping/experiments"


def sets(args):
    """The two sample sets: (name, pilot file, inputs, flat file, top-down folder)."""
    J, R = path(args.jev), path(args.rerank)
    return [("gated", f"{R}/pilot_k5anc.jsonl", f"{J}/inputs.json", f"{J}/flat.jsonl", J),
            ("random", f"{J}/pilot_all_k5anc.jsonl", f"{J}/inputs_all.json", f"{J}/all/flat.jsonl", f"{J}/all")]


def load(pilot_f, inputs_f, flat_f):
    pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pilot_f)}
    ctx = jp.Ctx(json.load(open(inputs_f)), SimpleNamespace(min_studies=1))
    flat = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(flat_f)} if os.path.exists(flat_f) else {}
    return pilot, ctx, flat


def cands(o):
    return [c["term_id"] for c in o["candidates"]]


def jev_rerank_right(o, rec):
    """(right, confidence) of the jev_rerank decision (rerank Choice + Nouls for 'none')."""
    c = cands(o)
    p, d = jp.decide_rerank(rec["rerank"], rec["fits"])
    return p < len(c) and c[p] == o["gold"], float(d[p])


# ----------------------------------------------------------------------------- blocks
def confidence(args):
    for name, pf, inp, fl, _ in sets(args):
        pilot, ctx, flat = load(pf, inp, fl)
        print(f"\n== confidence signals of the rerank Choice ({name} samples): AUROC for exact top-1")
        for slot in SLOTS:
            ks = [k for k in flat if k[0] == slot]
            ex, pmax, fpick, fmax, margin = [], [], [], [], []
            for k in ks:
                o, r = pilot[k], flat[k]
                p = np.array(r["rerank"])
                i = int(p.argmax())
                s = np.sort(p)
                ex.append(cands(o)[i] == o["gold"])
                pmax.append(p.max())
                fpick.append(r["fits"][i])
                fmax.append(max(r["fits"]))
                margin.append(s[-1] - s[-2] if len(s) > 1 else s[-1])
            ex, pmax = np.array(ex), np.array(pmax)
            print(f"  {slot:8} Choice alone exact {ex.mean():.3f} | top p {jp.auroc(pmax, ex):.3f}, margin {jp.auroc(margin, ex):.3f}, "
                  f"Noul of the pick {jp.auroc(fpick, ex):.3f}, max Noul {jp.auroc(fmax, ex):.3f}, "
                  f"top p x Noul {jp.auroc(pmax * np.array(fpick), ex):.3f}")
            bins = np.digitize(pmax, [0.5, 0.8, 0.95])
            print("           calibration, top p bin -> (mean top p, accuracy, n): " + ", ".join(
                f"({pmax[bins == b].mean():.2f}, {ex[bins == b].mean():.2f}, {int((bins == b).sum())})" for b in range(4) if (bins == b).any()))


def tokens(args):
    from rerank_prompt import count_tokens
    tot = {"state": 0, "vocab": 0, "rerank": 0, "fits": 0}
    n = 0
    with open(os.path.join(path(args.jev), "jev_cache.jsonl")) as handle:
        for line in handle:
            rec = json.loads(line)
            q = rec["request"]["questions"]
            if not any(k.startswith("vocab_") for k in q):
                continue
            tot["state"] += count_tokens(json.dumps(rec["request"]["state"]))
            for k, v in q.items():
                tot[k.split("_")[0]] += count_tokens(json.dumps(v))
            n += 1
            if n >= args.n_requests:
                break
    s = sum(tot.values())
    print(f"\n== token share by question kind, flat request ({n} requests, o200k estimate): "
          + ", ".join(f"{k} {v / s:.0%}" for k, v in tot.items()))


def vs_gpt(args):
    from rerank_prompt import parse
    name, pf, inp, fl, _ = sets(args)[0]
    pilot, ctx, flat = load(pf, inp, fl)
    gpt = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(os.path.join(path(args.rerank), args.gpt_file))}
    print(f"\n== Jev rerank - {args.gpt_file} (gated samples, paired, study bootstrap 95 % CI)")
    for slot in SLOTS:
        ks = [k for k in flat if k[0] == slot and k in gpt]
        a, b, st, same = [], [], [], []
        for k in ks:
            o, c = pilot[k], cands(pilot[k])
            pj, _ = jp.decide_rerank(flat[k]["rerank"], flat[k]["fits"])
            _, pg = parse(o, gpt[k])
            a.append(pj < len(c) and c[pj] == o["gold"])
            b.append(pg < len(c) and c[pg] == o["gold"])
            st.append(o["study"])
            same.append(pj == pg)
        print(f"  {slot:8} exact Jev {np.mean(a):.3f}, GPT {np.mean(b):.3f}, difference {jp.boot_diff(a, b, st)}, same pick {np.mean(same):.3f}")


def gate(args):
    name, pf, inp, fl, _ = sets(args)[1]
    pilot, ctx, flat = load(pf, inp, fl)
    print("\n== gate (random samples): rerank only the prototype's q least-confident samples; exact top-1, gain vs the prototype (95 % CI)")
    for slot in SLOTS:
        ks = [k for k in flat if k[0] == slot]
        conf = np.array([pilot[k]["base_maxp"] for k in ks])
        st = [pilot[k]["study"] for k in ks]
        base = np.array([cands(pilot[k])[0] == pilot[k]["gold"] for k in ks])
        jr = [jev_rerank_right(pilot[k], flat[k]) for k in ks]
        jev, jconf = np.array([a for a, _ in jr]), np.array([c for _, c in jr])
        order, n = np.argsort(conf), len(ks)
        cells = []
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
            g = np.zeros(n, bool)
            g[order[:int(round(q * n))]] = True
            mix = np.where(g, jev, base)
            d = jp.boot_diff(mix, base, st)
            cells.append(f"{int(q * 100)}% {mix.mean():.3f} ({d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}])")
        print(f"  {slot:8} base {base.mean():.3f} | " + " | ".join(cells))
        lo, mid, hi = order[:int(0.3 * n)], order[int(0.3 * n):int(0.7 * n)], order[int(0.7 * n):]
        print("           by prototype confidence (lowest 30 % / middle 40 % / top 30 %): prototype "
              + " / ".join(f"{base[b].mean():.2f}" for b in (lo, mid, hi)) + ", Jev " + " / ".join(f"{jev[b].mean():.2f}" for b in (lo, mid, hi)))
        lo_j = np.argsort(jconf)[:int(0.3 * n)]
        print(f"           Jev's least-confident 30 %: prototype {base[lo_j].mean():.2f}, Jev {jev[lo_j].mean():.2f} | either right "
              f"{np.mean(base | jev):.2f} | only one right: prototype {np.mean(base & ~jev):.2f}, Jev {np.mean(jev & ~base):.2f}; "
              f"in the prototype's lowest 30 %: prototype {np.mean((base & ~jev)[lo]):.2f}, Jev {np.mean((jev & ~base)[lo]):.2f}")


def zero_shot(args):
    import h5py
    from sklearn.preprocessing import normalize
    from common import load_npz, load_terms
    from _setup import DEFAULTS
    import hierarchical_backoff as hb
    terms = load_terms(DEFAULTS["ontology_terms"])
    trow = {t: i for i, t in enumerate(terms["term_id"])}
    lab = dict(zip(terms["term_id"], terms["label"]))
    TMs = hb.term_matrix(terms, DEFAULTS["term_vectors"])
    with h5py.File(path(args.plain_label_vectors)) as handle:
        txt = [t.decode() if isinstance(t, bytes) else t for t in handle["texts"][:]]
        V = normalize(handle["embeddings"][:])
    prow = {t: i for i, t in enumerate(txt)}
    (kr, K), (sr, B) = load_npz(DEFAULTS["keywords"]), load_npz(DEFAULTS["sub_biomes"])
    vec = lambda s: np.hstack([K[kr[s]], B[sr[s]]]) / np.sqrt(2)
    for name, pf, inp, fl, _ in sets(args):
        pilot, ctx, flat = load(pf, inp, fl)
        print(f"\n== zero-shot over Jev's vocabulary, uncentred cosine ({name} samples): exact / recall@3")
        for slot in SLOTS:
            out = {"plain label": [], "label; synonyms": [], "Jev": []}
            for k in [k for k in flat if k[0] == slot]:
                o = pilot[k]
                voc = ctx.vocab(slot, o["study"])[0]
                ok = [v for v in voc if lab[v] in prow]
                x, g = vec(o["sample_id"]), o["gold"]
                P = np.hstack([V[[prow[lab[v]] for v in ok]]] * 2) / np.sqrt(2)
                for nm, M, ids in [("plain label", P, ok), ("label; synonyms", TMs[[trow[v] for v in voc]], voc)]:
                    top = [ids[i] for i in np.argsort(-(M @ x))[:3]]
                    out[nm].append((top[0] == g, g in top))
                pj = flat[k]["vocab"]
                top = sorted(pj, key=lambda t: -pj[t])[:3]
                out["Jev"].append((top[0] == g, g in top))
            print(f"  {slot:8} " + " | ".join(f"{nm} {np.mean([a for a, _ in v]):.3f} / {np.mean([b for _, b in v]):.3f}" for nm, v in out.items()))


def topdown(args):
    for name, pf, inp, fl, folder in sets(args):
        pilot, ctx, _ = load(pf, inp, fl)
        print(f"\n== top-down walks ({name} samples)")
        for arm in ("topdown_root", "topdown_lca"):
            f = os.path.join(folder, f"{arm}.jsonl")
            if not os.path.exists(f):
                continue
            recs = jp.read_jsonl(f)
            for slot in SLOTS:
                rs = [r for r in recs if r["slot"] == slot]
                g_all = [pilot[(slot, r["sample_id"])]["gold"] for r in rs]
                top1 = np.mean([max(r["terminals"], key=r["terminals"].get, default=None) == g for r, g in zip(rs, g_all)])
                inbeam = np.mean([g in r["terminals"] for r, g in zip(rs, g_all)])
                where, rank = [], []
                for r in rs:
                    if not r["gold_reachable"]:
                        continue
                    o = pilot[(slot, r["sample_id"])]
                    tree, g = ctx.tree(slot, o["study"]), o["gold"]
                    line = ctx.anc.get(g, set()) | {g}
                    dec, out = 0, None
                    for parent, node in zip(r["best_path"], r["best_path"][1:]):
                        if len(jp.node_options(parent, tree, ctx)) > 1:
                            dec += 1
                        if node in line:
                            continue
                        out = ("too specific", dec) if g in ctx.anc.get(node, set()) else ("wrong branch", dec)
                        break
                    where.append(out)
                    for node, probs in (r["trace"][0] if r["trace"] else {}).items():  # the first request's question
                        target = {n: t for n, t, _ in jp.node_options(node, tree, ctx)}
                        ordered = sorted(probs, key=lambda n: -probs[n])
                        hits = [i for i, n in enumerate(ordered) if (target[n] is None and node in line)
                                or (target[n] is not None and (target[n] in line or g in tree["below"].get(target[n], [])))]
                        rank.append(hits[0] + 1 if hits else 99)
                        break
                n = len(where)
                wb = [d for x in where if x and x[0] == "wrong branch" for d in [x[1]]]
                ts = sum(1 for x in where if x and x[0] == "too specific")
                rank = np.array(rank)
                print(f"  {arm:13} {slot:8} exact {top1:.3f}, gold among finished paths {inbeam:.3f} | of {n} reachable: on gold's line "
                      f"{sum(x is None for x in where) / n:.0%}, too specific {ts / n:.0%}, wrong branch {len(wb) / n:.0%} (at decision 1 / 2 / 3+: "
                      f"{np.mean([d == 1 for d in wb]):.0%} / {np.mean([d == 2 for d in wb]):.0%} / {np.mean([d >= 3 for d in wb]):.0%}) | "
                      f"gold's branch at the first question ranked 1 / 2 / 3 / 4+: "
                      + " / ".join(f"{np.mean(m):.0%}" for m in (rank == 1, rank == 2, rank == 3, rank >= 4)))


def ontology_size(args):
    from rerank_prompt import count_tokens
    from common import load_terms, term_ancestors
    from _setup import DEFAULTS
    import jev_questions as W
    T = load_terms(DEFAULTS["ontology_terms"])
    anc = term_ancestors(T)
    rec = {r.term_id: r for r in T.itertuples()}

    def option(t, mode):
        r = rec[t]
        term = {"synonyms": [s for s in r.synonyms.split("||") if s][:3], "definition": r.definition[:220]}
        if mode == "label":
            return None
        if mode in ("label + synonyms", "label + synonyms + parents"):
            term["definition"] = ""
        d = W.describe(term) or {}
        if mode == "label + synonyms + parents":
            d["is a"] = "; ".join(rec[p].label for p in r.parents.split("||") if p in rec)
        return d or None
    roots = {"ENVO biome branch": "ENVO_00000428", "ENVO environmental material branch": "ENVO_00010483",
             "ENVO environmental system branch": "ENVO_01000254", "UBERON anatomical entity (incl. PO / FOODON subclasses)": "UBERON_0001062"}
    groups = {k: [t for t in rec if t == r or r in anc.get(t, ())] for k, r in roots.items()}
    groups["all ENVO"] = [t for t in rec if t.startswith("ENVO_")]
    groups["all UBERON"] = [t for t in rec if t.startswith("UBERON_")]
    groups["ENVO + UBERON"] = groups["all ENVO"] + groups["all UBERON"]
    groups["whole term table (+ PO, FOODON)"] = list(rec)
    inputs = json.load(open(os.path.join(path(args.jev), "inputs.json")))
    for slot in SLOTS:
        groups[f"Metalog {slot} vocabulary"] = [t for t in inputs["slots"][slot] if t in rec]
    modes = ["label", "label + synonyms", "label + synonyms + definition", "label + synonyms + parents"]
    rng = np.random.default_rng(0)
    print("\n== one Choice question per term set (o200k tokens; sets over 3,000 terms estimated from 3,000): " + " / ".join(modes))
    for k, ids in groups.items():
        sample = ids if len(ids) <= 3000 else list(rng.choice(ids, 3000, replace=False))
        toks = [count_tokens(json.dumps({rec[t].label: option(t, m) for t in sample})) * len(ids) / len(sample) for m in modes]
        print(f"  {k:58} {len(ids):7,} terms ({len(ids) / jp.MAX_OPTIONS:5.1f}x the option limit) | " + " / ".join(f"{t / 1000:,.1f}k" for t in toks))


def open_retrieval(args):
    from common import load_terms, load_npz
    from _setup import DEFAULTS
    import hierarchical_backoff as hb
    T = load_terms(DEFAULTS["ontology_terms"])
    ids = T["term_id"].to_numpy()
    keep = np.array([t.startswith(("ENVO_", "UBERON_")) for t in ids])
    TM = hb.term_matrix(T, DEFAULTS["term_vectors"])[keep].astype(np.float32)
    ids = ids[keep]
    gi = {t: i for i, t in enumerate(ids)}
    (kr, K), (sr, B) = load_npz(DEFAULTS["keywords"]), load_npz(DEFAULTS["sub_biomes"])
    ks = (10, 50, 100, 200, 255)
    for name, pf, *_ in sets(args):
        pilot = jp.read_jsonl(pf)
        print(f"\n== gold in the top k of embedding retrieval over ENVO + UBERON ({len(ids):,} terms, label; synonyms), {name} samples, k = {ks}")
        for slot in SLOTS:
            ps = [o for o in pilot if o["slot"] == slot and o["gold"] in gi]
            X = np.vstack([np.hstack([K[kr[o["sample_id"]]], B[sr[o["sample_id"]]]]) / np.sqrt(2) for o in ps]).astype(np.float32)
            S = X @ TM.T
            rank = np.array([(S[j] > S[j, gi[o["gold"]]]).sum() + 1 for j, o in enumerate(ps)])
            print(f"  {slot:8} n {len(ps)}: " + " / ".join(f"{np.mean(rank <= k):.2f}" for k in ks))


def open_hybrid(args):
    """Retrieve-then-choose (jev_followups.py open, run by run_jev_followups.sh): can its pick replace the
    prototype's when the gold label is one no other study used? Oracle (open pick exactly on unseen-label samples),
    a usable rule (open pick when it lies outside the training vocabulary with probability >= p) and the most
    frequent open-Choice errors."""
    import collections
    J = path(args.jev)
    F, O = f"{J}/followups", f"{J}/followups/results"
    for name, pf, *_ in sets(args):
        if not os.path.exists(f"{O}/{name}_{args.open_variant}.jsonl"):
            print(f"\n== {name}: no {O}/{name}_{args.open_variant}.jsonl (run run_jev_followups.sh)")
            continue
        pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pf)}
        opened = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(f"{F}/{name}_{args.open_variant}.jsonl")}
        inputs = json.load(open(f"{F}/inputs_{name}_{args.open_variant}.json"))
        ctx = jp.Ctx(inputs, SimpleNamespace(min_studies=1))
        label = lambda t: (inputs["terms"].get(t) or {}).get("label", t)
        resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(f"{O}/{name}_{args.open_variant}.jsonl")}
        print(f"\n== open Choice as a fallback for unseen labels, {name} samples")
        for slot in SLOTS:
            keys = [k for k in resp if k[0] == slot]
            if not keys:
                continue
            pro, ope, out, prob, uns, st, errors = [], [], [], [], [], [], collections.Counter()
            for k in keys:
                o = pilot[k]
                pr = np.array(resp[k]["rerank"])
                pick = cands(opened[k])[int(np.argmax(pr))]
                voc = set(ctx.vocab(slot, o["study"])[0])
                pro.append(o["candidates"][0]["term_id"] == o["gold"])
                ope.append(pick == o["gold"])
                out.append(pick not in voc)
                prob.append(float(pr.max()))
                uns.append(o["gold"] not in voc)
                st.append(o["study"])
                if pick != o["gold"]:
                    errors[(label(o["gold"]), label(pick))] += 1
            pro, ope, out, prob, uns, st = map(np.array, (pro, ope, out, prob, uns, st))
            print(f"  {slot}: prototype {pro.mean():.3f} | open pick outside the training vocabulary {out.mean():.3f} "
                  f"(gold unseen {uns.mean():.3f}) | oracle {np.where(uns, ope, pro).mean():.3f}")
            for th in (0.0, 0.5, 0.7, 0.9):
                use = out & (prob >= th)
                hyb = np.where(use, ope, pro)
                prec = f"{ope[use].mean():.3f}" if use.any() else "-"
                print(f"    open pick if outside the vocabulary and p >= {th}: fires {use.mean():.3f}, right {prec} "
                      f"-> exact {hyb.mean():.3f}, vs prototype {jp.boot_diff(hyb, pro, st)}")
            print("    most frequent errors (gold -> open pick): " + "; ".join(
                f"{g} -> {p} ({n})" for (g, p), n in errors.most_common(5)))


def open_detector(args):
    """When should the open Choice replace the prototype? Two signals that do not come from the open Choice itself:
    the coverage flag (held-out similarity to the nearest training sample of another project,
    atlas_coverage/metalog_held_out_sim.tsv.gz; flag < 0.611) and the prototype's own confidence (base_maxp).
    For each rule: share of samples it fires on, share of those with an unseen label, open / prototype right there,
    and overall exact top-1 vs the prototype (paired, study bootstrap)."""
    import gzip
    import csv
    J = path(args.jev)
    F, O = f"{J}/followups", f"{J}/followups/results"
    with gzip.open(path(args.coverage), "rt") as fh:
        sim = {r["sample_id"]: float(r["held_out_sim"]) for r in csv.DictReader(fh, delimiter="\t")}
    for name, pf, *_ in sets(args):
        if not os.path.exists(f"{O}/{name}_{args.open_variant}.jsonl"):
            continue
        pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pf)}
        opened = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(f"{F}/{name}_{args.open_variant}.jsonl")}
        ctx = jp.Ctx(json.load(open(f"{F}/inputs_{name}_{args.open_variant}.json")), SimpleNamespace(min_studies=1))
        resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(f"{O}/{name}_{args.open_variant}.jsonl")}
        print(f"\n== which samples get the open Choice's pick, {name} samples")
        for slot in SLOTS:
            keys = [k for k in resp if k[0] == slot and k[1] in sim]
            if not keys:
                continue
            o_ = [pilot[k] for k in keys]
            pro = np.array([o["candidates"][0]["term_id"] == o["gold"] for o in o_])
            ope = np.array([cands(opened[k])[int(np.argmax(resp[k]["rerank"]))] == pilot[k]["gold"] for k in keys])
            uns = np.array([o["gold"] not in ctx.vocab(slot, o["study"])[0] for o in o_])
            cov = np.array([sim[k[1]] for k in keys])
            conf = np.array([o["base_maxp"] for o in o_])
            st = np.array([o["study"] for o in o_])
            print(f"  {slot}: n {len(keys)} (of {sum(k[0] == slot for k in resp)} with a coverage value), unseen {uns.mean():.3f}, "
                  f"prototype {pro.mean():.3f}, oracle {np.where(uns, ope, pro).mean():.3f}")
            rules = [(f"coverage < {c}", cov < c) for c in (0.611, 0.7, 0.8)]
            rules += [(f"prototype confidence in its lowest {q:.0%}", conf <= np.quantile(conf, q)) for q in (0.1, 0.2, 0.3)]
            rules += [("coverage < 0.7 and confidence in the lowest 30%", (cov < 0.7) & (conf <= np.quantile(conf, 0.3)))]
            for label, use in rules:
                if not use.any():
                    continue
                hyb = np.where(use, ope, pro)
                print(f"    {label:48} fires {use.mean():.3f}, unseen there {uns[use].mean():.2f}, open / prototype right there "
                      f"{ope[use].mean():.2f} / {pro[use].mean():.2f} -> exact {hyb.mean():.3f}, vs prototype {jp.boot_diff(hyb, pro, st)}")


def open_examples(args):
    """Retrieve-then-choose with similar labelled samples (run_jev_followups.sh open_similar): where its gain comes from.
    Exact top-1 of the open Choice with examples vs a plain vote of the same 5 examples (1-NN and majority, ties to
    the most similar) and vs Jev reranking the prototype's candidates (the flat run); whether gold is among the
    examples' terms; how often it picks one of them; agreement with the reranker. Random samples: both as gated
    rerankers on the prototype's 30 % least-confident samples."""
    import collections
    J = path(args.jev)
    F, O = f"{J}/followups", f"{J}/followups/results"
    for name, pf, inputs_f, flat_f, _ in sets(args):
        if not os.path.exists(f"{O}/{name}_open_similar.jsonl"):
            print(f"\n== {name}: no {O}/{name}_open_similar.jsonl (run run_jev_followups.sh open_similar)")
            continue
        pilot, _, flat = load(pf, inputs_f, flat_f)
        opened = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(f"{F}/{name}_open_similar.jsonl")}
        resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(f"{O}/{name}_open_similar.jsonl")}
        terms = json.load(open(f"{F}/inputs_{name}_open_similar.json"))["terms"]
        by_label = {}
        for t, v in terms.items():
            by_label.setdefault(v["label"], t)
        print(f"\n== open Choice with similar samples, {name} samples")
        for slot in SLOTS:
            ks = [k for k in resp if k[0] == slot and k in flat]
            if not ks:
                continue
            gold = np.array([pilot[k]["gold"] for k in ks])
            st = np.array([pilot[k]["study"] for k in ks])
            ex_terms = [[by_label.get(x["term"]) for x in opened[k].get("similar", [])] for k in ks]
            sim = np.array([cands(opened[k])[int(np.argmax(resp[k]["rerank"]))] for k in ks])
            pro = np.array([pilot[k]["candidates"][0]["term_id"] for k in ks])
            rer = np.array([jev_rerank_pick(pilot[k], flat[k]) for k in ks], object)

            def vote(ts):
                cnt = collections.Counter(ts)
                return next(t for t in ts if cnt[t] == max(cnt.values())) if ts else None
            nn1 = np.array([ts[0] if ts else None for ts in ex_terms], object)
            knn = np.array([vote(ts) for ts in ex_terms], object)
            gin = np.array([g in ts for g, ts in zip(gold, ex_terms)])
            copied = np.array([p in ts for p, ts in zip(sim, ex_terms)])
            e = {n: (a == gold) for n, a in [("prototype", pro), ("Jev rerank", rer), ("1-NN", nn1), ("5-NN vote", knn),
                                              ("Jev with examples", sim)]}
            agree = sim == rer
            print(f"  {slot}: " + ", ".join(f"{n} {v.mean():.3f}" for n, v in e.items()))
            print(f"    with examples - 5-NN vote {jp.boot_diff(e['Jev with examples'], e['5-NN vote'], st)}; "
                  f"- Jev rerank {jp.boot_diff(e['Jev with examples'], e['Jev rerank'], st)}")
            print(f"    gold among the examples' terms {gin.mean():.3f} (right there {e['Jev with examples'][gin].mean():.2f}, "
                  f"elsewhere {e['Jev with examples'][~gin].mean():.2f}); picks one of the examples' terms {copied.mean():.3f}")
            print(f"    agrees with Jev rerank {agree.mean():.3f}: right when they agree {e['Jev with examples'][agree].mean():.2f}, "
                  f"when not {e['Jev with examples'][~agree].mean():.2f} (with examples) / {e['Jev rerank'][~agree].mean():.2f} (rerank)")
            if name == "random":
                conf = np.array([pilot[k]["base_maxp"] for k in ks])
                low = conf <= np.quantile(conf, 0.3)
                a, b = np.where(low, e["Jev rerank"], e["prototype"]), np.where(low, e["Jev with examples"], e["prototype"])
                print(f"    on the 30 % least-confident only: Jev rerank {a.mean():.3f}, with examples {b.mean():.3f}, "
                      f"difference {jp.boot_diff(b, a, st)}")


def jev_rerank_pick(o, rec):
    p, _ = jp.decide_rerank(rec["rerank"], rec["fits"])
    c = cands(o)
    return None if p == len(c) else c[p]


BLOCKS = {"confidence": confidence, "tokens": tokens, "vs_gpt": vs_gpt, "gate": gate, "zero_shot": zero_shot,
          "topdown": topdown, "ontology_size": ontology_size, "open_retrieval": open_retrieval,
          "open_hybrid": open_hybrid, "open_detector": open_detector,
          "open_examples": open_examples}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("block", choices=list(BLOCKS) + ["all"])
    p.add_argument("--jev", default=f"{E}/jev")
    p.add_argument("--rerank", default=f"{E}/rerank")
    p.add_argument("--gpt_file", default="resp_k5anc_gpt-4.1-mini.jsonl")
    p.add_argument("--plain_label_vectors",
                   default="~/MicrobeAtlasProject/ontology_mapping/slot_vocab_label__text-embedding-3-large__dim1024.h5")
    p.add_argument("--coverage", default="~/MicrobeAtlasProject/ontology_mapping/atlas_coverage/metalog_held_out_sim.tsv.gz",
                   help="open_detector: held-out similarity of each Metalog training sample (6b_coverage.py)")
    p.add_argument("--open_variant", default="open", help="open_hybrid / open_detector: open (default) or open_similar")
    p.add_argument("--n_requests", type=int, default=300, help="tokens: flat requests sampled from the cache")
    args = p.parse_args(argv)
    for name, fn in BLOCKS.items():
        if args.block in (name, "all"):
            fn(args)


if __name__ == "__main__":
    main()
