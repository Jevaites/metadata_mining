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

Flags (columns `flags`, `n_flags`): which proposals deserve a second look.
  disagree     knn and model top choices differ
  knn_close    knn top-2 vote shares within --close (default 15 points)
  model_close  model top-2 mean probabilities within --close
  weak         the proposal's branch (the proposal, its ancestors and descendants) gets < --weak
               (default 50 %) of the knn votes AND of the model probability
  study_split  the term is used by several studies and the proposal for one of them (same rules, on
               that study's samples only; studies with >= --min_study samples) lies outside the
               term-level proposal's branch: one answer may not fit every study
Per-study evidence is in `per_study`: "study (n): knn X 60% / model Y 80% -> Z", for every term
used by more than one study.

--llm asks an LLM (OpenAI-compatible API) to choose between the candidates, per term:
  candidates   knn top 2, model top 2, the 3 closest biome terms by label, one broader safe option
               (the deepest biome above both the knn and the model choice) and, for a multi-study term,
               each study's own statistical choice (studies shown with >= --min_study samples), each with
               its ENVO definition, in alphabetical order, plus "none of these" (then it may name another
               biome, for the whole term or for one study)
  evidence     the term and its definition, and per study (up to --llm_studies): sample count, the
               Metalog feature / material labels and --llm_examples sample texts. The knn / model
               percentages are NOT shown, so the LLM's choice is an independent check of the statistics
  answer       JSON: the chosen letter, whether one biome fits every study, the best letter per study
               when it does not, and a one-sentence reason
  --llm dry_run writes the prompts to <output stem>.llm_prompts.jsonl (no API call, no key needed);
  --llm run calls the API and caches the answers in <output stem>.llm.jsonl (resumable: a term whose
  prompt and model are unchanged is not asked again). Cost: ~55 prompts of ~1-3k tokens, a few cents.
With an LLM answer: to_id = the LLM's choice (stat_to_id keeps the statistical one); status
`confirmed (stats and LLM agree)` when both give the same biome, otherwise `review (...)`. When the
LLM says one biome does not fit every study, the term gets extra rows with a `study_code` (only the
studies whose biome differs from the term's to_id); 2b_clean_metalog.py applies those to that study
only, and the row without a study code to the other studies.

Output (TSV): slot, from_id, to_id, reason, study_code are the label-map format of
2b_clean_metalog.py --label_map; the other columns are evidence for the reviewer. Edit to_id (or set
it equal to from_id to keep a term) and fill `review_decision` / `review_notes`. Rows of other slots
in --base_map (e.g. the material rule rhizosphere -> soil) are carried over unchanged.

cd scripts/ontology_mapping
python3 experiments/draft_biome_label_map.py --output ~/MicrobeAtlasProject/metalog/clean/biome_label_map_draft.tsv
python3 experiments/draft_biome_label_map.py --output ~/MicrobeAtlasProject/metalog/clean/biome_label_map_draft.tsv --llm dry_run
python3 experiments/draft_biome_label_map.py --output ~/MicrobeAtlasProject/metalog/clean/biome_label_map_draft.tsv --llm run
(writing over clean/biome_label_map.tsv changes what 2b_clean_metalog.py --label_map applies)
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import load_npz, load_terms, path, read_tsv, select_samples, term_ancestors  # noqa: E402
from hierarchy import fit_temperature, softmax  # noqa: E402
from methods import prototype_model  # noqa: E402
from _setup import DEFAULTS  # noqa: E402
BIOME_ROOT = "ENVO_00000428"
KEEP = {"ENVO_01001002": "animal-associated environment", "ENVO_01001001": "plant-associated environment"}
# best-effort corrections where the automatic evidence is clearly wrong (still to be reviewed)
MANUAL = {"ENVO_00005797": ("ENVO_01000252", "lake bottom mud: a freshwater lake material, knn votes came from marine studies"),
          "ENVO_00003861": ("ENVO_01000219", "industrial building: no climate evidence; samples are soil at an industrial site")}
KIND_ROOTS = {"ENVO_00010483": "material", "ENVO_01000254": "environmental system",
              "ENVO_01000813": "astronomical body part (feature)", "ENVO_00000070": "human construction",
              "ENVO_02500000": "process", "BFO_0000015": "process"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAP_COLS = ["slot", "from_id", "to_id", "reason", "study_code"]
EVIDENCE_COLS = ["from_label", "to_label", "status", "flags", "n_flags", "n_samples", "n_studies", "kind",
                 "knn_top", "knn_second", "model_top", "model_second", "nearest_biome_terms", "feature_slot_top",
                 "material_slot_top", "studies", "per_study", "example_text"]
LLM_COLS = ["stat_to_id", "stat_to_label", "stat_status", "llm_to_id", "llm_to_label", "llm_one_for_all", "llm_reason"]


def read_rows(h5_path, texts):
    """{text: L2-normalised vector} for the texts present in the h5 (reads only those rows)."""
    import h5py
    with h5py.File(path(h5_path), "r") as handle:
        row = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        want = sorted({row[t] for t in texts if t in row})
        V = normalize(handle["embeddings"][want]) if want else np.zeros((0, 1))
    at = {i: j for j, i in enumerate(want)}
    return {t: V[at[row[t]]] for t in texts if t in row}


def fmt_share(label, ident, share):
    return f"{label.get(ident, ident)} {share:.0%}"


class Evidence:
    """knn votes, model probabilities and the rule-based proposal for any set of samples."""

    def __init__(self, X, y, study, good, vocab, mean, P, bias, T, k, anc):
        self.X, self.y, self.study, self.vocab, self.k, self.anc = X, y, study, vocab, k, anc
        self.good_idx = np.where(good)[0]
        self.mean, self.P, self.bias, self.T = mean, P, bias, T

    def knn(self, m):
        """Counter {biome: vote share} (sums to 1): knn over other studies' real-biome samples, one vote per study."""
        votes, cand = Counter(), self.good_idx
        for start in range(0, len(m), 500):
            q = m[start:start + 500]
            sim = self.X[q] @ self.X[cand].T
            for i, s in enumerate(q):
                sims = np.where(self.study[cand] != self.study[s], sim[i], -np.inf)
                nn = cand[np.argpartition(-sims, self.k)[:self.k]]
                per = Counter(self.study[nn])  # one vote per study, then each sample counts once
                for j in nn:
                    votes[self.y[j]] += 1 / per[self.study[j]] / len(per) / len(m)
        return votes

    def model(self, m):
        """Mean calibrated probability over vocab for the samples m."""
        return softmax(normalize(self.X[m] - self.mean) @ self.P.T + self.bias, self.T).mean(0)

    def same_branch(self, a, b):
        return a == b or a in self.anc.get(b, ()) or b in self.anc.get(a, ())

    def propose(self, votes, pr, near, near_cos, min_cos):
        """The rule-based proposal -> (to_id, status). See the module docstring."""
        knn = votes.most_common(2)
        k1, ks = knn[0] if knn else ("", 0)
        i1 = int(np.argmax(pr))
        m1, ms = self.vocab[i1], float(pr[i1])
        n1, c1 = (near[0], near_cos[0]) if near else ("", 0.0)
        if k1 == m1:
            more = n1 and n1 != k1 and k1 in self.anc.get(n1, ()) and c1 >= min_cos
            return (n1, "agree, more specific") if more else (k1, "agree")
        if n1 in (k1, m1):
            return n1, "review (2 of 3 sources)"
        if m1 in self.anc.get(k1, ()) or k1 in self.anc.get(m1, ()):  # same branch: the broader one is safe
            return (m1 if m1 in self.anc.get(k1, ()) else k1), "review (broader of knn / model)"
        return (k1 if ks >= ms else m1), "review"  # the better-supported source

    def support(self, to, votes, pr):
        """(knn share, model probability) of the proposal's branch: to, its ancestors, its descendants."""
        ks = sum(v for c, v in votes.items() if self.same_branch(c, to))
        ms = float(sum(p for c, p in zip(self.vocab, pr) if self.same_branch(c, to)))
        return ks, ms


def flags_for(votes, pr, to, ev, close, weak):
    knn = votes.most_common(2)
    top = np.sort(pr)[::-1]
    out = []
    if knn and knn[0][0] != ev.vocab[int(np.argmax(pr))]:
        out.append("disagree")
    if len(knn) > 1 and knn[0][1] - knn[1][1] < close:
        out.append("knn_close")
    if len(top) > 1 and top[0] - top[1] < close:
        out.append("model_close")
    ks, ms = ev.support(to, votes, pr)
    if ks < weak and ms < weak:
        out.append("weak")
    return out


# ----------------------------------------------------------------------------- LLM
SYSTEM = ("You are an expert curator of microbiome sample metadata. You annotate samples following MIxS: "
          "env_broad_scale (the 'biome' slot) takes a subclass of the ENVO class 'biome' (ENVO:00000428), the "
          "broad ecological context of the sample. Metalog, the curated database these samples come from, also "
          "uses 'animal-associated environment' and 'plant-associated environment' in that slot for "
          "host-associated samples.")


def llm_prompt(rec, cands, groups, label, definition):
    """Chat messages for one term. cands: candidate ids (lettered in this order). groups: per-study
    dicts {study, n, feature, material, examples}."""
    t = rec["from_id"]
    lines = [f"Metalog curators put the term below in the biome slot, but it is not an ENVO biome "
             f"(it is: {rec['kind']}). Choose the biome to use instead for these samples.", "",
             f"Term: {label.get(t, t)} ({t.replace('_', ':')})"]
    if definition.get(t):
        lines.append(f"Definition: {definition[t]}")
    lines.append(f"Used for {rec['n_samples']} samples from {rec['n_studies']} "
                 f"stud{'y' if rec['n_studies'] == 1 else 'ies'}.")
    lines += ["", "Studies (Metalog study code; the labels the curators gave in the other two slots; sample "
                  "descriptions):"]
    for g in groups:
        lines.append(f"- {g['study']}: {g['n']} samples; feature: {g['feature']}; material: {g['material']}")
        for e in g["examples"]:
            lines.append(f"    example: {e}")
    lines += ["", "Candidate biomes:"]
    for L, c in zip(LETTERS, cands):
        d = f": {definition[c]}" if definition.get(c) else ""
        lines.append(f"{L}. {label.get(c, c)} ({c.replace('_', ':')}){d}")
    none = LETTERS[len(cands)]
    lines.append(f"{none}. none of these fits")
    study_list = ", ".join(f'"{g["study"]}"' for g in groups)
    lines += ["", "Answer with a JSON object only:",
              '{"choice": "<letter of the best biome for all these samples>",',
              f' "suggested_label": "<only if you chose {none}: the ENVO biome label you would use>",']
    if rec["n_studies"] > 1:  # one term, several studies: may need one answer per study
        lines += [' "one_for_all": <true if that biome fits the samples of every study, false if some studies '
                  'need a different biome>,',
                  f' "per_study": {{<only if one_for_all is false: for each study, the letter of its best biome, or '
                  f'"{none}: <ENVO biome label>" if none of the candidates fits that study; keys from {study_list}>}},']
    lines.append(' "reason": "<one or two sentences>"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def prompt_hash(messages, model):
    return hashlib.md5((model + json.dumps(messages, sort_keys=True)).encode()).hexdigest()[:12]


def parse_answer(text, cands, studies, label_to_id):
    """-> {to_id, suggested, one_for_all, per_study {study: (to_id, suggested)}, reason}, or None if unusable.
    to_id is "" when the LLM chose 'none' and its suggested label is not a known biome (also per study:
    'G: freshwater river biome' with G = none -> (that biome's id, 'freshwater river biome'))."""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    letter = lambda v: (lambda c: c if c and c in LETTERS[:len(cands) + 1] else "")(str(v or "").strip().upper()[:1])
    pick = lambda v: cands[LETTERS.index(letter(v))] if letter(v) and LETTERS.index(letter(v)) < len(cands) else None
    if not letter(a.get("choice")):
        return None
    to = pick(a.get("choice"))
    suggested = str(a.get("suggested_label") or "").strip()
    if to is None:  # 'none of these'
        to = label_to_id.get(suggested.lower(), "")
    one = a.get("one_for_all")
    one = not (one is False or str(one).lower() == "false")
    per = {}
    none = LETTERS[len(cands)]
    for s, v in (a.get("per_study") or {}).items():
        if s not in studies or not letter(v):
            continue
        if letter(v) == none:  # none of the candidates fits this study: the label it names, if known
            named = str(v).split(":", 1)[1].strip() if ":" in str(v) else ""
            per[s] = (label_to_id.get(named.lower(), ""), named)
        else:
            per[s] = (pick(v), "")
    return {"to_id": to, "suggested": suggested, "one_for_all": one, "per_study": per,
            "reason": str(a.get("reason") or "").replace("\t", " ").replace("\n", " ")}


def call_llm(args, messages):
    from openai import OpenAI
    if not hasattr(call_llm, "client"):
        call_llm.client = OpenAI(api_key=open(path(args.api_key_path)).read().strip(), base_url=args.base_url,
                                 max_retries=8)
    body = {"model": args.model, "messages": messages, "max_completion_tokens": 4000,
            "response_format": {"type": "json_object"}}
    if args.model.startswith(("gpt-5", "o")):
        body["reasoning_effort"] = args.reasoning_effort
    else:
        body["temperature"] = 0
    r = call_llm.client.chat.completions.create(**body)
    return r.choices[0].message.content, (r.usage.prompt_tokens, r.usage.completion_tokens)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--term_vectors", default="~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5",
                   help="h5 with the plain-label and label_syn term texts (term_text_variants.py embed)")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--min_cos", type=float, default=0.6, help="label cosine needed to prefer a more specific biome term")
    p.add_argument("--close", type=float, default=0.15, help="knn_close / model_close: top-2 gap below this")
    p.add_argument("--weak", type=float, default=0.5, help="weak: branch support below this in knn and model")
    p.add_argument("--min_study", type=int, default=3, help="study_split: only studies with at least this many samples")
    p.add_argument("--base_map", default="~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv",
                   help="rows of other slots in this map are carried over ('' = none)")
    p.add_argument("--output", required=True)
    p.add_argument("--llm", choices=["off", "dry_run", "run"], default="off")
    p.add_argument("--model", default="gpt-5.1")
    p.add_argument("--reasoning_effort", default="low", help="gpt-5 / o models")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint (default: OpenAI)")
    p.add_argument("--api_key_path", default="~/MicrobeAtlasProject/my_api_key")
    p.add_argument("--llm_studies", type=int, default=8, help="studies shown per term (largest first)")
    p.add_argument("--llm_examples", type=int, default=2, help="sample texts shown per study")
    p.add_argument("--example_chars", type=int, default=300)
    args = p.parse_args()

    terms = load_terms(args.ontology_terms)
    ids = terms["term_id"].to_numpy()
    label = dict(zip(ids, terms["label"]))
    definition = {t: d for t, d in zip(ids, terms["definition"]) if d}
    anc = term_ancestors(terms)
    is_biome = lambda t: t != BIOME_ROOT and BIOME_ROOT in anc.get(t, ())

    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    S = select_samples(args.samples, [set(kr), set(sr)], max_per_study=0)
    S = S[S["biome"] != ""].reset_index(drop=True)
    X = np.hstack([K[[kr[s] for s in S["sample_id"]]], B[[sr[s] for s in S["sample_id"]]]]) / np.sqrt(2)
    y, study = S["biome"].to_numpy(), S["study_code"].to_numpy()
    feat, mat, text = S["feature"].to_numpy(), S["material"].to_numpy(), S["text"].astype(str).to_numpy()
    good = np.array([is_biome(t) or t in KEEP for t in y])  # labels that stay as they are
    targets = sorted(set(y[~good]) - {BIOME_ROOT})
    print(f"{len(S)} samples with a biome label, {good.mean():.1%} already a biome or a kept convention; "
          f"{len(targets)} other terms to review, plus the root 'biome' ({(y == BIOME_ROOT).sum()} samples)")

    # term vectors: label_syn texts of the model vocab, plain labels of every ENVO biome term and the targets
    vocab = np.unique(y[good])
    text_of = dict(zip(terms["term_id"], terms["text"]))
    biome_ids = [t for t in ids if is_biome(t)]
    vec = read_rows(args.term_vectors, [text_of[t] for t in vocab] + [label[t] for t in biome_ids + targets])
    TV = np.vstack([vec[text_of[t]] for t in vocab])
    TM = np.hstack([TV, TV]) / np.sqrt(2)  # as hierarchical_backoff.term_matrix
    mean, P, bias = prototype_model(X[good], y[good], TM, vocab, 0.5, 0.1)
    Sg = normalize(X[good] - mean) @ P.T + bias
    T = fit_temperature([(Sg, np.searchsorted(vocab, y[good]))])
    ev = Evidence(X, y, study, good, vocab, mean, P, bias, T, args.k, anc)

    biome_terms = [t for t in biome_ids if label[t] in vec]
    BV = np.vstack([vec[label[t]] for t in biome_terms])
    label_to_id = {label[t].lower(): t for t in biome_terms + list(KEEP)}

    def top2(m, n=2):
        return "; ".join(f"{label.get(v, v) or '(empty)'} {c / len(m):.0%}" for v, c in Counter(m).most_common(n))

    stem = os.path.splitext(path(args.output))[0]
    cache_path, prompts_path = stem + ".llm.jsonl", stem + ".llm_prompts.jsonl"
    cache = {}
    if args.llm == "run" and os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["from_id"], r["prompt_hash"])] = r
    prompts_out = open(prompts_path, "w") if args.llm == "dry_run" else None
    cost = [0, 0]

    rows = []
    for t in [BIOME_ROOT] + targets:
        m = np.where(y == t)[0]
        studies = Counter(study[m])
        kind = ", ".join(sorted({v for r, v in KIND_ROOTS.items() if r in anc.get(t, ())})) or "other"
        rec = {"slot": "biome", "from_id": t, "from_label": label.get(t, t), "n_samples": len(m),
               "n_studies": len(studies), "studies": "; ".join(f"{s} ({c})" for s, c in studies.most_common(4)),
               "kind": kind, "study_code": ""}
        if t == BIOME_ROOT:
            rec.update(to_id="", to_label="", reason="slot root: uninformative, dropped", status="drop")
            rows.append(rec)
            continue
        votes, pr = ev.knn(m), ev.model(m)
        knn = votes.most_common(2)
        top = np.argsort(-pr)[:2]
        tv = vec.get(label.get(t))
        cos = None if tv is None else BV @ tv
        order = [] if cos is None else np.argsort(-cos)[:3]
        near, near_cos = [biome_terms[i] for i in order], [float(cos[i]) for i in order]
        to, status = ev.propose(votes, pr, near, near_cos, args.min_cos)
        if t in MANUAL:
            to, status = MANUAL[t][0], "manual (Claude): " + MANUAL[t][1]
        flags = flags_for(votes, pr, to, ev, args.close, args.weak)

        # per study: the same evidence and rules on that study's samples only
        per_study, study_to = [], {}
        if len(studies) > 1:
            for s, n in studies.most_common():
                ms = m[study[m] == s]
                v_s, p_s = ev.knn(ms), ev.model(ms)
                to_s = ev.propose(v_s, p_s, near, near_cos, args.min_cos)[0]
                study_to[s] = (to_s, n)
                k_s = v_s.most_common(1)[0] if v_s else ("", 0)
                i_s = int(np.argmax(p_s))
                per_study.append(f"{s} ({n}): knn {fmt_share(label, *k_s)} / model "
                                 f"{fmt_share(label, ev.vocab[i_s], float(p_s[i_s]))} -> {label.get(to_s, to_s)}")
            if any(n >= args.min_study and not ev.same_branch(to_s, to) for to_s, n in study_to.values()):
                flags.append("study_split")

        rec.update(to_id=to, to_label=label.get(to, ""), status=status, flags=", ".join(flags), n_flags=len(flags),
                   reason=f"not an ENVO biome ({kind}); biome used by other studies for similar samples",
                   knn_top=fmt_share(label, *knn[0]) if knn else "",
                   knn_second=fmt_share(label, *knn[1]) if len(knn) > 1 else "",
                   model_top=fmt_share(label, vocab[top[0]], float(pr[top[0]])),
                   model_second=fmt_share(label, vocab[top[1]], float(pr[top[1]])),
                   nearest_biome_terms="; ".join(f"{label[n]} {c:.2f}" for n, c in zip(near, near_cos)),
                   feature_slot_top=top2(feat[m]), material_slot_top=top2(mat[m]), per_study=" | ".join(per_study),
                   example_text=str(text[m[0]])[:160].replace("\t", " ").replace("\n", " "))
        rows.append(rec)

        if args.llm == "off":
            continue
        # candidates: knn top 2, model top 2, 3 closest by label, one broader safe option; alphabetical
        k1, m1 = (knn[0][0] if knn else ""), vocab[top[0]]
        common = [a for a in ({k1} | set(anc.get(k1, ()))) & ({m1} | set(anc.get(m1, ()))) if is_biome(a)]
        if k1 == m1:  # agreement: the safe option is the parent biome
            common = [a for a in anc.get(k1, ()) if is_biome(a)]
        broader = max(common, key=lambda a: len(anc.get(a, ())), default="")
        # + each shown study's own statistical choice, so a per-study answer can be specific
        shown = studies.most_common(args.llm_studies)
        per_study_cands = [study_to[s][0] for s, n in shown if s in study_to and n >= args.min_study]
        pool = [c for c, _ in knn] + [vocab[i] for i in top] + near + [broader, to] + per_study_cands
        cands = sorted({c for c in pool if c}, key=lambda c: label.get(c, c).lower())[:len(LETTERS) - 1]
        groups = []
        for s, n in shown:
            ms = m[study[m] == s]
            ex = list(dict.fromkeys(text[ms]))[:args.llm_examples]
            groups.append({"study": s, "n": n, "feature": top2(feat[ms]), "material": top2(mat[ms]),
                           "examples": [e[:args.example_chars].replace("\n", " ") for e in ex]})
        rest = len(studies) - len(shown)
        messages = llm_prompt(rec, cands, groups, label, definition)
        if rest:
            messages[1]["content"] = messages[1]["content"].replace(
                "\n\nCandidate biomes:", f"\n- and {rest} smaller studies ({sum(c for s, c in studies.most_common()[len(shown):])} samples)\n\nCandidate biomes:")
        h = prompt_hash(messages, args.model)
        if prompts_out:
            prompts_out.write(json.dumps({"from_id": t, "prompt_hash": h, "candidates": cands, "messages": messages}) + "\n")
            continue
        if (t, h) not in cache:
            content, usage = call_llm(args, messages)
            cache[(t, h)] = {"from_id": t, "prompt_hash": h, "model": args.model, "candidates": cands,
                             "content": content, "usage": usage, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(cache_path, "a") as handle:
                handle.write(json.dumps(cache[(t, h)]) + "\n")
            cost[0] += usage[0]
            cost[1] += usage[1]
        ans = parse_answer(cache[(t, h)]["content"], cands, [g["study"] for g in groups], label_to_id)
        rec.update(stat_to_id=to, stat_to_label=label.get(to, ""), stat_status=status)
        if ans is None:
            rec.update(status=status + "; LLM answer unusable", llm_reason=cache[(t, h)]["content"][:300])
            continue
        lt = ans["to_id"]
        rec.update(llm_to_id=lt, llm_to_label=label.get(lt, ans["suggested"] and f"(not found) {ans['suggested']}"),
                   llm_one_for_all=str(ans["one_for_all"]), llm_reason=ans["reason"])
        if not lt:
            rec["status"] = "review (LLM: none of the candidates fits)"
        elif lt == to:
            rec["status"] = "confirmed (stats and LLM agree)"
        else:
            rel = ("broader" if lt in anc.get(to, ()) else "more specific" if to in anc.get(lt, ())
                   else "different branch")
            rec["status"] = f"review (LLM differs from stats: {rel})"
        if lt and lt not in cands:
            rec["status"] += "; LLM chose none and named this biome"
        if t in MANUAL:
            rec["status"] = status + ("; LLM agrees" if lt == to else f"; LLM prefers {label.get(lt, lt or 'none')}")
        else:
            rec.update(to_id=lt or to, to_label=label.get(lt or to, ""))
        if not ans["one_for_all"]:
            rec["status"] += "; studies differ (LLM)"
            for s, (ts, named) in ans["per_study"].items():
                if ts == rec["to_id"]:
                    continue
                st = "review (per study, LLM)"
                if named:  # 'none of these' for this study
                    st += f"; LLM chose none and named '{named}'" + ("" if ts else " (not an ENVO biome label: to_id kept)")
                rows.append({**{c: rec.get(c, "") for c in MAP_COLS + EVIDENCE_COLS + LLM_COLS},
                             "study_code": s, "to_id": ts or rec["to_id"], "to_label": label.get(ts or rec["to_id"], ""),
                             "llm_to_id": ts, "llm_to_label": label.get(ts, named), "n_samples": studies[s],
                             "n_studies": 1, "studies": f"{s} ({studies[s]})", "status": st,
                             "reason": f"not an ENVO biome ({kind}); this study's samples need another biome (LLM)"})
    if prompts_out:
        prompts_out.close()
        print(f"wrote {len(targets)} prompts to {prompts_path} (no API call)")
    if args.llm == "run":
        print(f"LLM tokens this run: {cost[0]:,} in, {cost[1]:,} out; answers cached in {cache_path}")

    cols = MAP_COLS + EVIDENCE_COLS + (LLM_COLS if args.llm == "run" else [])
    out = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    out["n_flags"] = out["n_flags"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["review_decision"] = ""
    out["review_notes"] = ""
    # biome root first, then terms by size with their per-study rows right after them
    size = dict(zip(out.loc[out["study_code"] == "", "from_id"], out.loc[out["study_code"] == "", "n_samples"]))
    out["_size"] = out["from_id"].map(size).astype(float)
    out["_sub"] = out["study_code"] != ""
    out = pd.concat([out.iloc[:1], out.iloc[1:].sort_values(["_size", "from_id", "_sub"], ascending=[False, True, True])])
    out = out.drop(columns=["_size", "_sub"])
    if args.base_map and os.path.exists(path(args.base_map)):
        base = read_tsv(args.base_map)
        other = base[base["slot"] != "biome"].reindex(columns=out.columns).fillna("")
        if len(other):
            print(f"carried over {len(other)} rows of other slots from {args.base_map}")
            out = pd.concat([out, other])
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    out.to_csv(path(args.output), sep="\t", index=False)
    main_rows = out[(out["slot"] == "biome") & (out["study_code"] == "")]
    print(main_rows["status"].str.replace(r"^(manual).*", r"\1", regex=True).str.split("; ").str[0]
          .value_counts().to_string())
    flagged = main_rows[main_rows["flags"] != ""]
    print(f"flagged: {len(flagged)} of {len(main_rows) - 1} terms; per flag: "
          + ", ".join(f"{f} {c}" for f, c in Counter(f for s in flagged["flags"] for f in s.split(", ")).most_common()))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
