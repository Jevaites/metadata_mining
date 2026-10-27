#!/usr/bin/env python3
"""
Descriptive checks on the atlas run and on MicrobeAtlas records, asked for in the deck review
(deck slides "problem", "evaluation checks", "confidence", "outside Metalog", "human biome"; report §7.5-7.6).
Needs the atlas outputs (6_predict_atlas.py --backoff, 6b_coverage.py) and the GPT outputs; no model is run.

hist          histogram of the calibrated top-1 probability of every atlas sample (0.05 bins), split by what
              the back-off did (top-1 / broader term / abstain); medians and share below 0.5.
flags         the out-of-coverage flag (6b_coverage.py) by GPT coarse biome, by 20 keyword-rule habitat
              categories of the GPT sub-biome (first match wins), and by exact GPT sub-biome string
              (>= 3,000 samples), with the answered share per slot.
human         atlas samples whose GPT sub-biome starts with "human": share whose top-1 biome is
              animal-associated environment, and share answered with it after back-off.
study_overlap SRA study of every atlas sample (sample.info `study=`) vs the studies of the Metalog-linked
              samples: how many atlas samples come from a study with no linked sample.
mixs_fields   how many MicrobeAtlas records fill an environment field (any sample_ key containing "env": MIxS
              env_*, "environment (biome)", env packages; values like "missing" ignored), how many of those carry
              an ENVO code, and how often the slot's field already contains the Metalog label (linked samples).

Findings (atlas of 3,437,058 samples): median top-1 p 0.88 / 0.68 / 0.78, below 0.5 for 18 / 37 / 29 %.
Flag rate: laboratory culture 87 %, frog skin 80 %, laboratory mock 77 %, mosquito gut 66 %; 114 of 132
frequent sub-biome strings never flagged. Human sub-biome: 1.2 M samples, top-1 animal-associated
environment 99.99 %, answered with it 93.4 %. ~95 % of atlas samples with a known SRA study come from a
study with no linked sample. 33 % of records fill an environment field, 14 % of those with an ENVO code.

python experiments/deck_review/atlas_summary.py hist        (and flags, human, study_overlap, mixs_fields)
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import SLOTS, iter_sample_info, path, read_tsv  # noqa: E402

P = "~/MicrobeAtlasProject"
ANIMAL_ENV = "ENVO_01001002"
RULES = [  # (habitat category, regex on the lower-cased GPT sub-biome); first match wins
    ("mock / control", r"\bmock\b|negative control|blank|\bcontrol\b.*(kit|extraction|pcr|lab)|laboratory control|zymo"),
    ("laboratory culture", r"laboratory culture|\bculture\b|isolate|enrichment"),
    ("human gut / stool", r"^human.*(gut|stool|feces|faeces|fecal|colon|rectum|intestin)"),
    ("mouse gut / feces", r"^(mouse|mice|rat)\b.*(gut|stool|feces|faeces|fecal|colon|cecum|caecum|ileum|intestin)"),
    ("fermented food", r"ferment|kimchi|kefir|sourdough|wine|beer|kombucha|vinegar"),
    ("cheese / dairy", r"cheese|milk|dairy|yogurt|yoghurt"),
    ("meat / food product", r"meat|beef|pork|chicken meat|sausage|food|carcass|poultry product|seafood"),
    ("root (endosphere)", r"\broots?\b|endosphere|rhizoplane"),
    ("rhizosphere", r"rhizosphere"),
    ("leaf / phyllosphere", r"leaf|leaves|phyllosphere|flower|fruit|seed|stem"),
    ("insect", r"insect|\bbee\b|\bant\b|mosquito|termite|beetle|fly\b|drosophila|tick|aphid|caterpillar|larva"),
    ("fish", r"\bfish|salmon|trout|zebrafish|cod\b|tilapia|carp"),
    ("amphibian / reptile", r"frog|toad|salamander|newt|amphibian|lizard|snake|turtle|tortoise|reptile|crocodil"),
    ("bird", r"bird|chicken|poultry|duck|goose|parrot|penguin|pigeon|sparrow"),
    ("wastewater / sludge / bioreactor", r"wastewater|waste water|sewage|sludge|bioreactor|digester|activated"),
    ("built environment (indoor, hospital)", r"indoor|hospital|built|surface|dust|office|house|building|kitchen"),
    ("air", r"\bair\b|aerosol|atmospher"),
    ("marine water / sediment", r"marine|sea\b|seawater|sea water|ocean|coastal|estuar"),
    ("lake / river water", r"lake|river|stream|reservoir|pond|freshwater|fresh water"),
    ("agricultural / other soil", r"soil"),
]
SLOT_WORDS = {"biome": ("biome", "broad"), "feature": ("feature", "local"), "material": ("material", "medium", "matter")}
MISSING = {"", "missing", "not applicable", "na", "n/a", "not collected", "unknown", "none", "-"}


def read_atlas(p, extra=()):
    cols = ["sample_id"] + [f"{s}_{c}" for s in SLOTS for c in ("p", "backoff_kind")] + list(extra)
    parts = [ch for ch in pd.read_csv(path(p), sep="\t", usecols=cols, chunksize=400000, dtype=str, keep_default_na=False)]
    a = pd.concat(parts, ignore_index=True)
    for s in SLOTS:
        a[f"{s}_p"] = pd.to_numeric(a[f"{s}_p"], errors="coerce")
    return a


def gpt_table(p, name):
    return pd.read_csv(path(p), sep="\t", header=None, names=["sample_id", name], dtype=str, quoting=3).drop_duplicates("sample_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["hist", "flags", "human", "study_overlap", "mixs_fields"])
    ap.add_argument("--atlas", default=f"{P}/ontology_mapping/atlas_backoff/atlas_predictions.tsv.gz")
    ap.add_argument("--coverage", default=f"{P}/ontology_mapping/atlas_coverage/atlas_coverage.tsv.gz")
    ap.add_argument("--gpt_sub_biomes", default=f"{P}/sidequest/latest/GPT_sub_biomes.txt")
    ap.add_argument("--gpt_biomes", default=f"{P}/sidequest/latest/GPT_biomes.txt")
    ap.add_argument("--sample_info", default=f"{P}/sample.info.gz")
    ap.add_argument("--training_set", default=f"{P}/metalog/clean/training_set.clean.tsv.gz")
    ap.add_argument("--project_groups", default=f"{P}/metalog/clean/project_groups.tsv")
    ap.add_argument("--ontology_terms", default=f"{P}/ontology_terms.tsv.gz")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    output = args.output or path(f"{P}/ontology_mapping/experiments/deck_review/atlas_summary_{args.mode}.json")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    res = {}
    if args.mode == "hist":
        a = read_atlas(args.atlas)
        bins = np.linspace(0, 1, 21)
        for s in SLOTS:
            p = a[f"{s}_p"].to_numpy()
            kind = a[f"{s}_backoff_kind"].to_numpy()
            res[s] = {"n": len(p), "median_p": float(np.nanmedian(p)), "median_p_answered": float(np.nanmedian(p[kind != "abstain"])),
                      "share_below_0.5": float(np.nanmean(p < 0.5)), "answered": float((kind != "abstain").mean()),
                      "hist": {k: np.histogram(p[kind == k], bins)[0].tolist() for k in ("top1", "coarser", "abstain")}}
            print(s, {k: round(v, 3) for k, v in res[s].items() if isinstance(v, float)}, flush=True)
    elif args.mode == "flags":
        a = read_atlas(args.atlas)
        cov = pd.read_csv(path(args.coverage), sep="\t", usecols=["sample_id", "in_coverage"], dtype=str)
        d = a.merge(cov, on="sample_id", how="left").merge(gpt_table(args.gpt_sub_biomes, "sb"), on="sample_id", how="left") \
             .merge(gpt_table(args.gpt_biomes, "gb"), on="sample_id", how="left")
        d["flag"] = d.in_coverage.eq("False")
        sb = d.sb.fillna("").str.lower().str.strip()
        cat = pd.Series("", index=d.index)
        for name, rx in RULES:
            cat[(cat == "") & sb.str.contains(rx, regex=True)] = name
        agg = dict(n=("flag", "size"), flag=("flag", "mean"),
                   **{f"answered_{s}": (f"{s}_backoff_kind", lambda x: (x != "abstain").mean()) for s in SLOTS})
        res["flag_overall"] = float(d.flag.mean())
        res["by_gpt_biome"] = d.groupby("gb").agg(**agg).sort_values("n", ascending=False).reset_index().to_dict("records")
        d["cat"], d["sbl"] = cat, sb
        res["categorised_share"] = float((cat != "").mean())
        res["by_category"] = d[d.cat != ""].groupby("cat").agg(**agg).sort_values("flag").reset_index().to_dict("records")
        g = d[d.sbl != ""].groupby("sbl").agg(**agg)
        g = g[g.n >= 3000].sort_values("flag", ascending=False)
        res["by_subbiome_min3000"] = g.reset_index().to_dict("records")
        res["subbiome_summary"] = {"strings": len(g), "atlas_share": float(g.n.sum() / len(d)), "flag_above_10pct": int((g.flag > 0.1).sum()),
                                   "never_flagged": int((g.flag == 0).sum())}
        print("flag overall", round(res["flag_overall"], 4), res["subbiome_summary"])
        print(g.head(12).round(3).to_string(), flush=True)
    elif args.mode == "human":
        a = read_atlas(args.atlas, extra=["biome", "biome_backoff"])
        d = a.merge(gpt_table(args.gpt_sub_biomes, "sb"), on="sample_id", how="left")
        h = d[d.sb.fillna("").str.lower().str.strip().str.startswith("human")]
        res = {"atlas": len(d), "human_subbiome": len(h), "top1_animal_associated": float(h.biome.eq(ANIMAL_ENV).mean()),
               "answered_animal_associated": float(h.biome_backoff.eq(ANIMAL_ENV).mean())}
        print(res)
    elif args.mode in ("study_overlap", "mixs_fields"):
        train = read_tsv(args.training_set)
        linked = set(train.sample_id)
        label_of = {s: dict(zip(train.sample_id, train[s])) for s in SLOTS}
        names = {}
        if args.mode == "mixs_fields":
            from common import load_terms
            terms = load_terms(args.ontology_terms)
            names = dict(zip(terms.term_id, terms.label.str.lower()))
        study_of, n, has_env, biome_vals, biome_envo = {}, 0, 0, 0, 0
        hits = {s: [0, 0] for s in SLOTS}
        for sid, lines in iter_sample_info(path(args.sample_info)):
            n += 1
            kv = {}
            for line in lines:
                k, _, v = line.partition("=")
                kv.setdefault(k, v)
            if args.mode == "study_overlap":
                if kv.get("study"):
                    study_of[sid] = kv["study"]
                continue
            env = {k: v for k, v in kv.items() if k.startswith("sample_") and "env" in k.lower()
                   and v.strip().lower() not in MISSING}  # env_*, "environment (biome)", env packages, ...
            has_env += bool(env)
            biome_envo += bool(env) and any("envo" in v.lower() for v in env.values())
            biome_vals += any(re.match(r"sample_env_(biome|feature|material|broad_scale|local_scale|medium)$", k) for k in env)
            if sid in linked:
                for s in SLOTS:
                    lab = label_of[s].get(sid, "")
                    vals = [v.lower() for k, v in env.items() if any(w in k.lower() for w in SLOT_WORDS[s])]
                    if lab and vals:
                        hits[s][1] += 1
                        hits[s][0] += any(lab.lower().replace("_", ":") in v or lab.lower() in v or names.get(lab, "#") in v for v in vals)
            if n % 500000 == 0:
                print(n, flush=True)
        if args.mode == "study_overlap":
            atlas = pd.read_csv(path(args.atlas), sep="\t", usecols=["sample_id"], dtype=str).sample_id
            linked_studies = {study_of[s] for s in linked if s in study_of}
            known = atlas[atlas.isin(study_of)]
            no_link = ~known.map(study_of).isin(linked_studies)
            groups = read_tsv(args.project_groups) if os.path.exists(path(args.project_groups)) else None
            res = {"records": n, "metalog_study_codes": int(train.study_code.nunique()),
                   "project_groups": int(groups.project_group.nunique()) if groups is not None else None,
                   "sra_studies_of_linked_samples": len(linked_studies), "atlas_samples": len(atlas),
                   "atlas_with_known_study": len(known), "share_from_study_without_linked_sample": float(no_link.mean())}
        else:
            res = {"records": n, "share_with_env_field": has_env / n, "share_with_mixs_env_field": biome_vals / n,
                   "share_of_those_with_envo_code": biome_envo / max(has_env, 1),
                   "linked_field_contains_metalog_label": {s: {"share": h[0] / max(h[1], 1), "n_with_field": h[1]} for s, h in hits.items()}}
        print(json.dumps(res, indent=1))
    with open(output, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", output)


if __name__ == "__main__":
    main()
