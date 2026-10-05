#!/usr/bin/env python3
"""
Step 2b: flag Metalog samples for cleaning. Nothing is deleted: every sample keeps its raw values,
and flag columns say what to use it for.

Inputs: one Metalog snapshot (*_core_wide_<date>.tsv.gz and *_all_long_<date>.tsv.gz), the term
index (step 1), and optionally the linked training set (step 2), reviewer overrides and a label map.

Outputs, in --output_dir:
  metalog_flags.tsv.gz        one row per Metalog sample: raw values + flags (all domains)
  audit_review.tsv            samples Metalog did NOT flag as artificial, but whose own metadata looks
                              like a control / culture / sort / mesocosm / ancient sample. One row per
                              study x category, with an empty `decision` column to fill in.
  training_set.clean.tsv.gz   step-2 rows usable for training (hard drops removed), cleaned labels in
                              biome/feature/material, raw ids in *_raw. Same columns 5_evaluate.py reads.
  training_set.gold.tsv.gz    the rows fit for evaluation: in_gold_eval and not a duplicate text
  summary.md                  counts for every flag

Artificial buckets (from Metalog's `artificial` field, then overrides):
  control    negative control, mock, spike-in, marked as contaminated -> dropped (not a habitat)
  perturbed  cultivation, enrichment, virome enrichment, sorted cells, mesocosm
             -> label = source environment: kept for training, excluded from gold eval
  degraded   paleosample, post-mortem, museum specimen -> kept for training, excluded from gold eval
  none       everything else

Main flag columns:
  drop_reason        ';'-joined hard drops: no_accession, conflicting_duplicate, duplicate_alias,
                     artificial_control. in_train = no drop_reason.
  audit_hits         audit categories matched by the sample's own metadata (study abstracts excluded)
  audit_unreviewed   an audit hit on a sample with no Metalog flag and no reviewer decision
  in_gold_eval       in_train, bucket none, no unreviewed audit hit. Use the same subset for
                     composition / relative-abundance work.
  <slot>_status      ok | obsolete | not_in_index | other_ontology | no_code | control_value | empty
  <slot>_clean       the usable ENVO / UBERON / PO / FOODON id after --label_map ("" when unusable)
  biome_in_biome_subtree  True when the clean biome is ENVO 'biome' (ENVO_00000428) or below it

Review loop:
  1. run once; open audit_review.tsv; set `decision` to control / perturbed / degraded / ok
     (ok = false alarm, a normal sample) on the rows you checked, leave the others empty;
  2. re-run with --overrides audit_review.tsv. Decided rows apply to the samples of that study whose
     audit text matches `pattern`. You can also add your own rows: scope=sample, id=<sample_alias>.
  Re-running is safe: decided rows are carried into the new audit_review.tsv, so the same command gives
  the same output; a review file with decisions is also copied to audit_review.previous.tsv first.
Label map (optional, TSV): slot (biome|feature|material|*), from_id, to_id, reason. Applied to every
matching raw id, including obsolete and other-ontology ids (e.g. ENVO_00009003 for human biomes).
An empty to_id blanks the label on purpose.

cd ~/github/metadata_mining/scripts/ontology_mapping
python 2b_clean_metalog.py \
  --metalog_dir ~/MicrobeAtlasProject/metalog \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --training_set ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --output_dir ~/MicrobeAtlasProject/metalog/clean
"""

import argparse
import glob
import os
import re
import shutil
import warnings
from collections import Counter

import pandas as pd

from common import SLOTS, path, read_tsv, term_ancestors

warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")

DOMAINS = ["human", "animal", "ocean", "environmental"]
METALOG_SLOT = {"biome": "environment_biome", "feature": "environment_feature", "material": "environment_material"}
BUCKET_OF = {
    "negative control": "control", "mock": "control", "spike-in": "control", "marked as contaminated": "control",
    "cultivation": "perturbed", "enrichment": "perturbed", "virome enrichment": "perturbed",
    "sorted cells": "perturbed", "mesocosm": "perturbed",
    "paleosample": "degraded", "post-mortem": "degraded", "museum specimen": "degraded",
}
DECISIONS = {"control", "perturbed", "degraded", "ok"}
KEEP_ONTOLOGIES = {"ENVO", "UBERON", "PO", "FOODON"}  # labels from other ontologies (CL) are unusable
BIOME_ROOT = "ENVO_00000428"
CODE = re.compile(r"\[([A-Za-z]+):(\d+)\]")  # "soil [ENVO:00001998]", "leaf [PO:0025034]"
CONTROL_VALUES = {"negative control", "mock", "blank", "control"}

# Sample-level signs of an artificial sample. Suggestions for review, not decisions:
# e.g. "blank" also matches the place name "Blank Spring".
AUDIT = {
    "negative_control": ("control", r"\b(negative[ -]?controls?|neg[ -]?ctrls?|(extraction|kit|reagent|pcr|library)[ -]?(blanks?|controls?)"
                                    r"|blanks?|ntc|no[ -]?template[ -]?controls?)\b"),
    "mock_or_spike_in": ("control", r"\b(mock[ -]?(community|communities)|mock|zymo\w*|standard[ -]?community|spike[ -]?ins?)\b"),
    "cultivation": ("perturbed", r"\b(enrichment[ -]?cultures?|enrichments?(?![ -]?(kit|of))|co[ -]?cultures?|subcultur\w*"
                                 r"|cultur(e|es|ed|ing)|incubat(ed|ion|ions)|in[ -]?vitro(?![ -]?fertili))\b"),
    "mesocosm": ("perturbed", r"\b(mesocosms?|microcosms?|chemostats?|continuous[ -]?cultures?)\b"),
    "sorted_cells": ("perturbed", r"\b(facs|flow[ -]?sort\w*|cell[ -]?sort\w*|sorted[ -]?cells?|single[ -]?cells?"
                                  r"|whole[ -]?genome[ -]?amplif\w*)\b"),
    "virome_enrichment": ("perturbed", r"\b(vlps?|virus[ -]?like[ -]?particles?|viral[ -]?(enrichment|fraction|concentrate)s?)\b"),
    "ancient_or_post_mortem": ("degraded", r"\b(ancient|pal(a)?eo(?![ -]?diet)\w*|archa?eolog\w*|post[ -]?mortem|cadavers?|mumm(y|ies|ified)"
                                          r"|museum|herbarium|coprolites?)\b"),
}
AUDIT_RE = {name: re.compile(rx, re.I) for name, (_, rx) in AUDIT.items()}
# Only these keys are audited (plus the sample alias): free-text fields that describe the sample itself.
# Questionnaire answers ("if none leave blank", "yogurt with active cultures") would drown the review.
AUDIT_KEYS = re.compile(r"^(sample[ _]?)?(title|description|isolation[ _]?source|sample[ _]?type|type|notes?|original[ _]sample[ _]name"
                        r"|env[ _](biome|feature|material|medium|local[ _]scale|broad[ _]scale)|environment.*|added[ _]matter"
                        r"|treatment|condition|comments?)$", re.I)
# where the study part starts in a step-2 text (sample fields come first)
STUDY_PART = re.compile(r"(^|; )(study_title|study_abstract|study_description|abstract):")


def audited_fields(text):
    """The AUDIT_KEYS fields of a step-2 text 'key: value; key: value' (study part cut off first)."""
    kept = []
    for pair in STUDY_PART.split(text)[0].split("; "):
        key, _, value = pair.partition(": ")
        if value and AUDIT_KEYS.search(key.strip()):
            kept.append(pair)
    return "; ".join(kept)


# ----------------------------------------------------------------------------- inputs
def snapshot_files(metalog_dir, date):
    """{domain: (core_wide, all_long)} for one snapshot date (default: the latest one)."""
    dates = sorted({re.search(r"_(\d{4}-\d{2}-\d{2})\.tsv\.gz$", f).group(1)
                    for f in glob.glob(os.path.join(metalog_dir, "*_core_wide_*.tsv.gz"))})
    if not dates:
        raise SystemExit(f"no *_core_wide_<date>.tsv.gz files in {metalog_dir}")
    date = date or dates[-1]
    files = {}
    for domain in DOMAINS:
        core = os.path.join(metalog_dir, f"{domain}_core_wide_{date}.tsv.gz")
        long = os.path.join(metalog_dir, f"{domain}_all_long_{date}.tsv.gz")
        missing = [f for f in (core, long) if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"missing for snapshot {date}: {missing}")
        files[domain] = (core, long)
    return date, files


def load_core(files):
    columns = ["sample_alias", "study_code", "spire_sample_name", "artificial", *METALOG_SLOT.values()]
    frames = []
    for domain, (core, _) in files.items():
        df = read_tsv(core)
        df = df[[c for c in columns if c in df.columns]].copy()
        df.insert(0, "domain", domain)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).fillna("")


def load_audit_text(files, samples):
    """Per sample: its alias plus the 'key: value' pairs of the extended/all tiers (study keys skipped)."""
    parts = []
    for domain, (_, long) in files.items():
        df = pd.read_csv(long, sep="\t", dtype=str, keep_default_na=False)
        df = df[df["curation_tier"].isin(["extended", "all"]) & (df["value"] != "")
                & df["metadata_item"].str.contains(AUDIT_KEYS)]
        parts.append((df["metadata_item"] + ": " + df["value"]).groupby(df["sample_alias"]).agg("; ".join))
    extra = pd.concat(parts)
    extra = extra[~extra.index.duplicated()]
    text = samples["sample_alias"] + "; " + samples["sample_alias"].map(extra).fillna("")
    return text


def load_overrides(p):
    if not p:
        return pd.DataFrame(columns=["scope", "id", "pattern", "decision", "reason"])
    df = read_tsv(p)
    need = {"scope", "id", "pattern", "decision"}
    if not need <= set(df.columns):
        raise SystemExit(f"{p}: needs columns {sorted(need)} (audit_review.tsv has them)")
    df["decision"] = df["decision"].str.strip().str.lower()
    df = df[df["decision"] != ""]
    bad = df[~df["decision"].isin(DECISIONS)]
    if len(bad):
        raise SystemExit(f"{p}: unknown decisions {sorted(set(bad['decision']))}; use {sorted(DECISIONS)}")
    if "reason" not in df.columns:
        df["reason"] = ""
    return df


def load_label_map(p):
    """{(slot, from_id): to_id}; slot '*' applies to every slot."""
    if not p:
        return {}
    df = read_tsv(p)
    return {(r.slot.strip(), r.from_id.strip()): r.to_id.strip() for r in df.itertuples()}


# ----------------------------------------------------------------------------- flags
def flag_labels(samples, terms, label_map):
    ontology_ok = set(terms.loc[terms["obsolete"] != "True", "term_id"])
    obsolete = set(terms.loc[terms["obsolete"] == "True", "term_id"])
    ancestors = term_ancestors(terms)
    unknown_targets = Counter()

    for slot, field in METALOG_SLOT.items():
        raw = samples[field].str.strip()
        code = raw.str.extract(CODE)
        term = (code[0].str.upper() + "_" + code[1]).fillna("")
        status = pd.Series("ok", index=samples.index)
        status[term.ne("") & ~code[0].str.upper().isin(KEEP_ONTOLOGIES)] = "other_ontology"
        status[term.ne("") & code[0].str.upper().isin(KEEP_ONTOLOGIES) & term.isin(obsolete)] = "obsolete"
        status[term.ne("") & code[0].str.upper().isin(KEEP_ONTOLOGIES) & ~term.isin(obsolete) & ~term.isin(ontology_ok)] = "not_in_index"
        status[term.eq("") & raw.ne("")] = "no_code"
        status[term.eq("") & raw.str.lower().isin(CONTROL_VALUES)] = "control_value"
        status[raw.eq("")] = "empty"

        def clean(t, s):
            for key in ((slot, t), ("*", t)):
                if key in label_map:
                    target = label_map[key]
                    if target and target not in ontology_ok:
                        unknown_targets[target] += 1
                    return target
            return t if s == "ok" else ""

        samples[f"{slot}_raw"] = term
        samples[f"{slot}_status"] = status
        samples[f"{slot}_clean"] = [clean(t, s) for t, s in zip(term, status)]
        samples[f"{slot}_ontology"] = samples[f"{slot}_clean"].str.split("_").str[0]

    samples["biome_in_biome_subtree"] = [
        "" if not b else str(b == BIOME_ROOT or BIOME_ROOT in ancestors.get(b, ()))
        for b in samples["biome_clean"]]
    if unknown_targets:
        print(f"WARNING: label map targets not valid in the term index: {dict(unknown_targets)}")


def flag_duplicates(samples):
    """no_accession; an accession under several aliases: conflicting labels -> drop all, else keep the first."""
    reasons = pd.Series("", index=samples.index)
    missing = samples["spire_sample_name"].str.strip().str.lower().isin(["", "na", "nan", "none"])
    reasons[missing] = "no_accession"
    has = samples[~missing]
    labels = has[list(METALOG_SLOT.values())].agg("|".join, axis=1)
    n_label_sets = labels.groupby(has["spire_sample_name"]).transform("nunique")
    dup = has["spire_sample_name"].duplicated(keep=False)
    reasons[dup[dup & (n_label_sets > 1)].index] = "conflicting_duplicate"
    later = has["spire_sample_name"].duplicated(keep="first") & (n_label_sets == 1)
    reasons[later[later].index] = "duplicate_alias"
    return reasons


def audit(samples, audit_text):
    norm = audit_text.str.replace(r"[_.]+", " ", regex=True)  # 'NTC_1' -> 'NTC 1' so \b works
    hits = pd.Series([[] for _ in range(len(samples))], index=samples.index)
    for name, rx in AUDIT_RE.items():
        found = norm.str.contains(rx)
        for i in found[found].index:
            hits[i].append(name)
    samples["audit_hits"] = hits.map(";".join)
    return norm


def apply_overrides(samples, norm_text, overrides):
    samples["artificial_bucket"] = samples["artificial"].map(lambda v: BUCKET_OF.get(v.strip().lower(), "perturbed") if v.strip() else "none")
    samples["artificial_source"] = samples["artificial"].map(lambda v: "metalog" if v.strip() else "")
    samples["reviewed"] = False
    unknown = sorted({v for v in samples["artificial"] if v.strip() and v.strip().lower() not in BUCKET_OF})
    if unknown:
        print(f"WARNING: unknown Metalog `artificial` values treated as perturbed: {unknown}")
    n_applied = 0
    # study rows first, then sample rows, so a sample decision wins over its study's decision
    for row in sorted(overrides.itertuples(), key=lambda r: r.scope != "study"):
        if row.scope == "study":
            mask = samples["study_code"] == row.id
        elif row.scope == "sample":
            mask = samples["sample_alias"] == row.id
        else:
            raise SystemExit(f"override scope must be study or sample, got {row.scope!r}")
        if row.pattern:
            mask &= norm_text.str.contains(re.compile(row.pattern, re.I))
        if not mask.any():
            print(f"WARNING: override matched no sample: {row.scope} {row.id} {row.pattern[:40]!r}")
        samples.loc[mask, "artificial_bucket"] = "none" if row.decision == "ok" else row.decision
        samples.loc[mask, "artificial_source"] = "override"
        samples.loc[mask, "reviewed"] = True
        n_applied += int(mask.sum())
    return n_applied


def review_table(samples, audit_text):
    """One row per (study, category) for audit hits on samples that Metalog did not flag and nobody reviewed."""
    todo = samples[samples["audit_unreviewed"]]
    rows = []
    for (study, category), group in todo.assign(cat=todo["audit_hits"].str.split(";")).explode("cat").groupby(["study_code", "cat"]):
        rx = AUDIT_RE[category]
        examples = []
        for i in group.index[:3]:
            text = audit_text[i]
            m = rx.search(text.replace("_", " ").replace(".", " "))
            s = max(0, m.start() - 60) if m else 0
            examples.append(text[s:(m.end() + 60) if m else 120].replace("\t", " "))
        study_rows = samples[samples["study_code"] == study]
        flagged = study_rows["artificial"][study_rows["artificial"] != ""].value_counts().to_dict()
        rows.append({"scope": "study", "id": study, "category": category,
                     "suggested": AUDIT[category][0], "n_hit": len(group), "n_study": len(study_rows),
                     "metalog_flags_in_study": "; ".join(f"{k}={v}" for k, v in flagged.items()),
                     "labels": group["environment_material"].value_counts().index[0],
                     "examples": " || ".join(examples), "pattern": AUDIT[category][1],
                     "decision": "", "reason": ""})
    columns = ["scope", "id", "category", "suggested", "n_hit", "n_study", "metalog_flags_in_study",
               "labels", "examples", "pattern", "decision", "reason"]
    return pd.DataFrame(rows, columns=columns).sort_values(["n_hit"], ascending=False)


def backup_review(review_path, overrides_path):
    """Before overwriting an audit_review.tsv that holds decisions, copy it to audit_review.previous.tsv."""
    if not os.path.exists(review_path):
        return
    old = read_tsv(review_path)
    if "decision" not in old.columns or not old["decision"].str.strip().ne("").any():
        return
    shutil.copyfile(review_path, review_path.replace(".tsv", ".previous.tsv"))
    same = overrides_path and os.path.abspath(path(overrides_path)) == os.path.abspath(review_path)
    if not same:
        print(f"WARNING: {review_path} had {old['decision'].str.strip().ne('').sum()} decisions that were not passed "
              f"with --overrides; saved them to audit_review.previous.tsv")


# ----------------------------------------------------------------------------- training set
def clean_training_set(training_path, samples):
    ts = read_tsv(training_path)
    canonical = samples[samples["drop_reason"].ne("duplicate_alias") & samples["spire_sample_name"].ne("")]
    canonical = canonical.drop_duplicates("spire_sample_name").set_index("spire_sample_name")
    keep = ["drop_reason", "artificial", "artificial_bucket", "audit_hits", "audit_unreviewed", "in_gold_eval",
            *[f"{s}_clean" for s in SLOTS]]
    missing = ~ts["spire_sample_name"].isin(canonical.index)
    if missing.any():
        print(f"WARNING: {missing.sum()} training rows have no Metalog sample in this snapshot; dropped")
    ts = ts[~missing].join(canonical[keep], on="spire_sample_name")
    for slot in SLOTS:
        ts[f"{slot}_raw"] = ts[slot]
        ts[slot] = ts[f"{slot}_clean"]
    ts["dup_text_in_study"] = ts.duplicated(["study_code", "text"], keep="first")
    n_before = len(ts)
    ts = ts[ts["drop_reason"] == ""]
    columns = ["sample_id", "spire_sample_name", "study_code", "domain", *SLOTS, *[f"{s}_raw" for s in SLOTS],
               "artificial", "artificial_bucket", "audit_hits", "audit_unreviewed", "in_gold_eval",
               "dup_text_in_study", "text"]
    ts = ts[columns]
    gold = ts[ts["in_gold_eval"] & ~ts["dup_text_in_study"]]
    return ts, gold, n_before


# ----------------------------------------------------------------------------- report
def counts(series):
    return "\n".join(f"| {k if k != '' else '(none)'} | {v:,} |" for k, v in series.value_counts().items())


def write_summary(p, date, samples, review, n_overrides, ts_info):
    lines = [f"# Metalog cleaning flags, snapshot {date}", "",
             f"{len(samples):,} Metalog samples in {samples['study_code'].nunique():,} studies.", "",
             "## Hard drops (drop_reason)", "", "| reason | samples |", "|---|---|", counts(samples["drop_reason"]), "",
             "## Artificial bucket", "", "| bucket | samples |", "|---|---|", counts(samples["artificial_bucket"]), "",
             f"Samples changed by overrides: {n_overrides:,}. "
             f"Controls that still carry a usable label (copied from the study default, or 'mock community culture'): {int(samples['control_has_habitat_label'].sum()):,}.", "",
             "## Audit (unflagged samples that look artificial)", "",
             f"{int(samples['audit_unreviewed'].sum()):,} unreviewed samples, {len(review):,} study x category rows "
             "in audit_review.tsv.", "",
             "## Labels", "", "| slot | " + " | ".join(sorted({s for sl in SLOTS for s in samples[f'{sl}_status']})) + " |"]
    statuses = sorted({s for sl in SLOTS for s in samples[f"{sl}_status"]})
    lines.append("|---" * (len(statuses) + 1) + "|")
    for slot in SLOTS:
        vc = samples[f"{slot}_status"].value_counts()
        lines.append(f"| {slot} | " + " | ".join(f"{vc.get(s, 0):,}" for s in statuses) + " |")
    for slot in SLOTS:
        other = samples.loc[samples[f"{slot}_status"].isin(["other_ontology", "no_code", "obsolete"]), METALOG_SLOT[slot]]
        if len(other):
            lines += ["", f"Unusable {slot} values (top 10): " + "; ".join(f"{k} ({v})" for k, v in other.value_counts().head(10).items())]
    b = samples["biome_in_biome_subtree"]
    lines += ["", f"Clean biome labels under ENVO 'biome': {(b == 'True').sum():,} of {(b != '').sum():,}.", "",
              "## Use", "", "| flag | samples |", "|---|---|",
              f"| in_train | {int(samples['in_train'].sum()):,} |", f"| in_gold_eval | {int(samples['in_gold_eval'].sum()):,} |"]
    if ts_info:
        ts, gold, n_before = ts_info
        lines += ["", "## Linked training set", "",
                  f"{n_before:,} step-2 rows → {len(ts):,} clean (training) → {len(gold):,} gold "
                  f"(in_gold_eval, first copy of each text per study), {gold['study_code'].nunique():,} studies."]
    with open(p, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metalog_dir", required=True)
    parser.add_argument("--ontology_terms", required=True, help="TSV from 1_build_term_index.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--training_set", help="TSV from 2_build_training_set.py (writes the clean/gold training sets)")
    parser.add_argument("--overrides", help="audit_review.tsv with decisions filled in, or your own TSV with the same columns")
    parser.add_argument("--label_map", help="TSV: slot, from_id, to_id, reason")
    parser.add_argument("--date", help="snapshot date YYYY-MM-DD (default: the latest in --metalog_dir)")
    args = parser.parse_args()

    os.makedirs(path(args.output_dir), exist_ok=True)
    out = lambda name: os.path.join(path(args.output_dir), name)
    date, files = snapshot_files(path(args.metalog_dir), args.date)
    print(f"snapshot {date}")

    samples = load_core(files)
    print(f"{len(samples):,} samples, {samples['study_code'].nunique():,} studies")
    terms = read_tsv(args.ontology_terms)
    flag_labels(samples, terms, load_label_map(args.label_map))

    audit_text = load_audit_text(files, samples)
    if args.training_set:  # the MicrobeAtlas record adds sample-level text for linked samples
        ts_text = read_tsv(args.training_set).groupby("spire_sample_name")["text"].first()
        sample_part = samples["spire_sample_name"].map(ts_text).fillna("").map(audited_fields)
        audit_text = audit_text + "; " + sample_part
    norm_text = audit(samples, audit_text)
    overrides = load_overrides(args.overrides)
    n_overrides = apply_overrides(samples, norm_text, overrides)

    reasons = flag_duplicates(samples)
    control = samples["artificial_bucket"] == "control"
    reasons[control] = (reasons[control] + ";artificial_control").str.strip(";")
    samples["drop_reason"] = reasons
    samples["control_has_habitat_label"] = control & samples[[f"{s}_status" for s in SLOTS]].eq("ok").any(axis=1)
    samples["audit_unreviewed"] = samples["audit_hits"].ne("") & samples["artificial"].eq("") & ~samples["reviewed"]
    samples["in_train"] = samples["drop_reason"] == ""
    samples["in_gold_eval"] = samples["in_train"] & samples["artificial_bucket"].eq("none") & ~samples["audit_unreviewed"]

    review = review_table(samples, audit_text)
    # Keep decided rows in the new review file, so re-running with --overrides audit_review.tsv is
    # idempotent (the decided samples are no longer "unreviewed", so review_table drops them).
    if len(overrides):
        decided = overrides.reindex(columns=review.columns, fill_value="")
        key = lambda df: df["scope"] + "\t" + df["id"] + "\t" + df["category"]
        review = pd.concat([decided, review[~key(review).isin(set(key(decided)))]], ignore_index=True)
    backup_review(out("audit_review.tsv"), args.overrides)
    review.to_csv(out("audit_review.tsv"), sep="\t", index=False)
    samples["audit_text"] = audit_text.str.slice(0, 1000)
    samples.to_csv(out("metalog_flags.tsv.gz"), sep="\t", index=False)
    print(f"wrote {out('metalog_flags.tsv.gz')} and {out('audit_review.tsv')} ({(review['decision'] == '').sum()} rows to review, {(review['decision'] != '').sum()} decided)")

    ts_info = None
    if args.training_set:
        ts_info = clean_training_set(args.training_set, samples)
        ts_info[0].to_csv(out("training_set.clean.tsv.gz"), sep="\t", index=False)
        ts_info[1].to_csv(out("training_set.gold.tsv.gz"), sep="\t", index=False)
        print(f"training set: {ts_info[2]:,} rows -> {len(ts_info[0]):,} clean -> {len(ts_info[1]):,} gold")
    write_summary(out("summary.md"), date, samples, review, n_overrides, ts_info)
    print(open(out("summary.md")).read())


if __name__ == "__main__":
    main()
