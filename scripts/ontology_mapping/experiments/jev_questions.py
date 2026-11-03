"""
Everything Jev sees in the Jev pilot (jev_pilot.py), in one place: the state, the question texts, the option
descriptions and the Noul thresholds. Changing anything here changes the experiment: rerun the arms.

Written after TypeSafe's integration guidance (the typesafe-ai skill, https://docs.typesafe.ai):
- Jev makes judgments, code makes decisions. How specific the final answer is (the curators' level or the most
  specific term the metadata supports) is decided in code by the hierarchical back-off, not asked of Jev; so the
  questions carry no "prefer the general term" rule, unlike the GPT prompt (rerank_prompt.GRANULARITY).
- Plain questions anchored to the goal (the term the curators would assign), one judgment each; what an option
  means goes in its criteria, not in the question.
- The null case is not a "none" option: one Noul per candidate says whether it fits, and code decides "none"
  when no candidate reaches NOUL_NO (TypeSafe's skill-suggestion cookbook).
- Named state fields instead of one 'key: value; key: value' string.
- Noul thresholds 0.7 (yes) / 0.3 (no); the band between is "not settled".
"""
import re

SLOT = {
    "biome": "broad-scale environment (MIxS env_broad_scale), that is, its biome",
    "feature": "local environment (MIxS env_local_scale), that is, the environmental feature or host body part it was taken from",
    "material": "environmental material (MIxS env_medium), that is, the material that was sampled",
}
NOUL_YES, NOUL_NO = 0.7, 0.3
CURATORS = "expert curators annotating microbiome samples with ENVO and UBERON terms"


# ----------------------------------------------------------------------------- state
def metadata_fields(text):
    """'title: X; host: Y; geo_loc_name: Brazil: Minas Gerais' -> {"title": "X", "host": "Y", "geo_loc_name":
    "Brazil: Minas Gerais"}. A piece without 'key: ' continues the previous value (a value containing '; '); a
    repeated key keeps both values joined by ' | '. Text without any 'key: ' stays one 'text' field."""
    fields = {}
    last = None
    for piece in (text or "").split("; "):
        m = re.match(r"^([A-Za-z0-9_ ()/.-]{1,60}): (.*)$", piece, re.S)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            fields[key] = f"{fields[key]} | {value}" if key in fields else value
            last = key
        elif last is not None:
            fields[last] += "; " + piece
        elif piece.strip():
            fields["text"] = (fields.get("text", "") + "; " + piece).strip("; ")
    return fields


MIXS = {"biome": "env_broad_scale", "feature": "env_local_scale", "material": "env_medium"}


def state(text, similar=None, slot=None):
    """The request state: the sample's metadata as named fields; optionally similar samples from other studies with
    the term the curators assigned them for this slot (a fact the model lacks: how Metalog's curators label such
    samples). similar = [{"text": ..., "term": label}, ...], most similar first."""
    s = {"sample_metadata": metadata_fields(text)}
    if similar:
        s["similar_samples_labelled_by_curators"] = {
            f"sample_{i + 1}": {"sample_metadata": metadata_fields(x["text"]), f"{MIXS[slot]}_assigned": x["term"]}
            for i, x in enumerate(similar)}
    return s


# ----------------------------------------------------------------------------- questions
def which_term(slot):
    """Choice over candidate terms or the whole slot vocabulary."""
    return f"Which term would {CURATORS} assign as this sample's {SLOT[slot]}?"


def which_branch(slot, parent_label=None):
    """Choice at one node of the top-down walk: its children (often abstract classes such as 'environmental
    system', so the question asks which one covers the sample, not which one curators would assign), plus the
    node itself when it is a slot term."""
    under = f" under '{parent_label}'" if parent_label else ""
    return f"Which of these terms{under} covers this sample's {SLOT[slot]}?"


def term_fits(slot, term):
    """Noul per candidate term: is it a correct description (any level of detail)? Several can be true (a term and
    its broader terms); code then picks the most specific one judged true."""
    return {"type": "noul",
            "instructions": {"question": f"Does `term` describe this sample's {SLOT[slot]}?", "term": term},
            "criteria": {"true": "The term describes the sample's environment correctly, at its level of detail or a "
                                 "more general one.",
                         "false": "The term names a different environment, body site or material than the one the "
                                  "sample comes from, or something more specific than the metadata supports."}}


# ----------------------------------------------------------------------------- option descriptions
def describe(term, example=None, narrower=None):
    """A Choice option's (or a Noul term's) description: synonyms, definition, an example sample the curators
    labelled with it, and a few slot terms below it (top-down)."""
    d = {}
    if term.get("synonyms"):
        d["synonyms"] = "; ".join(term["synonyms"])
    if term.get("definition"):
        d["definition"] = term["definition"]
    if example:
        d["a similar sample the curators labelled with this term"] = metadata_fields(example)
    if narrower:
        d["narrower terms include"] = "; ".join(narrower)
    return d or None


def itself_option(label):
    """Top-down: the option that stops at the current node."""
    return (f"{label} itself, nothing narrower",
            {"meaning": f"The sample's environment is '{label}', and none of the narrower terms offered applies."})
