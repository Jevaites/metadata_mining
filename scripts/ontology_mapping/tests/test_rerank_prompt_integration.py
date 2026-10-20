"""Integration tests for rerank_prompt.py: the round trip the LLM reranker makes for one sample,

    request record o  --prompt()-->  chat messages  --LLM-->  answer r  --parse()-->  (dist, pick)

with a FAKE LLM that only sees the prompt text, exactly like the real one. The fake therefore checks
that the letters written by prompt() and the letters read back by parse() agree.

THE WORKFLOW (as in 7_rerank_atlas.py and experiments/rerank_pilot.py)
  1. For one sample and one slot, build the request record
         o = {"slot": "biome", "text": <sample metadata>, "candidates": [...], "option_order": [...]}
     candidates: the embedding model's top terms, best first, each {label, synonyms, definition, example}.
     option_order: a random permutation of range(len(candidates)); option letter i shows
     candidates[option_order[i]], so the LLM cannot see the embedding model's ranking.
  2. messages = prompt(o, "logprobs")      -> chat API with logprobs=True, top_logprobs=5 (gpt-5) or 10
     (models that refuse log-probabilities: prompt(o, "verbal") -> answer like "B 70")
  3. store the answer as r = {"content": <answer text>, "top_logprobs_raw": [[token, logprob], ...]}
  4. dist, pick = parse(o, r)
         dist[i], i < k   LLM probability of candidates[i]     (CANDIDATE order, not letter order)
         dist[k]          probability of "none of these terms fits"
         pick             index into candidates, or k for "none"
         (None, None)     the answer is unusable: the caller keeps the embedding model's prediction

THE TOY SAMPLE (fake term ids)
  metadata    "isolation_source: seawater from the oxygen minimum zone; depth: 200 m"
  candidates  0 marine biome (ENVO_00000001)   1 freshwater biome (ENVO_00000002)   2 soil biome (ENVO_00000003)
  shown as    option_order [2, 0, 1]  ->  A = soil biome,  B = marine biome,  C = freshwater biome,  D = none
  fake LLM    believes marine biome 0.7, freshwater biome 0.2, soil biome 0.1
  expected    the LLM answers "B"; parse() gives [0.7, 0.2, 0.1, 0.0] and pick 0 = marine biome
"""

import itertools
import math
import re

import pytest

import rerank_prompt as rp

NONE_LABEL = "none of these terms fits"

SEAWATER = "isolation_source: seawater from the oxygen minimum zone; depth: 200 m"
CANDIDATES = [
    {"term_id": "ENVO_00000001", "label": "marine biome", "synonyms": ["ocean biome", "sea biome"],
     "definition": "A biome dominated by salt water.", "example": "isolation_source: surface seawater; depth: 5 m"},
    {"term_id": "ENVO_00000002", "label": "freshwater biome", "synonyms": [],
     "definition": "A biome dominated by fresh water.", "example": "isolation_source: lake water"},
    {"term_id": "ENVO_00000003", "label": "soil biome", "synonyms": [], "definition": "", "example": ""},
]


def request(text=SEAWATER, option_order=(2, 0, 1), slot="biome", candidates=CANDIDATES):
    return {"slot": slot, "text": text, "candidates": candidates, "option_order": list(option_order)}


def fake_llm(messages, beliefs, confidence="logprobs"):
    """Stands in for the chat API. beliefs: {option label: probability}, e.g. {"marine biome": 0.7}.
    It reads which letter each label got from the prompt (lines 'B. marine biome (synonyms: ...)') and
    answers like the API: the most likely letter, plus the top log-probabilities of that answer token
    ("logprobs"), or the letter and a 0-100 confidence without log-probabilities ("verbal")."""
    letter_of = {label: letter for letter, label in
                 re.findall(r"^([A-Z])\. (.+?)(?: \(synonyms: .*\))?$", messages[1]["content"], re.M)}
    probs = {letter_of[label]: p for label, p in beliefs.items()}  # KeyError = label not in the prompt
    best = max(probs, key=probs.get)
    if confidence == "logprobs":
        return {"content": best,
                "top_logprobs_raw": [[letter, math.log(p)] for letter, p in sorted(probs.items(), key=lambda x: -x[1])]}
    return {"content": f"{best} {round(100 * probs[best])}", "top_logprobs_raw": None}


BELIEFS_SEAWATER = {"marine biome": 0.7, "freshwater biome": 0.2, "soil biome": 0.1}


# ============================================================================= the basic round trip
def test_the_prompt_the_llm_sees():
    """The exact v2 prompt for the toy sample. The fusion weights fitted in step 7 are only valid for this
    exact text: if this test fails, the prompt changed and the weights must be refitted (new version)."""
    messages = rp.prompt(request(), "logprobs")
    assert messages[0] == {"role": "system", "content": (
        "You are an expert curator of microbiome sample metadata who annotates samples with ENVO and UBERON "
        "terms, following the conventions of the Metalog database.")}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == """\
Sample metadata:
isolation_source: seawater from the oxygen minimum zone; depth: 200 m

Choose the ontology term that best describes this sample's broad-scale environment (MIxS env_broad_scale): the biome or major environmental system the sample comes from.
Each option shows how expert curators used the term, with the most similar sample they labelled with it.
Several options can be true at different levels of detail (e.g. 'sediment' and 'marine sediment'). Choose the level of detail the curators would use: follow the examples, and prefer the more general term unless the metadata explicitly supports the more specific one.

A. soil biome
B. marine biome (synonyms: ocean biome; sea biome)
   definition: A biome dominated by salt water.
   example sample labelled with it: isolation_source: surface seawater; depth: 5 m
C. freshwater biome
   definition: A biome dominated by fresh water.
   example sample labelled with it: isolation_source: lake water
D. none of these terms fits

Answer with the single letter of the best option, nothing else."""


def test_round_trip_gives_the_llm_distribution_over_the_candidates():
    o = request()
    r = fake_llm(rp.prompt(o, "logprobs"), BELIEFS_SEAWATER)
    assert r["content"] == "B"  # marine biome is shown as B
    dist, pick = rp.parse(o, r)
    #                              marine freshwater soil none
    assert dist == pytest.approx([0.7, 0.2, 0.1, 0.0])
    assert o["candidates"][pick]["term_id"] == "ENVO_00000001"  # marine biome


@pytest.mark.parametrize("option_order", list(itertools.permutations([0, 1, 2])))
def test_the_shuffle_is_undone_whatever_the_option_order(option_order):
    """Marine biome gets a different letter in each of the 6 orders, but parse() always returns the
    same distribution in candidate order and the same pick."""
    o = request(option_order=option_order)
    r = fake_llm(rp.prompt(o, "logprobs"), BELIEFS_SEAWATER)
    assert r["content"] == "ABC"[option_order.index(0)]  # the letter showing candidate 0 (marine)
    dist, pick = rp.parse(o, r)
    assert dist == pytest.approx([0.7, 0.2, 0.1, 0.0])
    assert pick == 0


def test_the_llm_rejects_every_candidate():
    o = request(text="sample_type: mock community DNA standard")
    r = fake_llm(rp.prompt(o, "logprobs"), {NONE_LABEL: 0.85, "marine biome": 0.1, "freshwater biome": 0.05})
    assert r["content"] == "D"  # "none" is always the letter after the last candidate
    dist, pick = rp.parse(o, r)
    assert dist == pytest.approx([0.1, 0.05, 0.0, 0.85])
    assert pick == len(o["candidates"])  # k = "none": no candidate term is chosen


def test_a_model_without_logprobs_states_its_confidence():
    """Verbal mode: the stated confidence goes to the pick and the other 3 options share the rest
    equally, so the LLM's second choice (freshwater, 0.2) is lost."""
    o = request()
    messages = rp.prompt(o, "verbal")
    r = fake_llm(messages, BELIEFS_SEAWATER, confidence="verbal")
    assert r == {"content": "B 70", "top_logprobs_raw": None}
    dist, pick = rp.parse(o, r)
    assert dist == pytest.approx([0.7, 0.1, 0.1, 0.1])
    assert pick == 0


# ============================================================================= edge cases
def test_the_largest_request_22_candidates():
    """22 candidates use letters A..V and "none" is W. Shown in reverse order, candidate 0 is letter V."""
    candidates = [{"label": f"term {i}", "synonyms": [], "definition": "", "example": ""} for i in range(22)]
    o = request(candidates=candidates, option_order=list(reversed(range(22))))
    r = fake_llm(rp.prompt(o, "logprobs"), {"term 0": 0.9, NONE_LABEL: 0.1})
    assert r["content"] == "V"
    dist, pick = rp.parse(o, r)
    assert pick == 0
    assert dist[0] == pytest.approx(0.9) and dist[22] == pytest.approx(0.1)  # dist[22] = none


def test_a_single_candidate():
    """One candidate: A = the candidate, B = none."""
    o = request(candidates=CANDIDATES[:1], option_order=[0])
    r = fake_llm(rp.prompt(o, "logprobs"), {"marine biome": 0.6, NONE_LABEL: 0.4})
    assert rp.parse(o, r)[0] == pytest.approx([0.6, 0.4])

