"""
The LLM reranker's prompt and the parsing of its answer, shared by experiments/rerank_pilot.py (the
Metalog pilot) and 7_rerank_atlas.py (the atlas). The fitted fusion weights of step 7 are only valid
for this exact prompt (version v2) and the model they were fitted with.

The prompt shows the sample's metadata and the candidate terms as lettered options (label, synonyms,
definition, and the most similar sample curators labelled with the term), plus "none of these".
The model answers with one letter; its token log-probabilities give a distribution over the options.

Example (shortened):
  Sample metadata:
  isolation_source: Seawater from oxygen minimum zone; ...
  Choose the ontology term that best describes this sample's broad-scale environment (MIxS env_broad_scale) ...
  A. marine biome (synonyms: ...)
     definition: ...
     example sample labelled with it: ...
  B. ocean biome ...
  C. none of these terms fits
  Answer with the single letter of the best option, nothing else.
"""
import math
import re

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVW"  # up to 22 candidates + "none"
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4.1-nano": (0.10, 0.40),
          "gpt-5.1": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0)}  # $ / 1M input, output tokens
SLOT_TEXT = {
    "biome": "broad-scale environment (MIxS env_broad_scale): the biome or major environmental system the sample comes from",
    "feature": "local environment (MIxS env_local_scale): the environmental feature or host part the sample was taken from",
    "material": "environmental material (MIxS env_medium): the material that was sampled",
}
GRANULARITY = ("Several options can be true at different levels of detail (e.g. 'sediment' and 'marine sediment'). "
               "Choose the level of detail the curators would use: follow the examples, and prefer the more general "
               "term unless the metadata explicitly supports the more specific one.")


def prompt(o, confidence, version="v2"):
    """Chat messages for one request. o: {"slot", "text", "candidates": [{label, synonyms, definition,
    example}], "option_order": candidate index per letter (shuffled, so the base model's rank is hidden)}.
    confidence: "logprobs" (answer = one letter) or "verbal" (letter + 0-100). v1 lacks GRANULARITY."""
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown prompt version {version!r}: 'v1' or 'v2'")
    if confidence not in ("logprobs", "verbal"):
        raise ValueError(f"unknown confidence {confidence!r}: 'logprobs' or 'verbal'")
    opts = [o["candidates"][i] for i in o["option_order"]]
    lines = [f"Sample metadata:\n{o['text']}\n",
             f"Choose the ontology term that best describes this sample's {SLOT_TEXT[o['slot']]}.",
             "Each option shows how expert curators used the term, with the most similar sample they labelled with it."]
    if version == "v2":
        lines.append(GRANULARITY)
    lines.append("")
    for letter, c in zip(LETTERS, opts):
        syn = f" (synonyms: {'; '.join(c['synonyms'])})" if c["synonyms"] else ""
        lines.append(f"{letter}. {c['label']}{syn}")
        if c["definition"]:
            lines.append(f"   definition: {c['definition']}")
        if c["example"]:
            lines.append(f"   example sample labelled with it: {c['example']}")
    lines.append(f"{LETTERS[len(opts)]}. none of these terms fits")
    if confidence == "logprobs":
        lines.append("\nAnswer with the single letter of the best option, nothing else.")
    else:
        lines.append("\nAnswer with the letter of the best option and your confidence from 0 to 100 that it is right, "
                     "e.g. 'B 70', nothing else.")
    system = ("You are an expert curator of microbiome sample metadata who annotates samples with ENVO and UBERON "
              "terms, following the conventions of the Metalog database.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def count_tokens(text):
    """Tokens of `text` for the cost estimate (o200k_base; ~4 characters per token without tiktoken)."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


# a single letter standing alone: not touching letters, digits, "_" or apostrophes ("It's" has no "S")
_STANDALONE_LETTER = re.compile(r"(?<![\w'\u2019])([A-Za-z])(?![\w'\u2019])")


def answer_letter(text, letters):
    """The option letter written in the answer `text` (upper case), or None. letters: the valid letters.
    Takes the first stand-alone letter that is a valid option, skipping the English words "I" (pronoun)
    and "a" (article) when a lower-case word follows: "I think B" -> B, "a marine biome: C" -> C. If every
    valid letter looks like such a word, the first one is kept ("I is the best fit" -> I)."""
    found = [m for m in _STANDALONE_LETTER.finditer(text or "") if m.group(1).upper() in letters]
    if not found:
        return None
    is_word = lambda m: m.group(1) in ("I", "a") and re.match(r"\s+[a-z]", text[m.end():])
    return next((m for m in found if not is_word(m)), found[0]).group(1).upper()


def parse(o, r):
    """-> (reranker distribution over the k candidates + 'none', in candidate order; chosen index), or
    (None, None) without a usable answer. r: {"content": the answer text, "top_logprobs_raw": [[token,
    logprob], ...] of the answer token}. Example: options A-C + D = none, answer "B" with log-probabilities
    B -0.05, A -3.2 -> dist puts 0.95 on the candidate shown as B, 0.04 on A's, pick = B's candidate."""
    k = len(o["candidates"])
    letters = LETTERS[:k + 1]
    order = list(o["option_order"]) + [k]  # letter position -> candidate index (k = none); list() for numpy arrays
    text = (r["content"] or "").strip()
    choice_letter = answer_letter(text, letters)
    dist = np.zeros(k + 1)
    # log-probabilities of the answer token; tokens of the same letter (" B", "B") are summed.
    # Files written before 2026-10-01 kept a dict whose keys were stripped, so " B" overwrote "B":
    # those are not usable and only the chosen letter is kept.
    for tok, lp in (r.get("top_logprobs_raw") or []):
        t = tok.strip().upper().rstrip(".")
        if len(t) == 1 and t in letters:
            dist[order[letters.index(t)]] += math.exp(lp)
    if dist.sum() <= 0:  # no usable log-probabilities: stated confidence or plain choice
        if choice_letter is None:
            return None, None
        conf = 0.8
        nums = [int(x) for x in text.replace("%", " ").split()[1:] if x.isdigit()]
        if nums:
            conf = min(max(nums[0] / 100, 0.01), 0.99)
        dist[:] = (1 - conf) / k
        dist[order[letters.index(choice_letter)]] = conf
    dist = dist / dist.sum()
    # the pick is the letter the model wrote (greedy); the distribution only gives its confidence
    pick = order[letters.index(choice_letter)] if choice_letter else int(np.argmax(dist))
    return dist, pick

