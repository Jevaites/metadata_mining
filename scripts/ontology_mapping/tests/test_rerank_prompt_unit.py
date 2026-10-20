"""Unit tests for rerank_prompt.py: one class per function, each test checks one behaviour on tiny inputs.

Conventions (as in test_common_unit.py)
  * Inputs are written inline, small enough that the expected output can be checked by eye.
  * Log-probabilities are written as math.log(0.6), so the probabilities stay readable.
  * Tests named `..._quirk` pin down current behaviour that is surprising but harmless today;
    tests marked xfail describe behaviour that is probably a bug (they start failing, on purpose,
    once the bug is fixed: then remove the marker).

THE TOY REQUEST used by most tests: 3 candidates, shown to the LLM in shuffled order.

    candidate index   label                        option_order = [2, 0, 1] shows it as letter
          0           marine biome                   B
          1           freshwater biome               C
          2           soil biome                     A
          3 (= k)     none of these terms fits       D   (always the last letter)

    so: A -> candidate 2,  B -> candidate 0,  C -> candidate 1,  D -> "none" (index k = 3)

parse() returns its distribution in CANDIDATE order [marine, freshwater, soil, none], never in
letter order, and its pick as a candidate index (k = "none").

See test_rerank_prompt_integration.py for the full prompt -> LLM -> parse round trip.
"""

import math
import re
import sys
import types

import numpy as np
import pytest

import rerank_prompt as rp

MARINE, FRESHWATER, SOIL, NONE = 0, 1, 2, 3  # candidate indices of the toy request (NONE = k)


# ----------------------------------------------------------------------------- helpers
def cand(label, synonyms=(), definition="", example=""):
    """One candidate term as prompt() expects it."""
    return {"label": label, "synonyms": list(synonyms), "definition": definition, "example": example}


def toy_request(option_order=(2, 0, 1), slot="biome"):
    """The toy request of the module docstring (A = soil, B = marine, C = freshwater, D = none)."""
    return {"slot": slot, "text": "isolation_source: seawater",
            "candidates": [cand("marine biome"), cand("freshwater biome"), cand("soil biome")],
            "option_order": list(option_order)}


def request_with_n_candidates(n):
    """n candidates 'term 0' .. 'term n-1', shown in their own order (letter A = term 0, ...)."""
    return {"slot": "material", "text": "x",
            "candidates": [cand(f"term {i}") for i in range(n)], "option_order": list(range(n))}


def answer(content, token_probs=None):
    """An LLM response as stored by the callers: the answer text and the top log-probabilities of the
    answer token, e.g. answer("B", [("B", 0.6), ("A", 0.4)]). token_probs=None: no log-probabilities."""
    raw = None if token_probs is None else [[tok, math.log(p)] for tok, p in token_probs]
    return {"content": content, "top_logprobs_raw": raw}


def user_text(messages):
    return messages[1]["content"]


def option_lines(messages):
    """The 'X. label' lines of the prompt, in letter order."""
    return [line for line in user_text(messages).splitlines() if re.match(r"^[A-Z]\. ", line)]


# ============================================================================= constants
class TestConstants:
    def test_prices_are_input_and_output_dollars_per_million_tokens(self):
        for model, (price_in, price_out) in rp.PRICES.items():
            assert 0 < price_in <= price_out, model  # output tokens never cost less than input tokens


# ============================================================================= prompt
class TestPrompt:
    def test_returns_a_system_and_a_user_message(self):
        messages = rp.prompt(toy_request(), "logprobs")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "ENVO and UBERON" in messages[0]["content"]

    def test_starts_with_the_sample_metadata(self):
        text = user_text(rp.prompt(toy_request(), "logprobs"))
        assert text.startswith("Sample metadata:\nisolation_source: seawater\n")

    def test_options_follow_option_order_and_none_is_always_last(self):
        lines = option_lines(rp.prompt(toy_request(option_order=[2, 0, 1]), "logprobs"))
        assert lines == ["A. soil biome", "B. marine biome", "C. freshwater biome", "D. none of these terms fits"]

    def test_identity_order_shows_candidates_as_given(self):
        lines = option_lines(rp.prompt(toy_request(option_order=[0, 1, 2]), "logprobs"))
        assert lines == ["A. marine biome", "B. freshwater biome", "C. soil biome", "D. none of these terms fits"]

    def test_synonyms_definition_and_example_are_shown_only_when_present(self):
        o = {"slot": "biome", "text": "x", "option_order": [0, 1], "candidates": [
            cand("marine biome", synonyms=["ocean biome", "sea biome"], definition="A biome dominated by salt water.",
                 example="isolation_source: surface seawater"),
            cand("soil biome"),  # no synonyms, no definition, no example: a single line
        ]}
        assert ("A. marine biome (synonyms: ocean biome; sea biome)\n"
                "   definition: A biome dominated by salt water.\n"
                "   example sample labelled with it: isolation_source: surface seawater\n"
                "B. soil biome\n"
                "C. none of these terms fits\n") in user_text(rp.prompt(o, "logprobs"))

    @pytest.mark.parametrize("slot, mixs_field", [("biome", "env_broad_scale"),
                                                  ("feature", "env_local_scale"),
                                                  ("material", "env_medium")])
    def test_the_question_names_the_slot(self, slot, mixs_field):
        assert f"(MIxS {mixs_field})" in user_text(rp.prompt(toy_request(slot=slot), "logprobs"))

    def test_unknown_slot_raises(self):
        with pytest.raises(KeyError):
            rp.prompt(toy_request(slot="host"), "logprobs")

    def test_v2_adds_the_granularity_hint_v1_does_not(self):
        assert rp.GRANULARITY in user_text(rp.prompt(toy_request(), "logprobs", version="v2"))
        assert rp.GRANULARITY not in user_text(rp.prompt(toy_request(), "logprobs", version="v1"))

    def test_unknown_version_raises(self):
        # a typo or a future "v3" must not silently get the v1 prompt
        with pytest.raises(ValueError):
            rp.prompt(toy_request(), "logprobs", version="v3")

    def test_logprobs_asks_for_a_single_letter(self):
        text = user_text(rp.prompt(toy_request(), "logprobs"))
        assert text.endswith("\n\nAnswer with the single letter of the best option, nothing else.")

    def test_verbal_asks_for_a_letter_and_a_0_to_100_confidence(self):
        text = user_text(rp.prompt(toy_request(), "verbal"))
        assert text.endswith("your confidence from 0 to 100 that it is right, e.g. 'B 70', nothing else.")

    def test_unknown_confidence_raises(self):
        with pytest.raises(ValueError):
            rp.prompt(toy_request(), "typo")

    def test_22_candidates_is_the_maximum(self):
        lines = option_lines(rp.prompt(request_with_n_candidates(22), "logprobs"))
        assert lines[-2:] == ["V. term 21", "W. none of these terms fits"]
        with pytest.raises(IndexError):  # 23 candidates: no letter left for "none"
            rp.prompt(request_with_n_candidates(23), "logprobs")


# ============================================================================= count_tokens
class TestCountTokens:
    def test_without_tiktoken_4_characters_per_token_rounded_down(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tiktoken", None)  # `import tiktoken` now raises ImportError
        assert rp.count_tokens("abcdefgh") == 2
        assert rp.count_tokens("abcdefghij") == 2
        assert rp.count_tokens("abc") == 0
        assert rp.count_tokens("") == 0

    def test_uses_tiktoken_o200k_base_when_available(self, monkeypatch):
        asked = []
        fake = types.ModuleType("tiktoken")  # stand-in: one "token" per word

        def get_encoding(name):
            asked.append(name)
            return types.SimpleNamespace(encode=lambda s: s.split())

        fake.get_encoding = get_encoding
        monkeypatch.setitem(sys.modules, "tiktoken", fake)
        assert rp.count_tokens("one two three") == 3
        assert asked == ["o200k_base"]

    def test_falls_back_when_tiktoken_cannot_load_its_encoding(self, monkeypatch):
        # tiktoken downloads the encoding on first use: offline, get_encoding raises
        fake = types.ModuleType("tiktoken")
        fake.get_encoding = lambda name: (_ for _ in ()).throw(OSError("no network"))
        monkeypatch.setitem(sys.modules, "tiktoken", fake)
        assert rp.count_tokens("abcdefgh") == 2


# ============================================================================= parse
class TestParseWithLogprobs:
    """Responses that carry the answer token's top log-probabilities (the normal case); test_real_answer_lines
    also parses a real verbal-mode line."""

    def test_distribution_is_in_candidate_order_not_letter_order(self):
        # letters: B = marine 0.6, A = soil 0.3, C = freshwater 0.1
        dist, pick = rp.parse(toy_request(), answer("B", [("B", 0.6), ("A", 0.3), ("C", 0.1)]))
        #                              marine freshwater soil none
        assert dist == pytest.approx([0.6, 0.1, 0.3, 0.0])
        assert pick == MARINE

    def test_renormalised_when_the_top_tokens_do_not_cover_all_the_mass(self):
        dist, _ = rp.parse(toy_request(), answer("B", [("B", 0.5), ("A", 0.3)]))  # 0.2 on tokens not listed
        assert dist == pytest.approx([0.5 / 0.8, 0.0, 0.3 / 0.8, 0.0])

    def test_spellings_of_the_same_letter_are_summed(self):
        dist, _ = rp.parse(toy_request(),
                           answer("B", [("B", 0.3), (" B", 0.2), ("b", 0.1), ("B.", 0.1), ("A", 0.3)]))
        assert dist == pytest.approx([0.7, 0.0, 0.3, 0.0])  # B = 0.3 + 0.2 + 0.1 + 0.1

    def test_tokens_that_are_not_option_letters_are_ignored(self):
        # "E" is one past the last option (D = none); "The" and "**" are not letters
        dist, _ = rp.parse(toy_request(), answer("B", [("B", 0.5), ("E", 0.2), ("The", 0.2), ("**", 0.1)]))
        assert dist == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_none_of_these(self):
        dist, pick = rp.parse(toy_request(), answer("D", [("D", 0.9), ("B", 0.1)]))
        assert dist == pytest.approx([0.1, 0.0, 0.0, 0.9])
        assert pick == NONE == len(toy_request()["candidates"])

    def test_pick_is_the_written_letter_even_when_the_summed_distribution_prefers_another(self):
        # The API writes the single most likely TOKEN ("C", 0.4), but "B" and " B" together have 0.6.
        # parse() keeps the written answer as the pick; the distribution only gives the confidence.
        dist, pick = rp.parse(toy_request(), answer("C", [("C", 0.4), ("B", 0.3), (" B", 0.3)]))
        assert dist == pytest.approx([0.6, 0.4, 0.0, 0.0])
        assert pick == FRESHWATER

    @pytest.mark.parametrize("content", ["B", " B\n", "b", "B.", "(B)", "**B**", "Answer: B", "B 70"])
    def test_letter_is_found_in_usual_answer_spellings(self, content):
        _, pick = rp.parse(toy_request(), answer(content, [("B", 1.0)]))
        assert pick == MARINE

    @pytest.mark.parametrize("content", [None, "", "Z", "E"])  # no answer / not an option letter
    def test_without_a_usable_written_letter_the_pick_is_the_most_likely_option(self, content):
        dist, pick = rp.parse(toy_request(), answer(content, [("A", 0.3), ("B", 0.7)]))
        assert dist == pytest.approx([0.7, 0.0, 0.3, 0.0])
        assert pick == MARINE  # argmax of dist

    def test_logprobs_take_precedence_over_a_stated_confidence(self):
        dist, pick = rp.parse(toy_request(), answer("B 70", [("B", 0.9), ("A", 0.1)]))
        assert dist == pytest.approx([0.9, 0.0, 0.1, 0.0])  # not 0.7
        assert pick == MARINE

    def test_token_spellings_seen_in_real_answers(self):
        # gpt-4.1-mini's top tokens include "\tC", "c", '"C', "AE" and the Cyrillic "\u0421" (looks like C):
        # tab / space prefixes and lower case are letter C; a quoted letter, two letters and Cyrillic are not options
        dist, pick = rp.parse(toy_request(), answer("C", [("C", 0.5), ("\tC", 0.1), ("c", 0.1), ('"C', 0.1),
                                                          ("\u0421", 0.05), ("AE", 0.05), ("B", 0.1)]))
        assert dist == pytest.approx([0.1 / 0.8, 0.7 / 0.8, 0.0, 0.0])  # C = freshwater 0.7, B = marine 0.1
        assert pick == FRESHWATER

    @pytest.mark.parametrize("r, expected", [
        # rerank_atlas/responses.jsonl, first line (gpt-4.1-mini, letter log-probabilities): A 0.953, E 0.047
        ({"request_id": "biome-b709b812855995e5", "model": "gpt-4.1-mini-2025-04-14", "content": "A",
          "top_logprobs_raw": [["A", -0.048587825149297714], ["E", -3.0485877990722656], ["C", -15.298587799072266],
                               ["F", -16.298587799072266], ["B", -16.673587799072266], [" A", -17.673587799072266],
                               ["D", -17.798587799072266], ["AE", -18.048587799072266], [" E", -20.048587799072266],
                               ["EA", -21.923587799072266]], "usage": {"in": 979, "out": 1}},
         {0: 0.9526, 4: 0.0474}),
        # experiments/rerank/resp_gpt-5.1.jsonl, first line (verbal mode, older file): E 0.7, the 6 others 0.05 each
        ({"slot": "biome", "sample_id": "SRS1115139", "model": "gpt-5.1-2025-11-13", "mode": "verbal", "content": "E 70",
          "top_logprobs": None, "usage": {"in": 866, "out": 12}},
         {0: 0.05, 4: 0.7, 6: 0.05}),
    ])
    def test_real_answer_lines(self, r, expected):
        o = request_with_n_candidates(6)  # letters A-F = term 0-5 (identity order), G = none
        dist, pick = rp.parse(o, r)
        assert pick == max(expected, key=expected.get)
        assert {i: dist[i] for i in expected} == pytest.approx(expected, abs=1e-4)

    def test_option_order_may_be_a_numpy_array(self):
        # a numpy array must not be added element-wise to [k] ([2, 0, 1] + 3 = [5, 3, 4])
        o = toy_request()
        o["option_order"] = np.array([2, 0, 1])
        dist, pick = rp.parse(o, answer("B", [("B", 0.6), ("A", 0.4)]))
        assert dist == pytest.approx([0.6, 0.0, 0.4, 0.0])
        assert pick == MARINE


class TestParseWithoutLogprobs:
    """Responses without log-probabilities: a model asked for a stated confidence ('verbal'), or a file
    written before 2026-10-01. The pick gets the confidence, the other k options share the rest equally."""

    def test_letter_alone_gets_confidence_0_8(self):
        dist, pick = rp.parse(toy_request(), answer("B"))
        rest = 0.2 / 3  # shared by freshwater, soil and none
        assert dist == pytest.approx([0.8, rest, rest, rest])
        assert pick == MARINE

    @pytest.mark.parametrize("content", ["B 70", "B 70%", "B: 70", "b 70", "B 70 because seawater"])
    def test_stated_confidence(self, content):
        dist, pick = rp.parse(toy_request(), answer(content))
        assert dist == pytest.approx([0.7, 0.1, 0.1, 0.1])
        assert pick == MARINE

    def test_stated_confidence_is_clamped_to_0_01_0_99(self):
        assert rp.parse(toy_request(), answer("B 100"))[0][MARINE] == pytest.approx(0.99)
        assert rp.parse(toy_request(), answer("B 0"))[0][MARINE] == pytest.approx(0.01)

    @pytest.mark.parametrize("content", ["B (70)", "B 70.5", "B 0.7", "70 B"])
    def test_confidence_in_another_format_is_ignored_quirk(self, content):
        # only a whole number written after the letter is read; otherwise the default 0.8 is used
        assert rp.parse(toy_request(), answer(content))[0][MARINE] == pytest.approx(0.8)

    def test_none_of_these(self):
        dist, pick = rp.parse(toy_request(), answer("D 90"))
        assert dist == pytest.approx([0.1 / 3, 0.1 / 3, 0.1 / 3, 0.9])
        assert pick == NONE

    def test_empty_logprob_list_counts_as_no_logprobs(self):
        assert rp.parse(toy_request(), {"content": "B 70", "top_logprobs_raw": []})[0][MARINE] == pytest.approx(0.7)

    def test_old_files_with_a_top_logprobs_dict_use_only_the_written_letter(self):
        # before 2026-10-01 the log-probabilities were stored as a dict under another key: not read
        r = {"content": "B", "top_logprobs": {"B": math.log(0.6), "A": math.log(0.4)}}
        assert rp.parse(toy_request(), r)[0][MARINE] == pytest.approx(0.8)

    @pytest.mark.parametrize("content", [None, "", "Not enough information.", "Z", "E"])
    def test_no_usable_letter_gives_none_none(self, content):
        assert rp.parse(toy_request(), answer(content)) == (None, None)

    def test_logprobs_only_on_non_option_tokens_and_no_letter_gives_none_none(self):
        assert rp.parse(toy_request(), answer("Not sure", [("Not", 0.9), ("E", 0.1)])) == (None, None)


class TestAnswerLetter:
    """answer_letter(text, letters): which option letter the answer text names (used by parse())."""

    ABCD = "ABCD"            # 3 candidates + none
    A_TO_K = "ABCDEFGHIJK"   # 10 candidates + none: "I" is a valid option here

    @pytest.mark.parametrize("text, expected", [
        ("B", "B"), ("b", "B"), (" B\n", "B"), ("B.", "B"), ("(B)", "B"), ("**B**", "B"),
        ("Answer: B", "B"), ("The answer is B", "B"), ("B 70", "B"), ("B. marine biome", "B"),
        ("B is a good fit", "B"),        # "a" comes after the answer
        ("D", "D"),                      # the "none" letter is a valid answer
    ])
    def test_usual_answers(self, text, expected):
        assert rp.answer_letter(text, self.ABCD) == expected

    @pytest.mark.parametrize("text, expected", [
        ("I think B", "B"),              # pronoun "I" followed by a word: skipped
        ("I think B is right", "B"),
        ("I'd pick C", "C"),             # "I" touching an apostrophe is not a stand-alone letter
        ("It's C", "C"),                 # nor is the "s" of "It's"
        ("a marine biome: C", "C"),      # article "a" followed by a word: skipped
        ("I", "I"),                      # "I" alone is option I
        ("I.", "I"),
        ("I is the best fit", "I"),      # only "I": kept, even though it looks like the pronoun
        ("A is the best fit", "A"),      # capital "A" is never taken for the article
    ])
    def test_english_one_letter_words(self, text, expected):
        assert rp.answer_letter(text, self.A_TO_K) == expected

    @pytest.mark.parametrize("text", [None, "", "Not enough information.", "Z", "E", "B2", "x_B"])
    def test_no_valid_letter(self, text):
        assert rp.answer_letter(text, self.ABCD) is None  # E is one past "none"; "B2", "x_B" are not stand-alone

    def test_letters_that_are_not_options_are_skipped(self):
        assert rp.answer_letter("Z or B", self.ABCD) == "B"


class TestParseLetterExtraction:
    def test_the_word_I_is_not_mistaken_for_option_I(self):
        o = request_with_n_candidates(10)  # letters A..J = term 0..9, K = none: "I" is a valid option
        _, pick = rp.parse(o, answer("I think B"))
        assert pick == 1  # B = term 1, not 8 (option I)

