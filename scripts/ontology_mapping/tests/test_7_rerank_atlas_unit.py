"""Unit tests for 7_rerank_atlas.py: the pure functions, each on inputs small enough to check by eye.

What step 7 does, in one line per function (the order in which the steps use them):
    fit      pilot_rows   pilot JSONL + LLM responses -> one row per pilot sample (candidates, log base probs, ...)
             llm_logprob  the LLM's answer(s) -> summed log-probability of each candidate
             fuse         p ∝ exp((log p_base + w · log p_llm) / T) over the candidates; base mass outside kept aside
             fit_T        the temperature T that maximises the likelihood of the gold term
             outcomes     fused probabilities -> back-off answer -> exact / coarser / too specific / other / abstain
             fit_slot     the weight w (from a grid), T and tau with the most exact answers at a target accuracy
    build    parse_candidates   "id:p;id:p" (6_predict_atlas.py's <slot>_candidates) -> [(id, p)]
             estimate           token count and Batch API cost of the requests
    submit   body_for           one chat-completions request body per request
The commands themselves (fit, build, submit, collect, apply) are in test_7_rerank_atlas_integration.py.

Toy hierarchy used throughout (step7_toy.ANC):
    aquatic
    ├── marine
    │   └── ocean
    └── freshwater
        └── lake
"""

import math

import numpy as np
import pytest

from step7_toy import ANC, LABEL, load_rerank_atlas, write_jsonl

rr = load_rerank_atlas()
log = np.log
OUTCOME = {0: "abstain", 1: "exact", 2: "coarser", 3: "too_specific", 4: "other"}  # hierarchy's outcome codes


# ----------------------------------------------------------------------------- helpers
def request(cands, option_order=None, slot="biome"):
    """A pilot / atlas request as parse() and prompt() read it: candidates and the letter order.
    option_order[i] = the candidate shown as letter i (A, B, ...); default: in candidate order."""
    return {"slot": slot, "sample_id": "S1", "text": "salinity: 35 PSU",
            "candidates": [{"term_id": t, "label": LABEL[t], "synonyms": [], "definition": "", "example": ""} for t in cands],
            "option_order": list(range(len(cands))) if option_order is None else option_order}


def answer(**letter_probs):
    """An LLM response as collect() stores it: answer(A=0.9, B=0.1) -> content 'A' and the letters' log-probs."""
    best = max(letter_probs, key=letter_probs.get)
    return {"content": best, "top_logprobs_raw": [[letter, math.log(p)] for letter, p in letter_probs.items()]}


def row(cands, base, gold, llm=None):
    """One pilot row as pilot_rows() returns it (llm: the LLM's probabilities of the candidates, default uniform)."""
    llm = [1 / len(cands)] * len(cands) if llm is None else llm
    return {"cands": cands, "lb": log(base), "D": log(llm), "mass": float(sum(base)), "gold": gold, "study": "x",
            "gold_idx": cands.index(gold) if gold in cands else len(cands)}


def names(codes):
    return [OUTCOME[c] for c in codes]


# ============================================================================= fuse
class TestFuse:
    """fuse(lb, D, w, T, mass) -> [p(candidate 1), ..., p(candidate k), p(outside the candidates)]."""

    def test_docstring_example_the_llm_overturns_the_base_ranking(self):
        # base 0.5 vs 0.2, LLM 0.1 vs 0.9: products 0.05 vs 0.18 -> shares 0.05/0.23 and 0.18/0.23 of the
        # candidates' mass 0.7; the 0.3 outside the candidates is untouched.
        p = rr.fuse(log([0.5, 0.2]), log([0.1, 0.9]), w=1, T=1, mass=0.7)
        assert p == pytest.approx([0.05 / 0.23 * 0.7, 0.18 / 0.23 * 0.7, 0.3])  # = [0.152, 0.548, 0.3]

    def test_weight_zero_returns_the_base_probabilities(self):
        p = rr.fuse(log([0.5, 0.2]), log([0.1, 0.9]), w=0, T=1, mass=0.7)
        assert p == pytest.approx([0.5, 0.2, 0.3])  # the LLM is ignored

    def test_an_llm_that_cannot_decide_changes_nothing_whatever_the_weight(self):
        for w in (0.1, 1, 10):
            assert rr.fuse(log([0.5, 0.2]), log([0.5, 0.5]), w=w, T=1, mass=0.7) == pytest.approx([0.5, 0.2, 0.3])

    def test_temperature_below_one_sharpens_above_one_flattens(self):
        # T = 0.5 squares the probabilities: 0.25 vs 0.04 -> shares 0.25/0.29 and 0.04/0.29 of 0.7
        assert rr.fuse(log([0.5, 0.2]), log([1, 1]), w=0, T=0.5, mass=0.7) == pytest.approx([0.25 / 0.29 * 0.7, 0.04 / 0.29 * 0.7, 0.3])
        # a huge T makes the candidates equal: 0.7 split in two
        assert rr.fuse(log([0.5, 0.2]), log([1, 1]), w=0, T=1e6, mass=0.7) == pytest.approx([0.35, 0.35, 0.3])

    def test_the_mass_outside_the_candidates_is_never_given_to_them(self):
        # however sure the LLM is, 'outside' keeps 1 - mass: the reranker can only reshuffle the candidates' mass
        p = rr.fuse(log([0.4, 0.2]), log([0.999, 0.001]), w=5, T=0.1, mass=0.6)
        assert p[-1] == pytest.approx(0.4)
        assert p.sum() == pytest.approx(1)

    def test_mass_rounded_above_one_gives_zero_outside_never_negative(self):
        # <slot>_candidates holds probabilities with 4 significant digits, so they can sum to slightly more than 1
        p = rr.fuse(log([0.5001, 0.5001]), log([0.5, 0.5]), w=1, T=1, mass=1.0002)
        assert p[-1] == 0.0
        assert p[:-1] == pytest.approx([0.5001, 0.5001])


# ============================================================================= llm_logprob
class TestLlmLogprob:
    """llm_logprob(request, responses) -> per candidate, the sum over responses of log p_llm(candidate)."""

    def test_letters_are_mapped_back_to_candidates_through_option_order(self):
        # option_order [1, 0]: letter A shows candidate 1 (freshwater), letter B shows candidate 0 (marine)
        o = request(["marine", "freshwater"], option_order=[1, 0])
        D = rr.llm_logprob(o, [answer(A=0.9, B=0.1)])
        assert D == pytest.approx(log([0.1, 0.9]))  # in candidate order: marine 0.1, freshwater 0.9

    def test_two_llms_are_summed_in_log_space(self):
        o = request(["marine", "freshwater"])
        D = rr.llm_logprob(o, [answer(A=0.8, B=0.2), answer(A=0.5, B=0.5)])
        assert D == pytest.approx(log([0.8 * 0.5, 0.2 * 0.5]))  # a product of the two LLMs' probabilities

    def test_an_unusable_response_is_skipped_and_none_usable_gives_none(self):
        o = request(["marine", "freshwater"])
        unusable = {"content": "I am not sure", "top_logprobs_raw": None}  # no letter, no log-probabilities
        assert rr.llm_logprob(o, [unusable]) is None
        assert rr.llm_logprob(o, []) is None
        assert rr.llm_logprob(o, [unusable, answer(A=0.8, B=0.2)]) == pytest.approx(log([0.8, 0.2]))

    def test_a_candidate_missing_from_the_top_logprobs_gets_a_floor_of_1e_6(self):
        o = request(["marine", "freshwater", "aquatic"])
        D = rr.llm_logprob(o, [answer(A=0.75, B=0.25)])  # C (aquatic) not among the returned tokens
        assert D == pytest.approx(log([0.75, 0.25, 1e-6]))

    def test_quirk_the_llms_none_of_these_vote_is_dropped(self):
        # Options A, B + C = 'none of these'. The LLM puts half its mass on 'none'; only the candidates are
        # kept (0.25, 0.25), so after fusion that answer is the same as an LLM that cannot decide.
        o = request(["marine", "freshwater"])
        D = rr.llm_logprob(o, [answer(A=0.25, B=0.25, C=0.5)])
        assert D == pytest.approx(log([0.25, 0.25]))


# ============================================================================= fit_T
class TestFitT:
    """fit_T(rows, w): the temperature that maximises the likelihood of the gold candidate.
    Two candidates with base 0.75 / 0.25 (mass 1): at temperature T, p(first) = 1 / (1 + 3^(-1/T))."""

    def test_already_calibrated_base_keeps_temperature_one(self):
        # first candidate right in 3 of 4 rows, and the base says 0.75: nothing to correct
        rows = [row(["marine", "freshwater"], [0.75, 0.25], "marine")] * 3 + [row(["marine", "freshwater"], [0.75, 0.25], "freshwater")]
        assert rr.fit_T(rows, w=0) == pytest.approx(1.0, rel=1e-3)

    def test_underconfident_base_is_sharpened(self):
        # right in 9 of 10 rows but says 0.75: p(first) = 0.9 needs 3^(-1/T) = 1/9, i.e. T = 0.5
        rows = [row(["marine", "freshwater"], [0.75, 0.25], "marine")] * 9 + [row(["marine", "freshwater"], [0.75, 0.25], "freshwater")]
        assert rr.fit_T(rows, w=0) == pytest.approx(0.5, rel=1e-3)

    def test_a_coin_flip_base_goes_to_the_flattest_temperature_allowed(self):
        # right in only half the rows: the best p(first) is 0.5, reached as T -> infinity; T is bounded by e^4
        rows = [row(["marine", "freshwater"], [0.75, 0.25], "marine"), row(["marine", "freshwater"], [0.75, 0.25], "freshwater")]
        assert rr.fit_T(rows, w=0) == pytest.approx(math.exp(4), rel=1e-3)

    def test_always_right_goes_to_a_very_sharp_temperature(self):
        rows = [row(["marine", "freshwater"], [0.75, 0.25], "marine")] * 4
        assert rr.fit_T(rows, w=0) < 0.05  # towards the lower bound e^-4 = 0.018 (the optimiser stops near it)

    def test_the_llm_term_is_part_of_the_fitted_distribution(self):
        # base cannot decide (0.5 / 0.5); the LLM says 0.75 / 0.25 and is right in 3 of 4 rows -> T = 1 with w = 1
        rows = [row(["marine", "freshwater"], [0.5, 0.5], "marine", llm=[0.75, 0.25])] * 3 + \
               [row(["marine", "freshwater"], [0.5, 0.5], "freshwater", llm=[0.75, 0.25])]
        assert rr.fit_T(rows, w=1) == pytest.approx(1.0, rel=1e-3)

    def test_rows_whose_gold_is_outside_the_candidates_do_not_move_the_temperature(self):
        # such a row's likelihood is the 'outside' mass (here 1 - 0.5), the same at every T
        calibrated = [row(["marine", "freshwater"], [0.75, 0.25], "marine")] * 3 + [row(["marine", "freshwater"], [0.75, 0.25], "freshwater")]
        outside = row(["marine", "freshwater"], [0.25, 0.25], "lake")  # gold lake is not a candidate: gold_idx = 2
        assert outside["gold_idx"] == 2
        assert rr.fit_T(calibrated + [outside] * 5, w=0) == pytest.approx(1.0, rel=1e-3)


# ============================================================================= outcomes
class TestOutcomes:
    """outcomes(rows, w, T, tau, allowed, anc): back-off answer per row, graded against its gold term by
    hierarchy.relation (the grading of steps 5 and 6)."""

    ALL = {"aquatic", "marine", "ocean", "freshwater", "lake"}

    def test_the_four_outcome_kinds(self):
        # every row answers 'marine' (0.8 >= tau 0.5); only the gold term changes
        rows = [row(["marine", "freshwater"], [0.8, 0.2], gold) for gold in ("marine", "ocean", "aquatic", "freshwater")]
        assert names(rr.outcomes(rows, w=0, T=1, tau=0.5, allowed=self.ALL, anc=ANC)) == [
            "exact",         # marine = gold
            "coarser",       # marine is an ancestor of gold ocean: true, less specific
            "too_specific",  # marine is below gold aquatic: more specific than the curators were
            "other"]         # marine vs gold freshwater: another branch, wrong

    def test_below_tau_the_answer_backs_off_to_a_broader_allowed_term_or_abstains(self):
        r = row(["marine", "freshwater"], [0.6, 0.4], "marine")
        # tau 0.9: marine (0.6) is not enough; aquatic sums marine + freshwater = 1.0 -> 'aquatic', a coarser true answer
        assert names(rr.outcomes([r], w=0, T=1, tau=0.9, allowed=self.ALL, anc=ANC)) == ["coarser"]
        # the same row when 'aquatic' is not a term of the slot: nothing broader to fall back on -> abstain
        assert names(rr.outcomes([r], w=0, T=1, tau=0.9, allowed={"marine", "freshwater"}, anc=ANC)) == ["abstain"]

    def test_a_term_with_two_parents_is_graded_against_both(self):
        # ENVO is a DAG: toy 'estuary' sits below both marine and freshwater
        anc = {**ANC, "estuary": {"marine", "freshwater", "aquatic"}}
        allowed = self.ALL | {"estuary"}
        rows = [row(["estuary", "lake"], [0.8, 0.2], gold) for gold in ("marine", "freshwater", "lake")] + \
               [row(["freshwater", "lake"], [0.8, 0.2], "estuary")]
        assert names(rr.outcomes(rows, w=0, T=1, tau=0.5, allowed=allowed, anc=anc)) == [
            "too_specific",  # estuary is below gold marine
            "too_specific",  # ... and below gold freshwater (its second parent)
            "other",         # lake is in neither branch above estuary
            "coarser"]       # freshwater is above gold estuary

    def test_no_rows_and_all_abstaining_rows(self):
        assert len(rr.outcomes([], w=0, T=1, tau=0.5, allowed=self.ALL, anc=ANC)) == 0
        rows = [row(["marine", "freshwater"], [0.6, 0.4], "marine")] * 2
        assert list(rr.outcomes(rows, w=0, T=1, tau=0.9, allowed={"marine", "freshwater"}, anc=ANC)) == [0, 0]

    def test_the_llm_weight_can_turn_a_wrong_answer_into_an_exact_one(self):
        r = row(["marine", "freshwater"], [0.6, 0.4], "freshwater", llm=[0.1, 0.9])
        assert names(rr.outcomes([r], w=0, T=1, tau=0.5, allowed=self.ALL, anc=ANC)) == ["other"]  # base: marine
        assert names(rr.outcomes([r], w=1, T=1, tau=0.5, allowed=self.ALL, anc=ANC)) == ["exact"]  # 0.4 x 0.9 > 0.6 x 0.1


# ============================================================================= fit_slot
class TestFitSlot:
    """fit_slot(rows, allowed, anc, target, metric, grid) -> (best {w, T, tau, objective, ...}, one table row per w)."""

    ALL = {"aquatic", "marine", "freshwater"}

    @staticmethod
    def pilot(llm_right):
        """10 rows, half gold marine, half gold freshwater. The base always says marine 0.8 / freshwater 0.2;
        the LLM says 0.9 for the gold term (llm_right) or 0.9 for the other one."""
        rows = []
        for gold, other in [("marine", "freshwater"), ("freshwater", "marine")] * 5:
            pick = gold if llm_right else other
            rows.append(row(["marine", "freshwater"], [0.8, 0.2], gold, llm=[0.9, 0.1] if pick == "marine" else [0.1, 0.9]))
        return rows

    def test_a_helpful_llm_gets_the_smallest_weight_that_fixes_the_base(self):
        # Overturning the base on a freshwater row needs w · log(0.9 / 0.1) > log(0.8 / 0.2), i.e. w > 0.63:
        # w = 0.5 still answers marine there, w = 1 gets every row exactly right.
        best, table = rr.fit_slot(self.pilot(llm_right=True), self.ALL, ANC, target=0.9, metric="strict", grid=(0, 0.5, 1.0))
        assert [r["w"] for r in table] == [0, 0.5, 1.0]
        assert best["w"] == 1.0 and best["exact"] == 1.0
        assert best["tau"] == 0.05  # every answer is right, so the lowest tau of the grid already reaches 90 %

    def test_without_the_llm_the_base_can_only_back_off_to_the_common_parent(self):
        # w = 0: the base cannot tell the two halves apart (always marine, right half the time). The only way to
        # 90 % strict accuracy is to answer 'aquatic biome' (true for both): all coarser, none exact.
        best, _ = rr.fit_slot(self.pilot(llm_right=True), self.ALL, ANC, target=0.9, metric="strict", grid=(0,))
        assert (best["coverage"], best["exact"], best["coarser"], best["objective"]) == (1.0, 0.0, 1.0, 0.0)

    def test_a_misleading_llm_gets_weight_zero(self):
        best, _ = rr.fit_slot(self.pilot(llm_right=False), self.ALL, ANC, target=0.9, metric="strict", grid=(0, 0.5, 1.0))
        assert best["w"] == 0

    def test_ties_go_to_the_smaller_weight(self):
        # an LLM that always says 0.5 / 0.5 leaves the fused distribution equal to the base: same objective for all w
        rows = [row(["marine", "freshwater"], [0.8, 0.2], "marine")] * 4
        best, table = rr.fit_slot(rows, self.ALL, ANC, target=0.9, metric="strict", grid=(0, 0.5, 1.0))
        assert len({r["objective"] for r in table}) == 1
        assert best["w"] == 0

    def test_strict_backs_off_from_too_specific_answers_lenient_keeps_them(self):
        # Curators wrote 'aquatic biome'; the base prefers the more specific 'marine biome' (0.7).
        rows = [row(["marine", "aquatic"], [0.7, 0.3], "aquatic")] * 4
        strict, _ = rr.fit_slot(rows, self.ALL, ANC, target=0.9, metric="strict", grid=(0,))
        lenient, _ = rr.fit_slot(rows, self.ALL, ANC, target=0.9, metric="lenient", grid=(0,))
        # strict: 'too specific' counts as wrong, so tau rises until the answer climbs to aquatic (exact)
        assert strict["exact"] == 1.0 and strict["tau"] > 0.5
        # lenient: 'too specific' counts as right and as useful: answer marine at the lowest tau
        assert lenient["too_specific"] == 1.0 and lenient["tau"] == 0.05


# ============================================================================= pilot_rows
class TestPilotRows:
    """pilot_rows(pilot.jsonl, [responses.jsonl, ...]) -> {slot: [row, ...]}, rows only for answered samples."""

    @staticmethod
    def pilot_sample(sample_id, slot, cands, base, gold):
        return {**request(cands, slot=slot), "sample_id": sample_id, "study": "study_" + sample_id,
                "base_probs": base, "topk_mass": sum(base), "gold": gold}

    def test_one_row_per_answered_sample_with_its_fields(self, tmp_path):
        pilot = write_jsonl(tmp_path / "pilot.jsonl", [
            self.pilot_sample("P1", "biome", ["marine", "freshwater"], [0.6, 0.3], "freshwater"),
            self.pilot_sample("P2", "biome", ["marine", "freshwater"], [0.6, 0.3], "marine")])  # P2: no LLM answer
        responses = write_jsonl(tmp_path / "resp.jsonl", [{"slot": "biome", "sample_id": "P1", **answer(A=0.2, B=0.8)}])
        rows = rr.pilot_rows(pilot, [responses])
        assert list(rows) == ["biome"] and len(rows["biome"]) == 1  # P2 is skipped
        r = rows["biome"][0]
        assert r["cands"] == ["marine", "freshwater"]
        assert r["lb"] == pytest.approx(log([0.6, 0.3]))
        assert r["D"] == pytest.approx(log([0.2, 0.8]))
        assert (r["mass"], r["gold"], r["gold_idx"], r["study"]) == (pytest.approx(0.9), "freshwater", 1, "study_P1")

    def test_gold_outside_the_candidates_points_to_the_outside_entry_and_zero_base_is_floored(self, tmp_path):
        pilot = write_jsonl(tmp_path / "pilot.jsonl", [self.pilot_sample("P1", "biome", ["marine", "freshwater"], [0.9, 0.0], "lake")])
        responses = write_jsonl(tmp_path / "resp.jsonl", [{"slot": "biome", "sample_id": "P1", **answer(A=0.5, B=0.5)}])
        r = rr.pilot_rows(pilot, [responses])["biome"][0]
        assert r["gold_idx"] == 2                    # = number of candidates: fuse()'s last ('outside') entry
        assert r["lb"][1] == pytest.approx(log(1e-9))  # log(0) would be -inf

    def test_responses_are_matched_by_slot_and_sample_and_several_files_are_summed(self, tmp_path):
        pilot = write_jsonl(tmp_path / "pilot.jsonl", [
            self.pilot_sample("P1", "biome", ["marine", "freshwater"], [0.5, 0.5], "marine"),
            self.pilot_sample("P1", "feature", ["sea", "pond"], [0.5, 0.5], "sea")])  # same sample, another slot
        llm1 = write_jsonl(tmp_path / "llm1.jsonl", [{"slot": "biome", "sample_id": "P1", **answer(A=0.8, B=0.2)},
                                                     {"slot": "feature", "sample_id": "P1", **answer(A=0.3, B=0.7)}])
        llm2 = write_jsonl(tmp_path / "llm2.jsonl", [{"slot": "biome", "sample_id": "P1", **answer(A=0.5, B=0.5)}])
        rows = rr.pilot_rows(pilot, [llm1, llm2])
        assert rows["biome"][0]["D"] == pytest.approx(log([0.8, 0.2]) + log([0.5, 0.5]))  # both LLMs
        assert rows["feature"][0]["D"] == pytest.approx(log([0.3, 0.7]))                  # only llm1 answered


# ============================================================================= parse_candidates
class TestParseCandidates:
    def test_ids_and_probabilities(self):
        assert rr.parse_candidates("marine:0.5;freshwater:0.25;aquatic:0.25") == [
            ("marine", 0.5), ("freshwater", 0.25), ("aquatic", 0.25)]

    def test_the_last_colon_separates_the_probability(self):
        assert rr.parse_candidates("ENVO:00000447:0.5") == [("ENVO:00000447", 0.5)]

    def test_scientific_notation_from_the_4_significant_digit_format(self):
        assert rr.parse_candidates("marine:0.9999;lake:1e-05") == [("marine", 0.9999), ("lake", 1e-05)]

    def test_an_empty_string_is_an_error(self):
        # build() skips samples with an empty <slot>_candidates before calling this
        with pytest.raises(ValueError):
            rr.parse_candidates("")


# ============================================================================= estimate
class TestEstimate:
    def test_input_tokens_and_half_price_cost(self, monkeypatch):
        # pretend every chat message is 500,000 tokens: system + user = 1M input tokens per request
        monkeypatch.setattr(rr, "count_tokens", lambda text: 500_000)
        requests = [request(["marine", "freshwater"]), request(["sea", "pond"], slot="feature")]
        n_in, cost = rr.estimate(requests, "gpt-4.1-mini")  # $0.40 / 1M input, $1.60 / 1M output
        assert n_in == 2_000_000
        # (2M x $0.40 + 2 requests x 4 output tokens x $1.60 / 1M) x 0.5 (Batch API discount)
        assert cost == pytest.approx((2 * 0.40 + 2 * 4 / 1e6 * 1.60) * 0.5)

    def test_unknown_model_costs_nothing_and_a_generator_is_accepted(self, monkeypatch):
        monkeypatch.setattr(rr, "count_tokens", lambda text: 10)
        n_in, cost = rr.estimate((request(["marine"]) for _ in range(3)), "my-local-model")
        assert (n_in, cost) == (60, 0.0)


# ============================================================================= body_for
class TestBodyFor:
    @pytest.mark.parametrize("model, top_logprobs, temperature, reasoning", [
        ("gpt-4.1-mini", 10, 0, None),       # classic model: deterministic, 10 alternatives
        ("gpt-5-mini",   5,  None, None),    # gpt-5.x accepts at most 5 top log-probabilities, no temperature
        ("gpt-5.1",      5,  None, "none"),  # ... and gpt-5.1 is told not to spend tokens on reasoning
        ("o4-mini",      10, None, None),    # o-series: no temperature
    ])
    def test_model_specific_settings(self, model, top_logprobs, temperature, reasoning):
        body = rr.body_for(request(["marine", "freshwater"]), model)
        assert body["model"] == model and body["logprobs"] is True and body["max_completion_tokens"] == 16
        assert body["top_logprobs"] == top_logprobs
        assert body.get("temperature") == temperature
        assert body.get("reasoning_effort") == reasoning

    def test_messages_are_the_shared_v2_prompt(self):
        import rerank_prompt
        o = request(["marine", "freshwater"])
        assert rr.body_for(o, "gpt-4.1-mini")["messages"] == rerank_prompt.prompt(o, "logprobs", "v2")

