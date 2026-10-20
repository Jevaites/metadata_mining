"""Mutation tests for 7_rerank_atlas.py: deliberate one-line bugs, and proof that each one is caught.

Why: a test suite can pass because the code is right, or because the tests never look at the line that matters.
Each test below takes one line of 7_rerank_atlas.py, breaks it in a plausible way (a "mutant", loaded as a
separate module by step7_toy.load_mutant; the real script is not touched), runs the same small scenario on the
real module and on the mutant, and asserts:
    the real code gives the right answer, and the mutant gives a specific, documented wrong one.
So each test also documents why the line is written the way it is.

MUTANTS is the catalogue: name -> (the original text, its broken replacement). test_every_mutated_line_still_exists
fails when a line is rewritten, so the catalogue is kept in step with the script.

Scenarios reuse the toy world of test_7_rerank_atlas_integration.py (S1-S6, T1-T5: see its docstring).
"""

import json
import os
import sys

import numpy as np
import pytest

from step7_toy import ANC, FakeOpenAI, load_mutant, load_rerank_atlas, read_jsonl
from test_7_rerank_atlas_integration import (ATLAS, ATLAS_LLM, S2_TEXT, SAMPLE_INFO, apply_argv, build_argv, collect_lines,
                                             fit_argv, leading_newline_line, make_pilot, make_world, members_by_text,
                                             requests_by_text, s2_final, says, write_apply_inputs, write_params,
                                             write_single_study_pilot)
from test_7_rerank_atlas_unit import answer, request, row

rr = load_rerank_atlas()
log = np.log
S6_TEXT = "env_biome: beach; salinity: 35 PSU"  # S6: biome_p 0.6, exactly at the gate

MUTANTS = {
    # fit
    "fuse_outside_can_go_negative": ("max(1 - mass, 0.0)", "1 - mass"),
    "fuse_ignores_mass":            ("np.append(z / z.sum() * mass,", "np.append(z / z.sum(),"),
    "letters_reversed":             ("np.maximum(dist[:k], 1e-6)", "np.maximum(dist[:k][::-1], 1e-6)"),
    "ties_go_to_larger_w":          ('(r["objective"], -r["w"])', '(r["objective"], r["w"])'),
    "cv_trains_on_held_out_study":  ("tr = [R[i] for i in np.where(fold != k)[0]]", "tr = R"),
    "slots_option_ignored":         ('use = slot in args.slots and best["w"] > 0', 'use = best["w"] > 0'),
    # build
    "gate_not_strict":              ('atlas[f"{slot}_p"].astype(float) < params', 'atlas[f"{slot}_p"].astype(float) <= params'),
    "no_check_for_missing_text":    ("if sid not in texts or not cand:", "if not cand:"),
    "dedup_ignores_candidates":     (r'f"{slot}\t{texts[sid]}\t{cand}"', r'f"{slot}\t{texts[sid]}"'),
    "farthest_example":             ("idx[np.argmax(sim[i, idx])]", "idx[np.argmin(sim[i, idx])]"),
    # submit / collect
    "limit_ignored":                ("if args.limit and i >= args.limit:", "if False:"),
    "resends_requests_in_flight":   ('o["request_id"] not in done and o["request_id"] not in in_flight', 'o["request_id"] not in done'),
    "collects_running_batches":     ('if batch.status in ("validating", "in_progress", "finalizing", "cancelling"):', "if False:"),
    "keeps_failed_requests":        ('if resp.get("status_code") != 200:', "if False:"),
    "expired_request_crashes":      ('resp = r.get("response") or {}', 'resp = r["response"]'),
    "first_token_even_if_blank":    ('first = next((t for t in content_lp if t["token"].strip()), content_lp[0])',
                                     "first = content_lp[0]"),
    # apply
    "first_answer_wins":            ('responses[r["request_id"]] = r', 'responses.setdefault(r["request_id"], r)'),
    "reranked_labelled_backoff":    ('source[rows] = "rerank"', 'source[rows] = "backoff"'),
}


def mutant(name):
    old, new = MUTANTS[name]
    return load_mutant(old, new, name)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_mutated_line_still_exists(name):
    mutant(name)  # load_mutant asserts the original text occurs exactly once in 7_rerank_atlas.py


# ----------------------------------------------------------------------------- running a module's commands
@pytest.fixture
def cli(monkeypatch):
    """cli(module, "build", "--params", ...) = `python 7_rerank_atlas.py build ...` with that module (real or mutant)."""
    def run(module, *argv):
        monkeypatch.setattr(sys, "argv", ["7_rerank_atlas.py", *[str(a) for a in argv]])
        module.main()
    return run


@pytest.fixture
def fake_api(monkeypatch):
    """fake_api(module) -> a FakeOpenAI (answering with ATLAS_LLM) that module's submit / collect will use."""
    def install(module):
        api = FakeOpenAI(ATLAS_LLM)
        monkeypatch.setattr(module, "client_for", lambda args: api)
        return api
    return install


def built_world(d, module, cli, **params):
    """The toy atlas after `build` with that module; -> (world paths, params path)."""
    w = make_world(d)
    p = write_params(d, **params)
    cli(module, *build_argv(w, p))
    return w, p


# ============================================================================= fit
def test_outside_mass_is_clipped_at_zero():
    # candidates' probabilities rounded to 4 digits can sum to 1.0002; the 'outside' entry must not be negative
    real = rr.fuse(log([0.5001, 0.5001]), log([0.5, 0.5]), 1, 1, 1.0002)
    broken = mutant("fuse_outside_can_go_negative").fuse(log([0.5001, 0.5001]), log([0.5, 0.5]), 1, 1, 1.0002)
    assert real[-1] == 0.0
    assert broken[-1] == pytest.approx(-0.0002)


def test_the_candidates_keep_only_their_base_mass():
    # base 0.4 + 0.2 = 0.6 on the candidates, 0.4 outside. Without `* mass` the candidates get 1.0 and the
    # total is 1.4: back-off thresholds would be reached on probability that does not exist.
    real = rr.fuse(log([0.4, 0.2]), log([0.5, 0.5]), 1, 1, 0.6)
    broken = mutant("fuse_ignores_mass").fuse(log([0.4, 0.2]), log([0.5, 0.5]), 1, 1, 0.6)
    assert real == pytest.approx([0.4, 0.2, 0.4])
    assert broken == pytest.approx([2 / 3, 1 / 3, 0.4]) and broken.sum() == pytest.approx(1.4)


def test_llm_probabilities_are_kept_in_candidate_order():
    # option_order [1, 0]: letter A shows freshwater (candidate 1). The LLM says A 0.9 -> freshwater 0.9.
    o = request(["marine", "freshwater"], option_order=[1, 0])
    assert rr.llm_logprob(o, [answer(A=0.9, B=0.1)]) == pytest.approx(log([0.1, 0.9]))
    assert mutant("letters_reversed").llm_logprob(o, [answer(A=0.9, B=0.1)]) == pytest.approx(log([0.9, 0.1]))  # marine!


def test_ties_between_weights_go_to_the_smaller_one():
    # an LLM that always says 0.5 / 0.5 changes nothing, so every w scores the same: w = 0 must win
    # (the LLM is then switched off and its cost saved), not the largest w
    rows = [row(["marine", "freshwater"], [0.8, 0.2], "marine")] * 4
    allowed = {"aquatic", "marine", "freshwater"}
    assert rr.fit_slot(rows, allowed, ANC, 0.9, "strict", (0, 0.5, 1.0))[0]["w"] == 0
    assert mutant("ties_go_to_larger_w").fit_slot(rows, allowed, ANC, 0.9, "strict", (0, 0.5, 1.0))[0]["w"] == 1.0


def test_the_cv_estimate_does_not_train_on_the_held_out_study(tmp_path, cli):
    # write_single_study_pilot: the LLM helps in study0 only. An honest CV cannot learn that from study1-4, so
    # study0's freshwater sample stays wrong (estimate 0.9); training on every study claims 1.0.
    estimates = {}
    for name, module in [("real", rr), ("broken", mutant("cv_trains_on_held_out_study"))]:
        d = tmp_path / name
        d.mkdir()
        w = make_world(d / "world")
        cli(module, *fit_argv(w, *write_single_study_pilot(d), d / "params.json"))
        biome = json.load(open(d / "params.json"))["slots"]["biome"]
        estimates[name] = (sorted(biome["cv_w_per_fold"]), biome["cv_estimate"]["base + LLM (w by CV)"]["exact"])
    assert estimates["real"] == ([0, 1.0, 1.0, 1.0, 1.0], 0.9)
    assert estimates["broken"] == ([1.0] * 5, 1.0)  # optimistic: the estimate has seen its own test study


def test_a_slot_left_out_of_slots_never_uses_the_llm(tmp_path, cli):
    use = {}
    for name, module in [("real", rr), ("broken", mutant("slots_option_ignored"))]:
        d = tmp_path / name
        d.mkdir()
        w = make_world(d / "world")
        cli(module, *fit_argv(w, *make_pilot(d), d / "params.json", "--slots", "feature"))
        use[name] = json.load(open(d / "params.json"))["slots"]["biome"]["use_llm"]
    assert (use["real"], use["broken"]) == (False, True)  # biome's w is 1 in both; only --slots says no


# ============================================================================= build
def test_the_gate_is_strict(tmp_path, cli):
    # S6 has biome_p 0.6 = gate_p: "below the gate" means < , so S6 keeps its back-off answer
    real, _ = built_world(tmp_path / "real", rr, cli)
    broken, _ = built_world(tmp_path / "broken", mutant("gate_not_strict"), cli)
    assert S6_TEXT not in members_by_text(real)
    assert members_by_text(broken)[S6_TEXT] == ["S6"]


def test_a_gated_sample_without_metadata_text_is_skipped(tmp_path, cli):
    # S4 is gated but missing from sample.info: no request (it ends up backoff_no_llm_answer), not a crash
    real, _ = built_world(tmp_path / "real", rr, cli)
    assert "S4" not in sum(members_by_text(real).values(), [])
    with pytest.raises(KeyError, match="S4"):
        built_world(tmp_path / "broken", mutant("no_check_for_missing_text"), cli)


def test_requests_are_shared_only_when_text_and_candidates_are_the_same(tmp_path, cli):
    # S7: S2's metadata text, but step 6 ranked freshwater first. Sharing S2's request would give S7 the fused
    # answer of S2's base probabilities.
    s7 = ("S7", 0.5, "freshwater:0.5;marine:0.3;aquatic:0.2", "aquatic", 1.0, "pond", 1, 0)
    texts = {}
    for name, module in [("real", rr), ("broken", mutant("dedup_ignores_candidates"))]:
        d = tmp_path / name
        w = make_world(d, atlas=ATLAS + [s7], sample_info={**SAMPLE_INFO, "S7": SAMPLE_INFO["S2"]})
        cli(module, *build_argv(w, write_params(d)))
        texts[name] = sorted(r["text"] for r in read_jsonl(os.path.join(w["work"], "requests.jsonl")))
    assert texts["real"].count(S2_TEXT) == 2    # S2/S3 and S7
    assert texts["broken"].count(S2_TEXT) == 1  # S7 merged into S2's request


def test_the_example_is_the_nearest_training_sample(tmp_path, cli):
    # S2 points at 90 degrees: of the marine training samples, T2 (37 degrees) is nearer than T1 (0 degrees)
    real, _ = built_world(tmp_path / "real", rr, cli)
    broken, _ = built_world(tmp_path / "broken", mutant("farthest_example"), cli)
    example = lambda w: {c["term_id"]: c["example"] for c in requests_by_text(w)[S2_TEXT]["candidates"]}["marine"]
    assert example(real) == "coral reef water"       # T2
    assert example(broken) == "seawater, North Sea"  # T1, the farthest


# ============================================================================= submit / collect
def sent_ids(api):
    return [[line["custom_id"] for line in batch] for batch in api.sent]


def test_limit_sends_only_a_trial(tmp_path, cli, fake_api):
    for name, module, expected in [("real", rr, 1), ("broken", mutant("limit_ignored"), 2)]:
        w, _ = built_world(tmp_path / name, module, cli)
        api = fake_api(module)
        cli(module, "submit", "--work_dir", w["work"], "--limit", "1")
        assert sum(map(len, sent_ids(api))) == expected  # requests sent by a `--limit 1` trial


def test_a_request_in_a_pending_batch_is_not_sent_again(tmp_path, cli, fake_api):
    for name, module, n_batches_with_first in [("real", rr, 1), ("broken", mutant("resends_requests_in_flight"), 2)]:
        w, _ = built_world(tmp_path / name, module, cli)
        first = read_jsonl(os.path.join(w["work"], "requests.jsonl"))[0]["request_id"]
        api = fake_api(module)
        cli(module, "submit", "--work_dir", w["work"], "--limit", "1")  # batch-0 = [first], pending
        cli(module, "submit", "--work_dir", w["work"])                  # the rest
        assert sum(first in batch for batch in sent_ids(api)) == n_batches_with_first  # broken: paid for twice


def test_a_running_batch_is_left_for_the_next_collect(tmp_path, cli, fake_api):
    # Marking a running batch as collected loses its answers: they are never downloaded, and the next submit
    # sends (and pays for) the same requests again.
    sent = {}
    for name, module in [("real", rr), ("broken", mutant("collects_running_batches"))]:
        w, _ = built_world(tmp_path / name, module, cli)
        api = fake_api(module)
        cli(module, "submit", "--work_dir", w["work"])  # batch-0: both requests
        api.still_running.add("batch-0")
        cli(module, "collect", "--work_dir", w["work"])
        cli(module, "submit", "--work_dir", w["work"])
        sent[name] = sent_ids(api)
    assert len(sent["real"]) == 1                    # nothing new: both requests are still in batch-0
    assert sent["broken"][1] == sent["broken"][0]    # batch-1 = batch-0 again


def test_a_failed_request_is_skipped_by_collect(tmp_path, cli, fake_api):
    # S5's request fails (HTTP 500, no 'choices' in the body). Skipping it lets submit retry it later;
    # reading it like an answer crashes collect for the whole batch.
    real, _ = built_world(tmp_path / "real", rr, cli)
    fake_api(rr)
    cli(rr, "submit", "--work_dir", real["work"])
    cli(rr, "collect", "--work_dir", real["work"])
    assert len(read_jsonl(os.path.join(real["work"], "responses.jsonl"))) == 1  # S2's answer only
    broken_module = mutant("keeps_failed_requests")
    broken, _ = built_world(tmp_path / "broken", broken_module, cli)
    fake_api(broken_module)
    cli(broken_module, "submit", "--work_dir", broken["work"])
    with pytest.raises(KeyError, match="choices"):
        cli(broken_module, "collect", "--work_dir", broken["work"])


def test_an_expired_request_is_skipped_by_collect(tmp_path, cli, fake_api):
    # an expired request's error-file line has "response": null; reading it as a dict crashes collect
    for name, module in [("real", rr), ("broken", mutant("expired_request_crashes"))]:
        w, _ = built_world(tmp_path / name, module, cli)
        api = fake_api(module)
        api.expired.add(requests_by_text(w)[S2_TEXT]["request_id"])
        cli(module, "submit", "--work_dir", w["work"])
        if name == "real":
            cli(module, "collect", "--work_dir", w["work"])
            assert read_jsonl(os.path.join(w["work"], "responses.jsonl")) == []
        else:
            with pytest.raises(AttributeError):
                cli(module, "collect", "--work_dir", w["work"])


def test_the_log_probabilities_are_those_of_the_first_non_blank_token(tmp_path, cli, monkeypatch):
    # answer "\nE": the newline token's alternatives hold no letter, so parse() would fall back to the plain choice
    stored = {name: collect_lines(module, cli, str(tmp_path / name), [leading_newline_line()], monkeypatch)[0]["top_logprobs_raw"]
              for name, module in [("real", rr), ("broken", mutant("first_token_even_if_blank"))]}
    assert stored == {"real": [["E", -0.2], ["A", -1.8]], "broken": [["\n", -0.01]]}


# ============================================================================= apply
def test_the_latest_answer_of_a_request_wins(tmp_path, cli):
    # two answers for S2's request: first freshwater (B), then marine (A); the later one is the one to keep
    finals = {}
    for name, module in [("real", rr), ("broken", mutant("first_answer_wins"))]:
        d = tmp_path / name
        w = make_world(d)
        write_apply_inputs(w, [0.5, 0.25, 0.25], [says(A=0.2, B=0.6, C=0.2), says(A=0.9, B=0.05, C=0.05)])
        cli(module, *apply_argv(w, write_params(d)))
        finals[name] = s2_final(w)[1]
    assert (finals["real"], finals["broken"]) == ("marine", "freshwater")


def test_reranked_answers_are_labelled_rerank(tmp_path, cli):
    # the answer is the same; only <slot>_final_source tells which samples the LLM changed
    rows = {}
    for name, module in [("real", rr), ("broken", mutant("reranked_labelled_backoff"))]:
        d = tmp_path / name
        w = make_world(d)
        write_apply_inputs(w, [0.5, 0.25, 0.25], [says(A=0.2, B=0.6, C=0.2)])
        cli(module, *apply_argv(w, write_params(d)))
        rows[name] = s2_final(w)
    assert rows["real"] == ("S2", "freshwater", 0.5, "rerank")
    assert rows["broken"] == ("S2", "freshwater", 0.5, "backoff")  # indistinguishable from an untouched sample
