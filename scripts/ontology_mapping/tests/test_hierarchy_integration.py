"""Integration tests for hierarchy.py: the functions chained as the pipeline chains them, on toy data whose
every number can be worked out by hand.

THE THREE WORKFLOWS
  1. 5_evaluate.py (backoff_all)     out-of-fold scores per fold
                                       -> oof_backoff(folds, anc, targets) -> report (temperature, curve, tau per target)
                                       -> per fold: C.decode(P, tau) -> each sample's answer
                                       -> calibration.json entry {"temperature", "curve", "settings"}
  2. 6_predict_atlas.py              calibration entries (one per fold seed)
     (load_backoff, backoff_columns)   -> merge_calibrations -> choose_tau(curve, target) -> one tau per slot
                                       -> softmax(new scores, T) -> Closure(classes, anc).decode(P, tau)
                                       -> answer and its summed probability
  3. 7_rerank_atlas.py               candidate lists + reranked probabilities, with gold labels
     (outcomes, fit_slot)              -> decode_candidates at every tau of TAUS, graded with relation
                                       -> summarise -> choose_tau -> decode_candidates on new candidate lists
Only hierarchy.py functions are chained here; the step code around them (6_predict_atlas's top1 / coarser /
abstain kind, 7_rerank_atlas's base + LLM fusion and grading) is tested in that step's own test files.

THE TOY DATA (ontology: see toy_hierarchy.py)

    biome (root)
    ├── aquatic       ├── marine ── ocean        └── freshwater ── lake (not a label)
    └── terrestrial (not a label) ── soil, desert, forest, grassland, tundra

Three kinds of samples. The model gives the same probabilities to every sample of a kind, and the gold
labels are spread EXACTLY as those probabilities say: of 100 "open ocean" samples, 92 really are ocean and
8 are marine. The scores are the log of these probabilities. Such a model is perfectly calibrated, so the
fitted temperature is 1 and the calibrated probabilities are the numbers below.

    kind               model probabilities                     gold labels of 100 samples
    open ocean         ocean .92  marine .08                   92 ocean, 8 marine
    salt or fresh      ocean .52  marine .16  freshwater .32   52 ocean, 16 marine, 32 freshwater
    dry land           soil .62   desert .38                   62 soil, 38 desert

What the climb answers (q = probability at or below the node, on the path of the top-1 label):

    open ocean         ocean (q .92) -> marine (q 1)                    tau <= .92: ocean,  above: marine
    salt or fresh      ocean (.52) -> marine (.68) -> aquatic (1)       tau <= .52: ocean,  <= .68: marine,  above: aquatic
    dry land           soil (.62) -> nothing (terrestrial: no label,    tau <= .62: soil,   above: abstain
                                   biome: root)

So, for the 300 samples of one fold (E exact, C coarser, S too specific, O other branch):

    tau    open ocean      salt or fresh           dry land      coverage  strict acc.       lenient acc.
    .50    ocean 92E 8S    ocean 52E 16S 32O       soil 62E 38O  300/300   206/300 = .6867   230/300 = .7667
    .60    ocean 92E 8S    marine 16E 52C 32O      soil 62E 38O  300/300   222/300 = .74     230/300 = .7667
    .65    ocean 92E 8S    marine 16E 52C 32O      abstain       200/300   160/200 = .80     168/200 = .84
    .80    ocean 92E 8S    aquatic 100C            abstain       200/300   192/200 = .96     200/200 = 1
    .95    marine 8E 92C   aquatic 100C            abstain       200/300   200/200 = 1       1
    1.0    abstain (no allowed node has q = 1: the 1e-9 floor of log_scores leaves q a hair below 1)
"""

import numpy as np
import pytest

from toy_hierarchy import ANC, AQUATIC, DESERT, FOREST, FRESHWATER, LAKE, MARINE, OCEAN, SOIL, VOCAB, log_scores

import hierarchy
from hierarchy import COARSER, EXACT, OTHER

KINDS = {
    "open ocean":    {OCEAN: .92, MARINE: .08},
    "salt or fresh": {OCEAN: .52, MARINE: .16, FRESHWATER: .32},
    "dry land":      {SOIL: .62, DESERT: .38},
}


def make_fold(*kinds, n=100):
    """One fold as 5_evaluate.py passes it to oof_backoff: {"S": scores (samples x VOCAB), "vocab", "gold"},
    plus "kind" (not used by hierarchy.py; lets the tests group the answers).
    n samples per kind, all with the kind's scores; gold labels in exact proportion to its probabilities."""
    S, gold, kind = [], [], []
    for name in kinds:
        p = KINDS[name]
        S += [log_scores(p)[0]] * n
        gold += [label for label, share in p.items() for _ in range(round(share * n))]
        kind += [name] * n
    return {"S": np.array(S), "vocab": VOCAB, "gold": gold, "kind": kind}


def at(curve, tau):
    """The curve row of one tau."""
    return next(row for row in curve if abs(row["tau"] - tau) < 1e-9)


def answers_by_kind(fold, P, C, tau):
    """{kind: set of answers given to its samples} ("" = abstain), as 5_evaluate.py's "backoff" column."""
    pick, _ = C.decode(P, tau)
    answers = [C.nodes[k] if k >= 0 else "" for k in pick]
    out = {}
    for kind, a in zip(fold["kind"], answers):
        out.setdefault(kind, set()).add(str(a))
    return out


# ============================================================================= workflow 1: 5_evaluate.py
@pytest.fixture(scope="module")
def evaluation():
    """Three folds of identical make-up (100 samples of each kind), evaluated as 5_evaluate.py does."""
    folds = [make_fold("open ocean", "salt or fresh", "dry land") for _ in range(3)]
    report, fitted = hierarchy.oof_backoff(folds, ANC, targets=(0.75, 0.8, 0.95))
    return folds, report, fitted


class TestEvaluationWorkflow:
    def test_calibrated_model_gets_temperature_1(self, evaluation):
        _, report, _ = evaluation
        assert report["temperature_per_fold"] == pytest.approx([1, 1, 1], abs=1e-3)  # each fitted on the 2 others
        assert report["temperature_all"] == pytest.approx(1, abs=1e-3)              # for the final model

    @pytest.mark.parametrize("tau, coverage, strict, lenient", [
        # the table of the module docstring (pooled over the 3 identical folds = the same shares)
        (0.50, 1.0, 0.6867, 0.7667),
        (0.60, 1.0, 0.74, 0.7667),
        (0.65, 0.6667, 0.80, 0.84),
        (0.80, 0.6667, 0.96, 1.0),
        (0.95, 0.6667, 1.0, 1.0),
        (1.00, 0.0, None, None),
    ])
    def test_curve_matches_the_hand_computed_table(self, evaluation, tau, coverage, strict, lenient):
        _, report, _ = evaluation
        row = at(report["curve"], tau)
        assert (row["coverage"], row["accuracy_strict"], row["accuracy_lenient"]) == (coverage, strict, lenient)

    def test_outcome_shares_at_tau_0_8(self, evaluation):
        _, report, _ = evaluation
        row = at(report["curve"], 0.80)
        assert (row["exact"], row["coarser"], row["too_specific"], row["other_branch"]) == (.3067, .3333, .0267, 0)
        # 92 E + 100 C + 8 S + 0 O = 200 answered of 300

    @pytest.mark.parametrize("name, tau", [
        ("strict_0.95", 0.7),     # first tau above .68, where 'salt or fresh' climbs to aquatic (strict .96)
        ("strict_0.8", 0.625),    # first tau above .62, where 'dry land' abstains (strict .80)
        ("lenient_0.8", 0.625),   # lenient .84
        ("lenient_0.95", 0.7),    # lenient 1.0
        ("strict_0.75", 0.625),   # strict .74 below, .80 from .625
        ("lenient_0.75", 0.05),   # lenient .7667 from the lowest tau: strict and lenient taus differ
    ])
    def test_tau_per_target_is_the_lowest_reaching_it(self, evaluation, name, tau):
        _, report, _ = evaluation
        assert report[name]["tau_per_fold"] == [tau, tau, tau]  # folds alike, so the same tau for each

    def test_estimate_at_a_target_is_the_curve_at_its_tau(self, evaluation):
        _, report, _ = evaluation
        assert (report["strict_0.95"]["coverage"], report["strict_0.95"]["accuracy_strict"]) == (0.6667, 0.96)

    def test_answers_per_sample_at_the_chosen_tau(self, evaluation):
        # 5_evaluate.py decodes each fold's calibrated probabilities at that fold's tau
        folds, report, fitted = evaluation
        for fold, (P, C), tau in zip(folds, fitted, report["strict_0.95"]["tau_per_fold"]):
            assert answers_by_kind(fold, P, C, tau) == {
                "open ocean": {OCEAN},       # 92 % sure: answered as is
                "salt or fresh": {AQUATIC},  # unsure between salt and fresh water: the broader 'aquatic'
                "dry land": {""},            # unsure, and no allowed term above soil: abstain
            }


# ============================================================================= workflow 2: 6_predict_atlas.py
@pytest.fixture(scope="module")
def calibration_entries():
    """calibration.json entries of two 5_evaluate.py runs on the same samples with two fold seeds:
    seed 0 = 3 folds of 100 per kind, seed 1 = 6 folds of 50 per kind."""
    entries = []
    for seed, folds in [(0, [make_fold(*KINDS) for _ in range(3)]),
                        (1, [make_fold(*KINDS, n=50) for _ in range(6)])]:
        report, _ = hierarchy.oof_backoff(folds, ANC)
        entries.append({"temperature": report["temperature_all"], "curve": report["curve"],
                        "settings": {"method": "prototype", "fold_seed": seed}})
    return entries


class TestPredictionWorkflow:
    def test_merged_calibration_and_tau(self, calibration_entries):
        merged = hierarchy.merge_calibrations(calibration_entries)
        assert merged["temperature"] == pytest.approx(1, abs=1e-3)
        # every sample is scored once out of fold in each run, with the same answers: same curve
        assert at(merged["curve"], 0.8)["accuracy_strict"] == 0.96
        assert hierarchy.choose_tau(merged["curve"], 0.95, "accuracy_strict") == 0.7

    def test_new_samples_get_an_answer(self, calibration_entries):
        merged = hierarchy.merge_calibrations(calibration_entries)
        T, tau = merged["temperature"], hierarchy.choose_tau(merged["curve"], 0.95, "accuracy_strict")  # 1, .7
        closure = hierarchy.Closure(VOCAB, ANC)  # over the final model's classes
        # four new MicrobeAtlas samples (no gold); the final model's scores (log-probabilities again)
        scores = log_scores({OCEAN: .97, MARINE: .03},                    # deep sea water
                            {OCEAN: .45, MARINE: .30, FRESHWATER: .25},   # coastal water
                            {FRESHWATER: .50, OCEAN: .30, MARINE: .20},   # river or lake
                            {SOIL: .55, FOREST: .45})                     # garden soil
        # as 6_predict_atlas.backoff_columns:
        P = hierarchy.softmax(scores, T)
        pick, q = closure.decode(P, tau)
        answer = [str(closure.nodes[k]) if k >= 0 else "" for k in pick]
        assert list(zip(answer, np.round(q, 2))) == [
            (OCEAN, .97),       # q(ocean) .97 >= .7: the top-1 itself
            (MARINE, .75),      # ocean .45 < .7; marine .45 + .30 = .75
            (AQUATIC, 1.0),     # freshwater .50 < .7; its parent aquatic 1.0 (not marine: off its path)
            ("", 0.0),          # soil .55 < .7; terrestrial no label, biome root: abstain
        ]


# ============================================================================= workflow 3: 7_rerank_atlas.py
def reranked_rows(*kinds, n=100, drop=()):
    """Rows as 7_rerank_atlas.py holds them after fusing base model and LLM: candidates (the kind's labels,
    minus `drop`), their probabilities, and a gold label (same proportions as make_fold)."""
    rows = []
    for name in kinds:
        p = {label: share for label, share in KINDS[name].items() if label not in drop}
        for label, share in KINDS[name].items():
            rows += [{"kind": name, "cands": list(p), "p": list(p.values()), "gold": label}] * round(share * n)
    return rows


def rerank_outcomes(rows, tau):
    """Outcome code per row (0 = abstain): decode_candidates' answer graded with hierarchy.relation."""
    out = []
    for r in rows:
        node, _ = hierarchy.decode_candidates(r["cands"], r["p"], tau, set(VOCAB), ANC)
        out.append(0 if node is None else int(hierarchy.relation([node], [r["gold"]], ANC)[0, 0]))
    return np.array(out)


class TestRerankWorkflow:
    def test_fit_tau_then_answer(self):
        rows = reranked_rows(*KINDS)
        # as 7_rerank_atlas.fit_slot: a curve over TAUS, then the lowest tau reaching the target
        curve = [{"tau": float(t), **hierarchy.summarise(rerank_outcomes(rows, t))} for t in hierarchy.TAUS]
        assert at(curve, 0.8)["accuracy_strict"] == 0.96  # all probability on the candidates: same table
        tau = hierarchy.choose_tau(curve, 0.95, "accuracy_strict")
        assert tau == 0.7
        answers = {r["kind"]: hierarchy.decode_candidates(r["cands"], r["p"], tau, set(VOCAB), ANC)[0] for r in rows}
        assert answers == {"open ocean": OCEAN, "salt or fresh": AQUATIC, "dry land": None}

    def test_probability_left_off_the_candidates_makes_it_more_cautious(self):
        # 'salt or fresh' without its freshwater candidate: the candidates hold .52 + .16 = .68 only, so
        # aquatic (q 1 with Closure.decode) reaches only .68 here: abstain at tau .7
        rows = reranked_rows("salt or fresh", drop=(FRESHWATER,))
        assert hierarchy.decode_candidates(rows[0]["cands"], rows[0]["p"], 0.7, set(VOCAB), ANC) == (None, 0.0)
        assert hierarchy.decode_candidates(rows[0]["cands"], rows[0]["p"], 0.6, set(VOCAB), ANC)[0] == MARINE


# ============================================================================= edge cases
class TestEdgeCases:
    def test_tau_of_a_fold_is_chosen_on_the_other_folds(self):
        # fold 0 only 'open ocean', fold 1 only 'salt or fresh'; strict target .95
        #   fold 0 gets the tau that works for fold 1: aquatic from .7 on   -> .7
        #   fold 1 gets the tau that works for fold 0: marine from .925 on  -> .925
        folds = [make_fold("open ocean"), make_fold("salt or fresh")]
        report, fitted = hierarchy.oof_backoff(folds, ANC, targets=(0.95,))
        assert report["strict_0.95"]["tau_per_fold"] == [0.7, 0.925]
        assert answers_by_kind(folds[0], *fitted[0], 0.7) == {"open ocean": {OCEAN}}        # 92 E, 8 S
        assert answers_by_kind(folds[1], *fitted[1], 0.925) == {"salt or fresh": {AQUATIC}}  # 100 C
        # honest estimate: 192 strict-correct of 200 (on its own data fold 0 would have needed .925)
        assert report["strict_0.95"]["accuracy_strict"] == 0.96

    def test_temperature_of_a_fold_is_fitted_on_the_other_folds(self):
        # fold 0: calibrated 'open ocean' (92 % ocean). fold 1: same scores, but ALL 100 samples are ocean,
        # so on fold 1 the model is under-confident and a sharper softmax (T < 1) fits it better.
        calibrated = make_fold("open ocean")
        all_ocean = make_fold("open ocean")
        all_ocean["gold"] = [OCEAN] * 100
        report, _ = hierarchy.oof_backoff([calibrated, all_ocean], ANC)
        T_fold0, T_fold1 = report["temperature_per_fold"]
        assert T_fold1 == pytest.approx(1, abs=1e-3)  # fitted on fold 0 only (calibrated)
        assert T_fold0 < 0.9                          # fitted on fold 1 only (under-confident)

    def test_gold_label_the_model_never_saw(self):
        # 10 'lake water' samples per fold: lake is no label (no score column), the model says freshwater
        # .7 / ocean .3. Their rows are left out of the temperature fit (nothing to compare with), and an
        # answer on lake's path (freshwater, aquatic) is COARSER, i.e. correct.
        folds = []
        for _ in range(2):
            f = make_fold("open ocean")
            f["S"] = np.vstack([f["S"], log_scores({FRESHWATER: .7, OCEAN: .3})[[0] * 10]])
            f["gold"] += [LAKE] * 10
            f["kind"] += ["lake water"] * 10
            folds.append(f)
        report, fitted = hierarchy.oof_backoff(folds, ANC)
        assert report["temperature_per_fold"] == pytest.approx([1, 1], abs=1e-3)  # 'open ocean' alone sets T
        P, C = fitted[0]
        lake_rows = slice(100, 110)
        rel = hierarchy.relation(C.nodes, folds[0]["gold"], ANC)
        for tau, expected in [(0.5, FRESHWATER), (0.8, AQUATIC)]:
            pick, _ = C.decode(P, tau)
            assert set(C.nodes[pick[lake_rows]]) == {expected}
            assert set(hierarchy.outcome(rel, pick)[lake_rows]) == {COARSER}

    def test_target_the_model_never_reaches_switches_the_slot_off(self):
        # only 'dry land': strict accuracy .62 while it answers, nothing answered above tau .62
        folds = [make_fold("dry land"), make_fold("dry land")]
        report, _ = hierarchy.oof_backoff(folds, ANC, targets=(0.9,))
        assert at(report["curve"], 0.6)["accuracy_strict"] == 0.62
        assert at(report["curve"], 0.65)["accuracy_strict"] is None
        # choose_tau falls back to the highest tau (1.0), where every sample abstains
        assert report["strict_0.9"]["tau_per_fold"] == [1.0, 1.0]
        assert (report["strict_0.9"]["coverage"], report["strict_0.9"]["accuracy_strict"]) == (0.0, None)

    def test_folds_with_different_vocabularies(self):
        # the model of fold 1 never saw 'desert' (one column fewer). Its desert samples have no gold
        # column (left out of the temperature fit); an answer 'soil' for them is OTHER (another branch).
        f0 = make_fold("dry land")
        vocab1 = [label for label in VOCAB if label != DESERT]
        f1 = {"S": log_scores({SOIL: 1.0}, vocab=vocab1)[[0] * 100], "vocab": vocab1,
              "gold": [SOIL] * 62 + [DESERT] * 38, "kind": ["dry land"] * 100}
        report, fitted = hierarchy.oof_backoff([f0, f1], ANC)
        P1, C1 = fitted[1]
        assert P1.shape == (100, 8) and DESERT not in C1.nodes
        pick, _ = C1.decode(P1, 0.5)
        out = hierarchy.outcome(hierarchy.relation(C1.nodes, f1["gold"], ANC), pick)
        assert list(out).count(EXACT) == 62 and list(out).count(OTHER) == 38
