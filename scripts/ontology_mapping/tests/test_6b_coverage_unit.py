"""Unit tests for 6b_coverage.py: its two helper functions, on matrices small enough to check by eye.

Functions under test:
    top_k_mean(sim, k)         sim: (n_queries x n_training) similarity matrix
                               -> (mean of the k largest values of each row, column of the largest value)
                               main() uses it twice: atlas samples x training samples, and in calibrate().
    calibrate(kw, sb, groups, k, quantile, block)
                               kw, sb: one L2-normalised keyword / sub-biome vector per training sample
                               groups: study code (or project group) of each training sample
                               -> (threshold, {quantile: value} summary, held-out similarity per sample)
                               held-out similarity of sample i = top_k_mean over the samples of the
                               OTHER groups of  0.5 * (kw_i . kw_j + sb_i . sb_j)
                               (NaN when fewer than k samples are in other groups; ValueError if all are NaN)
                               threshold = the `quantile` of these held-out similarities (NaN left out)
main() (files, chunking, resume) is in test_6b_coverage_integration.py.

Conventions (as in the other test files of this folder):
  * Tests named `..._quirk` pin down current behaviour that is surprising; they document it, they do
    not endorse it. If the behaviour is changed on purpose, update the test.
  * Tests marked xfail(strict=True) are KNOWN BUGS (none in this file).
"""

import numpy as np
import pytest

from step_loader import load_step

cov = load_step("6b_coverage.py", "atlas_coverage")  # not "coverage": that name belongs to coverage.py / pytest-cov


# ============================================================================= top_k_mean()
class TestTopKMean:

    def test_k1_is_the_row_maximum_and_its_column(self):
        sim = np.array([[0.1, 0.9, 0.5],     # largest: 0.9 in column 1
                        [0.7, 0.2, 0.3]])    # largest: 0.7 in column 0
        value, column = cov.top_k_mean(sim, k=1)
        assert value.tolist() == [0.9, 0.7]
        assert column.tolist() == [1, 0]

    def test_k2_averages_the_two_largest_but_the_column_is_still_the_single_nearest(self):
        sim = np.array([[0.1, 0.9, 0.5]])
        value, column = cov.top_k_mean(sim, k=2)
        assert value == pytest.approx([0.7])   # (0.9 + 0.5) / 2
        assert column.tolist() == [1]          # nearest_study in the output = study of the single nearest

    def test_ties_go_to_the_first_column(self):
        value, column = cov.top_k_mean(np.array([[0.3, 0.8, 0.8]]), k=1)
        assert value.tolist() == [0.8]
        assert column.tolist() == [1]          # np.argmax: first of the equal maxima

    def test_k_equal_to_the_number_of_columns_is_the_row_mean(self):
        value, _ = cov.top_k_mean(np.array([[0.2, 0.4, 0.9]]), k=3)
        assert value == pytest.approx([0.5])   # (0.2 + 0.4 + 0.9) / 3

    @pytest.mark.parametrize("k", [0, 3, 4, 5])
    def test_k_must_be_between_1_and_the_number_of_columns(self, k):
        # Without the explicit check, np.partition (kth = n_columns - k) accepts a negative kth down to
        # -n_columns: k = 3 or 4 would silently return the mean of both columns (see the mutation tests).
        with pytest.raises(ValueError, match="only 2 training samples"):
            cov.top_k_mean(np.array([[1.0, 2.0]]), k=k)

    def test_minus_inf_entries_are_never_chosen_when_k_is_1(self):
        # calibrate() masks same-group training samples with -inf
        value, column = cov.top_k_mean(np.array([[-np.inf, 0.4, -np.inf]]), k=1)
        assert value.tolist() == [0.4]
        assert column.tolist() == [1]

    def test_minus_inf_enters_the_mean_when_fewer_than_k_entries_are_finite(self):
        value, _ = cov.top_k_mean(np.array([[-np.inf, 0.4, -np.inf]]), k=2)
        assert value.tolist() == [-np.inf]    # (0.4 + -inf) / 2


# ============================================================================= calibrate()
# Four training samples in 2-D. Keyword vector = sub-biome vector, so the similarity
# 0.5 * (kw . kw + sb . sb) is just the cosine of the two points:
#
#     s3 (0, 1)  study C
#      |    s2 (0.8, 0.6)  study B
#      |   /
#      |  /
#      +----- s0, s1 (1, 0)  study A (two identical samples)
#
#   cos(s0, s1) = 1     but same study: never compared
#   cos(s0, s2) = 0.8
#   cos(s2, s3) = 0.6
#   cos(s0, s3) = 0
X = np.array([[1.0, 0.0],     # s0  A
              [1.0, 0.0],     # s1  A
              [0.8, 0.6],     # s2  B
              [0.0, 1.0]])    # s3  C
STUDIES = np.array(["A", "A", "B", "C"])


class TestCalibrate:

    def test_each_sample_is_compared_only_with_other_studies(self):
        _, _, held_out = cov.calibrate(X, X, STUDIES, k=1, quantile=0.05)
        assert held_out == pytest.approx([
            0.8,   # s0: nearest other study is s2 (0.8), NOT its identical twin s1 (1.0, same study)
            0.8,   # s1: same as s0
            0.8,   # s2: nearest is s0 / s1 (0.8); s3 is at 0.6
            0.6,   # s3: nearest is s2 (0.6)
        ])

    def test_threshold_is_the_requested_quantile_of_the_held_out_similarities(self):
        # held-out = [0.8, 0.8, 0.8, 0.6]; np.quantile interpolates linearly between sorted values
        assert cov.calibrate(X, X, STUDIES, k=1, quantile=0.0)[0] == pytest.approx(0.6)   # the minimum
        assert cov.calibrate(X, X, STUDIES, k=1, quantile=0.5)[0] == pytest.approx(0.8)   # the median
        # default 0.05: position 0.05 * 3 = 0.15 between sorted values 0.6 and 0.8 -> 0.6 + 0.15 * 0.2
        assert cov.calibrate(X, X, STUDIES, k=1, quantile=0.05)[0] == pytest.approx(0.63)

    def test_the_summary_lists_fixed_quantiles_rounded_to_4_decimals(self):
        _, summary, _ = cov.calibrate(X, X, STUDIES, k=1, quantile=0.05)
        assert summary == {"0.01": 0.606, "0.02": 0.612, "0.05": 0.63, "0.1": 0.66, "0.25": 0.75, "0.5": 0.8}

    def test_similarity_is_the_mean_of_the_keyword_and_sub_biome_cosines(self):
        kw = np.array([[1.0, 0.0], [1.0, 0.0]])   # same keywords          -> keyword cosine 1
        sb = np.array([[1.0, 0.0], [0.0, 1.0]])   # unrelated sub-biomes   -> sub-biome cosine 0
        _, _, held_out = cov.calibrate(kw, sb, np.array(["A", "B"]), k=1, quantile=0.05)
        assert held_out == pytest.approx([0.5, 0.5])   # 0.5 * (1 + 0)

    def test_k2_averages_the_two_nearest_samples_of_other_studies(self):
        _, _, held_out = cov.calibrate(X, X, STUDIES, k=2, quantile=0.05)
        assert held_out == pytest.approx([
            0.4,   # s0: s2 0.8, s3 0      -> 0.4
            0.4,   # s1: same
            0.8,   # s2: s0 0.8, s1 0.8    -> 0.8
            0.3,   # s3: s2 0.6, s0 or s1 0 -> 0.3
        ])

    def test_grouping_studies_into_projects_removes_the_close_neighbours(self):
        # --fold_groups: studies A and B are one project, so s0/s1 can no longer use s2 (and vice versa)
        projects = np.array(["P1", "P1", "P1", "C"])
        _, _, held_out = cov.calibrate(X, X, projects, k=1, quantile=0.05)
        assert held_out == pytest.approx([0.0, 0.0, 0.6, 0.6])   # s0, s1 only see s3; s2 only s3; s3 sees s2

    def test_the_result_does_not_depend_on_the_block_size(self):
        # main() never sets `block`; it only bounds memory. One sample per block must give the same answer.
        default = cov.calibrate(X, X, STUDIES, k=1, quantile=0.05)
        one_by_one = cov.calibrate(X, X, STUDIES, k=1, quantile=0.05, block=1)
        assert one_by_one[0] == default[0]
        assert one_by_one[2].tolist() == default[2].tolist()

    def test_group_labels_can_be_any_type(self):
        as_ints = cov.calibrate(X, X, np.array([7, 7, 3, 5]), k=1, quantile=0.05)
        assert as_ints[2] == pytest.approx([0.8, 0.8, 0.8, 0.6])

    # ------------------------------------------------------------------------- degenerate inputs
    def test_a_sample_with_fewer_than_k_samples_outside_its_group_has_no_held_out_similarity(self):
        # k=3: s0 and s1 (study A) have only 2 samples in other studies (s2, s3). Their top 3 would include
        # a masked -inf, so their held-out similarity is NaN and they are left out of the quantiles.
        threshold, _, held_out = cov.calibrate(X, X, STUDIES, k=3, quantile=0.0)
        assert np.isnan(held_out[:2]).all()
        assert held_out[2:] == pytest.approx([
            (0.8 + 0.8 + 0.6) / 3,   # s2: s0, s1, s3
            (0.6 + 0.0 + 0.0) / 3,   # s3: s2, s0, s1  -> 0.2
        ])
        assert threshold == pytest.approx(0.2)   # quantile 0 = minimum of the samples that have a value
        # (without this, the -inf values made np.quantile return NaN, and `coverage_sim >= NaN` is False:
        #  the whole atlas was flagged out of coverage; see the mutation tests)

    def test_a_single_group_cannot_be_calibrated(self):
        # nobody has another group to be compared with
        with pytest.raises(ValueError, match="cannot be calibrated"):
            cov.calibrate(X, X, np.array(["A", "A", "A", "A"]), k=1, quantile=0.05)
