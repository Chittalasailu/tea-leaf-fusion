"""Unit tests for fusionlib on small inputs with known answers."""
import numpy as np
import pytest

import fusionlib as fl
from conftest import rand_probs

ONE_HOT = np.eye(3)


# --- input checks

@pytest.mark.parametrize("bad", [
    np.array([[0.5, np.nan, 0.5]]),
    np.array([[0.5, np.inf, 0.5]]),
    np.array([[0.7, 0.7, 0.7]]),
    np.array([[-0.2, 0.6, 0.6]]),
    np.zeros((0, 3)),
    np.array([0.2, 0.3, 0.5]),
], ids=["nan", "inf", "unnormalised", "negative", "empty", "one-dimensional"])
def test_check_probs_rejects_malformed(bad):
    with pytest.raises(ValueError):
        fl.check_probs(bad)


def test_check_probs_wrong_class_count():
    with pytest.raises(ValueError):
        fl.check_probs(rand_probs(5, 4), n_classes=7)


def test_check_probs_accepts_valid():
    p = rand_probs(10, 7)
    assert fl.check_probs(p, 7).shape == (10, 7)


@pytest.mark.parametrize("y", [np.array([0, 1, 7]), np.array([0, -1, 2]), np.array([0, 1]), np.array([0.0, 1.0, 2.0])],
                         ids=["too-large", "negative", "wrong-length", "float"])
def test_check_labels_rejects_invalid(y):
    with pytest.raises(ValueError):
        fl.check_labels(y, 3, 7)


# --- fusion rules

def test_weighted_average_known_answer():
    a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(fl.weighted_average([a, b], [3, 1]), [[0.75, 0.25]])


def test_weighted_average_rows_sum_to_one():
    p = fl.weighted_average([rand_probs(seed=i) for i in range(3)], [0.2, 0.5, 0.3])
    assert np.allclose(p.sum(1), 1)


def test_weighted_average_single_member_is_identity():
    p = rand_probs()
    assert np.allclose(fl.weighted_average([p], [1]), p)


def test_weighted_average_mismatched_shapes():
    with pytest.raises(ValueError):
        fl.weighted_average([rand_probs(10), rand_probs(11)], [1, 1])


@pytest.mark.parametrize("w", [[1, 1], [-1, 1, 1], [0, 0, 0]], ids=["wrong-length", "negative", "all-zero"])
def test_weighted_average_bad_weights(w):
    with pytest.raises(ValueError):
        fl.weighted_average([rand_probs(seed=i) for i in range(3)], w)


def test_two_confident_wrong_members_outvote_a_correct_one():
    right = np.array([[0.9, 0.05, 0.05]])
    wrong = np.array([[0.05, 0.9, 0.05]])
    assert fl.weighted_average([right, wrong, wrong], np.ones(3)).argmax() == 1


def test_one_very_confident_member_can_win_soft_voting():
    a = np.array([[0.98, 0.01, 0.01]])
    b = np.array([[0.4, 0.6, 0.0]])
    assert fl.weighted_average([a, b, b], np.ones(3)).argmax() == 0
    assert fl.majority_vote([a, b, b], np.ones(3))[0] == 1


def test_accuracy_weights_normalised_and_ordered():
    y = np.array([0, 1, 2, 0])
    good, bad = ONE_HOT[y], ONE_HOT[[1, 1, 2, 0]]
    w = fl.accuracy_weights([good, bad], y)
    assert np.isclose(w.sum(), 1) and w[0] > w[1]


def test_accuracy_weights_alpha_zero_is_equal():
    y = np.array([0, 1, 2, 0])
    w = fl.accuracy_weights([ONE_HOT[y], ONE_HOT[[1, 1, 2, 0]]], y, alpha=0)
    assert np.allclose(w, [0.5, 0.5])


def test_accuracy_weights_large_alpha_picks_best():
    y = np.array([0, 1, 2, 0])
    w = fl.accuracy_weights([ONE_HOT[y], ONE_HOT[[1, 1, 2, 0]]], y, alpha=100)
    assert w[0] > 0.99


def test_product_rule_known_answer():
    a, b = np.array([[0.8, 0.2]]), np.array([[0.5, 0.5]])
    expected = np.sqrt([0.8 * 0.5, 0.2 * 0.5])
    assert np.allclose(fl.weighted_product([a, b], [1, 1]), expected / expected.sum())


def test_product_rule_zero_probability_vetoes():
    a, b = np.array([[1.0, 0.0]]), np.array([[0.4, 0.6]])
    assert fl.weighted_product([a, b], [1, 1]).argmax() == 0


def test_max_rule_known_answer():
    a, b = np.array([[0.6, 0.4]]), np.array([[0.1, 0.9]])
    assert np.allclose(fl.max_rule([a, b]), [[0.4, 0.6]])


def test_majority_vote_plurality():
    p = [ONE_HOT[[0]], ONE_HOT[[1]], ONE_HOT[[1]]]
    assert fl.majority_vote(p, np.ones(3))[0] == 1


def test_majority_vote_tie_falls_back_to_weighted_average():
    p = [np.array([[0.6, 0.3, 0.1]]), np.array([[0.2, 0.7, 0.1]]), np.array([[0.1, 0.2, 0.7]])]
    assert fl.majority_vote(p, [0.1, 0.8, 0.1])[0] == 1


def test_simplex_grid_has_231_points():
    g = fl.simplex_grid(3, 0.05)
    assert len(g) == 231 and np.allclose(g.sum(1), 1)


def test_grid_weights_find_the_perfect_member():
    y = np.array([0, 1, 2, 1, 0, 2])
    perfect, noise = ONE_HOT[y] * 0.9 + 0.1 / 3, rand_probs(6, 3, 1)
    w = fl.grid_weights([noise, perfect], y)
    assert w[1] >= 0.5


def test_logloss_weights_on_simplex():
    y = np.random.default_rng(0).integers(0, 3, 40)
    good = 0.7 * ONE_HOT[y] + 0.1
    w = fl.logloss_weights([good, rand_probs(40, 3, 2)], y)
    assert np.isclose(w.sum(), 1) and (w >= 0).all() and w[0] > w[1]


def test_classwise_weights_columns_sum_to_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    wc = fl.classwise_weights([ONE_HOT[y], ONE_HOT[[0, 0, 0, 0, 0, 0]]], y, 3)
    assert np.allclose(wc.sum(0), 1)


def test_classwise_weights_favour_the_better_member_per_class():
    y = np.array([0, 1, 2, 0, 1, 2])
    wc = fl.classwise_weights([ONE_HOT[[0, 0, 0, 0, 0, 0]], ONE_HOT[y]], y, 3)
    assert wc[0, 0] == pytest.approx(0.5) and wc[1, 1] == pytest.approx(1.0)


def test_calibrate_then_fuse_rows_sum_to_one():
    y = np.random.default_rng(0).integers(0, 3, 60)
    V = [0.6 * ONE_HOT[y] + 0.4 * rand_probs(60, 3, i) for i in range(2)]
    p = fl.calibrate_then_fuse(V, y, V, [0.5, 0.5])
    assert np.allclose(p.sum(1), 1)


def test_stacker_learns_and_outputs_probabilities():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 150)
    V = [0.5 * ONE_HOT[y] + 0.5 * rand_probs(150, 3, i) for i in range(3)]
    st = fl.Stacker().fit(V, y)
    q = st.predict_proba(V)
    assert np.allclose(q.sum(1), 1) and (q.argmax(1) == y).mean() > 0.9


# --- calibration and metrics

def test_nll_perfect_is_zero():
    y = np.array([0, 1, 2])
    assert fl.nll(ONE_HOT[y], y) == pytest.approx(0, abs=1e-9)


def test_nll_uniform_is_log_k():
    y = np.array([0, 1, 2])
    assert fl.nll(np.full((3, 3), 1 / 3), y) == pytest.approx(np.log(3))


def test_brier_perfect_and_worst():
    y = np.array([0, 1])
    assert fl.brier(ONE_HOT[y], y) == 0
    assert fl.brier(ONE_HOT[[1, 0]], y) == pytest.approx(2)


def test_ece_zero_when_confidence_matches_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.5, 0.5, 0.0]] * 4)
    assert fl.ece(p, y) == pytest.approx(0, abs=1e-9)


def test_ece_known_answer():
    y = np.array([0, 0, 0, 0])
    p = np.array([[0.9, 0.1, 0.0]] * 2 + [[0.1, 0.9, 0.0]] * 2)
    # all four images fall in the same bin: accuracy 0.5, confidence 0.9
    assert fl.ece(p, y) == pytest.approx(0.4)


@pytest.mark.parametrize("bins", [10, 15, 20])
def test_ece_bounded(bins):
    p, y = rand_probs(200, 7), np.random.default_rng(0).integers(0, 7, 200)
    assert 0 <= fl.ece(p, y, bins) <= 1


def test_temperature_one_is_identity():
    p = rand_probs()
    assert np.allclose(fl.temperature_scale(p, 1.0), p)


def test_temperature_keeps_predicted_class():
    p = rand_probs()
    for t in (0.3, 2.5):
        assert (fl.temperature_scale(p, t).argmax(1) == p.argmax(1)).all()


def test_temperature_below_one_sharpens():
    p = rand_probs()
    assert (fl.temperature_scale(p, 0.5).max(1) >= p.max(1) - 1e-12).all()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        fl.temperature_scale(rand_probs(), 0)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    logits = rng.normal(0, 1, (500, 3))
    logits[np.arange(500), y] += 1.0
    true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    sharp = fl.temperature_scale(true, 0.4)
    assert fl.fit_temperature(sharp, y) > 1.5


def test_isotonic_outputs_probabilities():
    p, y = rand_probs(300, 4), np.random.default_rng(0).integers(0, 4, 300)
    q = fl.Isotonic().fit(p, y).predict(p)
    assert np.allclose(q.sum(1), 1) and (q >= 0).all()


def test_macro_auc_perfect_ranking():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert fl.macro_auc(ONE_HOT[y], y) == pytest.approx(1.0)


# --- diversity

def test_pair_diversity_identical_members():
    y = np.array([0, 1, 2, 0])
    pred = np.array([0, 1, 1, 0])
    d = fl.pair_diversity(pred, pred, y)
    assert d["q"] == 0.0 or d["q"] == pytest.approx(1.0)
    assert d["disagreement"] == 0 and d["double_fault"] == 0.25


def test_pair_diversity_q_known_answer():
    y = np.zeros(10, int)
    a = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    b = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    n11, n00, n10, n01 = 5, 3, 1, 1
    d = fl.pair_diversity(a, b, y)
    assert d["q"] == pytest.approx((n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10))
    assert d["disagreement"] == pytest.approx(0.2) and d["double_fault"] == pytest.approx(0.3)


def test_pair_diversity_complementary_members_negative_q():
    y = np.zeros(4, int)
    d = fl.pair_diversity(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), y)
    assert d["q"] == pytest.approx(-1.0) and d["double_fault"] == 0


def test_oracle_accuracy():
    y = np.array([0, 1, 2])
    assert fl.oracle_accuracy([np.array([0, 0, 0]), np.array([1, 1, 1])], y) == pytest.approx(2 / 3)


def test_agreement_breakdown_shares_sum_to_one():
    y = np.random.default_rng(0).integers(0, 3, 100)
    preds = [np.random.default_rng(i).integers(0, 3, 100) for i in range(3)]
    ab = fl.agreement_breakdown(preds, preds[0], y)
    assert sum(v[0] for v in ab.values()) == pytest.approx(1)


# --- significance

def test_mcnemar_identical_classifiers():
    c = np.array([True, False, True])
    assert fl.mcnemar(c, c) == (0, 0, 1.0)


def test_mcnemar_known_p_value():
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    n1, n2, p = fl.mcnemar(a, b)
    assert (n1, n2) == (10, 0) and p == pytest.approx(2 * 0.5 ** 10)


def test_mcnemar_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.random(200) > 0.3, rng.random(200) > 0.4
    assert fl.mcnemar(a, b)[2] == pytest.approx(fl.mcnemar(b, a)[2])


def test_holm_known_answer():
    assert fl.holm([0.01, 0.04, 0.03, 0.2]).tolist() == [True, False, False, False]


def test_holm_all_significant():
    assert fl.holm([0.001, 0.002, 0.003]).all()


def test_holm_stricter_than_raw():
    p = np.array([0.02, 0.03, 0.04])
    assert fl.holm(p).sum() <= (p < 0.05).sum()


def test_bootstrap_ci_contains_point_estimate():
    c = np.random.default_rng(0).random(500) > 0.1
    lo, hi = fl.bootstrap_ci(c, fl.bootstrap_indices(500, 500))
    assert lo <= c.mean() * 100 <= hi


def test_bootstrap_ci_constant_input():
    lo, hi = fl.bootstrap_ci(np.ones(50, bool), fl.bootstrap_indices(50, 200))
    assert lo == hi == 100


# --- selective prediction and unfamiliar-image scores

def test_advise_threshold():
    p = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert fl.advise(p, 0.8).tolist() == [True, False]


def test_advise_with_agreement_flag():
    p = np.array([[0.9, 0.1], [0.95, 0.05]])
    assert fl.advise(p, 0.8, np.array([True, False])).tolist() == [True, False]


def test_selective_stats_known_answer():
    p = np.array([[0.9, 0.1], [0.85, 0.15], [0.55, 0.45]])
    s = fl.selective_stats(p, np.array([0, 1, 0]), fl.advise(p, 0.8))
    assert s == {"coverage": pytest.approx(2 / 3), "selective_acc": 0.5, "wrong_advice": pytest.approx(1 / 3)}


def test_selective_stats_no_advice():
    p = np.array([[0.5, 0.5]])
    assert np.isnan(fl.selective_stats(p, np.array([0]), fl.advise(p, 0.9))["selective_acc"])


def test_risk_coverage_perfect_ranking():
    score = np.array([0.9, 0.8, 0.3])
    cov, risk = fl.risk_coverage(score, np.array([True, True, False]))
    assert risk[:2].tolist() == [0, 0] and risk[-1] == pytest.approx(1 / 3)


def test_aurc_better_ranking_is_lower():
    correct = np.array([True, True, False, False])
    assert fl.aurc(np.array([4, 3, 2, 1]), correct) < fl.aurc(np.array([1, 2, 3, 4]), correct)


def test_coverage_at_risk():
    assert fl.coverage_at_risk(np.array([3, 2, 1]), np.array([True, True, False]), 0.01) == pytest.approx(2 / 3)


def test_entropy_uniform_and_one_hot():
    assert fl.entropy(np.full((1, 4), 0.25))[0] == pytest.approx(np.log(4))
    assert fl.entropy(np.eye(4)[:1])[0] == pytest.approx(0, abs=1e-9)


def test_margin():
    assert fl.margin(np.array([[0.6, 0.3, 0.1]]))[0] == pytest.approx(0.3)


def test_js_divergence_zero_for_identical_members():
    p = rand_probs()
    assert np.allclose(fl.mean_pairwise_js([p, p, p]), 0, atol=1e-9)


def test_js_divergence_bounded_by_log2():
    a, b = np.eye(2)[:1], np.eye(2)[1:]
    assert fl.mean_pairwise_js([a, b])[0] == pytest.approx(np.log(2))


def test_ood_auroc_perfect_and_chance():
    assert fl.ood_auroc(np.zeros(10), np.ones(10)) == 1.0
    assert fl.ood_auroc(np.ones(10), np.ones(10)) == 0.5


def test_fpr_at_tpr():
    assert fl.fpr_at_tpr(np.linspace(0, 1, 101), np.full(20, 2.0)) == 0.0
