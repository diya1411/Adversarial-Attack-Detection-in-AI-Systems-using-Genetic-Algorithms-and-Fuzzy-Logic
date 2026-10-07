import numpy as np
import pytest

from advfuzz.fuzzy import (
    ADVERSARIAL,
    ARITY,
    MIN_SEPARATION,
    N_RULES,
    RULES,
    SAFE,
    FuzzyInferenceSystem,
    decode_partition,
    defuzzify,
    firing_strengths,
    memberships,
    uniform_partition,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("trial", range(20))
def test_decoded_partition_is_ordered_and_separated(trial):
    bp = decode_partition(rng.random(12))
    assert np.all(bp >= 0) and np.all(bp <= 1)
    assert np.all(np.diff(bp, axis=1) >= MIN_SEPARATION - 1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_partition_is_complete_and_sums_to_one(trial):
    bp = decode_partition(rng.random(12))
    M = memberships(rng.random((500, 4)), bp)
    np.testing.assert_allclose(M.sum(axis=2), 1.0)
    assert np.all(M >= 0) and np.all(M <= 1)


def test_uniform_partition_shapes():
    M = memberships(np.array([[0.0, 0.25, 0.5, 1.0]]), uniform_partition())[0]
    np.testing.assert_allclose(M[0], [1, 0, 0])   # 0.00 -> Low
    np.testing.assert_allclose(M[1], [1, 0, 0])   # 0.25 = p1 -> fully Low
    np.testing.assert_allclose(M[2], [0, 1, 0])   # 0.50 = p2 -> fully Medium
    np.testing.assert_allclose(M[3], [0, 0, 1])   # 1.00 -> High


def test_arity_correction_removes_antecedent_count_bias():
    # With every membership degree equal to m, every rule must fire at m,
    # whatever its number of antecedents.
    m = 0.6
    alpha = firing_strengths(np.full((1, 4, 3), m))[0]
    np.testing.assert_allclose(alpha, m)
    assert set(ARITY) == {2.0, 3.0}


def test_score_is_bounded_by_consequents():
    fis = FuzzyInferenceSystem(decode_partition(rng.random(12)), rng.random(N_RULES), 0.5)
    r = fis.score(rng.random((1000, 4)))
    assert np.all(r >= SAFE - 1e-12) and np.all(r <= ADVERSARIAL + 1e-12)


def test_weights_are_scale_invariant():
    Z = rng.random((200, 4))
    w = rng.random(N_RULES)
    a = FuzzyInferenceSystem(weights=w).score(Z)
    b = FuzzyInferenceSystem(weights=0.3 * w).score(Z)
    np.testing.assert_allclose(a, b)


def test_no_rule_firing_defaults_to_suspicious():
    assert defuzzify(np.zeros((1, N_RULES)), np.ones(N_RULES))[0] == pytest.approx(0.5)


def test_r9_flags_confident_but_atypical_input():
    # C High, U Low, P Low, F High: R9 (Adversarial) competes only with R1 (Safe).
    fis = FuzzyInferenceSystem()
    z = np.array([0.95, 0.05, 0.05, 0.95])
    explanation = fis.explain(z)
    fired = {r["rule"].split(":")[0] for r in explanation["rules"]}
    assert fired == {"R1", "R9"}
    assert explanation["score"] == pytest.approx(0.5)


def test_explain_shares_sum_to_one_and_match_score():
    fis = FuzzyInferenceSystem(weights=rng.random(N_RULES))
    z = rng.random(4)
    ex = fis.explain(z, top=N_RULES)
    assert sum(r["share"] for r in ex["rules"]) == pytest.approx(1.0)
    assert ex["score"] == pytest.approx(fis.score(z[None])[0])


def test_triage_is_consistent_with_binary_decision():
    fis = FuzzyInferenceSystem(weights=rng.random(N_RULES), theta=0.35)
    Z = rng.random((500, 4))
    tri, pred = fis.triage(Z), fis.predict(Z)
    assert np.all((tri == "Safe") == (pred == 0))
    assert np.all(fis.score(Z)[tri == "Adversarial"] >= fis.reject_threshold())


def test_dict_round_trip():
    fis = FuzzyInferenceSystem(decode_partition(rng.random(12)), rng.random(N_RULES), 0.42)
    back = FuzzyInferenceSystem.from_dict(fis.to_dict())
    np.testing.assert_allclose(back.breakpoints, fis.breakpoints)
    np.testing.assert_allclose(back.weights, fis.weights)
    assert back.theta == fis.theta


def test_rule_base_matches_table_ii():
    assert len(RULES) == 15
    assert RULES[8] == ({"C": "High", "F": "High"}, ADVERSARIAL)
