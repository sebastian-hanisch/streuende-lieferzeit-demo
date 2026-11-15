"""Kern von Hand geprüft (DEMO-PLAYBOOK §1b): das Handbeispiel k = 2, p = 2/3 (Mittel 3) in Brüchen, die Hurwitz-Grenzen des stetigen Kreises,
die Bestandsposition bei β = 1, dazu der Zufallsgenerator und die Zufallsziehungen. Die Handrechnung steht als Rechnung im Test, nicht als abgeschriebener Wert.

Rundung: Eigenwerte und Wurzeln werden mit Toleranzen verglichen, Brüche exakt. Kein Parameter ist eine Zweierpotenz außer der festen Lieferzeit (p = 1)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from refimpl import matrix_from_step, step_fractions

import slz_dynamics as K
import slz_rng as R

# --- H2: Lieferzeit-Verteilung der Kette ---------------------------------------------------------------------------------------------


def test_hand_lead_time_distribution_of_the_chain_in_fractions():
    """k = 2, p = 2/3: f_l = (l - 1) p^2 q^(l-2); eine Bestellung durch die Kette geschickt (α = 0, Brüche) kommt genau mit diesen Anteilen an."""
    p = F(2, 3)
    q = 1 - p
    f = {l: (l - 1) * p ** 2 * q ** (l - 2) for l in range(2, 14)}
    assert (f[2], f[3], f[4]) == (F(4, 9), F(8, 27), F(4, 27))
    i, x = F(0), [F(1), F(0)]                     # die Bestellung 1 liegt in Stufe 1
    arrivals = {}
    for t in range(12):
        i, x, _, arrival = step_fractions(i, x, F(0), F(0), p)
        arrivals[t + 1] = arrival
    assert arrivals[1] == 0 and all(arrivals[l] == f[l] for l in range(2, 13))
    pmf = K.erlang_pmf(2, 2 / 3, 12)
    assert all(abs(pmf[l - 1] - float(f[l])) < 1e-15 for l in range(2, 13))
    assert abs(sum(l * float(fl) for l, fl in f.items()) - 3.0) < 1e-2          # Mittel 3, Rest hinter l = 13


# --- H3: Grenze k = 2 -----------------------------------------------------------------------------------------------------------------


def test_hand_limit_8_9_polynomial_and_roots(hand):
    p, q = F(2, 3), F(1, 3)
    alpha = p * (2 - p)
    assert alpha == F(8, 9)                                                    # Jury (Grad 3): alpha < p (2 - p)
    coeffs = (-(2 * q + 1), q ** 2 + 2 * q + alpha * p ** 2, -q ** 2)
    assert coeffs == (F(-5, 3), F(95, 81), F(-1, 9))
    assert np.allclose(K.charpoly_chain(2, 2 / 3, 8 / 9, 0), [1.0] + [float(c) for c in coeffs], atol=1e-14)
    # (z^2 - (14/9) z + 1)(z - 1/9) hat dieselben Koeffizienten: Paar auf dem Einheitskreis mit Realteil 7/9, dazu die Wurzel 1/9
    assert (F(14, 9) + F(1, 9), 1 + F(14, 9) * F(1, 9), F(1, 9)) == (F(5, 3), F(95, 81), F(1, 9))
    assert F(7, 9) ** 2 + F(32, 81) == 1
    A, _ = K.chain_matrix(2, 2 / 3, 8 / 9, 0)
    assert np.allclose(A, matrix_from_step(F(8, 9), F(0), F(2, 3), 2), atol=1e-15)
    ev = np.linalg.eigvals(A)
    expected = sorted([complex(7 / 9, 4 * math.sqrt(2) / 9), complex(7 / 9, -4 * math.sqrt(2) / 9), complex(1 / 9, 0)], key=lambda z: (z.imag, z.real))
    assert np.allclose(sorted(ev, key=lambda z: (z.imag, z.real)), expected, atol=1e-9)
    assert abs(2 * math.pi / math.acos(7 / 9) - 9.24) < 0.01                   # Periode der kritischen Schwingung
    assert abs(K.critical_alpha_chain(2, 2 / 3) - hand["alpha_krit"]) < 1e-9
    below = K.spectral_radius(K.chain_matrix(2, 2 / 3, 8 / 9 * 0.99, 0)[0])
    above = K.spectral_radius(K.chain_matrix(2, 2 / 3, 8 / 9 * 1.01, 0)[0])
    assert below < 1 < above


def test_hand_fixed_delay_three_golden_ratio_and_the_factor_1_438():
    fixed = 2 * math.sin(math.pi / 10)
    assert abs(fixed - (math.sqrt(5) - 1) / 2) < 1e-14
    assert abs(K.critical_alpha_fixed(3) - fixed) < 1e-15
    assert abs(K.critical_alpha_chain(3, 1.0) - fixed) < 1e-9
    assert abs((8 / 9) / fixed - 1.438) < 1e-3


def test_hand_critical_oscillation_of_the_fixed_delay_has_period_4L_minus_2():
    for L in (2, 3, 5, 8):
        a = K.critical_alpha_fixed(L)
        ev = np.linalg.eigvals(K.chain_matrix(L, 1.0, a, 0)[0])
        crit = ev[np.argmax(np.abs(ev))]
        assert abs(abs(crit) - 1) < 1e-9
        assert abs(2 * math.pi / abs(np.angle(crit)) - (4 * L - 2)) < 1e-6


# --- H4: k = 1 -------------------------------------------------------------------------------------------------------------------------


def test_hand_geometric_lead_time_limit_10_with_root_minus_one():
    p, q = F(1, 3), F(2, 3)
    alpha = 2 * (2 - p) / p
    assert alpha == 10 == 4 * 3 - 2                                            # = 4 tau - 2 für tau = 3
    assert (-(1 + q - p * alpha), q) == (F(5, 3), F(2, 3))                     # z^2 + 5/3 z + 2/3 = (z + 1)(z + 2/3)
    assert abs(K.critical_alpha_chain(1, 1 / 3) - 10) < 1e-9
    roots = np.roots(K.charpoly_chain(1, 1 / 3, 10.0, 0))
    assert np.allclose(sorted(roots.real), [-1.0, -2 / 3], atol=1e-12)
    assert K.spectral_radius(K.chain_matrix(1, 1 / 3, 9.9, 0)[0]) < 1 < K.spectral_radius(K.chain_matrix(1, 1 / 3, 10.1, 0)[0])
    assert abs(10 / K.critical_alpha_fixed(3) - 16.18) < 0.01                  # Faktor gegen die feste Lieferzeit 3


# --- H1: stetiger Erlang-Kreis ---------------------------------------------------------------------------------------------------------


def test_hand_continuous_limits_by_hurwitz():
    # k = 2: s^3 + 4 s^2 + 4 s + 4a; Grad 3: a2 a1 > a0, also 16 > 4a, a < 4; bei a = 4: (s + 4)(s^2 + 4), Wurzel 2i
    assert abs((2j) ** 3 + 4 * (2j) ** 2 + 4 * (2j) + 16) < 1e-12
    assert abs(K.critical_continuous(2) - 4.0) < 1e-9
    # k = 3: s^4 + 9 s^3 + 27 s^2 + 27 s + 27 a; Grad 4: a3 a2 a1 > a1^2 + a3^2 a0 mit a3 = 9, a2 = 27, a1 = 27, a0 = 27 a
    a = F(9 * 27 * 27 - 27 ** 2, 9 ** 2 * 27)
    assert a == F(8, 3)
    assert abs(K.critical_continuous(3) - 8 / 3) < 1e-9
    assert K.critical_continuous(1) == math.inf                                # s^2 + s + a: nie instabil
    assert all(np.roots([1, 1, x]).real.max() < 0 for x in (0.1, 1.3, 10, 1000))
    assert abs((8 / 3) / (math.pi / 2) - 1.6977) < 1e-4


def test_hand_discrete_to_continuous_k2():
    """k = 2, p = 2/tau: alpha_krit * tau = 4 (tau - 1) / tau, also 4 im Grenzfall."""
    for tau in (3, 5, 12, 60, 600):
        a = K.critical_alpha_chain(2, 2 / tau)
        assert abs(a * tau - 4 * (tau - 1) / tau) < 1e-9
    assert abs(K.critical_alpha_chain(2, 2 / 600) * 600 - 4) < 0.01


# --- H5: Bestandsposition bei β = 1 ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p", [(1, F(1, 3)), (2, F(2, 3)), (3, F(3, 5)), (2, F(1, 5))])
def test_hand_inventory_position_identity_in_fractions(k, p):
    """β = 1: IP = i + Inhalt der Kette gehorcht IP' = (1 - alpha) IP - e für jede Kette (10 Schritte, Brüche)."""
    alpha = F(3, 5)
    i, x = F(0), [F(0)] * k
    for e in (F(3), F(-2), F(5), F(1), F(-4), F(2), F(0), F(7), F(-1), F(2)):
        ip = i + sum(x)
        i, x, _, _ = step_fractions(i, x, alpha, F(1), p, e)
        assert i + sum(x) == (1 - alpha) * ip - e


@pytest.mark.parametrize("k,p", [(1, 1 / 3), (2, 2 / 3), (3, 3 / 5), (3, 1.0)])
def test_hand_order_variance_3_7_for_every_chain(k, p):
    """α = 3/5: var(u)/var(e) = α / (2 - α) = 3/7, gleich für jede Lieferzeit-Verteilung (hier vier Ketten)."""
    A, c = K.chain_matrix(k, p, 0.6, 1)
    assert abs(K.variances(A, c)[0] - 3 / 7) < 1e-9
    assert np.allclose(A, matrix_from_step(F(3, 5), F(1), F(p).limit_denominator(1000), k), atol=1e-12)


def test_beta_one_polynomial_is_the_product_form():
    for k, p in ((1, 1 / 3), (2, 1 / 6), (3, 0.3), (6, 0.45), (4, 1.0)):
        q = 1 - p
        for a in (0.15, 0.35, 0.7, 1.3, 1.9):
            ch = np.poly(K.chain_matrix(k, p, a, 1)[0])
            ref = np.polymul([1.0, -(1 - a)], np.poly([q] * k))
            assert np.max(np.abs(ch - ref)) < 1e-12 and np.allclose(ch, K.charpoly_chain(k, p, a, 1), atol=1e-12)


# --- Einheiten --------------------------------------------------------------------------------------------------------------------------


def test_stages_clamps_k_to_tau_and_gives_p():
    assert K.stages(3, 2) == (2, 2 / 3) and K.stages(3, 12) == (3, 1.0) and K.stages(12, 1) == (1, 1 / 12) and K.stages(2, 5) == (2, 1.0)
    assert K.stages(5, 0)[0] == 1


def test_fixed_delay_one_is_a_single_feedback_step():
    """L = 1 (k = 1, p = 1): i' = i + x_1 + u mit x_1 = 0 auf jedem erreichbaren Zustand, also i' = (1 - α) i; die Matrix steht von Hand da
    (Zeile 1: 1 - α und 1, Zeile 2 leer, weil aus Stufe 1 alles weitergeht); Eigenwerte 0,4 und 0; die Grenze ist α = 2 (Überregeln)."""
    A, c = K.chain_matrix(1, 1.0, 0.6, 0)
    assert np.allclose(A, [[0.4, 1.0], [0.0, 0.0]], atol=1e-15) and np.allclose(c, [-0.6, 0.0])
    assert sorted(np.abs(np.linalg.eigvals(A))) == pytest.approx([0.0, 0.4], abs=1e-12)
    assert K.critical_alpha_chain(1, 1.0) == pytest.approx(2.0, abs=1e-9)


def test_decay_time_by_hand_for_L1():
    """i_t = -(1 - α)^(t-1): bei α = 0,6 liegt 0,4^5 = 0,01024 noch über 1 %, 0,4^6 darunter: letzte Überschreitung bei t = 6; α = 1: t = 1."""
    A, c = K.chain_matrix(1, 1.0, 0.6, 0)
    assert K.decay_time(A, c).time == 6
    A1, c1 = K.chain_matrix(1, 1.0, 1.0 - 1e-12, 0)
    assert K.decay_time(A1, c1).time == 1


def test_decay_time_none_for_unstable_and_not_decayed():
    A, c = K.chain_matrix(3, 1.0, 0.4, 0)                                         # α_krit(3) = 0,618: stabil, aber ρ = 0,88
    assert K.decay_time(A, c).time is not None
    A, c = K.chain_matrix(3, 1.0, 0.7, 0)
    assert K.decay_time(A, c).time is None                                        # instabil
    A, c = K.chain_matrix(12, 1.0, 0.0, 0)                                        # α = 0: ρ = 1, nichts klingt ab
    assert K.decay_time(A, c, n_max=500).time is None


def test_impulse_response_starts_with_minus_one_and_first_order():
    A, c = K.chain_matrix(2, 2 / 3, 0.7, 0)
    i, u = K.impulse_response(A, c, 5)
    assert i[0] == -1.0 and u[0] == pytest.approx(0.7)                          # u_1 = -α i_1


def test_distribution_builders_have_the_requested_mean():
    for tau in (2, 3, 5, 12):
        for d in (0, 1, 2, 11):
            for f in (K.two_point_pmf(tau, d), K.uniform_pmf(tau, d)):
                assert abs(K.mean_var(f)[0] - tau) < 1e-12 and abs(f.sum() - 1) < 1e-12
    f = K.two_point_pmf(4, 2)
    assert f[1] == 0.5 and f[5] == 0.5 and len(f) == 6                          # 2 und 6
    assert K.mean_var(K.uniform_pmf(4, 2))[1] == pytest.approx(2.0)               # gleichverteilt 2..6: Varianz (5^2 - 1)/12
    assert K.mean_var(K.two_point_pmf(4, 3))[1] == pytest.approx(9.0)
    assert np.array_equal(K.two_point_pmf(3, 0), K.fixed_pmf(3)) and np.array_equal(K.two_point_pmf(3, 99), K.two_point_pmf(3, 2))   # d wird auf tau - 1 begrenzt


def test_random_pmf_has_exact_mean_and_is_deterministic():
    for tau in (2, 3, 4, 7, 20):
        for form in K.RANDOM_FORMS:
            for seed in (35, 20261115, 7):
                f = K.random_pmf(seed, tau, form)
                assert abs(f.sum() - 1) < 1e-12 and (f > 0).all() and tau + 1 <= len(f) <= tau + 4
                assert abs(K.mean_var(f)[0] - tau) < 1e-9
    assert np.array_equal(K.random_pmf(35, 4, "gemischt"), K.random_pmf(35, 4, "gemischt"))
    assert not np.array_equal(K.random_pmf(35, 4, "gemischt"), K.random_pmf(36, 4, "gemischt"))


def test_erlang_pmf_mean_and_variance():
    for k, p in ((1, 0.25), (2, 0.4), (3, 0.3)):
        f = K.erlang_pmf(k, p, 600)
        m, v = K.mean_var(f)
        assert abs(f.sum() - 1) < 1e-9 and abs(m - k / p) < 1e-6 and abs(v - k * (1 - p) / p ** 2) < 1e-4


# --- Zufallsgenerator -------------------------------------------------------------------------------------------------------------------


def test_splitmix_reference_vector_and_vectorised_version():
    assert R.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF                      # veröffentlichter Referenzwert für den Zustand 0
    g = R.SplitMix64(12345)
    ref = [g.next_u64() for _ in range(50)]
    assert [int(v) for v in R.splitmix_array(12345, 50)] == ref
    g2 = R.SplitMix64(99)
    uni = [g2.random() for _ in range(30)]
    assert np.allclose(R.uniform_array(99, 30), uni, atol=0, rtol=0)
    g3 = R.SplitMix64(5)
    nor = [g3.normal() for _ in range(20)]
    assert np.allclose(R.normal_array(5, 20), nor, atol=1e-12)


def test_child_seeds_are_distinct_and_deterministic():
    seeds = {R.child_seed(35, i) for i in range(200)}
    assert len(seeds) == 200 and R.child_seed(35, 3) == R.child_seed(35, 3) and R.child_seed(35, 3) != R.child_seed(36, 3)


def test_geometric_failures_follow_q_to_the_g():
    u = R.uniform_array(11, 400000)
    p = 0.3
    g = R.geometric_failures(u, p)
    assert abs(g.mean() - (1 - p) / p) < 0.03
    for t in (1, 2, 5):
        assert abs((g >= t).mean() - (1 - p) ** t) < 0.005
    assert (R.geometric_failures(u[:100], 1.0) == 0).all()


def test_lead_times_have_the_chain_mean_and_respect_cap_and_minimum():
    for k, p in ((1, 0.25), (2, 2 / 3), (3, 0.3)):
        ell = K.draw_lead_times(5, k, p, (60000,), 10 ** 6)
        sd = math.sqrt(k * (1 - p)) / p
        assert ell.min() >= k and abs(ell.mean() - k / p) < 5 * sd / math.sqrt(60000)     # 5 Standardfehler
    capped = K.draw_lead_times(5, 1, 0.05, (20000,), 25)
    assert capped.max() == 25 and capped.min() >= 1
    assert (K.draw_lead_times(5, 4, 1.0, (50,), 99) == 4).all()                 # fest: genau k


# --- Randfälle --------------------------------------------------------------------------------------------------------------------------


def test_limit_as_a_band_never_exact_equality():
    for k, p in ((2, 2 / 3), (3, 1.0), (1, 1 / 3), (3, 3 / 5)):
        a = K.critical_alpha_chain(k, p)
        assert K.spectral_radius(K.chain_matrix(k, p, a * (1 - 1e-6), 0)[0]) < 1 + 1e-6
        assert K.spectral_radius(K.chain_matrix(k, p, a * (1 + 1e-3), 0)[0]) > 1
        assert K.spectral_radius(K.chain_matrix(k, p, a * (1 - 1e-3), 0)[0]) < 1


def test_beta_one_limit_is_two_for_every_chain_and_other_beta_is_rejected():
    for k, p in ((1, 0.25), (2, 0.4), (3, 1.0)):
        assert K.critical_alpha_chain(k, p, 1) == 2.0
    with pytest.raises(ValueError):
        K.critical_alpha_chain(2, 0.4, 0.3)
    with pytest.raises(ValueError):
        K.charpoly_chain(2, 0.4, 0.5, 0.3)


def test_sigma_zero_gives_zero_variance_for_the_fixed_delay_and_positive_for_spread():
    ex_fixed = K.ms_stationary(3, 1.0, 0.4, 0, 10.0, 0.0)
    ex_spread = K.ms_stationary(2, 2 / 3, 0.4, 0, 10.0, 0.0)
    assert ex_fixed == (0.0, 0.0) or max(abs(ex_fixed[0]), abs(ex_fixed[1])) < 1e-9
    assert ex_spread[0] > 100.0
    I, U = K.simulate_individual(3, 1.0, 0.4, 0, 10.0, 0.0, 20, 200, 3)
    assert np.abs(I).max() < 1e-9 and np.abs(U).max() < 1e-9                     # feste Lieferzeit, konstanter Bedarf: nichts schwankt


def test_individual_order_is_deterministic_per_seed_and_scales_with_demand():
    a, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 10.0, 1.0, 30, 100, 35)
    b, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 10.0, 1.0, 30, 100, 35)
    c, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 10.0, 1.0, 30, 100, 36)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    s1, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 1.0, 0.0, 30, 100, 35)
    s100, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 100.0, 0.0, 30, 100, 35)
    assert np.allclose(s100, 100.0 * s1, rtol=1e-9, atol=1e-9)                   # linear in D: gleiche Lieferzeiten, nur andere Skala


def test_lead_time_cap_formula():
    assert K.lead_time_cap(2) == 60 and K.lead_time_cap(1) == 40 and K.lead_time_cap(24) == 720 and K.lead_time_cap(60) == 720
