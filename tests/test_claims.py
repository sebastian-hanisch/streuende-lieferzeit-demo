"""Claims-Tests: jede Zahl der README steht hier als Band oder exakt (Spalte „Test“ der Befund-Tabelle nennt die Funktion).

Messbedingungen überall: Zeit in Perioden, linear, ohne Klemmung; anteilig (Erwartungswert-Modell), wo nicht „Einzelbestellung“ steht; β = 0, wo nichts anderes steht.
Rechnungen (Grenze, Lyapunov, Spektralradius) prüfen enge Bänder, Simulationen und Zählungen auf Rasterpunkten Bänder, die Plattformunterschiede der Fließkomma-Rechnung
(BLAS, libm) aushalten; die Abklingzeit ist eine Zählung und darf um 2 Perioden abweichen. Kein Parameter ist eine Zweierpotenz außer der festen Lieferzeit (p = 1)."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

import slz_constants as C
import slz_dynamics as K
import slz_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


def _near(value, expected, tol):
    assert abs(value - expected) <= tol, (value, expected, tol)


def _lim(tau, k):
    return K.critical_alpha_chain(k, k / tau)


# --- B1: Grenze bei gleichem Mittel -------------------------------------------------------------------------------------------------------


def test_limits_at_tau_12_and_60():
    fixed12, fixed60 = K.critical_alpha_fixed(12), K.critical_alpha_fixed(60)
    for k, expected in ((12, 0.1365), (6, 0.1541), (4, 0.1767), (3, 0.2065), (2, 0.3056)):
        _near(_lim(12, k), expected, 6e-5)
    _near(_lim(12, 1), 46.0, 1e-6)
    _near(_lim(12, 2) / fixed12, 2.24, 0.005)
    for k, expected in ((60, 0.0264), (10, 0.0295), (3, 0.0438), (2, 0.0656)):
        _near(_lim(60, k), expected, 6e-5)
    _near(_lim(60, 2) / fixed60, 2.48, 0.005)
    _near(_lim(60, 1), 238.0, 1e-6)                           # 4 tau - 2


def test_limit_formulas_k1_and_k2_in_eleven_cases():
    for tau in (2, 3, 5, 7, 12, 24, 60, 120, 240, 600, 1200):
        assert _lim(tau, 1) == pytest.approx(4 * tau - 2, rel=1e-9)
        assert _lim(tau, 2) == pytest.approx(4 * (tau - 1) / tau ** 2, rel=1e-9)


# --- B2: Annäherung an den stetigen Grenzfall ------------------------------------------------------------------------------------------------


def test_limit_times_tau_approaches_the_continuous_values():
    ks = (2, 3, 4, 6, 10)
    for tau, expected in ((12, (3.667, 2.478, 2.121, 1.850, 1.676)), (1200, (3.9967, 2.6648, 2.2727, 1.9782, 1.7916))):
        for k, e in zip(ks, expected):
            _near(_lim(tau, k) * tau, e, 6e-4 if tau == 12 else 6e-5)
    for k, e in zip(ks, (4.0000, 2.6667, 2.2742, 1.9794, 1.7927)):
        _near(K.critical_continuous(k), e, 6e-5)
    _near(K.critical_alpha_fixed(12) * 12, 1.638, 6e-4)
    _near(K.critical_alpha_fixed(60) * 60, 1.584, 6e-4)
    _near(math.pi / 2, 1.5708, 6e-5)


# --- B3: Verzögerung dritter Ordnung gegen fest ------------------------------------------------------------------------------------------


def test_third_order_against_fixed_ratios_and_percentages():
    for tau, expected in ((6, 1.335), (12, 1.513), (24, 1.604), (60, 1.660), (120, 1.679)):
        _near(E.k3_ratio(tau), expected, 6e-4)
    _near((8 / 3) / (math.pi / 2), 1.698, 6e-4)
    assert round(100 * (E.k3_ratio(12) - 1)) == 51 and round(100 * (E.k3_ratio(60) - 1)) == 66 and round(100 * ((8 / 3) / (math.pi / 2) - 1)) == 70


# --- B4: andere Formen ----------------------------------------------------------------------------------------------------------------------


def test_other_shapes_at_mean_4():
    fixed = K.critical_alpha_fixed(4)
    _near(fixed, 0.4450, 6e-5)
    for f, expected in ((K.two_point_pmf(4, 1), 0.4940), (K.two_point_pmf(4, 2), 0.7138), (K.two_point_pmf(4, 3), 2.0000),
                        (K.uniform_pmf(4, 1), 0.4765), (K.uniform_pmf(4, 2), 0.5496), (K.uniform_pmf(4, 3), 0.6932)):
        _near(K.critical_alpha_dist(f), expected, 6e-5)
    assert K.mean_var(K.two_point_pmf(4, 3))[1] == pytest.approx(9.0)


@pytest.fixture(scope="module")
def frozen():
    return E.load_frozen()


@pytest.fixture(scope="module")
def batch4():
    return E.random_batch(C.FROZEN_SEED, 4, C.FROZEN_N)


def test_random_distributions_mean_4_recomputed_equal_the_frozen_row(frozen, batch4):
    row = frozen["rows"]["4"]
    assert frozen["seed"] == C.FROZEN_SEED and frozen["n"] == C.FROZEN_N == batch4["n"] == 1000
    for key in ("above", "below", "equal", "no_crossing"):
        assert batch4[key] == row[key], key
    _near(batch4["ratio_median"], row["ratio_median"], 1e-6)
    _near(batch4["ratio_min"], row["ratio_min"], 1e-6)
    assert abs(batch4["returns"] - row["returns"]) <= 1


def test_random_distributions_mean_4_counts_and_the_exception(batch4):
    """999 von 1000 über der festen Grenze, 1 innerhalb 1e-6 (fast fest), 0 darunter; Median 1,33; Wiederkehr in 4 (Band 3 bis 5)."""
    assert (batch4["above"], batch4["below"], batch4["equal"]) == (999, 0, 1)
    assert batch4["no_crossing"] == 0
    _near(batch4["ratio_median"], 1.33, 0.005)
    _near(batch4["ratio_max"], 9.15, 0.01)
    assert 3 <= batch4["returns"] <= 5
    assert {k: round(v, 2) for k, v in batch4["median_by_form"].items()} == {"glatt": 1.39, "gemischt": 1.38, "spitz": 1.22}


def test_the_returning_distributions_have_two_crossings_and_stable_raster_points_above_the_first_limit(frozen):
    for i in frozen["rows"]["4"]["return_indices"]:
        f = E.frozen_pmf(C.FROZEN_SEED, 4, i)
        crossings = K.crossings_dist(f)
        assert len([c for c in crossings if c <= 2.61]) >= 2
        first, back = K.stability_returns(f)
        assert back >= 1 and first == crossings[0]


@pytest.mark.parametrize("mean", [3, 5, 6])
def test_random_distributions_other_means_partial_recomputation(frozen, mean):
    """Gegenprobe mit anderem Mittel: die vorgerechnete Reihe (1000 je Mittel) ist in sich stimmig, die ersten 250 Verteilungen werden neu gerechnet: keine unter der festen Grenze."""
    row = frozen["rows"][str(mean)]
    assert row["above"] + row["below"] + row["equal"] == 1000 and row["below"] == 0 and row["returns"] == 0
    part = E.random_batch(C.FROZEN_SEED, mean, 250)
    assert part["below"] == 0 and part["ratio_min"] >= 1 - 1e-9 and part["returns"] == 0
    assert part["above"] + part["equal"] == 250
    _near(part["ratio_median"], row["ratio_median"], 0.08)       # Teilstichprobe: Band, nicht die Zahl der Reihe
    assert round(row["ratio_median"], 2) == {3: 1.53, 5: 1.22, 6: 1.17}[mean]


# --- B5: Schwankung (anteilig) ---------------------------------------------------------------------------------------------------------------


def _var(tau, k, alpha, beta):
    p = k / tau
    A, c = K.chain_matrix(k, p, alpha, beta)
    return K.variances(A, c)


def test_noise_variances_tau_3_alpha_04_beta0():
    for k, vi, vu in ((1, 2.552, 0.408), (2, 3.415, 0.546), (3, 4.403, 0.705)):
        got_u, got_i = _var(3, k, 0.4, 0)
        _near(got_i, vi, 6e-4)
        _near(got_u, vu, 6e-4)


def test_noise_variances_tau_12_alpha_013_beta0():
    for k, vi in ((1, 9.61), (2, 13.70), (3, 18.07), (6, 34.9), (12, 103.1)):
        _near(_var(12, k, 0.13, 0)[1], vi, 0.06)


def test_noise_variances_beta1_order_variance_is_distribution_free():
    for k, vi in ((1, 2.409), (2, 2.912), (3, 3.254)):
        vu, got = _var(3, k, 0.55, 1)
        _near(vu, 0.55 / 1.45, 1e-9)
        _near(got, vi, 6e-4)
    _near(0.55 / 1.45, 0.379, 6e-4)
    for k, vi in ((1, 8.32), (2, 9.92), (3, 10.72), (6, 11.90), (12, 12.96)):
        vu, got = _var(12, k, 0.3, 1)
        _near(vu, 0.3 / 1.7, 1e-9)
        _near(got, vi, 0.006)


def test_smaller_bestand_variance_in_every_stable_cell_of_the_whole_alpha_raster():
    """Zählung (DEMO-PLAYBOOK §11a Punkt 1): β = 0, α von 0,01 bis 0,99 α_krit(fest) im Schritt 0,01, jedes k unter τ gegen die feste Lieferzeit; τ = 3, 4, 6, 12.
    Ergebnis: var(i) und var(u) der streuenden Lieferzeit sind in allen 403 Zellen kleiner (Gilt nur im anteiligen Modell)."""
    total = smaller_i = smaller_u = 0
    for tau, ks in ((3, (1, 2)), (4, (1, 2, 3)), (6, (1, 2, 3)), (12, (1, 2, 3, 4, 6))):
        top = 0.99 * K.critical_alpha_fixed(tau)
        for a in np.arange(0.01, top + 1e-12, 0.01):
            uf, vf = _var(tau, tau, float(a), 0)
            for k in ks:
                A, _ = K.chain_matrix(k, k / tau, float(a), 0)
                if K.spectral_radius(A) >= 1 - 1e-9:
                    continue
                u, v = _var(tau, k, float(a), 0)
                total += 1
                smaller_i += v < vf
                smaller_u += u < uf
    assert total == 403 and smaller_i == total and smaller_u == total


def test_beta_one_bestand_variance_smaller_and_order_variance_equal_in_every_cell():
    """β = 1, α von 0,01 bis 1,99 im Schritt 0,01, τ = 3 und 12: var(i) kleiner in allen Zellen, var(u) gleich (Abweichung höchstens 1e-9)."""
    total = smaller_i = 0
    worst = 0.0
    for tau, ks in ((3, (1, 2)), (12, (1, 2, 3, 6))):
        for a in np.arange(0.01, 1.995, 0.01):
            uf, vf = _var(tau, tau, float(a), 1)
            for k in ks:
                u, v = _var(tau, k, float(a), 1)
                total += 1
                smaller_i += v < vf
                worst = max(worst, abs(u - uf))
    assert smaller_i == total and total == 199 * 2 + 199 * 4 and worst < 1e-9


# --- B6: Einzelbestellung --------------------------------------------------------------------------------------------------------------------


def test_single_order_variance_at_constant_demand_sigma_zero():
    """σ = 0, D = 10, τ = 3, β = 0: var(i)/D² exakt (zweites Moment); die feste Lieferzeit gibt 0."""
    for alpha, expected in ((0.2, (1.288, 0.773, 0.0)), (0.4, (1.5625, 1.1997, 0.0))):
        for k, e in zip((1, 2, 3), expected):
            ex = K.ms_stationary(k, k / 3, alpha, 0, 10.0, 0.0)
            _near(ex[0] / 100.0, e, 6e-4)
    vals = [K.ms_stationary(2, 2 / 3, 0.4, 0, D, 0.0)[0] / D ** 2 for D in (1.0, 10.0, 100.0)]
    _near(max(vals) - min(vals), 0.0, 1e-9)
    _near(vals[0], 1.1997, 6e-5)


def test_second_moment_limits_are_below_the_mean_limits():
    expected = {(3, 1): (0.872, 10.0), (3, 2): (0.678, 0.889), (3, 3): (0.618, 0.618), (12, 1): (0.416, 46.0), (12, 2): (0.238, 0.306), (12, 3): (0.188, 0.206),
                (12, 6): (0.151, 0.154), (24, 1): (0.291, 94.0)}
    for (tau, k), (ms, mean) in expected.items():
        _near(K.ms_critical(k, k / tau), ms, 6e-4)
        _near(_lim(tau, k), mean, 6e-4 if mean < 5 else 1e-6)
        assert K.ms_critical(k, k / tau) <= _lim(tau, k) + 1e-9


def test_single_order_with_noise_numbers():
    """σ = 1, D = 10, β = 0: τ = 3, α = 0,4 und τ = 12, k = 2, α = 0,1; dazu der Faktor gegen den anteiligen Wert (36 bis 62)."""
    exact = {k: K.ms_stationary(k, k / 3, 0.4, 0, 10.0, 1.0)[0] for k in (1, 2, 3)}
    for k, e in ((1, 159.4), (2, 124.0), (3, 4.40)):
        _near(exact[k], e, 0.06)
    fluid = {k: _var(3, k, 0.4, 0)[1] for k in (1, 2, 3)}
    _near(exact[3], fluid[3], 1e-9)                                      # fest: Einzelbestellung = anteilig
    assert round(exact[1] / fluid[1]) == 62 and round(exact[2] / fluid[2]) == 36
    _near(K.ms_stationary(2, 1 / 6, 0.1, 0, 10.0, 1.0)[0], 652.0, 0.1)
    _near(_var(12, 2, 0.1, 0)[1], 13.4, 0.05)


def test_single_order_variance_larger_than_fixed_in_every_cell():
    """Zählung: σ = 1, D = 10, β = 0, α von 0,01 bis 0,99 α_krit(fest), τ = 3 (k = 1, 2) und τ = 12 (k = 2, 3, 6): die Bestandsvarianz der Einzelbestellung liegt in allen 161 Zellen über der festen."""
    total = bigger = 0
    for tau, ks in ((3, (1, 2)), (12, (2, 3, 6))):
        for a in np.arange(0.01, 0.99 * K.critical_alpha_fixed(tau), 0.01):
            vf = K.ms_stationary(tau, 1.0, float(a), 0, 10.0, 1.0)
            for k in ks:
                ex = K.ms_stationary(k, k / tau, float(a), 0, 10.0, 1.0)
                if ex is None or vf is None:
                    continue
                total += 1
                bigger += ex[0] > vf[0]
    assert (bigger, total) == (161, 161)


def test_second_moment_limit_with_leitung_is_two_in_all_checked_pairs():
    """β = 1: Grenze des zweiten Moments 2,000 in allen 15 geprüften (τ, k) mit k ≤ 8 (Moment-System der App); gleich der Grenze des Mittelwerts."""
    pairs = [(3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 4), (12, 1), (12, 2), (12, 3), (12, 4), (12, 6), (24, 1), (24, 2), (24, 4), (24, 8)]
    for tau, k in pairs:
        _near(K.ms_critical(k, k / tau, 1), 2.0, 1e-6)
    assert len(pairs) == 15


# --- B8: Abklingzeit ---------------------------------------------------------------------------------------------------------------------------


def _decay(tau, k, alpha, beta):
    A, c = K.chain_matrix(k, k / tau, alpha, beta)
    return K.spectral_radius(A), K.decay_time(A, c).time


def test_decay_time_tau_12_alpha_02_beta0():
    for k, rho, t in ((1, 0.9574, 106), (2, 0.9854, 300), (3, 0.9986, 3193)):
        r, got = _decay(12, k, 0.2, 0)
        _near(r, rho, 6e-5)
        assert abs(got - t) <= 2, (k, got)
    for k, rho in ((6, 1.0147), (12, 1.0241)):
        r, got = _decay(12, k, 0.2, 0)
        _near(r, rho, 6e-5)
        assert got is None


def test_decay_time_tau_3_alpha_06_beta0():
    for k, rho, t in ((3, 0.9914, 542), (2, 0.9176, 51), (1, 0.8165, 23)):
        r, got = _decay(3, k, 0.6, 0)
        _near(r, rho, 6e-5)
        assert abs(got - t) <= 2


def test_decay_time_with_leitung_tau_12_alpha_03():
    for k, rho, t in ((1, 0.917, 56), (2, 0.833, 41), (3, 0.75, 34), (6, 0.70, 28), (12, 0.70, 24)):
        r, got = _decay(12, k, 0.3, 1)
        _near(r, rho, 5e-3)
        assert abs(got - t) <= 2


def test_smallest_spectral_radius_over_alpha():
    alphas = np.arange(0.01, 2.0001, 0.01)
    for tau, expected in ((3, {3: 0.6667, 2: 0.7287, 1: 0.8165}), (12, {12: 0.9167, 6: 0.9236, 3: 0.9343, 2: 0.9422, 1: 0.9574})):
        for k, e in expected.items():
            _near(K.best_rho(k, k / tau, 0, alphas)[0], e, 6e-5)
    _near(K.best_rho(1, 1 / 3, 0, alphas)[0], math.sqrt(2 / 3), 1e-9)                # k = 1: ρ = √q
    for k in (1, 2, 3, 6):
        assert K.best_rho(k, k / 12, 0, alphas)[0] > K.best_rho(12, 1.0, 0, alphas)[0]


def test_k1_spectral_radius_is_sqrt_q_on_a_wide_alpha_range():
    """k = 1: ρ = √q = √(1 − 1/τ) unabhängig von α, gezählt über das Raster: τ = 3 für α von 0,2 bis 1,9 (Schritt 0,1), τ = 12 für α von 0,05 bis 0,3 (Schritt 0,01)."""
    for tau, grid in ((3, np.arange(0.2, 1.9001, 0.1)), (12, np.arange(0.05, 0.3001, 0.01))):
        hits = [abs(K.spectral_radius(K.chain_matrix(1, 1 / tau, float(a), 0)[0]) - math.sqrt(1 - 1 / tau)) < 1e-9 for a in grid]
        assert all(hits) and len(hits) == (18 if tau == 3 else 26), (tau, sum(hits), len(hits))


def test_hand_example_alpha_07():
    for k, rho in ((3, 1.0369), (2, 0.9476), (1, 0.8165)):
        _near(_decay(3, k, 0.7, 0)[0], rho, 6e-5)
    _near(K.spectral_radius(K.chain_matrix(3, 1.0, 0.8, 0)[0]), 1.0780, 6e-5)
    _near(K.spectral_radius(K.chain_matrix(2, 2 / 3, 0.8, 0)[0]), 0.9760, 6e-5)
    _near(K.critical_alpha_chain(2, 2 / 3), 0.8889, 6e-5)
    _near(K.critical_alpha_fixed(3), 0.6180, 6e-5)


# --- App und README: gleiche Texte ----------------------------------------------------------------------------------------------------------


def test_assumption_table_rows_of_the_app_are_in_the_readme():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = [line for line in app.splitlines() if line.startswith("| **")]
    assert len(rows) == 7
    for row in rows:
        assert row in readme, row


def test_readme_numbers_of_the_summary_line_match_the_constants():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Zufalls-Seed {C.FROZEN_SEED}" in readme and "1 000 Zufallsverteilungen" in readme
    assert re.search(r"\b" + str(C.NOISE_RUNS) + r" Läufe", readme)
