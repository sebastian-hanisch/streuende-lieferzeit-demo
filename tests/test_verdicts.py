"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen statt Seeds."""

import slz_evaluation as ev


def test_number_formatting_uses_decimal_commas():
    assert ev.de(1.5, 2) == "1,50" and ev.de(-0.05, 4, True) == "-0,0500" and ev.de(0.9476, 3) == "0,948"
    assert ev.pct(0.512, 0) == "51 %" and ev.pct(0.6601, 0) == "66 %" and ev.pct(0.5076, 1) == "50,8 %"
    assert ev.fmt_int(3193) == "3 193" and ev.fmt_int(1000) == "1 000" and ev.fmt_int(46) == "46"
    assert ev.sig(652.0159) == "652" and ev.sig(13.444) == "13,4" and ev.sig(0.1234) == "0,123" and ev.sig(0) == "0" and ev.sig(46.0) == "46,0"


def _model(name, k, rho, alpha_krit=1.0):
    return {"name": name, "k": k, "p": 0.5, "rho": rho, "alpha_krit": alpha_krit}


def _hand(rf, rc, r1, tau=3, k=2, alpha=0.7, beta=0):
    return {"tau": tau, "k": k, "alpha": alpha, "beta": beta, "fixed": _model("fest", tau, rf), "chosen": _model(f"k = {k}", k, rc), "one": _model("k = 1", 1, r1)}


def test_verdict_hand_fixed_unstable_spread_stable_is_success():
    kind, text = ev.verdict_hand(_hand(1.037, 0.948, 0.8165))
    assert kind == "success"
    assert "Mittlere Lieferzeit 3, α = 0,70" in text and "instabil (ρ = 1,037, der Bestand wächst um 3,7 % je Periode)" in text
    assert "mit k = 2 ist stabil (ρ = 0,948)" in text and "k = 1: ρ = 0,8165" in text


def test_verdict_hand_threshold_is_rho_one_exactly():
    assert ev.verdict_hand(_hand(1.0, 0.9, 0.8))[0] == "success"                 # ρ = 1 zählt als instabil
    assert ev.verdict_hand(_hand(0.9999, 0.9, 0.8))[0] == "info"
    assert ev.verdict_hand(_hand(1.2, 1.0, 0.8))[0] == "warning"                   # beide nicht stabil


def test_verdict_hand_all_branches():
    kind, text = ev.verdict_hand(_hand(0.88, 0.85, 0.82))
    assert kind == "info" and "Beide stabil, fest ρ = 0,8800, k = 2 ρ = 0,8500" in text
    kind, text = ev.verdict_hand(_hand(1.05, 1.03, 0.9))
    assert kind == "warning" and "Beide instabil, fest ρ = 1,050, k = 2 ρ = 1,030" in text
    kind, text = ev.verdict_hand(_hand(0.9, 1.1, 0.8))
    assert kind == "warning" and "instabiler als die feste" in text and "ρ = 1,100" in text and "ρ = 0,900" in text
    kind, text = ev.verdict_hand(_hand(1.02, 1.02, 0.9, tau=3, k=3))
    assert kind == "info" and "Mit k = τ ist die Lieferzeit fest" in text and "instabil" in text and "k = 1 (geometrisch) ρ = 0,9000" in text


def test_verdict_hand_beta_one_names_the_identity():
    kind, text = ev.verdict_hand(_hand(0.45, 0.45, 0.667, alpha=0.55, beta=1))
    assert kind == "info" and "ρ = max(|1 − α|, q)" in text and "α = 0,55" in text


def test_verdict_limit_names_51_66_70_percent_and_the_geometric_limit():
    kind, text = ev.verdict_limit({"tau": 12})
    assert kind == "success"
    assert "bei τ = 12 um 51 %" in text and "bei τ = 60 um 66 %" in text and "stetig (τ → ∞) sind es 70 %" in text
    assert "α_krit = 4τ − 2 = 46" in text and "z = −1" in text
    assert "α_krit = 4τ − 2 = 238" in ev.verdict_limit({"tau": 60})[1]


def _forms(ratios):
    rows = [{"name": "fest", "ratio": 1.0}] + [{"name": f"Form {j}", "ratio": r} for j, r in enumerate(ratios, start=1)]
    return {"tau": 5, "fixed": 0.347, "rows": rows}


_FROZEN = {"seed": 20261115, "n": 1000, "rows": {"4": {"above": 999, "below": 0, "ratio_median": 1.3285, "returns": 4}}}


def test_verdict_forms_names_range_and_the_counts():
    kind, text = ev.verdict_forms(_forms([1.13, 1.31, 1.66]), _FROZEN)
    assert kind == "info"
    assert "Mittel 5" in text and "festen Lieferzeit bei 0,347" in text and "dem 1,13fachen (Form 1)" in text and "dem 1,66fachen (Form 3)" in text
    assert "1 000 Zufallsverteilungen, Zufalls-Seed 20261115" in text and "in 999 von 1 000 über der festen (Median 1,33fach)" in text and "unter ihr in 0" in text
    assert "in 4 von 1 000 kehrt die Stabilität" in text


def _noise_row(name, rho, vi, vu):
    return {"name": name, "k": 1, "p": 0.5, "rho": rho, "stable": rho < 1 - 1e-9, "var_i": vi, "var_u": vu}


def _noise(rows, chosen, alpha=0.4, beta=0, tau=3, k=2):
    return {"tau": tau, "k": k, "alpha": alpha, "beta": beta, "rows": rows, "chosen": chosen, "sim": None}


def test_verdict_noise_smaller_variance_for_the_spread_delay():
    r1, r2, rf = _noise_row("k = 1", 0.8, 2.552, 0.408), _noise_row("k = 2", 0.85, 3.4145, 0.546), _noise_row("fest", 0.88, 4.403, 0.705)
    kind, text = ev.verdict_noise(_noise([r1, r2, rf], r2))
    assert kind == "info" and "τ = 3, α = 0,40, β = 0" in text and "k = 1: 2,55, k = 2: 3,41, fest: 4,40" in text
    assert "kleiner (Bestellvarianz 0,55 gegen 0,70 bei fester Lieferzeit)" in text


def test_verdict_noise_not_smaller_unstable_and_beta_one():
    r2, rf = _noise_row("k = 2", 0.85, 5.0, 0.8), _noise_row("fest", 0.88, 4.0, 0.7)
    kind, text = ev.verdict_noise(_noise([r2, rf], r2))
    assert kind == "warning" and "nicht kleiner (Bestellvarianz 0,80 gegen 0,70)" in text
    ru = _noise_row("k = 2", 1.2, None, None)
    kind, text = ev.verdict_noise(_noise([ru, rf], ru))
    assert kind == "warning" and "(k = 2) ist hier instabil (ρ = 1,200)" in text
    rfu = _noise_row("fest", 1.05, None, None)
    r2s = _noise_row("k = 2", 0.9, 6.1, 0.9)
    kind, text = ev.verdict_noise(_noise([r2s, rfu], r2s))
    assert kind == "success" and "feste Lieferzeit ist hier instabil" in text
    b1 = [_noise_row("k = 2", 0.45, 2.912, 0.3793), _noise_row("fest", 0.45, 3.254, 0.3793)]
    kind, text = ev.verdict_noise(_noise(b1, b1[0], alpha=0.55, beta=1))
    assert kind == "info" and "α/(2 − α) = 0,379" in text and "k = 2: 2,91, fest: 3,25" in text


def _single(alpha, am=0.306, a2=0.238, beta=0, ms=True):
    return {"tau": 12, "k": 2, "alpha": alpha, "beta": beta, "D": 10.0, "sigma": 1.0, "ms_available": ms, "alpha_mean": am, "alpha_ms": a2,
            "exact": {"var_i": 652.016, "var_u": 6.52}, "fluid": {"var_i": 13.444, "var_u": 0.134}}


def test_verdict_single_below_between_above_and_unavailable():
    kind, text = ev.verdict_single(_single(0.1))
    assert kind == "info" and "Mittelwert stabil bis α = 0,306, zweites Moment nur bis α = 0,238; α liegt darunter" in text
    assert "Bedarf 10 und Rauschen 1,0" in text and "Einzelbestellung 652 gegen 13,4 im anteiligen Modell" in text
    assert ev.verdict_single(_single(0.238))[0] == "warning"                       # genau auf der Grenze des zweiten Moments: nicht mehr darunter
    kind, text = ev.verdict_single(_single(0.27))
    assert kind == "warning" and "zwischen den Grenzen" in text and "zweite Moment divergiert" in text
    kind, text = ev.verdict_single(_single(0.4))
    assert kind == "warning" and "über beiden Grenzen" in text
    kind, text = ev.verdict_single(_single(0.4, ms=False))
    assert kind == "info" and "k > 8" in text and "nicht gerechnet" in text
    assert "ebenfalls bis α = 2,00" in ev.verdict_single(_single(0.55, am=2.0, a2=2.0, beta=1))[1]


def _decay_row(label, k, rho0, d0, rho1=0.5, d1=20, best=0.9):
    return {"label": label, "k": k, "rho0": rho0, "decay0": d0, "rho1": rho1, "decay1": d1, "best_rho": best, "best_alpha": 0.03}


def _decay(rows, tau=12, alpha=0.2):
    return {"tau": tau, "alpha": alpha, "rows": rows}


def test_verdict_decay_slow_stable_form_is_a_warning_with_the_3193_periods():
    rows = [_decay_row("k = 1", 1, 0.9574, 106, best=0.9574), _decay_row("k = 2", 2, 0.9854, 300, best=0.95), _decay_row("k = 3", 3, 0.9986, 3193, best=0.95),
            _decay_row("k = 6", 6, 1.0147, None, best=0.95), _decay_row("fest", 12, 1.0241, None, best=0.9167)]
    kind, text = ev.verdict_decay(_decay(rows), 0)
    assert kind == "warning"
    assert "k = 1: 106 Perioden (ρ = 0,9574); k = 2: 300 Perioden (ρ = 0,9854); k = 3: 3 193 Perioden (ρ = 0,9986)" in text
    assert "Instabil: k = 6 (ρ = 1,015), fest (ρ = 1,024)" in text
    assert "fest mit 0,9167, der kleinste aller gerechneten Formen" in text


def test_verdict_decay_fast_is_info_and_threshold_is_ten_times():
    rows = [_decay_row("k = 1", 1, 0.8, 20), _decay_row("fest", 3, 0.9, 200, best=0.8)]
    assert ev.verdict_decay(_decay(rows), 0)[0] == "info"                           # genau das Zehnfache: noch nicht langsam
    rows[1]["decay0"] = 201
    assert ev.verdict_decay(_decay(rows), 0)[0] == "warning"
    rows = [_decay_row("k = 1", 1, 0.8, 20, best=0.7), _decay_row("fest", 3, 0.9, 25, best=0.8)]
    kind, text = ev.verdict_decay(_decay(rows), 0)
    assert kind == "info" and "k = 1 mit 0,7000 (fest: 0,8000)" in text


def test_verdict_decay_not_decayed_all_unstable_and_beta_one():
    rows = [_decay_row("k = 1", 1, 0.99999, None), _decay_row("fest", 3, 0.9, 30)]
    kind, text = ev.verdict_decay(_decay(rows), 0)
    assert kind == "warning" and "k = 1: nicht abgeklungen (ρ = 0,99999)" in text
    assert ev.verdict_decay(_decay([_decay_row("fest", 3, 1.2, None)]), 0) == ("warning", "τ = 12, α = 0,20, β = 0: Alle gerechneten Formen sind instabil.")
    rows = [_decay_row("k = 1", 1, 0.8, 20, rho1=0.917, d1=56), _decay_row("fest", 12, 0.9, 30, rho1=0.7, d1=24)]
    kind, text = ev.verdict_decay(_decay(rows), 1)
    assert kind == "info" and "β = 1" in text and "k = 1: 56 Perioden (ρ = 0,9170); fest: 24 Perioden (ρ = 0,7000)" in text and "Kleinster Spektralradius" not in text
