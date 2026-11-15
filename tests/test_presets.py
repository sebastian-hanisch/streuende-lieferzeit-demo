"""Presets: Vollständigkeit, gültige Werte, keine Zweierpotenz, Standard-Preset = Standardeinstellungen, und je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""

import math

import pytest

import slz_constants as C
import slz_dynamics as K
import slz_evaluation as ev
import slz_presets as P


def _settings(preset: dict) -> ev.Settings:
    return ev.Settings(**{("spread" if key == "dist_spread" else key): value for key, value in preset.items()})


def _is_power_of_two(x: float) -> bool:
    if x <= 0:
        return False
    m, _ = math.frexp(x)
    return m == 0.5


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert {"tau", "k", "alpha", "beta"} <= set(preset), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_no_preset_uses_a_power_of_two_for_alpha_or_the_derived_stage_probability():
    """Linie Rückkopplung V10/V11: kein Regler-Wert als Zweierpotenz. α und die abgeleitete Fortschrittswahrscheinlichkeit p = k/τ der Presets sind keine; β ∈ {0, 1} ist die Modellwahl."""
    for name, preset in C.PRESETS.items():
        assert not _is_power_of_two(preset["alpha"]), name
        _, p = K.stages(preset["tau"], preset["k"])
        assert not _is_power_of_two(p), (name, p)
        assert preset["beta"] in (0, 1)


def test_presets_are_distinct_settings():
    keyed = {tuple(sorted(_settings(p).__dict__.items())) for p in C.PRESETS.values()}
    assert len(keyed) == len(C.PRESETS)


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Von Hand (k = 2, τ = 3)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Von Hand (k = 2, τ = 3)"


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_limits_values_and_options():
    spec = P.SETTING_SPECS["alpha_slider"]
    assert P.limit_value(spec, 9.0) == C.ALPHA_MAX and P.limit_value(spec, -1.0) == C.ALPHA_MIN
    spec = P.SETTING_SPECS["tau_map_select"]
    assert P.limit_value(spec, 60) == 60 and P.limit_value(spec, 7) == C.DEFAULT_TAU_MAP
    spec = P.SETTING_SPECS["random_form_select"]
    assert P.limit_value(spec, "spitz") == "spitz" and P.limit_value(spec, "x") == C.DEFAULT_RANDOM_FORM


# --- je Preset die erwartete Verdikt-Zeile -----------------------------------------------------------------------------------------------


def _hand(name):
    s = _settings(C.PRESETS[name])
    return ev.verdict_hand(ev.hand_analysis(s.tau, s.k, s.alpha, s.beta))


def _noise(name):
    s = _settings(C.PRESETS[name])
    return ev.verdict_noise(ev.noise_analysis(s.tau, s.k, s.alpha, s.beta, s.seed))


def _single(name):
    s = _settings(C.PRESETS[name])
    return ev.verdict_single(ev.single_analysis(s.tau, s.k, s.alpha, s.beta, s.d, s.sigma, s.runs, s.seed))


def _decay(name):
    s = _settings(C.PRESETS[name])
    return ev.verdict_decay(ev.decay_analysis(s.tau, s.alpha, s.k), s.beta)


def test_von_hand_shows_fixed_unstable_and_spread_stable():
    kind, text = _hand("Von Hand (k = 2, τ = 3)")
    assert kind == "success" and "instabil (ρ = 1,037" in text and "mit k = 2 ist stabil (ρ = 0,948)" in text and "k = 1: ρ = 0,8165" in text


def test_zwoelf_perioden_shows_slow_decay_with_3193_periods():
    assert _hand("Zwölf Perioden")[0] == "success"
    kind, text = _decay("Zwölf Perioden")
    assert kind == "warning" and "k = 1: 106 Perioden" in text and "k = 2: 300 Perioden" in text and "k = 3: 3 193 Perioden" in text and "Instabil: k = 4" in text and "fest (ρ = 1,024)" in text


def test_dritte_ordnung_is_stable_while_the_fixed_delay_is_not():
    kind, text = _hand("Dritte Ordnung")
    assert kind == "success" and "Mittlere Lieferzeit 20" in text and "mit k = 3 ist stabil" in text
    s = _settings(C.PRESETS["Dritte Ordnung"])
    assert K.critical_alpha_fixed(20) < s.alpha < K.critical_alpha_chain(3, 3 / 20)
    assert s.tau_map == 60


def test_andere_verteilungen_preset_compares_the_three_shapes_at_mean_5():
    s = _settings(C.PRESETS["Andere Verteilungen"])
    fa = ev.forms_analysis(s.tau, s.spread, s.random_form, s.seed)
    assert [r["name"] for r in fa["rows"][:3]] == ["fest", "Zweipunkt 3 / 7", "gleichverteilt 3 bis 7"]
    assert all(r["ratio"] > 1 for r in fa["rows"][1:])
    kind, text = ev.verdict_forms(fa, ev.load_frozen())
    assert kind == "info" and "Mittel 5" in text and "999 von 1 000" in text


def test_schwankung_preset_shows_smaller_variance_for_the_spread_delay():
    kind, text = _noise("Schwankung")
    assert kind == "info" and "k = 1: 2,55" in text and "k = 2: 3,41" in text and "fest: 4,40" in text and "kleiner" in text


def test_einzelbestellung_preset_shows_652_against_13_4():
    kind, text = _single("Einzelbestellung")
    assert kind == "info" and "Einzelbestellung 652 gegen 13,4" in text and "zweites Moment nur bis α = 0,238" in text


def test_mit_leitung_preset_names_the_distribution_free_order_variance():
    kind, text = _noise("Mit Leitung")
    assert kind == "info" and "α/(2 − α) = 0,379" in text and "k = 1: 2,41" in text and "k = 2: 2,91" in text and "fest: 3,25" in text
    kind2, text2 = _hand("Mit Leitung")
    assert kind2 == "info" and "ρ = max(|1 − α|, q)" in text2
    assert "zweites Moment ebenfalls bis α = 2,00" in _single("Mit Leitung")[1]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_through_all_five_analyses_without_error(name):
    s = _settings(C.PRESETS[name])
    for kind, text in (_hand(name), _noise(name), _single(name), _decay(name),
                       ev.verdict_forms(ev.forms_analysis(s.tau, s.spread, s.random_form, s.seed), ev.load_frozen())):
        assert kind in ("success", "info", "warning") and text
