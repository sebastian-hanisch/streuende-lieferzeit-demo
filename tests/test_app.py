"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_choice_option_does_not_raise():
    """Jede Option der Auswahlregler (β, Karten-τ, Form der Zufallsverteilung) einzeln."""
    import slz_constants as C
    for widget_kind, key, options in (("radio", "beta_radio", C.BETA_OPTIONS), ("select_slider", "tau_map_select", C.TAU_MAP_OPTIONS),
                                      ("selectbox", "random_form_select", C.RANDOM_FORM_OPTIONS)):
        for option in options:
            at = _fresh_app()
            getattr(at, widget_kind)(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])
            assert getattr(at, widget_kind)(key=key).value == option


def test_worst_case_corners_do_not_raise():
    """Rand- und Eckfälle gleichzeitig: größtes τ mit zu großem k (wird auf τ begrenzt), α am Rand, β = 1, größte Läufe-Zahl, größter Bedarf und stärkstes Rauschen."""
    for tau, k, alpha, beta in ((24, 12, 2.0, 1), (24, 12, 2.0, 0), (24, 1, 0.05, 0), (2, 12, 1.99, 0), (24, 2, 0.4, 1)):
        at = _fresh_app()
        at.slider(key="tau_slider").set_value(tau)
        at.slider(key="k_slider").set_value(k)
        at.slider(key="alpha_slider").set_value(alpha)
        at.radio(key="beta_radio").set_value(beta)
        at.slider(key="runs_slider").set_value(600)
        at.slider(key="d_slider").set_value(100)
        at.slider(key="sigma_slider").set_value(3.0)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (tau, k, alpha, beta, [str(e) for e in at.exception])


def test_every_preset_shows_the_verdicts_and_no_exception():
    from slz_constants import PRESETS
    for name in PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        verdicts = [e.value for e in list(at.success) + list(at.info) + list(at.warning)]
        assert len(verdicts) >= 6, (name, len(verdicts))                       # Kernfrage, Grenzkarte, Formen, Schwankung, Einzelbestellung, Abklingen


def test_permalink_values_are_limited_to_the_slider_bounds():
    at = AppTest.from_file(APP_PATH)
    at.query_params["tau"] = "99"
    at.query_params["alpha"] = "7.5"
    at.query_params["form"] = "unsinn"
    at.query_params["taumap"] = "13"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="tau_slider").value == 24 and at.slider(key="alpha_slider").value == 2.0
    assert at.select_slider(key="tau_map_select").value == 12
