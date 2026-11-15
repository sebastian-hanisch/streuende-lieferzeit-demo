"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Nie ein Regler-Wert als Zweierpotenz voreingestellt (kein α = 0,5, kein p = 1/2 als Preset, DEMO-PLAYBOOK und Linie Rückkopplung V10/V11): die Presets nutzen
α = 0,7; 0,2; 0,1; 0,3; 0,4; 0,55 und die Mittel 3, 5, 12, 20. Die Fortschrittswahrscheinlichkeit p = k / τ ist abgeleitet; sie ist bei einzelnen
Reglerstellungen (etwa τ = 4, k = 2) eine Zweierpotenz, das ist hier unschädlich (lineare Rechnung, Tests prüfen Rundungsrauschen mit)."""

# --- Seitenleiste: Lager und Lieferzeit
TAU_MIN, TAU_MAX, DEFAULT_TAU = 2, 24, 3
K_MIN, K_MAX, DEFAULT_K = 1, 12, 2          # k > τ heißt fest (k wird auf τ begrenzt)
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.05, 2.0, 0.01, 0.7
BETA_OPTIONS = (0, 1)                       # 0: Leitung nicht mitrechnen, 1: Bestand plus Leitung (Bestandsposition) zurückführen
DEFAULT_BETA = 0
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Experiment Grenzkarte
TAU_MAP_OPTIONS = (12, 60, 240, 1200)
DEFAULT_TAU_MAP = 12

# --- Experiment andere Formen (alle drei Formen werden immer gerechnet: kein Regler wird je zum No-Op)
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 1, 11, 1   # Abstand d (Zweipunkt) und Halbbreite w (gleichverteilt): gleiche Zahl, auf höchstens τ - 1 begrenzt
RANDOM_FORM_OPTIONS = ("glatt", "gemischt", "spitz")
DEFAULT_RANDOM_FORM = "gemischt"

# --- Experiment Einzelbestellung
D_MIN, D_MAX, DEFAULT_D = 1, 100, 10
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 3.0, 0.1, 1.0
RUNS_MIN, RUNS_MAX, RUNS_STEP, DEFAULT_RUNS = 100, 600, 100, 300
SIM_PERIODS, SIM_BURN = 1500, 300            # Perioden je Lauf und Einschwingen (Perioden davor zählen nicht)
SHOCK_RUNS, SHOCK_PERIODS = 2000, 40         # Stoßantwort der Einzelbestellung (Mittel über Läufe)

# --- Feste Messbedingungen
FROZEN_SEED = 20261115                       # Zufalls-Seed der vorgerechneten Zufallsverteilungen
FROZEN_N = 1000
FROZEN_MEANS = (3, 4, 5, 6)                  # Mittel der Gegenprobe; Mittel 4 ist die Hauptreihe
NOISE_RUNS, NOISE_PERIODS, NOISE_BURN = 200, 2000, 400   # Simulation des anteiligen Modells (Normalrauschen)
DECAY_FRAC = 0.01                            # Abklingzeit: Bestand unter 1 % seiner Spitze

COLOR_FIXED = "#4c78a8"
COLOR_SPREAD = "#2F6B65"
COLOR_ONE = "#c6783b"
COLOR_SIM = "rgba(150,150,150,0.75)"
COLOR_BAD = "#b5453a"

PRESETS = {
    "Von Hand (k = 2, τ = 3)": {"tau": 3, "k": 2, "alpha": 0.7, "beta": 0},
    "Zwölf Perioden": {"tau": 12, "k": 2, "alpha": 0.2, "beta": 0},
    "Dritte Ordnung": {"tau": 20, "k": 3, "alpha": 0.1, "beta": 0, "tau_map": 60},
    "Andere Verteilungen": {"tau": 5, "k": 2, "alpha": 0.3, "beta": 0, "dist_spread": 2},
    "Schwankung": {"tau": 3, "k": 2, "alpha": 0.4, "beta": 0},
    "Einzelbestellung": {"tau": 12, "k": 2, "alpha": 0.1, "beta": 0, "d": 10, "sigma": 1.0},
    "Mit Leitung": {"tau": 3, "k": 2, "alpha": 0.55, "beta": 1},
}
PRESET_HELP = {
    "Von Hand (k = 2, τ = 3)": "Mittlere Lieferzeit 3 Perioden, zwei Stufen (Fortschrittswahrscheinlichkeit 2/3), α = 0,7: feste Lieferzeit schwingt auf, die streuende nicht.",
    "Zwölf Perioden": "Mittlere Lieferzeit 12, zwei Stufen, α = 0,2: stabil, aber das Abklingen nach einem Stoß dauert Hunderte Perioden; die feste Lieferzeit ist hier instabil.",
    "Dritte Ordnung": "Mittlere Lieferzeit 20, drei Stufen (die Verzögerung dritter Ordnung der Systemdynamik-Software), α = 0,1; die Grenzkarte steht auf τ = 60.",
    "Andere Verteilungen": "Mittlere Lieferzeit 5: Lieferzeit 3 oder 7 mit je 50 %, verglichen mit der festen Lieferzeit 5.",
    "Schwankung": "Weißes Rauschen im Bedarf bei τ = 3, k = 2, α = 0,4: Bestands- und Bestellvarianz, fest gegen streuend.",
    "Einzelbestellung": "Jede Bestellung kommt als Ganzes nach einer gezogenen Lieferzeit (τ = 12, zwei Stufen, α = 0,1, Bedarf 10, Rauschen 1): die Streuung der Lieferzeit selbst wird zur Schwankung.",
    "Mit Leitung": "Bestand plus Leitung zurückführen (β = 1), α = 0,55: die Bestellvarianz hängt nicht an der Verteilung, die Bestandsvarianz schon.",
}
