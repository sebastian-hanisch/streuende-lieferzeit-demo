"""Eigener Zufallsgenerator (SplitMix64): gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen.

Herkunft: Kopie von `lgc_rng.py` aus lager-chaos-demo (dort Kopie und Erweiterung von `rtm_rng.py`), mit Präfix umbenannt; neu hier: `child_seed`
(ein Zufalls-Seed je Zweck und je Index, ohne Python-`hash()`) und `geometric_failures` (Anzahl der Fehlversuche vor dem ersten Erfolg, vektorisiert).
`splitmix_array` erzeugt die ersten n Ausgaben vektorisiert (der Zustand ist ein Zähler, die Ausgabe eine Mischfunktion des Zählers); die Folge
stimmt mit der Klasse `SplitMix64` überein (siehe Test). Python-`hash()` und `random` sind für feste Zufalls-Seeds nicht geeignet
(pro Prozess gesalzen bzw. nicht versionsstabil)."""

import math

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def normal(self) -> float:
        """Standardnormalverteilt (Box-Muller, nur der Kosinus-Zweig)."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def child_seed(seed: int, index: int) -> int:
    """Abgeleiteter Zufalls-Seed: die erste Ausgabe von SplitMix64 zum Zustand seed + index * GAMMA (je Index eine unabhängige Folge)."""
    return SplitMix64((seed & MASK) + (int(index) & MASK) * GAMMA).next_u64()


def splitmix_array(seed: int, n: int) -> np.ndarray:
    """Die ersten n Ausgaben von SplitMix64(seed) als uint64-Feld (vektorisiert, gleiche Folge wie die Klasse)."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + k * np.uint64(GAMMA)          # uint64-Felder rechnen modulo 2^64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, n: int) -> np.ndarray:
    """n Gleichverteilte in [0, 1) mit 53 Bit."""
    return (splitmix_array(seed, n) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def normal_array(seed: int, n: int) -> np.ndarray:
    """n Standardnormalwerte; Wert j nutzt die Ausgaben 2j und 2j + 1 (wie `SplitMix64.normal`)."""
    u = uniform_array(seed, 2 * n)
    u1 = np.maximum(u[0::2], 1e-12)
    u2 = u[1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def geometric_failures(u: np.ndarray, p: float) -> np.ndarray:
    """Fehlversuche vor dem ersten Erfolg (Erfolgswahrscheinlichkeit p je Versuch) aus Gleichverteilten u in [0, 1): floor(ln u / ln q), q = 1 - p.

    P(G >= g) = q^g, denn floor(ln u / ln q) >= g genau dann, wenn u <= q^g. Für p >= 1 immer 0."""
    q = 1.0 - p
    if q <= 0.0:
        return np.zeros(np.shape(u), dtype=np.int64)
    g = np.floor(np.log(np.maximum(u, 1e-300)) / math.log(q))
    return np.minimum(g, 1e9).astype(np.int64)
