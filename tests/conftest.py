"""Gemeinsame Fixtures: das Handbeispiel (k = 2, p = 2/3, Mittel 3), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Handbeispiel: zwei Stufen, Fortschrittswahrscheinlichkeit 2/3 (Mittel 3), Grenze α_krit = p (2 - p) = 8/9 (Jury, von Hand)."""
    return {"k": 2, "p": 2.0 / 3.0, "alpha_krit": 8.0 / 9.0}
