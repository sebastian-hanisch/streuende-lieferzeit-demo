"""Erzeugt data/slz_frozen.json: die vorgerechnete Reihe der Zufallsverteilungen (Experiment „Andere Formen“).

Aufruf (aus dem Repo-Ordner, mit dem Python des Test-venv oder des Laufzeit-venv):  python tools/make_frozen.py
Rechnet für jedes Mittel in `slz_constants.FROZEN_MEANS` die Grenze (β = 0) von `FROZEN_N` Zufallsverteilungen (Zufalls-Seed `FROZEN_SEED`)
gegen die feste Lieferzeit gleichen Mittels und zählt, in wie vielen die Stabilität oberhalb der ersten Grenze auf dem α-Raster (Schritt 0,01
bis 2,6) zurückkehrt. Dauer etwa 30 Sekunden. Die Tests (`tests/test_claims.py`) rechnen die Reihe für Mittel 4 neu und vergleichen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import slz_constants as C  # noqa: E402
import slz_evaluation as E  # noqa: E402


def main() -> None:
    rows = {}
    for mean in C.FROZEN_MEANS:
        rows[str(mean)] = E.random_batch(C.FROZEN_SEED, mean, C.FROZEN_N)
        r = rows[str(mean)]
        print(f"Mittel {mean}: über fest {r['above']}, unter fest {r['below']}, gleich {r['equal']}, Median {r['ratio_median']:.4f}, Wiederkehr {r['returns']}")
    out = {
        "beschreibung": "Vorgerechnet von tools/make_frozen.py: Grenze (beta = 0) von Zufallsverteilungen gegen die feste Lieferzeit gleichen Mittels.",
        "seed": C.FROZEN_SEED, "n": C.FROZEN_N, "rows": rows,
    }
    target = ROOT / "data" / "slz_frozen.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print("geschrieben:", target)


if __name__ == "__main__":
    main()
