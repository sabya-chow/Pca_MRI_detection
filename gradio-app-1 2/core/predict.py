from __future__ import annotations

from pathlib import Path


ACTIVE_PROJECT = Path(__file__).resolve().parents[3] / "prostate_pirads_gradio"


class LegacyAppMovedError(RuntimeError):
    pass


def raise_moved_error():
    raise LegacyAppMovedError(
        "The active prostate PI-RADS predictor is in "
        f"{ACTIVE_PROJECT}. Run that project's app.py instead."
    )
