# tests/test_slices.py
import pytest, json
from statistics import mean

GLOBAL_BLEU_REQ = 0.25    # global BLEU must be ≥ 0.25
RELATIVE_FLOOR  = 0.90    # each slice ≥ 90 % of global

def test_slice_bleu(slice_bleu):
    """Every question_type slice should not lag far behind global."""
    global_bleu = mean(slice_bleu.values())
    assert global_bleu >= GLOBAL_BLEU_REQ, (
        f"Global BLEU {global_bleu:.3f} < required {GLOBAL_BLEU_REQ}"
    )

    min_allowed = global_bleu * RELATIVE_FLOOR
    bad = {k: v for k, v in slice_bleu.items() if v < min_allowed}

    assert not bad, (
        "Slices below threshold: "
        + ", ".join(f"{k} ({v:.3f})" for k, v in bad.items())
        + f"  (threshold {min_allowed:.3f})"
    )
