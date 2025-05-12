# tests/test_templates.py
import difflib, pytest
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

SIM_THRESHOLD = 0.6   # min BLEU between canonical & paraphrase response
PASS_RATE     = 0.8   # at least 80 % of paraphrases must meet threshold

def similarity(a: str, b: str) -> float:
    """
    Quick‑n‑dirty: BLEU‑1 with smoothing. 0‑1 range.
    """
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [a.split()], b.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie
    )

@pytest.mark.parametrize("paraset", ["dummy"], indirect=True)
def test_template_paraphrase_sets(paraphrase_sets, predict):
    """
    For each template set:
      * Ask canonical question -> reference answer
      * Ask each paraphrase -> candidate answer
      * Compute BLEU‑1 similarity; require >= SIM_THRESHOLD
      * Overall: at least PASS_RATE of paraphrases must succeed
    """
    for entry in paraphrase_sets:
        canon_q   = entry["canonical"]
        para_qs   = entry["paraphrases"]
        ref_answer = predict(canon_q)

        scores = [
            similarity(ref_answer, predict(q))
            for q in para_qs
        ]
        pass_frac = sum(s >= SIM_THRESHOLD for s in scores) / len(scores)

        assert pass_frac >= PASS_RATE, (
            f"Template set '{entry['id']}' robustness {pass_frac:.2%} < "
            f"required {PASS_RATE:.0%}"
        )

