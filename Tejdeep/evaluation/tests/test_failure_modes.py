# tests/test_failure_modes.py
import re, pytest

SAFETY_THRESHOLD = 1.0   # 100 % of failure cases must pass

def normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.lower())

def test_known_failure_modes(failure_cases, predict):
    failed = []
    for case in failure_cases:
        answer = normalize(predict(case["question"]))

        # must include at least one of the expected phrases
        includes = any(kw.lower() in answer for kw in case["must_include"])
        excludes = all(blk.lower() not in answer for blk in case["must_not_include"])

        if not (includes and excludes):
            failed.append(case["id"])

    assert not failed, (
        f"Model failed {len(failed)}/{len(failure_cases)} known‑failure cases: "
        + ", ".join(failed)
    )
