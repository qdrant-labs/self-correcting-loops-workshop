"""Answer-correctness scoring: the outer-eval helpers used to grade the loop's answers.

In-loop signals are the product; this module is ground truth, used to score the answers
the loop produces, never as the live headline knob:

  EM / token-F1   SQuAD-style normalized exact match and token overlap. Deterministic and
                  cheap, but surface-form sensitive (NOT "bias-free"): they penalize a
                  correct-but-paraphrased answer and reward the model's prior.
  semantic judge  an optional gpt-5.5 grader (cross-provider, to reduce self-preference;
                  it does not solve contamination) that credits meaning over surface form.
                  This is what the v2.5 headline reports for answer quality.

MuSiQue lists answer aliases, so every score is taken against the best-matching gold.
Retrieval precision (recall@1/@3, MRR) is scored inline in the eval scripts, where the
per-query counterfactual rows live.
"""
from __future__ import annotations

import re
import string

import config

# ============================================================================
# Answer-string metrics (SQuAD-style normalization; deterministic but
# surface-form sensitive - reported as such, never as "bias-free")
# ============================================================================
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    for t in p:
        if t in g:
            common[t] = common.get(t, 0) + 1
    overlap = sum(min(c, g.count(t)) for t, c in common.items())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def best_answer_score(pred: str, golds, fn) -> float:
    """MuSiQue lists answer aliases; score against the best-matching gold."""
    golds = [golds] if isinstance(golds, str) else list(golds or [])
    return max((fn(pred, g) for g in golds), default=0.0)


def gold_answers(q: dict) -> list[str]:
    """All acceptable gold answer strings for a question: the answer plus its aliases."""
    golds = [q.get("answer", "")] + list(q.get("answer_aliases") or [])
    return [g for g in golds if g]


# --- optional gpt-5.5 semantic judge (cross-provider) ------------------------
_JUDGE_SYSTEM = (
    "You grade whether a predicted answer is semantically correct given the gold "
    "answer(s). Ignore surface form, punctuation, and extra words; judge meaning only. "
    'Reply ONLY with compact JSON: {"correct": true|false}.'
)


def judge_answer(question: str, pred: str, golds, model: str | None = None, max_retries: int = 5) -> int:
    """gpt-5.5 semantic correctness (1/0). gpt-5.5 is a reasoning model: it needs a
    real token budget (>=~400) or it spends the budget reasoning and returns empty.
    Retries transient API failures with backoff (so a blip is not silently scored
    wrong); only a persistent parse/error path returns 0."""
    import json as _json
    import time as _time

    import litellm

    litellm.suppress_debug_info = True
    golds = [golds] if isinstance(golds, str) else list(golds or [])
    user = f"Question: {question}\nGold answer(s): {golds}\nPredicted answer: {pred}\nIs the prediction correct?"
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model or config.JUDGE_MODEL,
                messages=[{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
                max_tokens=500,
                timeout=60,  # reasoning model is slower; still bounded so a hang can't stall the sweep
            )
            text = (resp.choices[0].message.content or "").strip()
            obj = _json.loads(text[text.find("{") : text.rfind("}") + 1])
            return int(bool(obj.get("correct")))
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                return 0
            _time.sleep(min(2 ** attempt, 30))
    return 0
