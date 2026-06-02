"""The agent: deterministic self-correcting loop + single-shot baseline.

Two agents share the retrieval / signals primitives:

  run_baseline   single-shot: search(hybrid) -> answer. No loop, no correction. The
                 CP1 comparison point - hybrid fusion, NO cross-encoder, the one that
                 fails quietly. Signals are recorded anyway so CP1 can SEE the failure.
  run_loop       deterministic self-correcting loop:
                   search(hybrid) -> read_signals -> policy.decide
                     -> DECOMPOSE : IRCoT iterative decompose (the taught gated action)
                     -> RERANK    : cross-encode the pool (offline/docs ladder only)
                     -> ANSWER/STOP : terminal SUFFICIENCY decides (system | evidence |
                                      autorater) - the v2 selling point.
                   always terminates (step budget).

The taught loop is deterministic so the headline numbers are reproducible and
ablatable. Claude (Sonnet 4.6 via LiteLLM) does three focused sub-tasks: generating
the next IRCoT sub-query, the final grounded answer, and (via the FAST_MODEL haiku)
the sufficiency autorater. Every step appends a trace.StepRecord.
"""
from __future__ import annotations

import json
import os
import re
import time

import config
import policy as policy_mod
import retrieval
import signals as signals_mod
from policy import Action, Decision, LoopState
from trace import StepRecord, Trace

os.environ.setdefault("LITELLM_LOG", "ERROR")

ABSTAIN_TEXT = "INSUFFICIENT_CONTEXT"

# Autorater health: track parse/API failures so the abstention study can report the rate
# (a silent default-to-sufficient would otherwise hide bias toward answering).
AUTORATER_STATS = {"calls": 0, "failures": 0}


# --- LLM helper (via LiteLLM) ------------------------------------------------
def _complete(system: str, user: str, max_tokens: int, model: str | None = None,
              temperature: float = 0.0, max_retries: int = 5) -> tuple[str, int]:
    """One LLM turn. Returns (text, total_tokens). Deterministic at temperature=0.
    Retries transient API failures (503 / reset / rate limit / timeout) with backoff so
    a long offline run is not killed by one blip."""
    import litellm

    litellm.suppress_debug_info = True
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model or config.AGENT_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=45,  # client-side: a hung socket fails fast -> retry on a fresh connection
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
            return text, tokens
        except Exception as exc:  # noqa: BLE001 - retry transient, re-raise persistent
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise last_exc  # unreachable


# --- IRCoT: iterative decompose (the headline gated action) ------------------
_IRCOT_SYSTEM = (
    "You are running an iterative retrieve-and-reason loop to answer a multi-hop "
    "question. Given the main question, the evidence retrieved so far, and the "
    "sub-questions already asked, output the NEXT single sub-question whose answer is "
    "still MISSING and is needed to answer the main question. Make it self-contained: "
    "name entities explicitly, resolving any bridge entity from the evidence so far. "
    "If the evidence already contains everything needed to answer the main question, "
    "reply with exactly: ENOUGH. Output ONLY the sub-question text or ENOUGH - no prose."
)


def _evidence_digest(results: list[retrieval.RetrievalResult], max_docs: int = 6, max_chars: int = 160) -> str:
    """A short, deduped digest of the evidence retrieved so far (titles + snippets) to
    condition the next IRCoT sub-query. Kept small to control tokens."""
    seen, lines = set(), []
    for res in results:
        for c in res.candidates:
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            lines.append(f"- {c.title}: {(c.text or '')[:max_chars]}")
            if len(lines) >= max_docs:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(none)"


def _next_subquery(question: str, results, sub_queries, model=None) -> tuple[str | None, int]:
    user = (
        f"Main question: {question}\n\n"
        f"Evidence so far:\n{_evidence_digest(results)}\n\n"
        f"Sub-questions already asked:\n" + ("\n".join(f"- {s}" for s in sub_queries) or "(none)") +
        "\n\nNext sub-question (or ENOUGH):"
    )
    text, tokens = _complete(_IRCOT_SYSTEM, user, max_tokens=80, model=model)
    t = (text or "").strip()
    if not t or t.upper().startswith("ENOUGH"):
        return None, tokens
    t = re.sub(r"^[\-\d\.\)\s]+", "", t.splitlines()[0]).strip()  # first line, strip bullet
    return (t or None), tokens


def ircot_search(question: str, k: int = config.TOP_K, max_hops: int = config.IRCOT_MAX_HOPS,
                 fusion: str | None = None, model: str | None = None, encoded=None) -> tuple[retrieval.RetrievalResult, int]:
    """IRCoT iterative decompose: retrieve the question, then up to max_hops-1 follow-up
    sub-queries, EACH conditioned on the evidence accumulated so far (the v2 upgrade over
    v1's one-shot parallel split). Union the per-hop evidence (max fusion score per doc)
    -> top-K. Returns (RetrievalResult mode='ircot', tokens). The original question's raw
    rankings (for divergence) come from hop 0. Pools FUSION scores (no reranker), so the
    IRCoT lift is attributable to evidence recovery, not reranking."""
    enc = encoded or retrieval.encode_query(question)
    t0 = time.perf_counter()
    r0 = retrieval.search(question, mode="hybrid", k=k, fusion=fusion, encoded=enc, with_raw=True)
    results = [r0]
    sub_queries: list[str] = []
    tokens = 0
    for _hop in range(1, max(1, max_hops)):
        nxt, tok = _next_subquery(question, results, sub_queries, model=model)
        tokens += tok
        if nxt is None:
            break
        sub_queries.append(nxt)
        results.append(retrieval.search(nxt, mode="hybrid", k=k, fusion=fusion, with_raw=False))
        if len(sub_queries) >= config.MAX_SUBQUERIES:
            break
    pooled = retrieval.union_pool(results, k)
    timings = {"ircot_ms": (time.perf_counter() - t0) * 1000}
    res = retrieval.RetrievalResult(question, "ircot", sub_queries, pooled, r0.raw, timings,
                                    score_kind="fusion", pool=pooled)
    return res, tokens


# --- sufficiency autorater (the STOP upgrade; FAST_MODEL) --------------------
_SUFFICIENCY_SYSTEM = (
    "You judge whether the provided context contains ENOUGH information to answer the "
    "question with certainty. First decompose the question into the facts it requires; "
    "the context is SUFFICIENT only if EVERY required fact is explicitly present in the "
    "context. Use ONLY the context, not outside knowledge. If any required fact is "
    'missing, it is insufficient. Reply ONLY with compact JSON: {"sufficient": true|false}.'
)


def sufficiency_judge(question: str, candidates, model: str | None = None, max_chars: int = 600) -> tuple[bool, int]:
    """Sufficient-Context Autorater (arXiv 2411.06037), one FAST_MODEL call: does the
    retrieved context contain every fact the question requires? Returns (sufficient,
    tokens). On parse/empty failure, defaults to sufficient=True (do not over-abstain)."""
    if not candidates:
        return False, 0
    ctx = "\n".join(f"[{i}] {c.title}. {(c.text or '')[:max_chars]}" for i, c in enumerate(candidates, 1))
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nIs the context sufficient to answer the question?"
    AUTORATER_STATS["calls"] += 1
    try:
        # The fast model often reasons BEFORE the JSON (the prompt asks it to decompose the
        # question first), so give it room (512) and find the verdict ROBUSTLY by regex -
        # otherwise a correct judgment is lost when the closing brace is truncated/wrapped.
        text, tokens = _complete(_SUFFICIENCY_SYSTEM, user, max_tokens=512,
                                 model=model or config.FAST_MODEL)
        m = re.search(r'"?sufficient"?\s*[:=]\s*"?(true|false)"?', text, re.I)
        if m:
            return m.group(1).lower() == "true", tokens
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return bool(obj.get("sufficient")), tokens
    except Exception:  # noqa: BLE001 - default sufficient (do not over-abstain), but COUNT it
        AUTORATER_STATS["failures"] += 1
        return True, 0


def terminal_sufficiency(result, question: str, stop_mode: str, thresholds: dict | None,
                         model: str | None = None) -> tuple[bool, int]:
    """The STOP mechanism (swappable). Returns (sufficient, tokens).
      system   - always 'sufficient'; generate_answer self-abstains (weak negative).
      evidence - cheap calibrated gate: entity coverage of the FINAL set >= floor.
      autorater- the fast LLM sufficiency judge (the upgrade)."""
    if stop_mode == "system":
        return True, 0
    if stop_mode == "evidence":
        th = thresholds or signals_mod.load_thresholds()
        cov = signals_mod.evidence_coverage(result)
        return cov >= th.get("evidence_coverage", 0.5), 0
    if stop_mode == "autorater":
        return sufficiency_judge(question, result.candidates, model=model)
    raise ValueError(f"unknown stop_mode: {stop_mode!r}")


# --- answer: grounded short answer or honest abstention ----------------------
_ANSWER_SYSTEM = (
    "You answer a question using ONLY the numbered context passages provided. "
    "Reply with ONLY the final answer on a single line: a name, date, number, or short "
    "noun phrase, usually one to six words. Do NOT show reasoning or steps, do NOT "
    "restate the question, do NOT write 'I need to find', do NOT explain. Output just "
    "the answer text. If the passages do not contain the information needed to answer, "
    "reply with exactly: INSUFFICIENT_CONTEXT"
)


def _clean_answer(text: str) -> str:
    """Keep answers short and EM/F1-meaningful even if the model adds a stray line."""
    t = (text or "").strip()
    if "INSUFFICIENT_CONTEXT" in t.upper():
        return ABSTAIN_TEXT
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ABSTAIN_TEXT
    return re.sub(r"^(answer|the answer is|final answer)[:\-\s]+", "", lines[-1], flags=re.I).strip()


def _context_block(candidates, max_docs: int = config.ANSWER_K, max_chars: int = 700) -> str:
    return "\n".join(f"[{i}] {c.title}. {(c.text or '')[:max_chars]}" for i, c in enumerate(candidates[:max_docs], 1))


def generate_answer(question: str, candidates, model: str | None = None) -> tuple[str, int]:
    """Claude answers grounded in the FOCUSED top-ANSWER_K passages (v2.5 precision
    regime: a small, cheap context where ranking precision matters), or emits
    INSUFFICIENT_CONTEXT."""
    if not candidates:
        return ABSTAIN_TEXT, 0
    user = f"Context:\n{_context_block(candidates)}\n\nQuestion: {question}\nAnswer (answer only):"
    text, tokens = _complete(_ANSWER_SYSTEM, user, max_tokens=150, model=model)
    return _clean_answer(text), tokens


def is_abstention(answer: str) -> bool:
    return answer.strip().upper().startswith("INSUFFICIENT_CONTEXT") or not answer.strip()


# --- single-shot baseline (CP1) ----------------------------------------------
def run_baseline(question: str, qid: str = "", answerable: bool | None = None,
                 gold_doc_ids: list | None = None, thresholds: dict | None = None,
                 k: int = config.TOP_K, fusion: str | None = None, model: str | None = None) -> Trace:
    """One hybrid search (dense + BM25 fusion, NO reranker), then answer. No loop, no
    correction. Signals are computed + recorded so CP1 can SEE the quiet failure - using
    the CALIBRATED detector/fusion (not the bm25 default) so the demo readings match the
    taught selection. latency_ms includes the answer-generation LLM call (end-to-end)."""
    gold = gold_doc_ids or []
    th = thresholds or signals_mod.load_thresholds()
    detector = th.get("_detector", signals_mod.DEFAULT_DETECTOR)
    fusion = fusion or th.get("_fusion")
    tr = Trace(question=question, question_id=qid, answerable=answerable, gold_doc_ids=gold)
    result = retrieval.search(question, mode=config.BASELINE_MODE, k=k, fusion=fusion)
    report = signals_mod.read_signals(result, th, detector=detector)
    t = time.perf_counter()
    answer, tokens = generate_answer(question, result.candidates, model=model)
    ans_ms = (time.perf_counter() - t) * 1000
    rec = StepRecord.from_objects(1, result, report,
                                  Decision(Action.ANSWER, "single-shot baseline (no loop)"),
                                  tokens=tokens, gold_doc_ids=gold)
    rec.latency_ms = round(rec.latency_ms + ans_ms, 1)
    tr.add(rec)
    tr.answer = answer
    tr.stopped = is_abstention(answer)
    return tr


# --- the self-correcting loop (CP3) ------------------------------------------
def loop_retrieve(question: str, gold_doc_ids: list | None = None, thresholds: dict | None = None,
                  policy: str = "ladder",
                  k: int = config.TOP_K, fusion: str | None = None, model: str | None = None) -> dict:
    """Run the loop's RETRIEVAL phase (Decision A: the cost-escalation ladder) up to the
    terminal, WITHOUT the answer/stop decision. Returns {steps, result, report,
    terminal_rec, tokens}. Split out so the abstention study can run the (expensive) IRCoT
    once and then apply the STOP mechanisms to the SAME retrieval. `policy` selects the
    adaptive ladder (default) or a fixed baseline (always_answer / always_colbert /
    always_rerank / always_decompose) for the policy comparison."""
    gold = gold_doc_ids or []
    state = LoopState()
    enc = retrieval.encode_query(question)
    th = thresholds or signals_mod.load_thresholds()
    detector = th.get("_detector", signals_mod.DEFAULT_DETECTOR)
    fusion = fusion or th.get("_fusion")

    result = retrieval.search(question, mode=config.BASELINE_MODE, encoded=enc, k=k, fusion=fusion)
    report = signals_mod.read_signals(result, th, detector=detector)
    steps: list[StepRecord] = []
    tokens = 0

    while True:
        decision = policy_mod.decide(report, state, question, policy=policy)

        rec = StepRecord.from_objects(state.step + 1, result, report, decision, gold_doc_ids=gold)
        steps.append(rec)
        state.step += 1

        if decision.action == Action.COLBERT:
            # tier 2: ColBERT late-interaction re-retrieval (token-level precision fix).
            # Terminal: its MaxSim scores are off the fusion/raw-dense scale, so decide()
            # treats `colberted` as terminal and the next step answers/stops.
            result = retrieval.colbert_search(question, n_prefetch=config.RETRIEVE_N, k=k)
            state.colberted = True
            report = signals_mod.read_signals(result, th, detector=detector)
            continue
        if decision.action == Action.RERANK:
            # the cross-encoder tier-2 alternative (measured peer of ColBERT). Also terminal.
            result = retrieval.rerank(result, query=question, k=k)
            state.reranked = True
            report = signals_mod.read_signals(result, th, detector=detector)
            continue
        if decision.action == Action.DECOMPOSE:
            # tier 3: IRCoT re-pools FUSION scores like the baseline -> signals re-read after
            # it stay on-scale, so the loop can re-gate (answer if recovered, else rerank).
            result, dtok = ircot_search(question, k=k, fusion=fusion, model=model, encoded=enc)
            rec.tokens += dtok
            tokens += dtok
            state.decompose_count += 1
            report = signals_mod.read_signals(result, th, detector=detector)
            continue
        return {"steps": steps, "result": result, "report": report, "terminal_rec": rec, "tokens": tokens}


def finalize_trace(core: dict, question: str, qid: str = "", answerable: bool | None = None,
                   gold_doc_ids: list | None = None, stop_mode: str = "system",
                   thresholds: dict | None = None, model: str | None = None,
                   answer: str | None = None, answer_tokens: int = 0, answer_ms: float = 0.0,
                   sufficient: bool | None = None, suff_tokens: int = 0, suff_ms: float = 0.0) -> Trace:
    """Apply the terminal SUFFICIENCY decision to a loop_retrieve core and build the Trace.
    `answer`/`sufficient` may be precomputed (shared across stop modes) to avoid redundant
    LLM calls; pass their `*_ms` so end-to-end latency is still accounted. A fresh terminal
    step is built per call so traces for different modes do not alias one record."""
    import dataclasses

    gold = gold_doc_ids or []
    tr = Trace(question=question, question_id=qid, answerable=answerable, gold_doc_ids=gold)
    for s in core["steps"][:-1]:
        tr.add(s)
    rec = dataclasses.replace(core["terminal_rec"])  # copy: do not mutate the shared core
    result = core["result"]
    extra_ms = 0.0

    if sufficient is None:
        t = time.perf_counter()
        sufficient, stok = terminal_sufficiency(result, question, stop_mode, thresholds, model=model)
        extra_ms += (time.perf_counter() - t) * 1000
        rec.tokens += stok
    else:
        rec.tokens += suff_tokens
        extra_ms += suff_ms

    if sufficient:
        if answer is None:
            t = time.perf_counter()
            answer, atok = generate_answer(question, result.candidates, model=model)
            extra_ms += (time.perf_counter() - t) * 1000
            rec.tokens += atok
        else:
            rec.tokens += answer_tokens
            extra_ms += answer_ms
        tr.answer = answer
        tr.stopped = is_abstention(answer)
        if tr.stopped:  # the generator itself abstained
            rec.action = rec.decision = Action.STOP.value
            rec.reason = "grounded generation abstained (insufficient context)"
    else:
        tr.stopped = True
        tr.answer = ABSTAIN_TEXT
        rec.action = rec.decision = Action.STOP.value
        rec.reason = f"{stop_mode} sufficiency gate: insufficient context; abstain"
    rec.latency_ms = round(rec.latency_ms + extra_ms, 1)  # include answer + sufficiency LLM wall-clock
    tr.add(rec)
    return tr


def run_loop(question: str, qid: str = "", answerable: bool | None = None,
             gold_doc_ids: list | None = None, thresholds: dict | None = None,
             stop_mode: str = "system",
             policy: str = "ladder", k: int = config.TOP_K,
             fusion: str | None = None, model: str | None = None) -> Trace:
    """The self-correcting loop. Decision A is the cost-escalation ladder (`policy`: the
    adaptive 'ladder' or a fixed baseline always_answer/always_colbert/always_rerank/
    always_decompose); Decision B is the STOP mechanism (`stop_mode`: system | evidence |
    autorater). Default is the gentle `system` stop (the generator self-abstains), which
    keeps answers non-degraded; the `autorater` is the abstention-MAXIMIZER studied
    separately - it catches the most unanswerables but over-abstains on answerables
    (bounded by retrieval completeness). Always terminates."""
    th = thresholds or signals_mod.load_thresholds()
    core = loop_retrieve(question, gold_doc_ids=gold_doc_ids, thresholds=th,
                         policy=policy, k=k, fusion=fusion, model=model)
    return finalize_trace(core, question, qid=qid, answerable=answerable, gold_doc_ids=gold_doc_ids,
                          stop_mode=stop_mode, thresholds=th, model=model)
