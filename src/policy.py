"""Decision policy + action space (v2.5): the cost-escalation LADDER.

On a MIXED workload (single-hop + multi-hop + unanswerable) the agent should pay only
as much as a query needs. The ladder is cost-ordered (Tier 1 cheapest, Tier 3 dearest)
and the route through it is matched to the failure mode:

  Tier 0  retrieve hybrid (dense bge + miniCOIL, RRF), read cheap fusion signals.
  Tier 1  ANSWER if confident      - the confidence gate clears -> answer now, no extra
                                      spend. The cheap path v2 lacked entirely.
  Tier 2  RERANK (jina cross-encoder rescore of the EXISTING pool) - a genuine in-place
                                      precision fix for a weak SINGLE-HOP lookup (the doc
                                      is likely in the pool but mis-ranked). Terminal.
  Tier 3  DECOMPOSE (IRCoT)        - escalate for a weak MULTI-HOP query (a missing hop;
                                      rerank cannot help - the hop is not in the pool).
  Terminal ANSWER or STOP          - the sufficiency decision (Decision B, in agent.py).

Why FAILURE-MATCHED, not a strict answer->rerank->decompose escalation: IRCoT's hop-0
re-retrieves the question from scratch, so decompose REPLACES
the pool - running rerank first would be discarded work, and re-reading the height/spread
signals on cross-encoder logits is off the fusion scale the thresholds were calibrated on.
So rerank (single-hop precision) and decompose (multi-hop hop-recovery) are routed by the
gold-free `looks_multi_hop` heuristic. DECOMPOSE pools FUSION scores like the baseline, so
signals re-read after it stay comparable and the loop can re-gate; RERANK changes the score
scale, so it is a terminal fix (no further gating). The cost ordering and the
adaptive-beats-fixed story are preserved; the routing is now honest.

Decision A (route through the tiers, here) stays separate from Decision B (answer vs stop,
the sufficiency mechanism in agent.py). Budget caps (one decompose, hard step budget)
guarantee termination.

The same module also exposes the FIXED policies (always-answer / always-rerank /
always-decompose) used as the headline comparison baselines in the policy study.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import config

# Adaptive ladder + the fixed-policy baselines for the cost/quality comparison. The
# three pure tiers are always_answer (T1) / always_colbert (T2) / always_decompose (T3);
# always_rerank is the measured cross-encoder alternative to the ColBERT tier-2.
LADDER = "ladder"
FIXED_POLICIES = ("always_answer", "always_colbert", "always_decompose", "always_rerank")


class Action(str, Enum):
    COLBERT = "colbert"       # ColBERT late-interaction (tier-2 single-hop precision fix)
    RERANK = "rerank"         # cross-encoder rescore (the measured tier-2 alternative)
    DECOMPOSE = "decompose"   # IRCoT iterative decompose
    ANSWER = "answer"
    STOP = "stop"


@dataclass
class LoopState:
    mode: str = config.BASELINE_MODE
    step: int = 0
    reranked: bool = False
    colberted: bool = False
    decompose_count: int = 0

    def precision_fixed(self) -> bool:
        """A terminal precision fix (ColBERT or rerank) has been applied."""
        return self.colberted or self.reranked

    def can_decompose(self) -> bool:
        return self.decompose_count < config.DECOMPOSE_BUDGET

    def budget_left(self) -> bool:
        return self.step < config.STEP_BUDGET


@dataclass
class Decision:
    action: Action
    reason: str
    tier: int = 0          # 1=answer-if-confident, 2=rerank, 3=decompose (cost rung)


def looks_multi_hop(question: str) -> bool:
    """Cheap, GOLD-FREE route signal: does this query look like it needs a hop recovered?
    True if the question names >= 2 entities or is long. On the MIXED workload this is a
    real router (single-hop lookups route to rerank, multi-hop to decompose); using the
    gold n_hops would be cheating (eval integrity)."""
    import signals as signals_mod

    return len(signals_mod.question_entities(question)) >= 2 or len(question.split()) >= 12


def _weak_names(report) -> str:
    active = report.weakness or ("retriever_divergence", "score_variance")
    weak = [n for n in active if report.fired.get(n)]
    return ", ".join(weak) if weak else "weak"


def decide(report, state: LoopState, question: str = "", policy: str = LADDER) -> Decision:
    """One step of Decision A. `policy` selects the adaptive ladder (default) or one of the
    fixed baselines. Returns the next corrective action, or ANSWER meaning 'stop escalating;
    go to the terminal answer/stop step'."""
    if not state.budget_left():
        return Decision(Action.ANSWER, "step budget spent; go to the answer/stop step", tier=1)

    if policy == "always_answer":
        return Decision(Action.ANSWER, "fixed policy: answer from the hybrid baseline (no correction)", tier=1)

    if policy == "always_colbert":
        if not state.colberted:
            return Decision(Action.COLBERT, "fixed policy: always ColBERT late-interaction", tier=2)
        return Decision(Action.ANSWER, "fixed policy: colberted; answer/stop", tier=2)

    if policy == "always_rerank":
        if not state.reranked:
            return Decision(Action.RERANK, "fixed policy: always rerank the pool", tier=2)
        return Decision(Action.ANSWER, "fixed policy: reranked; answer/stop", tier=2)

    if policy == "always_decompose":
        if state.can_decompose() and state.decompose_count == 0:
            return Decision(Action.DECOMPOSE, "fixed policy: always IRCoT decompose", tier=3)
        return Decision(Action.ANSWER, "fixed policy: decomposed; answer/stop", tier=3)

    # --- the adaptive ladder ---------------------------------------------------
    # The tier-2 fixes (ColBERT / rerank) are TERMINAL: each is a fresh retrieval or a
    # cross-encoder rescore whose scores are off the fusion/raw-dense scale the thresholds
    # use, so we do not re-gate on them. Decompose, by contrast, re-pools FUSION scores, so
    # the loop CAN re-gate after it (answer if a hop was recovered, else a terminal ColBERT).
    if state.precision_fixed():
        return Decision(Action.ANSWER, "precision fix applied (terminal); go to answer/stop", tier=2)

    if report.healthy:
        return Decision(Action.ANSWER, "tier 1: confident; answer now (cheapest path)", tier=1)

    # weak -> route by failure mode (cost-ordered): a multi-hop query likely has a MISSING
    # hop -> decompose (recover it); a single-hop lookup is likely mis-ranked in the pool ->
    # ColBERT late-interaction (token-level precision fix to promote it into the focused context).
    if looks_multi_hop(question) and state.can_decompose() and state.decompose_count == 0:
        return Decision(Action.DECOMPOSE, f"tier 3: weak ({_weak_names(report)}) + multi-hop; IRCoT to recover a hop", tier=3)
    # ColBERT is the SINGLE-HOP precision fix; do not apply it after a decompose (the
    # multi-hop fix already ran - escalating a multi-hop query to a single-hop tool is
    # incoherent and muddies the tiers). So tier 2 fires only for non-decomposed queries.
    if not state.colberted and state.decompose_count == 0:
        return Decision(Action.COLBERT, f"tier 2: weak ({_weak_names(report)}); ColBERT late-interaction (precision fix)", tier=2)

    return Decision(Action.ANSWER, "fixes exhausted; go to the answer/stop step", tier=1)
