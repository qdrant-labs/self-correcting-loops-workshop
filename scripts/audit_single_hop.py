"""v2.5 offline data-quality pass: judge standalone-answerability of the derived
single-hop questions with a fast LLM (claude-haiku-4-5).

WHY: the derived single-hop set is the foundation of the v2.5 cost story (the cheap
tier must answer genuinely standalone lookups). The deterministic natural-language
filter in prepare_mixed.py removes `#N` placeholders, relation triples, and dangling
PERSONAL pronouns, but it cannot catch generic anaphoric noun phrases ("the creature",
"the black community", "the candidate") or fragmentary questions - those secretly
depend on a prior hop. A hand-sample of 75 put that residual at ~5-7%. This scales the
hand-check (verify a sample by hand; drop any that stay ambiguous).

INTEGRITY GUARD (applied in prepare_mixed, not here): we never trust the LLM alone.
This script emits only the RAW per-question verdict. prepare_mixed drops a question
only when the LLM says not-standalone AND the heuristic finds no capitalized proper-noun
anchor (the AND-gate). That protects obscure named entities ("Andre Bloc", "The Genius
of Victory") - which the LLM over-rejects because it does not recognize them - so the
filter never biases the set toward famous, easy-to-retrieve entities (which would inflate
single-hop recall). The LLM's job is only to catch the no-name danglers the heuristic misses.

The judge is CONSERVATIVE (default standalone=true; fail only on a clear no-name dangler
or fragment). Validated 16/17 on a labeled probe. Results are cached + resumable.

Usage:
  python scripts/audit_single_hop.py            # judge all unjudged unique question texts
  python scripts/audit_single_hop.py --limit 20 # smoke
  (run with the sandbox disabled - it makes LLM calls)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import config  # noqa: E402
import data  # noqa: E402

PROMPT_VERSION = "v3-conservative-anchor"
QUALITY_PATH = config.ARTIFACTS_DIR / "single_hop_quality.json"

_SYSTEM = (
    "Decide if a single-hop question can STAND ALONE, or whether it secretly depends on some PRIOR "
    "question's answer. DEFAULT to standalone=true. "
    "Mark standalone=FALSE ONLY when the main thing being asked about is referenced WITHOUT a name: an "
    "anaphoric pronoun (he/she/it/they/him/her/them/his/their) or a generic definite phrase ('the creature', "
    "'the candidate', 'the company', 'the black community') that has NO proper noun in the question to anchor "
    "it; OR the question is an incomplete fragment with no clear single answer. "
    "If the question contains a proper noun (a name, a named title, a place) that anchors the subject, mark "
    "standalone=TRUE EVEN IF you do not recognize it or cannot answer it. Judge self-containedness, never difficulty. "
    'Reply STRICT JSON only: {"standalone": true|false, "reason": "<=8 words"}.'
)


def _norm(q: str) -> str:
    return " ".join((q or "").split())


def judge_text(text: str, max_retries: int = 5) -> dict:
    """One conservative standalone-answerability verdict for a question text."""
    import litellm

    litellm.suppress_debug_info = True
    last = None
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=config.FAST_MODEL,
                messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": text}],
                max_tokens=80,
                temperature=0,
                timeout=45,
            )
            raw = (resp.choices[0].message.content or "").strip()
            obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            return {"standalone": bool(obj["standalone"]), "reason": str(obj.get("reason", ""))[:80]}
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == max_retries - 1:
                return {"standalone": True, "reason": f"JUDGE_ERROR:{type(exc).__name__}"}
            time.sleep(min(2 ** attempt, 20))
    return {"standalone": True, "reason": f"JUDGE_ERROR:{type(last).__name__}"}  # default-keep on failure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="judge at most N unjudged texts (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sh = [r for r in data.load_questions_mixed() if r.get("query_type") == "single_hop"]
    texts = sorted({_norm(r["question"]) for r in sh})

    prior = {}
    if QUALITY_PATH.exists():
        cached = json.loads(QUALITY_PATH.read_text())
        if cached.get("prompt_version") == PROMPT_VERSION:
            prior = cached.get("verdicts_by_text", {})
    todo = [t for t in texts if t not in prior]
    if args.limit:
        todo = todo[: args.limit]
    print(f"single-hop records: {len(sh)} | unique texts: {len(texts)} | cached: {len(prior)} | to judge: {len(todo)}")

    verdicts = dict(prior)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_text, t): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            verdicts[t] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  judged {done}/{len(todo)} ...")

    n_false = sum(1 for v in verdicts.values() if not v["standalone"])
    n_err = sum(1 for v in verdicts.values() if str(v["reason"]).startswith("JUDGE_ERROR"))
    out = {
        "model": config.FAST_MODEL,
        "prompt_version": PROMPT_VERSION,
        "n_texts": len(verdicts),
        "n_not_standalone": n_false,
        "n_judge_errors": n_err,
        "note": "Raw conservative LLM verdicts per UNIQUE question text. prepare_mixed applies the "
        "AND-gate (drop only if standalone=false AND no capitalized proper-noun anchor).",
        "verdicts_by_text": verdicts,
    }
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote -> {QUALITY_PATH}")
    print(f"not-standalone (raw LLM): {n_false}/{len(verdicts)}  | judge errors: {n_err}")
    if n_false:
        print("\nsample flagged (raw, before AND-gate):")
        shown = 0
        for t, v in verdicts.items():
            if not v["standalone"] and not str(v["reason"]).startswith("JUDGE_ERROR"):
                print(f"  - {v['reason'][:42]:44s} | {t}")
                shown += 1
                if shown >= 30:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
