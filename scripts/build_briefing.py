"""Generate briefing.html: the INTERNAL reference for the v2.5 build (post-Codex, honest).

Reframed after the Codex adversarial review: the headline is COST-EFFICIENT adaptive routing
(a Pareto point), not "beats every fixed policy." Leads with recall@3 / full_gold@3; MRR is
shown but labeled (first-gold, lenient on multi-hop). Restores the outer-eval-against-baseline
LIFT view, stratified by query type, so the multi-hop full_gold gain is visible. Shows the
full-workload selective accuracy (where the ladder does NOT win) honestly. Rates are rendered as
PERCENTAGES for readability. Written for IR-literate colleagues: domain terms kept, wording plain,
caveats unstacked. No em dashes.

Reads the corrected artifacts (headline_final_v25.json + abstention_study_v25 + signal_analysis_mixed
+ thresholds_mixed + mixed_manifest). Re-runnable:  python scripts/build_briefing.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"
OUT = REPO / "briefing.html"

meta = json.loads((REPO / "data" / "dataset_meta.json").read_text())
H = json.loads((ART / "headline_final_v25.json").read_text())
sig = json.loads((ART / "signal_analysis_mixed.json").read_text())
abst = json.loads((ART / "abstention_study_v25.json").read_text())
tgt = json.loads((ART / "targeted_stop_v25.json").read_text())["variants"]
jdoc = json.loads((ART / "judge_eval_v25.json").read_text())
jud, jci = jdoc["by_policy"], jdoc["ci_ladder"]
thr = json.loads((ART / "thresholds_mixed.json").read_text())

ov, bt = H["overall"], H["by_type"]
ans, sel, ci = H["answers"], H["selective_accuracy"], H["ci_vs"]
av, selsig = sig["auc_validation"], sig["selection"]
g, a = abst["gentle"], abst["autorater"]
AK = H["answer_k"]
BANNER = ('<div style="background:#fff3cd;border:1px solid #ffe69c;border-radius:8px;padding:11px 15px;margin:12px 0;font-size:14px">'
          '<b>PROVISIONAL build.</b> The numbers are placeholders carried from the prior run, for reviewing the TEXT and MESSAGING only. '
          'They refresh automatically when the corrected headline run finishes and this is rebuilt.</div>') if H.get("_provisional") else ""
FRONTIER = ["always_answer", "always_colbert", "always_rerank", "always_decompose", "ladder"]
PRETTY = {"always_answer": "hybrid baseline", "always_colbert": "always ColBERT",
          "always_rerank": "always cross-encoder", "always_decompose": "always decompose", "ladder": "ladder (adaptive)"}


def pct(x):
    # for accuracy / rate metrics (EM, F1, abstention, abstain rate, selective accuracy)
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x)


def num(x):
    # for retrieval-relevance metrics (recall@k, full_gold, MRR): read as a 0-1 number
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def dpct(x):
    if not isinstance(x, (int, float)):
        return str(x)
    cls = "pos" if x > 0.0005 else ("neg" if x < -0.0005 else "flat")
    return f'<span class="{cls}">{"+" if x >= 0 else ""}{x*100:.1f} pts</span>'


def dnum(x):
    # signed delta for retrieval-relevance metrics, as a 0-1 number
    if not isinstance(x, (int, float)):
        return str(x)
    cls = "pos" if x > 0.0005 else ("neg" if x < -0.0005 else "flat")
    return f'<span class="{cls}">{"+" if x >= 0 else ""}{x:.3f}</span>'


def ci_flag(c):
    lo, hi = c["ci95"]
    if lo > 0:
        return "clears 0"
    if hi < 0:
        return "clears 0 (negative)"
    return "crosses 0"


# multi-hop full_gold lift, the per-slice headline gain
mh = bt["multi_hop"]
mh_ladder_lift = mh["ladder"]["full_gold@3"] - mh["always_answer"]["full_gold@3"]
mh_dec_lift = mh["always_decompose"]["full_gold@3"] - mh["always_answer"]["full_gold@3"]
cost_ratio = ov["ladder"]["llm_calls"] / ov["always_decompose"]["llm_calls"] if ov["always_decompose"]["llm_calls"] else 0


def frontier_rows(block):
    out = ""
    for p in FRONTIER:
        v = block[p]
        cls = "badge-green" if p == "ladder" else "badge-grey"
        out += (f"<tr><td><b>{PRETTY[p]}</b></td><td>{num(v['recall@3'])}</td><td>{num(v['full_gold@3'])}</td>"
                f"<td>{num(v['mrr_first'])}</td><td>{v['llm_calls']:.2f}</td>"
                f"<td><span class='badge {cls}'>{'ADAPTIVE' if p=='ladder' else 'fixed'}</span></td></tr>")
    return out


def lift_rows():
    # ladder & always_decompose vs the always_answer baseline, by query type
    out = ""
    for qt, blk in (("single-hop", bt["single_hop"]), ("multi-hop", bt["multi_hop"]), ("overall", ov)):
        b = blk["always_answer"]; l = blk["ladder"]; d = blk["always_decompose"]
        out += (f"<tr><td><b>{qt}</b></td>"
                f"<td>{num(b['full_gold@3'])}</td><td>{num(l['full_gold@3'])} ({dnum(l['full_gold@3']-b['full_gold@3'])})</td>"
                f"<td>{num(d['full_gold@3'])} ({dnum(d['full_gold@3']-b['full_gold@3'])})</td>"
                f"<td>{num(b['recall@3'])}</td><td>{num(l['recall@3'])} ({dnum(l['recall@3']-b['recall@3'])})</td></tr>")
    return out


def answer_rows():
    out = ""
    for slice_ in ("overall", "single_hop", "multi_hop"):
        b, l, d = jud["always_answer"][slice_], jud["ladder"][slice_], jud["always_decompose"][slice_]
        out += (f"<tr><td>{slice_.replace('_',' ')}</td><td>{pct(b['judge'])}</td>"
                f"<td><b>{pct(l['judge'])}</b> ({dpct(l['judge']-b['judge'])})</td><td>{pct(d['judge'])}</td></tr>")
    return out


def sel_rows():
    out = ""
    for p in ("always_answer", "ladder", "always_decompose"):
        if p in sel:
            v = sel[p]
            out += (f"<tr><td><b>{PRETTY[p]}</b></td><td>{pct(v['selective_accuracy'])}</td>"
                    f"<td>{pct(v['answerable_em'])}</td><td>{pct(v['abstain_rate_unans'])}</td></tr>")
    return out


def stop_rows():
    out = ""
    for k, label, note in [
        ("gentle", "gentle stop (default)", "Generator self-abstains. Keeps the most answers; abstains less once the ladder escalates an unanswerable (more context, fewer self-abstentions)."),
        ("autorater", "haiku sufficiency autorater", "Reads whether the passages answer the question. Recovers abstention but over-abstains on answerables, so EM drops."),
    ]:
        m = abst[k]
        out += (f"<tr><td><b>{label}</b><br><span class='sub'>{note}</span></td>"
                f"<td>{pct(m['em'])}</td><td>{pct(m['abstention_f1'])}</td><td>{pct(m['abstain_rate_unans'])}</td>"
                f"<td>{pct(m['false_answer_unans'])}</td><td>{pct(m['false_stop_answerable'])}</td></tr>")
    return out


# Full catalog of confidence signals we tested (kept for teammates: useful elsewhere).
SIGNALS = [
    ("dense_variance", "spread, raw dense", "pstdev of the top-K raw dense cosine scores",
     "The gate signal here. Generalizes as a confidence/peakedness reading on any dense retriever."),
    ("dense_gap", "spread, raw dense", "rank-1 minus rank-K raw dense cosine",
     "Same axis as dense_variance (about 0.99 correlated), so we kept one."),
    ("score_variance", "spread, fused", "pstdev of the top-K fused (RRF) scores",
     "The fused-score spread; separated the multi-hop slice independently, so kept alongside dense spread."),
    ("confidence_gap", "spread, fused", "rank-1 minus rank-K fused score",
     "Fused spread; redundant twin of score_variance. Which of the pair wins flips by dataset."),
    ("max_score", "height, fused", "the top-1 fused score",
     "A classic query-performance-prediction signal when the retriever emits calibrated, comparable scores (e.g. a single dense retriever)."),
    ("evidence_coverage", "coverage", "fraction of the question's named entities present in the top-K",
     "Single-hop, entity-lookup, or lexical-mismatch retrieval, where a missing named entity is a visible failure. Cheap, no LLM."),
    ("retriever_divergence", "agreement", "dense vs miniCOIL rank disagreement (pre-fusion)",
     "An uncertainty signal across retrieval stacks: when two retrievers disagree, retrieval is shakier. Below the AUC bar here."),
]


def signal_rows():
    out = ""
    for k, axis, what, useful in SIGNALS:
        auc = av.get(k)
        kept = k in selsig["weakness_signals"]
        badge = "badge-green" if kept else "badge-grey"
        verdict = "KEPT (gates the loop)" if kept else "catalog only"
        out += (f"<tr><td><span class='mono'>{k}</span><br><span class='sub'>{axis} &middot; {what}</span></td>"
                f"<td>{auc:.3f}</td><td><span class='badge {badge}'>{verdict}</span></td>"
                f"<td style='text-align:left'>{useful}</td></tr>")
    return out


def reuse_line():
    r = H.get("test_reuse", {})
    parts = []
    if "v2" in r:
        parts.append(f"{r['v2']['multi_in_prior_ans']} multi-hop + {r['v2']['unans_in_prior_unans']} unanswerable + {r['v2']['single_src_in_prior_ans']} single-hop sources also in the v2 test set")
    return "; ".join(parts) if parts else "see mixed_manifest.json"


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-Correcting Agentic Retrieval Loops (v2.5) - Internal Briefing</title>
<style>
:root {{ --ink:#1a1a22; --muted:#5b6270; --line:#e5e7eb; --bg:#fafafb; --card:#fff;
  --accent:#d6336c; --accent2:#1c7ed6; --pos:#138a52; --neg:#c92a2a; --amber:#b8860b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
.wrap {{ display:grid; grid-template-columns:240px 1fr; max-width:1180px; margin:0 auto; }}
nav {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto; padding:28px 18px; border-right:1px solid var(--line); font-size:14px; }}
nav h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 10px; }}
nav a {{ display:block; color:var(--muted); text-decoration:none; padding:5px 0; }}
nav a:hover {{ color:var(--accent); }}
main {{ padding:40px 48px 100px; max-width:900px; }}
h1 {{ font-size:30px; line-height:1.2; margin:0 0 6px; }}
h2 {{ font-size:23px; margin:44px 0 14px; padding-top:10px; border-top:1px solid var(--line); }}
h3 {{ font-size:17px; margin:24px 0 8px; }}
p, li {{ color:#23262e; }}
.sub {{ color:var(--muted); font-size:14px; }}
.tag {{ display:inline-block; background:#fdeef4; color:var(--accent); border:1px solid #f6c6da; border-radius:999px; padding:2px 11px; font-size:12.5px; font-weight:600; margin-right:6px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin:16px 0; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
.grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.kpi {{ text-align:center; }} .kpi .n {{ font-size:23px; font-weight:700; }} .kpi .l {{ font-size:12.5px; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; margin:10px 0; font-size:14px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
td:not(:first-child), th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pos {{ color:var(--pos); font-weight:600; }} .neg {{ color:var(--neg); font-weight:600; }} .flat {{ color:var(--muted); }}
.badge {{ font-size:11px; padding:2px 8px; border-radius:6px; font-weight:600; white-space:nowrap; }}
.badge-green {{ background:#e6f4ec; color:var(--pos); }} .badge-grey {{ background:#eef0f2; color:var(--muted); }}
.pipe {{ font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; background:#0f1117; color:#e6e6e6; border-radius:10px; padding:16px 18px; overflow:auto; }}
.pipe .hi {{ color:#ff7aa8; }} .pipe .di {{ color:#6cc0ff; }} .pipe .gr {{ color:#7bd88f; }}
.callout {{ border-left:4px solid var(--accent); background:#fff; padding:12px 16px; border-radius:0 8px 8px 0; margin:14px 0; }}
.callout.good {{ border-color:var(--pos); }} .callout.warn {{ border-color:var(--amber); }}
.mono {{ font-family:ui-monospace,Menlo,monospace; font-size:13px; background:#f2f3f5; padding:1px 5px; border-radius:4px; }}
ul.tight li {{ margin:4px 0; }}
.foot {{ color:var(--muted); font-size:13px; margin-top:50px; border-top:1px solid var(--line); padding-top:16px; }}
</style></head><body><div class="wrap">
<nav>
<h2>Contents</h2>
<a href="#tldr">TL;DR</a>
<a href="#what">What we built</a>
<a href="#data">The mixed workload</a>
<a href="#signal">The confidence signal</a>
<a href="#frontier">The cost/quality frontier</a>
<a href="#lift">Lift vs the baseline</a>
<a href="#answers">Answer quality</a>
<a href="#stop">The stop decision</a>
<a href="#lessons">Takeaways</a>
</nav>
<main>

<div><span class="tag">Internal briefing</span><span class="tag">v2.5</span><span class="tag">June 10, 2026 - SF</span><span class="tag">post-Codex review</span></div>
<h1>Self-Correcting Agentic Retrieval Loops (v2.5)</h1>
<p class="sub">A hands-on workshop on the method for finding the right retrieval-quality signals and corrective actions for your own stack. The agent reads its retrieval confidence and climbs a cost-escalation ladder (answer, ColBERT precision fix, decompose, stop) only as far as each query needs. Co-presented with Google. This page reports the honest results, including where the ladder does not win.</p>
{BANNER}

<h2 id="tldr">TL;DR</h2>
<div class="card grid4">
  <div class="kpi"><div class="n">{pct(cost_ratio)} <span class="sub">of cost</span></div><div class="l">ladder LLM cost vs always routing towards the costliest path, at similar answerable quality</div></div>
  <div class="kpi"><div class="n pos">{num(mh['always_answer']['full_gold@3'])} &rarr; {num(mh['ladder']['full_gold@3'])}</div><div class="l">full_gold@3 on difficult questions (baseline &rarr; ladder)</div></div>
  <div class="kpi"><div class="n pos">{pct(jud['always_answer']['overall']['judge'])} &rarr; {pct(jud['ladder']['overall']['judge'])}</div><div class="l">answer accuracy (gpt-5.5 semantic judge): baseline &rarr; ladder</div></div>
  <div class="kpi"><div class="n">{meta['n_corpus_docs']:,}</div><div class="l">passages in Qdrant</div></div>
</div>
<div class="callout"><b>The honest headline</b><br><br>
On a mixed workload, the adaptive ladder is <b>cost-efficient</b>: it reaches about the same answerable-retrieval quality as always-decompose for only <b>{pct(cost_ratio)} of the LLM cost</b>.<br><br>
<b>What it beats:</b> the hybrid baseline (retrieve and answer, no correction), clearly on the multi-hop slice.<br>
<b>Where it leads the whole field:</b> MRR {num(ov['ladder']['mrr_first'])} - it gets the right passage to the top better than any fixed policy, which is what counts when the LLM reads only the top few.<br>
<b>What it does NOT beat:</b> always-decompose on completeness (full_gold@3 {num(ov['always_decompose']['full_gold@3'])} vs the ladder's {num(ov['ladder']['full_gold@3'])}), if you pay its higher cost.<br>
<b>On the full workload:</b> with the gentle stop the ladder trails the baseline (escalation makes it answer unanswerables it should refuse); swap in an LLM sufficiency stop and it handles more of the whole workload correctly than the baseline ({pct(tgt['ladder_autorater_all']['selective_accuracy'])} vs {pct(tgt['baseline_hybrid_gentle']['selective_accuracy'])}), trading some answers for honest refusals (see the stop section).<br><br>
<b>The lesson is the method:</b> build the loop, measure on cost AND quality, and route only when routing pays, including the parts where the loop loses.</div>

<h2 id="what">What we built (the stack in one line)</h2>
<div class="pipe">query &rarr; dense (bge-base) + sparse (<span class="di">miniCOIL</span>) &rarr; Qdrant fuse (Query API, RRF) &nbsp;[baseline, no cross-encoder]<br>
&nbsp;&nbsp;&rarr; read the confidence signal on the RAW DENSE scores<br>
&nbsp;&nbsp;&rarr; <span class="hi">ladder</span>: confident &rarr; answer &middot; weak single-hop &rarr; <span class="di">ColBERT</span> &middot; weak multi-hop &rarr; <span class="gr">decompose (IRCoT)</span><br>
&nbsp;&nbsp;&rarr; answer from the focused top-{AK} &middot; sufficiency check &rarr; answer or stop</div>
<p class="sub">The agent answers from a focused top-{AK}, so ranking precision is the metric. ColBERT lives in a separate <span class="mono">musique_colbert</span> collection (Qdrant native multivector, MaxSim). Reusable modules in <span class="mono">src/</span>.</p>

<h2 id="data">The mixed workload</h2>
<p>MuSiQue-Full over one Qdrant corpus, recast as a mixed workload: single-hop lookups derived from the per-hop sub-questions, full multi-hop questions, and native unanswerables (about two-thirds single-hop, one-third multi-hop among answerables). Three disjoint splits: calibration sets the threshold, validation selects the signal and runs the policy comparison, test reports the result.</p>
<p class="sub"><b>Why we measure at the focused top-{AK}:</b> this is an agentic pipeline, so the LLM reads only the top few passages to answer. A correct passage sitting at rank 10 is of no use to it, so ranking precision (recall@1/@3), not recall@10, is the metric that matters. It is also where the corrective tiers have room to work: at top-10 the easy lookups are already about 98% solved, so there is nothing to fix.</p>

<h2 id="signal">The confidence signal</h2>
<p>The gate reads the <b>spread of the raw dense scores</b> (<span class="mono">dense_variance</span>): a peaked top means a confident retrieval. One finding worth carrying: read it on the raw dense scores, not the rank-fused score, which discards the spread. We selected <b>{', '.join(selsig['weakness_signals'])}</b>.</p>
<p class="sub">Reference for teammates: the full catalog we tested, scored by AUC (how well each separates good from weak retrieval; 0.5 = chance, 1.0 = perfect). Most were weak on this data but are useful on other stacks, which is why we keep the list. Skim or skip.</p>
<table>
<tr><th>Signal (axis &middot; what it reads)</th><th>AUC</th><th>Verdict</th><th style="text-align:left">Useful when (elsewhere)</th></tr>
{signal_rows()}
</table>

<h2 id="frontier">The cost/quality frontier (test, answerable)</h2>
<p class="sub">Metrics, plainly: <b>recall@3</b> = fraction of a question's needed passages that land in the top 3; <b>full_gold@3</b> = all of them in the top 3 (the strict bar); <b>LLM calls/q</b> = the cost. Higher quality, lower cost is better.</p>
<table>
<tr><th>Policy</th><th>recall@3</th><th>full_gold@3</th><th>MRR*</th><th>LLM calls/q</th><th></th></tr>
{frontier_rows(ov)}
</table>
<p class="sub">The two quality metrics measure different things: <b>MRR</b> is how high the right passage ranks (the ladder leads it, getting the right passage to the top best), and <b>full_gold@3</b> is whether ALL of a question's supporting passages are present (always-decompose leads it). For multi-hop, MRR counts the first supporting passage found, so read it as rank quality, not completeness. LLM-calls counts only decompose sub-query calls; ColBERT and the cross-encoder are model rescoring passes, not LLM calls.</p>
<div class="callout good"><b>Read this as a cost/quality tradeoff, not a single winner.</b> Always-decompose has the highest recall@3 ({num(ov['always_decompose']['recall@3'])}) and full_gold@3 ({num(ov['always_decompose']['full_gold@3'])}), at {ov['always_decompose']['llm_calls']:.2f} LLM calls/query. The ladder reaches {num(ov['ladder']['recall@3'])} / {num(ov['ladder']['full_gold@3'])} at {ov['ladder']['llm_calls']:.2f} calls/query, about {pct(cost_ratio)} of the cost. So the ladder buys nearly all of decompose's answerable quality far more cheaply by sending decomposition only where it helps. Nothing else is both cheaper and better at once: that is what makes the ladder a good tradeoff. It does not beat decompose on quality, and our significance test (below) confirms it beats the hybrid baseline, not decompose.</div>

<h2 id="lift">Lift vs the baseline (full_gold@3), by query type</h2>
<p class="sub">The overall lift is small because single-hop dominates the mix and is already strong. The real gain is on the multi-hop slice, where decomposition recovers the missing hop. This is the outer-eval-against-baseline view.</p>
<table>
<tr><th>Slice</th><th>baseline full_gold@3</th><th>ladder full_gold@3</th><th>decompose full_gold@3</th><th>baseline recall@3</th><th>ladder recall@3</th></tr>
{lift_rows()}
</table>
<div class="callout"><b>Where the gain lives:</b> on multi-hop, full_gold@3 goes from {num(mh['always_answer']['full_gold@3'])} (baseline) to {num(mh['ladder']['full_gold@3'])} (ladder, {dnum(mh_ladder_lift)}) and {num(mh['always_decompose']['full_gold@3'])} (always-decompose, {dnum(mh_dec_lift)}). Decomposition recovering the missing hop is the real win; it is concentrated in the multi-hop slice, so it washes out in the overall number. Significance vs the baseline (paired bootstrap, answerable): recall@3 {dnum(ci['always_answer']['recall@3']['lift'])} ({ci_flag(ci['always_answer']['recall@3'])}), full_gold@3 {dnum(ci['always_answer']['full_gold@3']['lift'])} ({ci_flag(ci['always_answer']['full_gold@3'])}). Vs always-decompose: recall@3 {dnum(ci['always_decompose']['recall@3']['lift'])} ({ci_flag(ci['always_decompose']['recall@3'])}).</div>

<h2 id="answers">Answer quality (semantic judge, gentle stop)</h2>
<p class="sub">Scored by a gpt-5.5 semantic judge (credits a correct answer regardless of phrasing), not exact match. Higher is better.</p>
<table>
<tr><th>Slice</th><th>baseline</th><th>ladder</th><th>always-decompose</th></tr>
{answer_rows()}
</table>
<p class="sub">The ladder answers competitively: {pct(jud['ladder']['overall']['judge'])} correct vs the baseline's {pct(jud['always_answer']['overall']['judge'])} (lift {dpct(jci['vs_baseline']['lift'])}, {ci_flag(jci['vs_baseline'])} - the fair judge also credits the baseline's correctly-phrased answers, so the gap is marginally short of significance). always-decompose answers best ({pct(jud['always_decompose']['overall']['judge'])}; the ladder trails it by {dpct(jci['vs_decompose']['lift'])}, {ci_flag(jci['vs_decompose'])}). As with retrieval, the real gain is on difficult questions: multi-hop {pct(jud['always_answer']['multi_hop']['judge'])} &rarr; {pct(jud['ladder']['multi_hop']['judge'])} (always-decompose {pct(jud['always_decompose']['multi_hop']['judge'])}).</p>

<h2 id="stop">The stop decision (a smaller, separate lever)</h2>
<p>Stopping is a separate decision from routing: not which fix to apply, but whether to answer at all or abstain. We use the <b>gentle stop</b> by default (the generator answers from its context, or says it lacks enough); it keeps answers. For workloads where you want the highest confidence and abstaining out of caution is fine, swap in an <b>LLM sufficiency check</b>: it reads whether the passages actually answer the question, catches far more unanswerables, and handles more of the full workload correctly, at the cost of occasionally refusing a question it could have answered.</p>
<table>
<tr><th>Stop method</th><th>catches unanswerables</th><th>over-refuses answerables</th><th>full-workload correct</th></tr>
<tr><td><b>gentle (default)</b></td><td>{pct(tgt['ladder_gentle']['abstain_unans'])}</td><td>{pct(tgt['ladder_gentle']['false_stop_ans'])}</td><td>{pct(tgt['ladder_gentle']['selective_accuracy'])}</td></tr>
<tr><td><b>LLM sufficiency check</b></td><td>{pct(tgt['ladder_autorater_all']['abstain_unans'])}</td><td>{pct(tgt['ladder_autorater_all']['false_stop_ans'])}</td><td>{pct(tgt['ladder_autorater_all']['selective_accuracy'])}</td></tr>
</table>

<h2 id="lessons">Takeaways</h2>
<ul class="tight">
<li><b>Easy single-hop IS the cost story, not a limitation.</b> The modest overall lift and the cost win are the same fact: single-hop queries are easy, so the ladder answers them cheaply at tier 1 and there is little to lift on them. You cannot have both a big single-hop lift and cheap single-hop; the realistic mix is the point.</li>
<li><b>The right substrate makes a weak signal strong.</b> Reading the spread on the raw dense scores (AUC about {av['dense_variance']:.2f}) separates good from weak retrieval far better than the same idea on the rank-fused score, which discards the spread.</li>
<li><b>Nothing transfers as a constant.</b> The threshold, the winning signal, and the winning rung are corpus-specific. The transferable asset is the method: build cheap signals, route only when routing pays, and measure on cost and quality. (One caveat to carry: decompose's lift is an upper bound, since its LLM sub-queries may already know the bridge entities, so recalibrate on your own corpus.)</li>
</ul>

<div class="foot">Generated from the corrected v2.5 artifacts in <span class="mono">artifacts/</span> by <span class="mono">scripts/build_briefing.py</span>.</div>
</main></div></body></html>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({len(HTML):,} bytes)")
