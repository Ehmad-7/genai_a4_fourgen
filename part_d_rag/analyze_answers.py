"""Part D: look at the 100 generated answers. Prints the numbers for analysis questions 6/7, and writes a DRAFT of
results/samples.md (5 RAG questions: >= 1 retrieval failed, >= 1 'I don't know'). READ the draft before you submit it.
Run from repo root: python part_d_rag/analyze_answers.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results"
recs = [json.loads(l) for l in open(res / "rag_answers.jsonl", encoding="utf-8")]
ret = json.load(open(res / "rag_retrieval.json"))
fail_q = {f["qid"] for f in ret["failures"]}                 # no relevant abstract in the top 10 (retrieval metric)

n = len(recs)
print(f"{n} answers")
print("\nidk  x  does the top-5 context contain a relevant abstract?")
for ctx in (True, False):
    for idk in (False, True):
        c = sum(1 for r in recs if r["context_has_relevant"] == ctx and r["idk"] == idk)
        print(f"  context has relevant = {str(ctx):5s} | {'I-dont-know' if idk else 'answered   '} : {c}")
unc = [r for r in recs if not r["cited"]]
print(f"\nanswers without any [n] citation: {len(unc)} (of which flagged I-dont-know: {sum(r['idk'] for r in unc)})")
fails = [r for r in recs if r["qid"] in fail_q]
print(f"retrieval failures (no relevant abstract in top 10) among these {n} questions: {len(fails)}; "
      f"model said I-dont-know on {sum(r['idk'] for r in fails)} of them, answered the other {sum(not r['idk'] for r in fails)}")

print("\nanswers flagged as I-dont-know (check these by eye: is the flag right?):")
for r in [r for r in recs if r["idk"]]:
    print(f"  qid {r['qid']}: {r['answer'][:140]!r}")

print("\nFIRST THREE RETRIEVAL FAILURES (analysis question 6):")
for r in fails[:3]:
    print(f"\n- Q: {r['question']}\n  reference: {r['reference'][:300]}\n  model answer: {r['answer'][:500]}")
    print(f"  I-dont-know: {r['idk']} | cited passages: {r['cited']} | relevant abstracts: {r['relevant_ids'][:3]}")
    print("  retrieved abstract ids:", [p["doc_id"] for p in r["passages"]])


def block(r, why):
    cited = [p for p in r["passages"] if p["n"] in r["cited"]] or r["passages"][:2]
    lines = [f"## {r['question']}", f"*({why})*", "",
             f"**Reference answer:** {r['reference']}", "", f"**Model answer:** {r['answer']}", "",
             f"**Relevant abstract ids (ground truth):** {r['relevant_ids']}",
             f"**Retrieved chunks (passage -> abstract id, relevant?):** "
             + ", ".join(f"[{p['n']}] -> {p['doc_id']} ({'yes' if p['relevant'] else 'no'})" for p in r["passages"]), ""]
    lines.append("**Cited passages**" if r["cited"] else "**No citation in the answer; the two top passages:**")
    for p in cited:
        t = p["text"] if len(p["text"]) < 600 else p["text"][:600] + " ..."
        lines.append(f"- [{p['n']}] (abstract {p['doc_id']}) {t}")
    return "\n".join(lines) + "\n"


used = set()
def pick(pred):
    for r in recs:
        if r["qid"] not in used and pred(r):
            used.add(r["qid"]); return r
    return None

chosen = [(pick(lambda r: r["qid"] in fail_q and not r["idk"]) or pick(lambda r: r["qid"] in fail_q),
           "retrieval failed: no relevant abstract in the top 10"),
          (pick(lambda r: r["idk"]), "the model said it does not know")]
for _ in range(3):
    chosen.append((pick(lambda r: r["context_has_relevant"] and not r["idk"] and r["cited_correct"]),
                   "retrieval worked and the answer cites a relevant passage"))
out = ["# Qualitative RAG samples (draft: check every item before submitting)\n"]
for r, why in chosen:
    if r is not None:
        out.append(block(r, why))
(res / "samples.md").write_text("\n".join(out), encoding="utf-8")
print("\nwrote results/samples.md with", sum(r is not None for r, _ in chosen), "questions")
