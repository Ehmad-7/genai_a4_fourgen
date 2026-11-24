"""Part D: add your human labels (20 answers) to results/human_eval.csv. Fill results/rag_human_labels.csv first
(supported = everything the answer says is in the passages; correct = agrees with the reference; write 1 or 0).
Run from repo root: python part_d_rag/score_human_rag.py"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results"
labels = list(csv.DictReader(open(res / "rag_human_labels.csv", encoding="utf-8")))
bad = [r["item"] for r in labels if r["supported"].strip() not in ("0", "1") or r["correct"].strip() not in ("0", "1")]
if bad:
    sys.exit(f"Items with a missing or invalid label (use 1 or 0): {bad}")

hum = list(csv.DictReader(open(res / "human_eval.csv", encoding="utf-8")))
keep = [c for c in hum[0].keys() if not c.startswith("rag_")]
new = ["rag_item", "rag_qid", "rag_member1_supported", "rag_member1_correct",
       "rag_member2_supported", "rag_member2_correct"]
for i, row in enumerate(hum):
    row.pop("rag_item", None)
    for c in new:
        row[c] = ""
    if i < len(labels):
        l = labels[i]
        row.update(rag_item=l["item"], rag_qid=l["qid"], rag_member1_supported=l["supported"].strip(),
                   rag_member1_correct=l["correct"].strip(), rag_member2_supported="NA", rag_member2_correct="NA")
with open(res / "human_eval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keep + new)
    w.writeheader()
    for row in hum:
        w.writerow({k: row.get(k, "") for k in keep + new})
sup = sum(int(l["supported"]) for l in labels)
cor = sum(int(l["correct"]) for l in labels)
json.dump({"member1_supported": sup, "member1_correct": cor, "total": len(labels), "member2": "NA (solo)"},
          open(res / "rag_human_summary.json", "w"), indent=2)
print(f"member 1: supported {sup}/{len(labels)}, correct {cor}/{len(labels)}   (member 2: N/A, solo)")
print("added the RAG columns to results/human_eval.csv")
