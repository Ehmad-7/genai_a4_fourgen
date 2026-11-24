"""Part D: retrieval evaluation on all test questions: Hit@k, Recall@k (k = 1, 5, 10), MRR@10, Recall@k curve (k = 1..20).
Run from repo root: python part_d_rag/evaluate.py"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bioasq import load_corpus, load_questions
from part_d_rag.chunking import build_chunks
from part_d_rag.index import Embedder
from part_d_rag.search import Index
from part_d_rag.metrics import evaluate_rankings, recall_at_k

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="", help="'smoke_' to use the smoke-test index")
args = ap.parse_args()

corpus = load_corpus()
questions = load_questions("test", corpus)
texts, ids = build_chunks(corpus)
emb = np.load(ROOT / "weights" / f"{args.prefix}rag_index.npy")
chunk_ids = np.load(ROOT / "weights" / f"{args.prefix}rag_chunk_doc_ids.npy")
if args.prefix:                                           # smoke index covers only the first abstracts
    keep = set(chunk_ids.tolist())
    questions = [dict(q, relevant_ids=[i for i in q["relevant_ids"] if i in keep]) for q in questions]
    questions = [q for q in questions if q["relevant_ids"]]
else:
    assert len(emb) == len(texts) and np.array_equal(chunk_ids, np.array(ids)), "index does not match the chunks"
print(f"{len(questions)} test questions, {len(emb)} chunks, {len(set(chunk_ids.tolist()))} abstracts")

device = "cuda" if torch.cuda.is_available() else "cpu"
q_emb = Embedder(device).encode([q["question"] for q in questions])
index = Index(emb, chunk_ids, device)
rankings = index.search_docs(q_emb, k=20)
rels = [q["relevant_ids"] for q in questions]

res = evaluate_rankings(rankings, rels)
for k, v in res.items():
    print(f"{k:10s} {v:.4f}")

curve = [sum(recall_at_k(r, rel, k) for r, rel in zip(rankings, rels)) / len(rankings) for k in range(1, 21)]
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results" / f"{args.prefix}rag_recall_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["k", "recall"]); w.writerows([[k, v] for k, v in zip(range(1, 21), curve)])
plt.figure(figsize=(5.5, 3.8))
plt.plot(range(1, 21), curve, marker="o", ms=3)
plt.xlabel("k"); plt.ylabel("Recall@k"); plt.title("Retrieval: Recall@k"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(ROOT / "results" / "figures" / f"{args.prefix}rag_recall_at_k.png", dpi=150)

# questions where no relevant abstract is in the top 10 (for the analysis)
fails = [dict(qid=q["qid"], question=q["question"], relevant_ids=q["relevant_ids"], top10=r[:10])
         for q, r in zip(questions, rankings) if not set(r[:10]) & set(q["relevant_ids"])]
print(f"retrieval failures (no relevant abstract in top 10): {len(fails)} of {len(questions)}")
json.dump({"metrics": res, "n_questions": len(questions), "n_chunks": int(len(emb)), "failures": fails},
          open(ROOT / "results" / f"{args.prefix}rag_retrieval.json", "w"), indent=2)
json.dump([dict(qid=q["qid"], ranked=r) for q, r in zip(questions, rankings)],
          open(ROOT / "results" / f"{args.prefix}rag_rankings.json", "w"))
for f_ in fails[:3]:
    print("\nQ:", f_["question"], "\n relevant:", f_["relevant_ids"][:3], "\n top-1 retrieved id:", f_["top10"][0])
    print(" relevant abstract starts:", corpus[f_["relevant_ids"][0]][:200].replace("\n", " "))
    print(" top-1 retrieved starts:  ", corpus[f_["top10"][0]][:200].replace("\n", " "))
