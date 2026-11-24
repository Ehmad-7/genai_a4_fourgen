"""Part D correctness checks: chunks <= 150 words (30-word overlap), embedding norms = 1, padding ignored in pooling,
and a perfect retriever gives Hit@k = Recall@k = MRR@10 = 1.   python part_d_rag/checks.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
from bioasq import load_corpus
from part_d_rag.chunking import chunk_words, build_chunks
from part_d_rag.metrics import evaluate_rankings

# 1) chunking
corpus = load_corpus()
texts, ids = build_chunks(corpus)
lens = np.array([len(t.split()) for t in texts])
assert lens.max() <= 150, "a chunk has more than 150 words"
print(f"OK: {len(corpus)} abstracts -> {len(texts)} chunks; max words in a chunk = {lens.max()}, mean {lens.mean():.1f}")
multi = 0
for text in corpus.values():
    ch = chunk_words(text)
    if not ch:
        continue
    for a, b in zip(ch, ch[1:]):
        assert a.split()[-30:] == b.split()[:30], "consecutive chunks must share 30 words"
    multi += len(ch) > 1
    assert ch[-1].split()[-1] == text.split()[-1], "the last chunk must end at the end of the abstract"
print(f"OK: consecutive chunks share exactly 30 words ({multi} abstracts have more than one chunk)")

# 2) a perfect retriever (relevant ids first) -> every metric is 1
rng = np.random.default_rng(0)
rankings, rels = [], []
for size in [1, 1, 2, 3, 7, 12, 25]:
    rel = [int(x) for x in rng.choice(100000, size=size, replace=False)]
    others = [int(x) for x in rng.choice(np.arange(100000, 200000), size=30, replace=False)]
    rankings.append(rel + others)
    rels.append(rel)
res = evaluate_rankings(rankings, rels)
assert all(abs(v - 1.0) < 1e-12 for v in res.values()), res
print("OK: perfect retriever ->", {k: round(v, 3) for k, v in res.items()})

# 3) embeddings: unit norm; padding must not change a vector
idx_path = ROOT / "weights" / "rag_index.npy"
if idx_path.exists():
    E = np.load(idx_path)
    norms = np.linalg.norm(E, axis=1)
    assert E.shape == (len(texts), 384) and np.abs(norms - 1).max() < 1e-4
    print(f"OK: index {E.shape}, every embedding has norm 1 (min {norms.min():.6f}, max {norms.max():.6f})")
else:
    print("(index not built yet: skipped the embedding-norm check, rerun after part_d_rag/index.py)")

from part_d_rag.index import Embedder
emb = Embedder("cpu")
short = "Is capmatinib effective for glioblastoma?"
alone = emb.encode([short])[0]
batched = emb.encode([short, " ".join(texts[0].split() * 3)])
diff = np.abs(batched[0] - alone).max()                  # batched[0] = the short text, now padded
assert diff < 1e-4, "padding changed the embedding"
print(f"OK: embedding of a text is the same alone and padded in a batch (max diff {diff:.1e}); norm {np.linalg.norm(alone):.6f}")
