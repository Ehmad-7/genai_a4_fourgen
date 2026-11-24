"""Part D: embed every chunk with all-MiniLM-L6-v2 (via transformers, NOT sentence-transformers) and save the index.
Mean pooling over the real tokens (attention mask), then L2-normalise. Truncate at 256 tokens.
Run from repo root: python part_d_rag/index.py"""
import argparse, json, sys, time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        self.model = AutoModel.from_pretrained(EMB_MODEL).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, bs=128, max_len=256):
        """-> float32 array (len(texts), 384), every row has L2 norm 1. Questions are encoded the same way."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths together = less padding
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            batch = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt").to(self.device)
            h = self.model(**batch).last_hidden_state                    # (B, L, 384)
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)        # 1 = real token, 0 = padding
            pooled = (h * m).sum(dim=1) / m.sum(dim=1)                   # mean over the real tokens only
            out[idx] = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        return out


if __name__ == "__main__":
    from bioasq import load_corpus
    from part_d_rag.chunking import build_chunks

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N abstracts (smoke test)")
    args = ap.parse_args()
    prefix = "smoke_" if args.limit else ""
    (ROOT / "weights").mkdir(exist_ok=True)

    corpus = load_corpus()
    if args.limit:
        corpus = dict(islice(corpus.items(), args.limit))
    texts, ids = build_chunks(corpus)
    print(f"{len(corpus)} abstracts -> {len(texts)} chunks")

    emb = Embedder()
    print("device:", emb.device)
    t0 = time.time()
    E = emb.encode(texts)
    elapsed = time.time() - t0
    norms = np.linalg.norm(E, axis=1)
    print(f"embedded in {elapsed:.0f}s; shape {E.shape}; norm min {norms.min():.6f} max {norms.max():.6f}")

    np.save(ROOT / "weights" / f"{prefix}rag_index.npy", E)
    np.save(ROOT / "weights" / f"{prefix}rag_chunk_doc_ids.npy", np.array(ids, dtype=np.int64))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"abstracts": len(corpus), "chunks": len(texts), "seconds": elapsed, "device": emb.device},
              open(ROOT / "results" / f"{prefix}rag_index_stats.json", "w"), indent=2)
    print("saved weights/rag_index.npy and weights/rag_chunk_doc_ids.npy")
