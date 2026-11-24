"""Part D: cosine-similarity search. With normalised vectors cosine = dot product, so one matmul per batch of
questions. An abstract's score = the highest score of its chunks."""
import numpy as np
import torch


class Index:
    def __init__(self, emb, chunk_doc_ids, device="cpu"):
        self.device = device
        self.emb = torch.as_tensor(emb, device=device)                       # (N chunks, 384)
        self.doc_ids, inv = np.unique(np.asarray(chunk_doc_ids), return_inverse=True)
        self.chunk_pos = torch.as_tensor(inv, dtype=torch.long, device=device)   # chunk -> abstract position
        self.n_docs = len(self.doc_ids)

    @torch.no_grad()
    def search_docs(self, q, k=20, bs=64):
        """q (Q, 384) normalised -> list (one per question) of the top-k abstract ids, best first."""
        q = torch.as_tensor(q, device=self.device)
        out = []
        for s in range(0, len(q), bs):
            scores = q[s : s + bs] @ self.emb.T                              # (b, N) cosine similarities
            doc_scores = torch.full((scores.size(0), self.n_docs), -2.0, device=self.device)
            doc_scores = doc_scores.scatter_reduce(1, self.chunk_pos.expand(scores.size(0), -1), scores,
                                                   reduce="amax", include_self=True)   # max over chunks
            top = doc_scores.topk(k, dim=1).indices.cpu().numpy()
            out.extend(self.doc_ids[top].tolist())
        return out

    @torch.no_grad()
    def search_chunks(self, q, k=5):
        """Top-k CHUNKS (not abstracts) for each question: (chunk indices, scores)."""
        scores = torch.as_tensor(q, device=self.device) @ self.emb.T
        s, i = scores.topk(k, dim=1)
        return i.cpu().numpy(), s.cpu().numpy()
