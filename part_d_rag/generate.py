"""Part D generation: top-5 CHUNKS -> numbered prompt -> Qwen2.5-1.5B-Instruct (fp16, chat template, greedy, <= 256 new tokens).
Evaluates the first 100 test questions: ROUGE-L F, citation precision, "I don't know" rate.
Run from repo root: python part_d_rag/generate.py        (smoke test: --n 3 --prefix smoke_)"""
import re

GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IDK_TEXT = "I don't know based on the provided passages."
SYSTEM = ("You are a biomedical research assistant. Answer the question using ONLY the numbered passages provided. "
          "After each statement, cite the passage number(s) you used in square brackets, for example [1] or [2][3]. "
          f"If the passages do not contain the answer, reply exactly: \"{IDK_TEXT}\" "
          "Do not use outside knowledge.")

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_IDK = re.compile(r"(don['’]t know|do not know|not contain|does not (?:provide|mention|specify|say)|"
                  r"no (?:relevant )?information|cannot (?:be )?(?:answer|determine)|not (?:enough|sufficient) information)", re.I)


def build_messages(question, passages):
    """passages: list of 5 chunk texts -> chat messages (passages numbered [1]..[5])."""
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    user = f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer using only the passages and cite passage numbers."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_citations(answer):
    """Distinct passage numbers cited in the answer, e.g. 'x [1][3] and [2, 3]' -> [1, 2, 3]."""
    nums = set()
    for m in _CITE.findall(answer):
        nums.update(int(x) for x in re.split(r"\s*,\s*", m))
    return sorted(nums)


def strip_citations(answer):
    return re.sub(r"\s+", " ", _CITE.sub("", answer)).strip()


def is_idk(answer):
    """True if the model says the passages do not contain the answer."""
    return bool(_IDK.search(answer))


class Generator:
    def __init__(self, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        try:
            model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float16)
        except TypeError:                                   # older transformers
            model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16)
        self.model = model.to(self.device).eval()

    def generate(self, messages, max_new_tokens=256):
        torch = self.torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)   # greedy
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    import argparse, json, random, sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "starter"))

    import numpy as np
    import torch
    from rouge_score import rouge_scorer
    from bioasq import load_corpus, load_questions
    from part_d_rag.chunking import build_chunks
    from part_d_rag.index import Embedder
    from part_d_rag.search import Index

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    res_dir = ROOT / "results"
    ans_path = res_dir / f"{args.prefix}rag_answers.jsonl"

    corpus = load_corpus()
    questions = load_questions("test", corpus)[: args.n]
    texts, ids = build_chunks(corpus)
    emb = np.load(ROOT / "weights" / "rag_index.npy")
    chunk_ids = np.load(ROOT / "weights" / "rag_chunk_doc_ids.npy")
    assert len(emb) == len(texts) and np.array_equal(chunk_ids, np.array(ids)), "index does not match the chunks"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_emb = Embedder(device).encode([q["question"] for q in questions])
    index = Index(emb, chunk_ids, device)
    top_idx, top_scores = index.search_chunks(q_emb, k=5)
    gen = Generator(device)

    done = {}
    if ans_path.exists():                                   # resume after an interrupted run
        for line in open(ans_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["qid"]] = r
    with open(ans_path, "a", encoding="utf-8") as f:
        for n, (q, idx) in enumerate(zip(questions, top_idx)):
            if q["qid"] in done:
                continue
            passages = [texts[i] for i in idx]
            answer = gen.generate(build_messages(q["question"], passages))
            cited = parse_citations(answer)
            rel = set(q["relevant_ids"])
            rec = {"qid": q["qid"], "question": q["question"], "reference": q["answer"], "answer": answer,
                   "relevant_ids": q["relevant_ids"],
                   "passages": [{"n": j + 1, "doc_id": int(chunk_ids[i]), "relevant": int(chunk_ids[i]) in rel,
                                 "text": texts[i]} for j, i in enumerate(idx)],
                   "cited": cited, "idk": is_idk(answer)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            done[q["qid"]] = rec
            if (n + 1) % 10 == 0:
                print(f"{n + 1}/{len(questions)} answered", flush=True)

    # ---- evaluation ----
    recs = [done[q["qid"]] for q in questions]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for r in recs:                                          # ROUGE-L F on the answer text without [n] markers
        r["rougeL"] = scorer.score(r["reference"], strip_citations(r["answer"]))["rougeL"].fmeasure
        valid = [c for c in r["cited"] if 1 <= c <= 5]
        r["cited_correct"] = [c for c in valid if r["passages"][c - 1]["relevant"]]
        r["context_has_relevant"] = any(p["relevant"] for p in r["passages"])
    n_cit = sum(len(r["cited"]) for r in recs)
    n_ok = sum(len(r["cited_correct"]) for r in recs)
    non_idk = [r for r in recs if not r["idk"]]
    summary = {
        "n_questions": len(recs),
        "rougeL_f": float(np.mean([r["rougeL"] for r in recs])),
        "rougeL_f_excluding_idk": float(np.mean([r["rougeL"] for r in non_idk])) if non_idk else None,
        "citation_precision": n_ok / n_cit if n_cit else None,
        "n_citations": n_cit, "n_correct_citations": n_ok,
        "answers_without_citation": sum(1 for r in recs if not r["cited"]),
        "idk_rate": sum(r["idk"] for r in recs) / len(recs),
        "context_has_relevant_chunk": sum(r["context_has_relevant"] for r in recs) / len(recs),
    }
    json.dump(summary, open(res_dir / f"{args.prefix}rag_generation.json", "w"), indent=2)
    with open(ans_path, "w", encoding="utf-8") as f:       # rewrite with the computed fields
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in summary.items():
        print(f"{k:28s} {v if not isinstance(v, float) else round(v, 4)}")

    # ---- sheet for the human evaluation: 20 answers, fixed seed ----
    pick = sorted(random.Random(0).sample(range(len(recs)), min(20, len(recs))))
    lines = ["# RAG human evaluation: mark each answer `supported` (everything it says is in the passages) "
             "and `correct` (agrees with the reference). Write 1/0 in results/rag_human_labels.csv\n"]
    for k, i in enumerate(pick, start=1):
        r = recs[i]
        lines += [f"\n---\n## Item {k} (qid {r['qid']})\n**Question:** {r['question']}\n",
                  f"**Reference answer:** {r['reference']}\n", f"**Model answer:** {r['answer']}\n", "**Passages:**\n"]
        lines += [f"[{p['n']}] (abstract {p['doc_id']}) {p['text']}\n" for p in r["passages"]]
    (res_dir / f"{args.prefix}rag_human_eval_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    with open(res_dir / f"{args.prefix}rag_human_labels.csv", "w", encoding="utf-8") as f:
        f.write("item,qid,supported,correct\n")
        for k, i in enumerate(pick, start=1):
            f.write(f"{k},{recs[i]['qid']},,\n")
    print("saved rag_answers.jsonl, rag_generation.json, rag_human_eval_sheet.md, rag_human_labels.csv")
