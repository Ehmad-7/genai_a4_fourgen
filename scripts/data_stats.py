"""Table 1 helper: mean number of relevant abstracts per BioASQ test question (questions with a relevant abstract
in the corpus, as returned by starter/bioasq.py). Writes results/data_stats.json.   python scripts/data_stats.py"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from bioasq import load_corpus, load_questions

corpus = load_corpus()
qs = load_questions("test", corpus)
mean_rel = sum(len(q["relevant_ids"]) for q in qs) / len(qs)
stats = {"test_questions": len(qs), "mean_relevant_per_question": mean_rel,
         "max_relevant_per_question": max(len(q["relevant_ids"]) for q in qs)}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(stats, open(ROOT / "results" / "data_stats.json", "w"), indent=2)
print(stats)
