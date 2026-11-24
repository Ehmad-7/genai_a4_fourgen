"""Fill the {{...}} numbers in README.md from the result files (run after data_stats.py and score_human_rag.py).
python scripts/fill_readme.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
values = {}
f = ROOT / "results" / "data_stats.json"
if f.exists():
    values["{{MEAN_REL}}"] = f"{json.load(open(f))['mean_relevant_per_question']:.2f}"
f = ROOT / "results" / "rag_human_summary.json"
if f.exists():
    h = json.load(open(f))
    values["{{RAG_SUPPORTED}}"] = f"{h['member1_supported']} of {h['total']}"
    values["{{RAG_CORRECT}}"] = f"{h['member1_correct']} of {h['total']}"
for token, v in values.items():
    text = text.replace(token, v)
readme.write_text(text, encoding="utf-8")
left = [t for t in ("{{MEAN_REL}}", "{{RAG_SUPPORTED}}", "{{RAG_CORRECT}}") if t in text]
print("filled:", ", ".join(values) or "nothing", "| still missing:", ", ".join(left) or "none")
