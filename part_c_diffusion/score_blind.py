"""Score the blind recognisability test (solo: member 1 only).
Run AFTER you filled results/blind_test/guesses.csv without having seen the labels.
Writes results/human_eval.csv.   python part_c_diffusion/score_blind.py"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "results" / "blind_test"


def norm(s):
    return " ".join(s.lower().split())


classes = [norm(l) for l in (d / "class_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
key = {int(r["id"]): norm(r["true_class"]) for r in csv.DictReader(open(d / "_key_do_not_open.csv", encoding="utf-8"))}
guess = {int(r["id"]): norm(r["guess"]) for r in csv.DictReader(open(d / "guesses.csv", encoding="utf-8"))}

empty = [i for i, g in guess.items() if not g]
unknown = [(i, g) for i, g in guess.items() if g and g not in classes]
if empty:
    sys.exit(f"Empty guesses for ids {empty}. Fill all 40 (make a best guess) before scoring.")
if unknown:
    sys.exit(f"These guesses are not in class_list.txt (check the spelling): {unknown}")

rows, correct = [], 0
for i in sorted(key):
    ok = guess[i] == key[i]
    correct += ok
    rows.append([i, key[i], guess[i], int(ok), "NA", "NA", "NA"])      # solo: member 2 / agreement not applicable
with open(ROOT / "results" / "human_eval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "true_class", "member1_guess", "member1_correct",
                "member2_guess", "member2_correct", "both_agreed"])
    w.writerows(rows)
json.dump({"member1_correct": correct, "total": len(rows), "member2": "NA (solo)", "both_agreed": "NA (solo)"},
          open(ROOT / "results" / "blind_test_summary.json", "w"), indent=2)

print(f"member 1 correct: {correct}/{len(rows)} ({100 * correct / len(rows):.1f}%)   chance level = 2.5%")
print("member 2 and 'both agreed': N/A (solo submission)")
print("wrong guesses:")
for i, k, g, ok, *_ in rows:
    if not ok:
        print(f"  #{i:2d}: guessed '{g}', true '{k}'")
print("saved results/human_eval.csv")
