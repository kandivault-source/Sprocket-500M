"""Fetch risky PROMPTS from an existing open safety dataset.

WHY THIS SCRIPT EXISTS (locked project decision): Sprocket's safety refusals must
be trained against prompts written by someone else. We do NOT self-generate
harmful prompts and we do NOT reword prompts to slip past a safety filter. The
only thing we generate is the REFUSAL.

SOURCE: LibrAI/do-not-answer (arXiv:2308.13387), 939 prompts, **Apache-2.0**.

Apache-2.0 is the reason this dataset was chosen over the two obvious
alternatives. BeaverTails and PKU-SafeRLHF are both CC-BY-NC-4.0 -
non-commercial - which is a bad footing for a model published under a company.
do-not-answer is also prompts-only by construction ("curated and filtered to
consist only of prompts to which responsible language models do not answer"),
so nothing harmful is carried across; the response columns are ignored here.

Its taxonomy (risk_area / types_of_harm / specific_harms) is kept so the
generator can vary its handling by harm type - a self-harm prompt must get
warmth and a crisis line, not the flat decline a malware request gets.

    py scripts/fetch_safety_prompts.py
    -> data/safety/dna_prompts.jsonl
"""
import argparse
import io
import json
import os
import sys
import urllib.request

PARQUET = ("https://huggingface.co/datasets/LibrAI/do-not-answer/resolve/"
           "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
SOURCE = {
    "dataset": "LibrAI/do-not-answer",
    "url": "https://huggingface.co/datasets/LibrAI/do-not-answer",
    "license": "apache-2.0",
    "citation": "arXiv:2308.13387",
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/safety/dna_prompts.jsonl")
    a = ap.parse_args()

    import pandas as pd

    print(f"downloading {SOURCE['dataset']} ({SOURCE['license']}) ...")
    req = urllib.request.Request(PARQUET, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    print(f"  {len(raw)/1e6:.2f} MB")

    df = pd.read_parquet(io.BytesIO(raw))
    print(f"  {len(df):,} rows, columns: {list(df.columns)[:6]}...")

    # Prompts + taxonomy ONLY. The model-response columns are deliberately
    # dropped: we want other people's questions, not other models' answers.
    keep = ["id", "risk_area", "types_of_harm", "specific_harms", "question"]
    df = df[[c for c in keep if c in df.columns]]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            q = str(rec.get("question", "")).strip()
            if not q:
                continue
            f.write(json.dumps({
                "id": int(rec.get("id", n)),
                "question": q,
                "risk_area": str(rec.get("risk_area", "")).strip(),
                "types_of_harm": str(rec.get("types_of_harm", "")).strip(),
                "specific_harms": str(rec.get("specific_harms", "")).strip(),
                "source": SOURCE["dataset"],
            }, ensure_ascii=False) + "\n")
            n += 1

    with open(os.path.join(os.path.dirname(a.out), "SOURCE.json"), "w",
              encoding="utf-8") as f:
        json.dump({**SOURCE, "rows": n}, f, indent=2)

    print(f"\nwrote {n:,} prompts -> {a.out}")

    counts = df["risk_area"].value_counts() if "risk_area" in df else {}
    print("\nrisk areas:")
    for k, v in dict(counts).items():
        print(f"  {v:5,}  {k}")
    harms = df["types_of_harm"].value_counts() if "types_of_harm" in df else {}
    print(f"\ntypes of harm ({len(harms)}):")
    for k, v in list(dict(harms).items())[:20]:
        print(f"  {v:5,}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
