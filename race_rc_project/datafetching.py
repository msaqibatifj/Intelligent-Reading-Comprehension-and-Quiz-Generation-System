"""
datafetching.py — Download MCQ datasets and export as CSV matching the
pipeline's expected format:

    article, question, A, B, C, D, answer

Datasets:
  RACE          1. Hugging Face (ehovy/race)  2. Kaggle (ankitdhiman7/race-dataset)
  DREAM         1. GitHub (nlpdata/dream)     2. — (HF loading scripts no longer supported)
  CommonsenseQA 1. Hugging Face (commonsense_qa)
  ARC           1. Hugging Face (allenai/ai2_arc)
  SocialIQA     1. Google Storage              (HF loading scripts no longer supported)
  MultiRC       1. super_glue (Parquet)
  OpenBookQA    1. Hugging Face (allenai/openbookqa)
  SciQ          1. Hugging Face (sciq Parquet)
  MMLU          1. Hugging Face (cais/mmlu Parquet)
  MedQA         1. Hugging Face (GBaker/MedQA-USMLE-4-options-hf JSONL)
  QASC          1. Hugging Face (qasc Parquet, 8→4 options)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — works in Kaggle/Colab notebooks and local scripts
# ---------------------------------------------------------------------------

try:
    _THIS_DIR = Path(__file__).resolve().parent
except NameError:
    _THIS_DIR = Path(os.getcwd())
DATA_RAW = _THIS_DIR / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_options(opts: List[str], n: int = 4) -> List[str]:
    while len(opts) < n:
        opts.append("(none of the above)")
    return opts[:n]


def _answer_letter(idx: int) -> str:
    return chr(65 + idx)


# ---------------------------------------------------------------------------
# 1. RACE
# ---------------------------------------------------------------------------

def _race_from_hf() -> Optional[pd.DataFrame]:
    """Try loading RACE via Hugging Face datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    print("[RACE] trying HuggingFace ehovy/race (config='all') …")
    try:
        ds = load_dataset("ehovy/race", "all")
    except Exception as e:
        print(f"  HF failed: {e}")
        return None

    rows = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            opts = ex["options"]
            rows.append({
                "article": ex["article"],
                "question": ex["question"],
                "A": opts[0] if len(opts) > 0 else "",
                "B": opts[1] if len(opts) > 1 else "",
                "C": opts[2] if len(opts) > 2 else "",
                "D": opts[3] if len(opts) > 3 else "",
                "answer": str(ex["answer"]).strip().upper(),
            })
    return pd.DataFrame(rows)


def _race_from_kaggle() -> Optional[pd.DataFrame]:
    """Fallback: download RACE CSV from Kaggle."""
    print("[RACE] trying Kaggle ankitdhiman7/race-dataset …")
    dest = DATA_RAW / "kaggle_race"
    dest.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "ankitdhiman7/race-dataset", "-p", str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Kaggle API failed. Download manually from:")
        print("    https://www.kaggle.com/datasets/ankitdhiman7/race-dataset")
        print(f"  Error: {result.stderr[:200]}")
        return None

    zips = list(dest.glob("*.zip"))
    if not zips:
        return None
    with zipfile.ZipFile(zips[0], "r") as zf:
        zf.extractall(dest)

    csvs = list(dest.rglob("*.csv")) + list(dest.rglob("*.CSV"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    required = {"article", "question", "A", "B", "C", "D", "answer"}
    if not required.issubset(df.columns):
        print(f"  CSV missing columns: {required - set(df.columns)}")
        return None
    return df


def fetch_race() -> pd.DataFrame:
    df = _race_from_hf()
    if df is not None and len(df) > 0:
        print(f"[RACE]   → {len(df):,} rows (HuggingFace)")
        return df
    df = _race_from_kaggle()
    if df is not None and len(df) > 0:
        print(f"[RACE]   → {len(df):,} rows (Kaggle)")
        return df
    print("[RACE]   WARNING: no data loaded")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. DREAM  (GitHub — nlpdata/dream)
# ---------------------------------------------------------------------------

def fetch_dream() -> pd.DataFrame:
    """
    Download DREAM from GitHub, parse JSON → flat DataFrame.
    DREAM has 3-option MCQs; we pad to 4 with placeholders.
    """
    url = "https://github.com/nlpdata/dream/archive/refs/heads/master.zip"
    zip_path = DATA_RAW / "dream-master.zip"
    extract_dir = DATA_RAW / "dream-master"

    print(f"[DREAM] downloading from GitHub …")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW)

    all_rows = []
    for split_name in ("train", "dev", "test"):
        json_path = extract_dir / "data" / f"{split_name}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            dialogue = " ".join(str(t) for t in entry[0])
            for q in entry[1]:
                choices = q.get("choice", [])
                ans_text = q.get("answer", choices[0] if choices else "")
                opts = _pad_options(choices, 4)
                try:
                    ans_idx = choices.index(ans_text)
                except ValueError:
                    ans_idx = 0
                all_rows.append({
                    "article": dialogue,
                    "question": q.get("question", ""),
                    "A": opts[0],
                    "B": opts[1],
                    "C": opts[2],
                    "D": opts[3],
                    "answer": _answer_letter(ans_idx),
                })
    df = pd.DataFrame(all_rows)
    print(f"[DREAM]   → {len(df):,} rows (GitHub)")
    return df


# ---------------------------------------------------------------------------
# 3. CommonsenseQA  (HuggingFace — 5 options, no passage)
# ---------------------------------------------------------------------------

def fetch_commonsenseqa() -> pd.DataFrame:
    """Download CommonsenseQA from HuggingFace. 5 options → 4."""
    print("[CSQA] loading commonsense_qa …")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets")
        return pd.DataFrame()

    try:
        ds = load_dataset("commonsense_qa")
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()

    rows = []
    for split in ds:
        for ex in ds[split]:
            texts = list(ex["choices"]["text"])
            labels = list(ex["choices"]["label"])
            ans_key = str(ex["answerKey"]).strip().upper()
            try:
                ans_idx = labels.index(ans_key)
            except ValueError:
                ans_idx = 0
            if len(texts) > 4:
                if ans_idx >= 4:
                    texts[3], texts[ans_idx] = texts[ans_idx], texts[3]
                    ans_idx = 3
                opts = texts[:4]
            else:
                opts = _pad_options(texts, 4)
            rows.append({
                "article": ex["question"],
                "question": "",
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx),
            })
    df = pd.DataFrame(rows)
    print(f"[CSQA]   → {len(df):,} rows (HuggingFace)")
    return df


# ---------------------------------------------------------------------------
# 4. ARC (AI2 Reasoning Challenge)  — HuggingFace
# ---------------------------------------------------------------------------

def fetch_arc() -> pd.DataFrame:
    """Download ARC (Challenge + Easy) from HuggingFace."""
    print("[ARC] loading allenai/ai2_arc …")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets")
        return pd.DataFrame()

    rows = []
    for config in ("ARC-Challenge", "ARC-Easy"):
        try:
            ds = load_dataset("allenai/ai2_arc", config)
        except Exception as e:
            print(f"  {config} failed: {e}")
            continue
        for split in ds:
            for ex in ds[split]:
                texts = list(ex["choices"]["text"])
                labels = list(ex["choices"]["label"])
                ans_raw = str(ex.get("answerKey", "")).strip()
                if not ans_raw:
                    continue
                try:
                    ans_idx = labels.index(ans_raw)
                except ValueError:
                    ans_idx = 0
                opts = _pad_options(texts, 4)
                if ans_idx >= len(opts):
                    ans_idx = 0
                rows.append({
                    "article": ex["question"],
                    "question": "",
                    "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                    "answer": _answer_letter(ans_idx),
                })
    df = pd.DataFrame(rows)
    print(f"[ARC]     → {len(df):,} rows (HuggingFace)")
    return df


# ---------------------------------------------------------------------------
# 5. SocialIQA  (HuggingFace — 3 options, has passage context)
# ---------------------------------------------------------------------------

def fetch_socialiqa() -> pd.DataFrame:
    """Download SocialIQA from Google Storage (HF loading script no longer supported)."""
    print("[SIQA] downloading from Google Storage …")
    url = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
    zip_path = DATA_RAW / "socialiqa-train-dev.zip"
    extract_dir = DATA_RAW / "socialiqa-train-dev"

    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  Download failed: {e}")
        return pd.DataFrame()

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW)

    rows = []
    for split_name in ("train", "dev"):
        jsonl_path = extract_dir / "socialiqa-train-dev" / f"{split_name}.jsonl"
        labels_path = extract_dir / "socialiqa-train-dev" / f"{split_name}-labels.lst"
        if not jsonl_path.exists() or not labels_path.exists():
            continue

        with open(labels_path) as f:
            labels = [line.strip() for line in f if line.strip()]

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if i >= len(labels):
                    break
                ex = json.loads(line)
                ans_idx = int(labels[i]) - 1
                if ans_idx < 0:
                    ans_idx = 0
                opts = [ex["answerA"], ex["answerB"], ex["answerC"]]
                opts_padded = _pad_options(opts, 4)
                rows.append({
                    "article": ex["context"],
                    "question": ex.get("question", ""),
                    "A": opts_padded[0], "B": opts_padded[1],
                    "C": opts_padded[2], "D": opts_padded[3],
                    "answer": _answer_letter(ans_idx),
                })
    df = pd.DataFrame(rows)
    print(f"[SIQA]    → {len(df):,} rows (Google Storage)")
    return df


# ---------------------------------------------------------------------------
# 6. MultiRC  (HuggingFace — variable answers, multiple correct per Q)
# ---------------------------------------------------------------------------

def fetch_multirc() -> pd.DataFrame:
    """Download MultiRC from super_glue (Parquet). Pick first correct answer per Q."""
    print("[MultiRC] loading super_glue/multirc …")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("  pip install huggingface_hub pyarrow")
        return pd.DataFrame()

    rows = []
    for split in ("train", "validation", "test"):
        try:
            path = hf_hub_download(
                repo_id="super_glue",
                filename=f"multirc/{split}-00000-of-00001.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  {split} download failed: {e}")
            continue

        table = pq.read_table(path)
        df_split = table.to_pandas()

        for (para, q_text), group in df_split.groupby(["paragraph", "question"], sort=False):
            answers = group["answer"].tolist()
            labels = group["label"].tolist()

            correct_texts = [
                a for a, lbl in zip(answers, labels) if lbl == 1
            ]
            if not correct_texts:
                continue
            correct = correct_texts[0]

            opts = _pad_options(answers, 4)
            try:
                ans_idx = answers.index(correct)
            except ValueError:
                ans_idx = 0
            if ans_idx >= 4:
                ans_idx = 0

            rows.append({
                "article": para,
                "question": q_text,
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx),
            })

    df = pd.DataFrame(rows)
    print(f"[MultiRC] → {len(df):,} rows (super_glue Parquet)")
    return df


# ---------------------------------------------------------------------------
# 7. OpenBookQA  (HuggingFace — 4 options, no passage)
# ---------------------------------------------------------------------------

def fetch_openbookqa() -> pd.DataFrame:
    """Download OpenBookQA from HuggingFace."""
    print("[OBQA] loading allenai/openbookqa …")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets")
        return pd.DataFrame()

    try:
        ds = load_dataset("allenai/openbookqa", "main")
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()

    rows = []
    for split in ds:
        for ex in ds[split]:
            texts = list(ex["choices"]["text"])
            labels = list(ex["choices"]["label"])
            ans_key = str(ex.get("answerKey", "")).strip().upper()
            try:
                ans_idx = labels.index(ans_key)
            except ValueError:
                ans_idx = 0
            opts = _pad_options(texts, 4)
            if ans_idx >= len(opts):
                ans_idx = 0
            rows.append({
                "article": ex["question_stem"],
                "question": "",
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx),
            })
    df = pd.DataFrame(rows)
    print(f"[OBQA]    → {len(df):,} rows (HuggingFace)")
    return df


# ---------------------------------------------------------------------------
# 8. SciQ  (HuggingFace Parquet — 4 options, has support passage)
# ---------------------------------------------------------------------------

def fetch_sciq() -> pd.DataFrame:
    """Download SciQ from HuggingFace Parquet. Has support passage + 4 options."""
    print("[SciQ] loading sciq Parquet …")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("  pip install huggingface_hub pyarrow")
        return pd.DataFrame()

    import random

    rows = []
    for split in ("train", "validation", "test"):
        try:
            path = hf_hub_download(
                repo_id="sciq",
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  {split} download failed: {e}")
            continue

        table = pq.read_table(path)
        df_split = table.to_pandas()

        for _, ex in df_split.iterrows():
            opts = [
                ex["correct_answer"],
                ex["distractor1"],
                ex["distractor2"],
                ex["distractor3"],
            ]
            random.shuffle(opts)
            ans_idx = opts.index(ex["correct_answer"])
            rows.append({
                "article": str(ex.get("support", "")),
                "question": ex["question"],
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx),
            })

    df = pd.DataFrame(rows)
    print(f"[SciQ]    → {len(df):,} rows (HuggingFace Parquet)")
    return df


# ---------------------------------------------------------------------------
# 9. MMLU  (HuggingFace Parquet — all subjects, 4 options)
# ---------------------------------------------------------------------------

def fetch_mmlu() -> pd.DataFrame:
    """Download MMLU from HuggingFace Parquet. All 57 subjects, 4 options each."""
    print("[MMLU] loading cais/mmlu Parquet …")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("  pip install huggingface_hub pyarrow")
        return pd.DataFrame()

    rows = []
    for split in ("auxiliary_train", "dev", "validation", "test"):
        try:
            path = hf_hub_download(
                repo_id="cais/mmlu",
                filename=f"all/{split}-00000-of-00001.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  {split} download failed: {e}")
            continue

        table = pq.read_table(path)
        df_split = table.to_pandas()

        for _, ex in df_split.iterrows():
            opts = list(ex["choices"])
            ans_idx = int(ex["answer"])
            if ans_idx < 0 or ans_idx >= len(opts):
                ans_idx = 0
            rows.append({
                "article": ex["question"],
                "question": "",
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx),
            })

    df = pd.DataFrame(rows)
    print(f"[MMLU]    → {len(df):,} rows (HuggingFace Parquet)")
    return df


# ---------------------------------------------------------------------------
# 10. MedQA  (HuggingFace JSONL — USMLE medical MCQs, 4 options)
# ---------------------------------------------------------------------------

def fetch_medqa() -> pd.DataFrame:
    """Download MedQA (USMLE) from HuggingFace JSONL. 4 options, medical context."""
    print("[MedQA] loading GBaker/MedQA-USMLE-4-options-hf …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  pip install huggingface_hub")
        return pd.DataFrame()

    import json

    rows = []
    for split in ("train", "dev", "test"):
        try:
            path = hf_hub_download(
                repo_id="GBaker/MedQA-USMLE-4-options-hf",
                filename=f"{split}.json",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  {split} download failed: {e}")
            continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                opts = [ex["ending0"], ex["ending1"], ex["ending2"], ex["ending3"]]
                ans_idx = int(ex["label"])
                if ans_idx < 0 or ans_idx >= 4:
                    ans_idx = 0
                rows.append({
                    "article": ex["sent1"],
                    "question": ex.get("sent2", ""),
                    "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                    "answer": _answer_letter(ans_idx),
                })

    df = pd.DataFrame(rows)
    print(f"[MedQA]   → {len(df):,} rows (HuggingFace JSONL)")
    return df


# ---------------------------------------------------------------------------
# 11. QASC  (HuggingFace Parquet — 8 options trimmed to 4)
# ---------------------------------------------------------------------------

def fetch_qasc() -> pd.DataFrame:
    """Download QASC from HuggingFace Parquet. 8 options → keep correct + 3."""
    print("[QASC] loading qasc Parquet …")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("  pip install huggingface_hub pyarrow")
        return pd.DataFrame()

    import random

    rows = []
    for split in ("train", "validation", "test"):
        try:
            path = hf_hub_download(
                repo_id="qasc",
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  {split} download failed: {e}")
            continue

        table = pq.read_table(path)
        df_split = table.to_pandas()

        for _, ex in df_split.iterrows():
            texts = list(ex["choices"]["text"])
            labels = list(ex["choices"]["label"])
            ans_key = str(ex["answerKey"]).strip().upper()
            try:
                ans_idx = labels.index(ans_key)
            except ValueError:
                continue

            fact_parts = [str(ex.get("fact1", "")), str(ex.get("fact2", ""))]
            article = " ".join(f for f in fact_parts if f and f != "nan" and f != "None")

            if len(texts) > 4:
                keep = {ans_idx}
                candidates = [i for i in range(len(texts)) if i != ans_idx]
                random.shuffle(candidates)
                keep.update(candidates[:3])
                final_indices = sorted(keep)
                opts = [texts[i] for i in final_indices]
                ans_idx_new = final_indices.index(ans_idx)
            else:
                opts = _pad_options(texts, 4)
                ans_idx_new = ans_idx if ans_idx < 4 else 0

            rows.append({
                "article": article if len(article) > 20 else ex["question"],
                "question": ex["question"] if len(article) > 20 else "",
                "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "answer": _answer_letter(ans_idx_new),
            })

    df = pd.DataFrame(rows)
    print(f"[QASC]    → {len(df):,} rows (HuggingFace Parquet)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all(
    include_race: bool = True,
    include_dream: bool = True,
    include_commonsenseqa: bool = True,
    include_arc: bool = True,
    include_socialiqa: bool = True,
    include_multirc: bool = True,
    include_openbookqa: bool = True,
    include_sciq: bool = True,
    include_mmlu: bool = True,
    include_medqa: bool = True,
    include_qasc: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    parts = []

    if include_race:
        df = fetch_race()
        if len(df):
            df["source"] = "race"
            parts.append(df)

    if include_dream:
        df = fetch_dream()
        if len(df):
            df["source"] = "dream"
            parts.append(df)

    if include_commonsenseqa:
        df = fetch_commonsenseqa()
        if len(df):
            df["source"] = "commonsenseqa"
            parts.append(df)

    if include_arc:
        df = fetch_arc()
        if len(df):
            df["source"] = "arc"
            parts.append(df)

    if include_socialiqa:
        df = fetch_socialiqa()
        if len(df):
            df["source"] = "socialiqa"
            parts.append(df)

    if include_multirc:
        df = fetch_multirc()
        if len(df):
            df["source"] = "multirc"
            parts.append(df)

    if include_openbookqa:
        df = fetch_openbookqa()
        if len(df):
            df["source"] = "openbookqa"
            parts.append(df)

    if include_sciq:
        df = fetch_sciq()
        if len(df):
            df["source"] = "sciq"
            parts.append(df)

    if include_mmlu:
        df = fetch_mmlu()
        if len(df):
            df["source"] = "mmlu"
            parts.append(df)

    if include_medqa:
        df = fetch_medqa()
        if len(df):
            df["source"] = "medqa"
            parts.append(df)

    if include_qasc:
        df = fetch_qasc()
        if len(df):
            df["source"] = "qasc"
            parts.append(df)

    if not parts:
        print("No datasets loaded. Aborting.")
        sys.exit(1)

    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["article", "question"])
    if len(combined) != before:
        print(f"  Dedup removed {before - len(combined):,} rows")

    combined = combined[combined["article"].str.len() > 20]
    combined = combined[(combined["question"].str.len() > 3) | (combined["question"] == "")]
    combined = combined[combined["answer"].isin(["A", "B", "C", "D"])]
    combined = combined.reset_index(drop=True)

    out_path = Path(output_path or (DATA_RAW / "train.csv"))
    combined.to_csv(out_path, index=False)
    print(f"\n  Written {len(combined):,} rows → {out_path}")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MCQ datasets → CSV"
    )
    parser.add_argument("--no-race", action="store_true")
    parser.add_argument("--no-dream", action="store_true")
    parser.add_argument("--no-csqa", action="store_true")
    parser.add_argument("--no-arc", action="store_true")
    parser.add_argument("--no-siqa", action="store_true")
    parser.add_argument("--no-multirc", action="store_true")
    parser.add_argument("--no-obqa", action="store_true")
    parser.add_argument("--no-sciq", action="store_true")
    parser.add_argument("--no-mmlu", action="store_true")
    parser.add_argument("--no-medqa", action="store_true")
    parser.add_argument("--no-qasc", action="store_true")
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    fetch_all(
        include_race=not args.no_race,
        include_dream=not args.no_dream,
        include_commonsenseqa=not args.no_csqa,
        include_arc=not args.no_arc,
        include_socialiqa=not args.no_siqa,
        include_multirc=not args.no_multirc,
        include_openbookqa=not args.no_obqa,
        include_sciq=not args.no_sciq,
        include_mmlu=not args.no_mmlu,
        include_medqa=not args.no_medqa,
        include_qasc=not args.no_qasc,
        output_path=args.output,
    )
