import re
from pathlib import Path
import ijson
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_FILES = 4
RAW_FILES = sorted(DATA.glob("part-*.json"))[:MAX_FILES]
PARTS = DATA / "parts"
OUT = DATA / "reviews.pkl"
BATCH = 50_000  
YEAR_RE = re.compile(r"\((\d{4})\s*([–-])?\s*(\d{4})?\s*\)")

def parse_helpful(x):
    if isinstance(x, list) and len(x) == 2:
        return pd.to_numeric(x[0], errors="coerce"), pd.to_numeric(x[1], errors="coerce")
    return float("nan"), float("nan")


def clean_batch(records):
    df = pd.DataFrame(records)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["review_date"], errors="coerce")
    h = df["helpful"].apply(parse_helpful)
    df["helpful_yes"] = [a for a, _ in h]
    df["helpful_total"] = [b for _, b in h]
    m = df["movie"].str.extract(YEAR_RE)
    df["year"] = pd.to_numeric(m[0], errors="coerce")
    df["is_series"] = m[1].notna()
    df["title"] = df["movie"].str.replace(YEAR_RE, "", regex=True).str.strip()
    df["n_words"] = df["review_detail"].str.split().str.len()

    text = df[["review_id", "review_summary", "review_detail"]]
    meta = df.drop(columns=["review_summary", "review_detail", "helpful", "review_date"])
    return meta, text

def main():
    PARTS.mkdir(exist_ok=True)
    for old in PARTS.glob("*.pkl"):  
        old.unlink()
    batch, part, total = [], 0, 0

    def flush():
        nonlocal batch, part, total
        meta, text = clean_batch(batch)
        meta.to_pickle(PARTS / f"meta_{part:03d}.pkl")
        text.to_pickle(PARTS / f"text_{part:03d}.pkl")
        total += len(batch)
        print(f"  batch {part:03d} done, {total:,} reviews so far", flush=True)
        batch, part = [], part + 1

    for raw in RAW_FILES:
        print("streaming", raw.name, flush=True)
        with open(raw, "rb") as f:
            for record in ijson.items(f, "item"):
                batch.append(record)
                if len(batch) >= BATCH:
                    flush()
    if batch:
        flush()

    print("combining metadata...", flush=True)
    metas = [pd.read_pickle(p) for p in sorted(PARTS.glob("meta_*.pkl"))]
    df = pd.concat(metas, ignore_index=True)
    df = df.drop_duplicates("review_id").reset_index(drop=True)
    df.to_pickle(OUT)

    print("\nsaved", OUT)
    print("shape:", df.shape)
    print("date range:", df["date"].min(), "->", df["date"].max())
    print("rated reviews:", df["rating"].notna().sum())
    print("titles:", df["movie"].nunique(), "| reviewers:", df["reviewer"].nunique())
    print(df.dtypes)


if __name__ == "__main__":
    main()