"""
Step 1: load both datasets and check what we actually have.

Setup:
    pip install numpy pandas matplotlib

Files expected in ./data/
    Dataset.npy   (IEEE DataPort: userID, movieID, rating, review date)
    reviews.csv   (Kaggle: Reviews of IMDB Movies)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
DATA = Path(r"D:\rating_filter\data")


# ---------- Dataset A: IEEE ratings (no text) ----------
def load_ieee():
    arr = np.load(DATA / "Dataset.npy", allow_pickle=True)
    df = pd.DataFrame(arr, columns=["user_id", "movie_id", "rating", "date"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # e.g. "22 January 2019"
    return df


# ---------- Dataset B: Kaggle reviews (with text) ----------


def load_kaggle():
    df = pd.read_json(DATA / "part-01.json")
    return df.drop_duplicates("review_id")

def report(name, df):
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    print("\ncolumns / dtypes:\n", df.dtypes)
    print("\nnulls:\n", df.isna().sum())
    print("\nhead:\n", df.head())
    if "date" in df.columns:
        print("\ndate range:", df["date"].min(), "->", df["date"].max())
    if "rating" in df.columns:
        print("\nrating distribution:\n",
              df["rating"].value_counts(dropna=False).sort_index())


def daily_profile(df, movie_id, id_col="movie_id"):
    """Reviews per day + mean rating per day for one title.
    A bombing shows up as a spike in count with the mean rating collapsing."""
    sub = df[df[id_col] == movie_id].dropna(subset=["date", "rating"])
    daily = sub.groupby(sub["date"].dt.date).agg(
        n=("rating", "size"),
        mean_rating=("rating", "mean"),
        pct_1star=("rating", lambda r: (r <= 1).mean()),
    )
    return daily


def plot_title(daily, title=""):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].bar(daily.index, daily["n"])
    ax[0].set_ylabel("reviews/day")
    ax[1].plot(daily.index, daily["mean_rating"])
    ax[1].set_ylabel("mean rating")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cache = DATA / "reviews_clean.pkl"

    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = load_kaggle()
        df["date"] = pd.to_datetime(df["review_date"], errors="coerce")
        # helpful looks like ["1", "1"] -> helpful_yes, helpful_total
        pairs = df["helpful"].apply(lambda x: x if isinstance(x, list) and len(x) == 2 else [None, None])
        df[["helpful_yes", "helpful_total"]] = pd.DataFrame(pairs.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce")
        df.to_pickle(cache)

    print("shape:", df.shape)
    print("\nnulls:\n", df.isna().sum())
    print("\ndate range:", df["date"].min(), "->", df["date"].max())
    print("\nrating distribution:\n", df["rating"].value_counts(dropna=False).sort_index())
    print("\nunique movies:", df["movie"].nunique(), "| unique reviewers:", df["reviewer"].nunique())
    print("\nmost-reviewed movies:\n", df["movie"].value_counts().head(15))
    print("\nsample row:\n", df.iloc[0].to_dict())