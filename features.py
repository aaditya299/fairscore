from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent
DATA=ROOT/"data"

def mark_burst_reviews(df,flagged):
    key=flagged[["movie","date","direction"]]
    out=df.merge(key,on=["movie","date"],how="left")
    out["in_burst"]=out["direction"].notna()
    return out

def reviewer_stats(df):
    rated=df.dropna(subset=["rating"]).copy()
    rated["extreme"]=rated["rating"].isin([1,10]).astype(float)
    stats=(
        rated.groupby("reviewer")
        .agg(n_reviews=("rating","size"),
            n_titles=("movie","nunique"),
            mean_rating=("rating","mean"),
            extreme_share=("extreme","mean"),
            first_review=("date","min"),
            last_review=("date","max"))
        .reset_index()
    )
    return stats

def add_reviewwe_features(df,stats):
    out=df.merge(stats,on="reviewer",how="left")
    out["days_active"]=(out["last_review"]-out["first_review"]).dt.days
    out["one_shot"]=out["n_reviews"]<=2
    return out

def main():
    df=pd.read_pickle(DATA/"reviews.pkl")
    flagged=pd.read_pickle(DATA/"flagged_days.pkl")
    out= mark_burst_reviews(df,flagged)
    stats=reviewer_stats(df)
    out=add_reviewwe_features(out,stats)
    out.to_pickle(DATA/"features.pkl")
    summary=out.groupby(out["direction"].fillna("none")).agg(
        reviews=("review_id","size"),
        one_shot=("one_shot","mean"),
        extreme_share=("extreme_share","mean"),
        median_n_reviews=("n_reviews","median"),
        median_days_active=("days_active","median"),
    )
    print(summary.round(3).to_string())

if __name__=="__main__":
    main()
