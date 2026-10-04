from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent
DATA=ROOT/"data"
MIN_TITLE_REVIEWS=300
MIN_DAY_N=15
VOL_Z=3.0
LOW_SHARE=0.70
HIGH_SHARE=0.85
SHIFT=0.30

def robust_z(s):
    med=s.median()
    mad=(s-med).abs().median()
    scale=1.4826*mad if mad>0 else max(s.std(ddof=0),1.0)
    return (s-med)/scale

def build_daily(df):
    rated=df.dropna(subset=["rating","date"]).copy()
    counts=rated["movie"].value_counts()
    keep=counts[counts>=MIN_TITLE_REVIEWS].index
    rated=rated[rated["movie"].isin(keep)]
    rated["low"]=(rated["rating"]<=2).astype(float)
    rated["high"]=(rated["rating"]>=9).astype(float)
    daily=(
        rated.groupby(["movie","date"])
        .agg(n=("rating","size"),
            mean=("rating","mean"),
            low_share=("low","mean"),
            high_share=("high","mean"))
        .reset_index()
    )
    return daily

def flag(daily):
    g=daily.groupby("movie")
    daily["vol_z"]=g["n"].transform(robust_z)
    base_low = (daily["low_share"] * daily["n"]).sum() / daily["n"].sum()
    base_high = (daily["high_share"] * daily["n"]).sum() / daily["n"].sum()
    print("global baseline  low:", round(base_low, 3), " high:", round(base_high, 3))
    big = (daily["n"]>=MIN_DAY_N) & (daily["vol_z"]>=VOL_Z)
    bomb = big & (daily["low_share"] >= LOW_SHARE) & \
        (daily["low_share"] - base_low >= SHIFT)
    inflate = big & (daily["high_share"] >= HIGH_SHARE) & \
        (daily["high_share"] - base_high >= SHIFT)
    daily["direction"]=None
    daily.loc[bomb,"direction"]="bomb"
    daily.loc[inflate,"direction"]="inflate"
    return daily[daily["direction"].notna()].copy()

def main():
    df=pd.read_pickle(DATA/"reviews.pkl")
    daily=build_daily(df)
    flagged=flag(daily)
    flagged.to_pickle(DATA/"flagged_days.pkl")
    print("titles analysed: ",daily["movie"].nunique())
    print("title-days analyzed: ",len(daily))
    print("flagged title-days: ",len(flagged))
    print(flagged["direction"].value_counts().to_string())
    summary=(
        flagged.groupby(["movie","direction"])
        .agg(days=("date","size"),reviews=("n","sum"),
            first=("date","min"),last=("date","max"))
        .reset_index()
        .sort_values("reviews",ascending=False)
    )
    print("\ntop flagged titles:")
    print(summary.head(25).to_string(index=False))

if __name__=="__main__":
    main()