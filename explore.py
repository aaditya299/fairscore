import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).parent
DATA=ROOT/"data"/"reviews.pkl"
PLOTS=ROOT/"plots"
PLOTS.mkdir(exist_ok=True)

def daily_profile(df,movie):
    sub=df[df["movie"]==movie].dropna(subset=["rating","date"])
    g=sub.groupby(sub["date"].dt.date)
    return pd.DataFrame({
        "n":g.size(),
        "mean":g["rating"].mean(),
        "p1":g["rating"].apply(lambda r:(r==1).mean()),
        "p10":g["rating"].apply(lambda r: (r==10).mean()),
    })

def plot(movie,daily):
    fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    ax[0].bar(daily.index,daily["n"])
    ax[0].set_ylabel("review/day")
    ax[1].plot(daily.index,daily["mean"])
    ax[1].set_ylabel("mean rating")
    ax[2].plot(daily.index,daily["p1"],label="% 1-star")
    ax[2].plot(daily.index,daily["p10"],label="% 10-star")
    ax[2].set_ylabel("share")
    ax[2].legend()
    fig.suptitle(movie)
    plt.tight_layout()
    safe="".join(c if c.isalnum() else "_" for c in movie)
    fig.savefig(PLOTS/f"{safe}.png")
    plt.close(fig)

def main():
    df=pd.read_pickle(DATA)
    titles=sys.argv[1:] or df["movie"].value_counts().head(8).index.tolist()
    for t in titles:
        daily=daily_profile(df,t)
        if daily.empty:
            print(f"{t}: no rated reviews found")
            continue
        sub=df[df["movie"]==t]
        print(f"\n{t}: {len(sub)} reviews, mean rating {sub['rating'].mean():.2f}")
        print(daily.sort_values("n",ascending=False).head(5).round(2))
        plot(t,daily)
    print("\nplots saved in", PLOTS)

if __name__=="__main__":
    main()