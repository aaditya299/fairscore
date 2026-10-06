from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MIN_TITLE_REVIEWS=300
DOWNWEIGHT=0.25

def mark_suspect(df):
    rated=df.dropna(subset=["rating"]).copy()
    bomb=(rated["direction"]=="bomb") & (rated["rating"]<=2)
    inflate=(rated["direction"]=="inflate") & (rated["rating"]>=9)
    rated["suspect"]=(bomb|inflate)& rated["one_shot"]
    rated["weight"]=1.0
    rated.loc[rated["suspect"],"weight"]=DOWNWEIGHT
    return rated

def adjusted_rating(rated):
    rated["weighted"]=rated["rating"]*rated["weight"]
    rated["kept_rating"]=rated["rating"].where(~rated["suspect"])
    g=rated.groupby("movie").agg(
        n=("rating","size"),
        n_suspect=("suspect","sum"),
        original=("rating","mean"),
        removed=("kept_rating","mean"),
        w_sum=("weighted","sum"),
        w_total=("weight","sum"),
    )
    g["downweighted"]=g["w_sum"]/g["w_total"]
    g["suspect_pct"]=g["n_suspect"]/g["n"]
    g["shift"]=g["removed"]-g["original"]
    g=g[g["n"]>=MIN_TITLE_REVIEWS]
    return g.drop(columns=["w_sum","w_total"]).reset_index()

def main():
    df=pd.read_pickle(DATA/"features.pkl")
    rated=mark_suspect(df)
    result=adjusted_rating(rated)
    result.to_pickle(DATA/"adjusted.pkl")
    cols=["movie","n","n_suspect","suspect_pct","original","removed","downweighted"]
    cases=["Laxmii (2020)","Dil Bechara (2020)","Wonder Woman 1984 (2020)",
           "Coolie No. 1 (2020)","Tenet (2020)"]
    print("test cases:")
    print(result[result["movie"].isin(cases)][cols].round(2).to_string(index=False))
    order = result["shift"].abs().sort_values(ascending=False).index
    print("\nbiggest changes:")
    print(result.loc[order, cols].head(15).round(2).to_string(index=False))   

if __name__=="__main__":
    main()