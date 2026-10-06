from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent
DATA=ROOT/"data"
PARTS=DATA/"parts"
MIN_ESTABLISHED=3
NEG_PER_POS=3

def make_labels(df):
    low=df[df["rating"]<=2]
    sus=low[(low["direction"]=="bomb") & low["one_shot"]]
    bomb_titles = sus["movie"].unique()
    gen = low[(~low["in_burst"]) & (low["n_reviews"] >= MIN_ESTABLISHED)
              & low["movie"].isin(bomb_titles)]
    print("suspicious:", len(sus), "| matched genuine:", len(gen))
    gen=gen.sample(n=min(len(gen),NEG_PER_POS*len(sus)),random_state=42)
    sus=sus.assign(label=1)
    gen=gen.assign(label=0)
    return pd.concat([sus,gen],ignore_index=True)

def add_text(labeled):
    ids=set(labeled["review_id"])
    chunks=[]
    for p in sorted(PARTS.glob("text_*.pkl")):
        t=pd.read_pickle(p)
        chunks.append(t[t["review_id"].isin(ids)])
    text=pd.concat(chunks,ignore_index=True).drop_duplicates("review_id")
    return labeled.merge(text,on="review_id",how="left")

def main():
    df=pd.read_pickle(DATA/"features.pkl")
    labeled=make_labels(df)
    labeled=add_text(labeled)
    labeled.to_pickle(DATA/"labels.pkl")
    print(labeled["label"].value_counts())
    print(labeled.groupby("label")["n_words"].median())
    for lab in (1,0):
        print(f"\nlabel {lab} samples:")
        for s in labeled[labeled["label"]==lab]["review_detail"].head(3):
            print("-",s[:200].replace("\n"," "))

if __name__=="__main__":
    main()