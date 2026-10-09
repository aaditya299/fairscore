from pathlib import Path
import pandas as pd
ROOT = Path(__file__).parent
DATA = ROOT / "data"
MIN_DAY_N = 15
VOL_Z = 3.0
DOWNWEIGHT = 0.25

CASES = ["Laxmii (2020)", "Dil Bechara (2020)", "Coolie No. 1 (2020)",
         "Tenet (2020)", "Wonder Woman 1984 (2020)"]
LEVELS = {"loose": (0.60, 0.75), "default": (0.70, 0.85), "strict": (0.80, 0.95)}
SHIFTS = [0.2, 0.3, 0.4]
def flag_days(daily, low_share, high_share, shift):
    base_low = (daily["low_share"] * daily["n"]).sum() / daily["n"].sum()
    base_high = (daily["high_share"] * daily["n"]).sum() / daily["n"].sum()
    big = (daily["n"] >= MIN_DAY_N) & (daily["vol_z"] >= VOL_Z)
    bomb = big & (daily["low_share"] >= low_share) & \
        (daily["low_share"] - base_low >= shift)
    inflate = big & (daily["high_share"] >= high_share) & \
        (daily["high_share"] - base_high >= shift)
    out = daily[["movie", "date"]].copy()
    out["direction"] = None
    out.loc[bomb, "direction"] = "bomb"
    out.loc[inflate, "direction"] = "inflate"
    return out[out["direction"].notna()]

def adjust_cases(reviews, flagged):
    key = flagged[["movie", "date", "direction"]]
    r = reviews.merge(key, on=["movie", "date"], how="left")
    bomb = (r["direction"] == "bomb") & (r["rating"] <= 2)
    inflate = (r["direction"] == "inflate") & (r["rating"] >= 9)
    r["suspect"] = (bomb | inflate) & r["one_shot"]
    r["weight"] = 1.0
    r.loc[r["suspect"], "weight"] = DOWNWEIGHT
    r["kept"] = r["rating"].where(~r["suspect"])
    r["wr"] = r["rating"] * r["weight"]
    g = r.groupby("movie").agg(
        n=("rating", "size"), suspect=("suspect", "sum"),
        original=("rating", "mean"), removed=("kept", "mean"),
        wr=("wr", "sum"), w=("weight", "sum"))
    g["downweighted"] = g["wr"] / g["w"]
    g["suspect_pct"] = g["suspect"] / g["n"]
    return g[["suspect_pct", "original", "downweighted", "removed"]]

def main():
    daily = pd.read_pickle(DATA / "daily.pkl")
    df = pd.read_pickle(DATA / "features.pkl")
    reviews = (df[df["movie"].isin(CASES)].dropna(subset=["rating"])
               [["movie", "date", "rating", "one_shot"]].copy())
    del df
    parts, order = [], []
    for level, (low, high) in LEVELS.items():
        for shift in SHIFTS:
            flagged = flag_days(daily, low, high, shift)
            res = adjust_cases(reviews, flagged).reset_index()
            res["setting"] = f"{level}, shift {shift}"
            res["flagged_days"] = len(flagged)
            parts.append(res)
            order.append(res["setting"].iloc[0])
    out = pd.concat(parts, ignore_index=True)
    out["movie"] = out["movie"].str.replace(r" \(\d{4}\)", "", regex=True)
    print("flagged title-days per setting:")
    print(out.groupby("setting")["flagged_days"].first().reindex(order).to_string())
    for col, title in [("suspect_pct", "suspect share"),
                       ("downweighted", "down-weighted rating")]:
        print(f"\n{title}:")
        table = out.pivot(index="setting", columns="movie", values=col).reindex(order)
        print(table.round(2).to_string())


if __name__ == "__main__":
    main()