from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).parent
DATA = ROOT / "data"
st.set_page_config(page_title="IMDb review-bomb filter", layout="wide")

@st.cache_data
def load():
    adjusted=pd.read_pickle(DATA/"adjusted.pkl")
    daily=pd.read_pickle(DATA/"daily.pkl")
    return adjusted,daily

adjusted,daily=load()
st.title("IMDb review-bomb filter")
st.caption("A sensitivity analysis: how would this title's rating change if "
           "suspicious burst reviews counted less? It is not a verdict on any review.")
titles = [str(t) for t in adjusted.sort_values("n", ascending=False)["movie"]]
choice = st.sidebar.selectbox("Pick a title", titles)
row = adjusted[adjusted["movie"] == choice].iloc[0]

st.subheader(choice)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original rating", f"{row['original']:.2f}")
c2.metric("Down-weighted", f"{row['downweighted']:.2f}",
          f"{row['downweighted'] - row['original']:+.2f}")
c3.metric("Suspect removed", f"{row['removed']:.2f}",
          f"{row['removed'] - row['original']:+.2f}")
c4.metric("Suspect reviews", f"{row['suspect_pct']:.0%}",
          f"{int(row['n_suspect']):,} of {int(row['n']):,}", delta_color="off")

d = daily[daily["movie"] == choice].sort_values("date").copy()
d["normal days"] = d["n"].where(d["direction"].isna(), 0)
d["flagged burst days"] = d["n"].where(d["direction"].notna(), 0)

span = (d["date"].max() - d["date"].min()).days
if span <= 60:
    axis = alt.Axis(format="%b %d", tickCount="day", labelAngle=-45)
elif span <= 400:
    axis = alt.Axis(format="%b %Y", tickCount="month")
else:
    axis = alt.Axis(format="%Y", tickCount="year")

st.markdown("**Reviews per day** (red = flagged burst day)")
long = d.melt(id_vars="date", value_vars=["normal days", "flagged burst days"],
              var_name="type", value_name="reviews")
bars = alt.Chart(long).mark_bar().encode(
    x=alt.X("date:T", axis=axis, title=None),
    y=alt.Y("reviews:Q", title="reviews/day"),
    color=alt.Color("type:N", legend=alt.Legend(title=None),
                    scale=alt.Scale(domain=["normal days", "flagged burst days"],
                                    range=["#4c78a8", "#e45756"])),
)
st.altair_chart(bars, use_container_width=True)

rule = "W" if span > 120 else "D"
st.markdown("**Average rating per week**" if rule == "W" else "**Average rating per day**")
d["total"] = d["mean"] * d["n"]
w = d.set_index("date")[["total", "n"]].resample(rule).sum()
w = w[w["n"] >= (5 if rule == "W" else 1)]
w["mean_rating"] = w["total"] / w["n"]
line = alt.Chart(w.reset_index()).mark_line().encode(
    x=alt.X("date:T", axis=axis, title=None),
    y=alt.Y("mean_rating:Q", title="mean rating", scale=alt.Scale(domain=[1, 10])),
)
st.altair_chart(line, use_container_width=True)

dirs=set(d["direction"].dropna())
if "bomb" in dirs:
    st.warning("This title has flagged bombing bursts:  a spike in reviews that "
               "are almost all 1-2 stars.")
if "inflate" in dirs:
    st.info("This title has flagged inflation bursts. A wave of 9-10 star reviews "
            "can be genuine fan excitement, so read these as enthusiasm bursts, "
            "not proof of fakes.")

st.divider()
st.subheader("Titles whose rating changes most")
top = (adjusted.assign(change=(adjusted["downweighted"] - adjusted["original"]).abs())
       .sort_values("change", ascending=False).head(20))
st.dataframe(top[["movie", "n", "suspect_pct", "original", "downweighted", "removed"]]
             .round(2), hide_index=True)

with st.expander("How the adjusted rating works, and its limits"):
    st.markdown("""
  A review counts as suspect only if all three are true:
1. It was written on a flagged burst day, meaning an unusual spike in reviews that are almost all very low (bombing) or very high (inflation).
2. Its author has 2 or fewer rated reviews in the dataset.
3. Its rating points the same way as the burst (1-2 stars on a bomb day, 9-10 on an inflate day).

  Three numbers:   the original average, a down-weighted average where suspect reviews count 25% as much, and an average with suspect reviews removed. Treat them as a range.

  Limits:
- There are no verified fake reviews, so this can't be called accurate, only consistent with obvious cases.
- First-time reviewers can be genuine, so the suspect share is an upper bound.
- Ratings come from a Kaggle sample of IMDb reviews, not IMDb's official scores.
- A drop in rating without a clear burst (for example a polarizing film) is not flagged.
""")