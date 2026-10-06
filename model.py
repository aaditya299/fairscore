from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).parent
DATA=ROOT/"data"
SEED=42

def make_text(df):
    return (df["review_summary"].fillna("")+". "+df["review_detail"].fillna(""))

def split_by_title(df):
    key=df["title"].str.split(" Season").str[0].str.split(":").str[0].str.lower().str.strip()
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED)
    train_idx,test_idx=next(splitter.split(df,df["label"],groups=key))
    return df.iloc[train_idx],df.iloc[test_idx]
    
def report(name,y_true,y_prob):
    roc=roc_auc_score(y_true,y_prob)
    pr=average_precision_score(y_true,y_prob)
    print(f"{name:<22} ROC-AUC {roc:.3f}   PR-AUC {pr:.3f}")

def length_baseline(train,test):
    x_train=np.log1p(train["n_words"].to_numpy()).reshape(-1,1)
    x_test=np.log1p(test["n_words"].to_numpy()).reshape(-1,1)
    model=LogisticRegression()
    model.fit(x_train,train["label"])
    return model.predict_proba(x_test)[:,1]

def text_model(train,test):
    vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),
                         min_df=3,sublinear_tf=True)
    x_train=vec.fit_transform(make_text(train))
    x_test=vec.transform(make_text(test))
    clf=LogisticRegression(max_iter=1000,class_weight="balanced")
    clf.fit(x_train,train["label"])
    return clf,vec,clf.predict_proba(x_test)[:,1]

def main():
    df=pd.read_pickle(DATA/"labels.pkl")
    train,test=split_by_title(df)
    print("train: ",len(train),"reviews,",train["movie"].nunique(),"titles")
    print("test: ",len(test),"reviews,",test["movie"].nunique(),"titles")
    print("suspicious share train: ",round(train["label"].mean(),3),
          " test: ",round(test["label"].mean(),3))
    print("\ntest suspicious titles:")
    print(test[test["label"] == 1]["movie"].value_counts().head(8))
    print("train suspicious titles:")
    print(train[train["label"] == 1]["movie"].value_counts().head(8))
    y=test["label"]
    report("length only",y,length_baseline(train,test))
    clf,vec,prob=text_model(train,test)
    report("text model",y,prob)
    words=np.array(vec.get_feature_names_out())
    order=np.argsort(clf.coef_[0])
    print("\nmost 'genuine' terms:",", ".join(words[order[:15]]))
    print("most 'suspicious' terms:",", ".join(words[order[-15:]]))

if __name__=="__main__":
    main()