from pathlib import Path
import numpy as np
import pandas as pd

from joblib import parallel_backend
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ---------- Pen utskrift ----------
def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def explain_metrics():
    print(
        "\nMETRIKK-FORKLARING:"
        "\n  • Accuracy  = (riktige / alle)"
        "\n  • Precision = Andel av det modellen kalte FAKE som faktisk var FAKE"
        "\n  • Recall    = Andel av alle FAKE som modellen fant (sensitivitet)"
        "\n  • F1        = Balansert snitt av precision og recall"
        "\n  • ROC-AUC   = Hvor godt modellen skiller klasser uansett terskel (1.0 er best, 0.5 ~ gjetting)"
    )

def print_confusion(y_true, y_pred, labels=(0, 1)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    correct = tn + tp
    incorrect = fp + fn

    def pct(x): 
        return f"{(100*x/total):.1f}%"

    print("\nCONFUSION MATRIX (rad = fasit, kolonne = modell):")
    print("              Pred REAL   Pred FAKE   |  Rad-sum")
    print(f"Fasit REAL     {tn:9d}   {fp:9d}   |  {tn+fp:7d}")
    print(f"Fasit FAKE     {fn:9d}   {tp:9d}   |  {fn+tp:7d}")
    print(f"Kol-sum        {tn+fn:9d}   {fp+tp:9d}   |  {total:7d}")

    print("\nENKEL OPPSUMMERING:")
    print(f"- Riktig gjetting : {correct} artikler ({pct(correct)})")
    print(f"- Feil gjetting   : {incorrect} artikler ({pct(incorrect)})")
    print(f"  • Riktig REAL (TN): {tn}")
    print(f"  • Riktig FAKE (TP): {tp}")
    print(f"  • Feil: REAL→FAKE (FP): {fp}")
    print(f"  • Feil: FAKE→REAL (FN): {fn}")

# ---------- Data ----------
def load_data():
    here = Path(__file__).parent
    df = pd.read_csv(here / "english_fake_news_2212.csv", encoding="utf-8")

    if {"headline", "body_text"}.issubset(df.columns):
        df["text"] = (df["headline"].fillna("").astype(str) + " ") * 3 + df["body_text"].fillna("").astype(str)
        df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        raise ValueError("Fant ikke både 'headline' og 'body_text' i CSV.")

    X = df["text"].astype(str)
    y = df["label"].astype(str).str.lower().map({"fake": 1, "real": 0}).astype(int)
    return X, y, df

# ---------- Modell ----------
def build_rf_pipeline(n_components=150):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ])

def crossval_report(pipe, X_train, y_train, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {"acc": "accuracy", "prec": "precision_macro", "rec": "recall_macro", "f1": "f1_macro", "auc": "roc_auc"}
    with parallel_backend("threading"):
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    summary = {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std()) for m in scoring}
    banner("5-fold KRYSSVALIDERING (gjennomsnitt ± std)")
    for k, (mean, std) in summary.items():
        print(f"{k.upper():>4}: {mean:.4f} ± {std:.4f}")
    explain_metrics()
    return summary

def fit_and_test(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    banner("TESTRESULTATER")
    print(f"Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision(mac): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall   (mac): {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1       (mac): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"ROC-AUC      : {auc:.4f}")

    print("\nKLASSIFIKASJONSRAPPORT (per klasse):")
    print(classification_report(y_test, y_pred, digits=3, target_names=["REAL(0)", "FAKE(1)"]))

    print_confusion(y_test, y_pred)
    explain_metrics()

def main():
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    banner("DATAOVERSIKT")
    print(f"Fil: english_fake_news_2212.csv")
    print(f"Antall rader totalt: {len(df)}")
    print(f"Klassefordeling: REAL(0)={(y==0).sum()} | FAKE(1)={(y==1).sum()}")
    print(f"Train/Test-splitt: {len(X_train)}/{len(X_test)}")

    tmp_vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)
    tmp_vec.fit(X_train)
    n_features = len(tmp_vec.get_feature_names_out())
    svd_k = max(100, min(500, n_features - 10))
    print(f"\n[Info] TF-IDF features på train: {n_features} -> SVD n_components: {svd_k}")

    pipe = build_rf_pipeline(n_components=svd_k)
    crossval_report(pipe, X_train, y_train)
    fit_and_test(pipe, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
