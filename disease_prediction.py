"""
Disease Prediction from Medical Data
Datasets : Breast Cancer (built into scikit-learn / UCI WDBC),
           Pima Indians Diabetes, Heart Disease (Cleveland)
Models   : SVM, Logistic Regression, Random Forest, XGBoost

Usage:
    python disease_prediction.py                       # all three datasets
    python disease_prediction.py --dataset heart       # breast_cancer | diabetes | heart
    python disease_prediction.py --csv my.csv --target label

Positive class (1) = disease present. Recall on this class is the key metric:
a missed diagnosis (false negative) is usually costlier than a false alarm.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42

URLS = {
    "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "heart": "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
}
PIMA_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
             "insulin", "bmi", "pedigree", "age", "outcome"]


# ------------------------------------------------------------------ data
def load_dataset(name):
    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        df = d.frame.copy()
        df["target"] = 1 - df["target"]          # sklearn: 1 = benign -> flip so 1 = malignant
        return df, "target"
    if name == "diabetes":
        df = pd.read_csv(URLS["diabetes"], header=None, names=PIMA_COLS)
        # physiologically impossible zeros are really missing values
        for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
            df[c] = df[c].replace(0, np.nan)
        return df, "outcome"
    if name == "heart":
        df = pd.read_csv(URLS["heart"], encoding="utf-8-sig")
        return df, "target"
    raise ValueError(name)


# ----------------------------------------------------- feature engineering
def add_features(df, name):
    df = df.copy()
    if name == "diabetes":
        df["glucose_bmi"] = df["glucose"] * df["bmi"] / 100
        df["age_group"] = pd.cut(df["age"], [0, 30, 45, 60, 120], labels=False)
    elif name == "heart":
        df["hr_reserve"] = (220 - df["age"]) - df["thalach"]    # age-predicted max HR minus achieved
        df["chol_age"] = df["chol"] / df["age"]
    return df


# ----------------------------------------------------------------- models
def build_models(pos_weight):
    def pipe(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", clf)])
    return {
        "SVM": pipe(SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=SEED)),
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                     class_weight="balanced",
                                                     n_jobs=-1, random_state=SEED)),
        "XGBoost": pipe(XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      scale_pos_weight=pos_weight, eval_metric="logloss",
                                      random_state=SEED, n_jobs=-1)),
    }


def run(name, df, target, tune=False):
    print(f"\n{'=' * 70}\nDATASET: {name}  |  {len(df)} rows  |  disease rate = {df[target].mean():.1%}")
    df = add_features(df, name)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = build_models(pos_weight)
    if tune:  # optional light hyper-parameter search for the SVM
        gs = GridSearchCV(models["SVM"], {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
                          cv=5, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        models["SVM"] = gs.best_estimator_
        print("Tuned SVM params:", gs.best_params_)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, curves, fitted = [], {}, {}
    for mname, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"Model": mname, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred), "Recall": recall_score(y_te, pred),
                     "F1": f1_score(y_te, pred), "ROC_AUC": roc_auc_score(y_te, proba),
                     "CV_AUC": cv_auc})
        curves[mname] = roc_curve(y_te, proba)
        fitted[mname] = (m, pred)

    res = pd.DataFrame(rows).set_index("Model").round(3)
    print(res.to_string())
    best = res["F1"].idxmax()
    tn, fp, fn, tp = confusion_matrix(y_te, fitted[best][1]).ravel()
    print(f"Best by F1: {best}  ->  TP={tp} FP={fp} FN={fn} TN={tn}")
    res.to_csv(f"results_{name}.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for mname, (fpr, tpr, _) in curves.items():
        ax[0].plot(fpr, tpr, label=f"{mname} ({res.loc[mname, 'ROC_AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[0].set(title=f"ROC - {name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax[0].legend(title="AUC")
    rf = fitted["Random Forest"][0].named_steps["clf"]
    pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10).plot.barh(ax=ax[1])
    ax[1].set_title("Random Forest - top features")
    plt.tight_layout(); plt.savefig(f"plots_{name}.png", dpi=150); plt.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["breast_cancer", "diabetes", "heart", "all"], default="all")
    ap.add_argument("--csv"); ap.add_argument("--target", default="target")
    ap.add_argument("--tune", action="store_true", help="grid-search the SVM")
    a = ap.parse_args()

    if a.csv:
        run("custom", pd.read_csv(a.csv), a.target, a.tune)
        return
    names = ["breast_cancer", "diabetes", "heart"] if a.dataset == "all" else [a.dataset]
    for n in names:
        df, t = load_dataset(n)
        run(n, df, t, a.tune)


if __name__ == "__main__":
    main()
