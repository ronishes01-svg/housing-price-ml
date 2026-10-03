"""שלב 2 — פריפרוססינג לנתוני הדירות.

שני חלקים:
1. clean() — ניקוי דטרמיניסטי שמותר לעשות לפני החלוקה (חריגים, הנדסת משתנים).
2. build_transformer() — קידוד + נרמול. *לא* מתאימים אותו על כל הדאטה:
   הוא מותאם (fit) רק על נתוני האימון בשלב הבא, כדי שלא תהיה דליפת מידע.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "housing.csv"
REFERENCE_YEAR = 2015  # השנה האחרונה בנתונים — ממנה מחשבים גיל בניין

TARGET = "price"
NUMERIC = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
           "waterfront", "sqft_basement", "age"]
CATEGORICAL = ["view", "condition"]
FEATURES = NUMERIC + CATEGORICAL


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        lo, hi = iqr_bounds(df[c])
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"עמודה": c, "גבול תחתון": round(lo), "גבול עליון": round(hi),
                     "חריגים": n, "% מהנתונים": round(100 * n / len(df), 1)})
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """מחזיר דאטה נקי + דוח של מה שנעשה בכל צעד."""
    report = {"rows_before": len(df)}

    report["missing"] = int(df.isna().sum().sum())
    report["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # חריגים — רק במחיר (ההחלטה שאושרה): שאר העמודות מתארות בתים אמיתיים
    lo, hi = iqr_bounds(df[TARGET])
    mask = df[TARGET].between(lo, hi)
    report["price_bounds"] = (lo, hi)
    report["price_outliers"] = int((~mask).sum())
    df = df[mask].copy()

    # sqft_above = sqft_living - sqft_basement בדיוק → מידע כפול
    report["above_is_redundant"] = bool(
        (df["sqft_living"] == df["sqft_above"] + df["sqft_basement"]).all())
    df = df.drop(columns=["sqft_above"])

    # שנת בנייה → גיל בניין (מקדם פרשני יותר)
    df["age"] = REFERENCE_YEAR - df["yr_built"]
    df = df.drop(columns=["yr_built"])

    report["rows_after"] = len(df)
    return df.reset_index(drop=True), report


def build_transformer() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # רשת ביטחון לשדה חסר בתחזית
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ], verbose_feature_names_out=False).set_output(transform="pandas")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
