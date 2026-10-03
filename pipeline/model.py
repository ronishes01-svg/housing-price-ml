"""שלבים 3–6 — חלוקה, אימון, הערכה ופרדיקציה."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import preprocessing as pp

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split(clean: pd.DataFrame):
    X, y = pp.split_xy(clean)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_model() -> Pipeline:
    # הקידוד והנרמול בתוך ה-Pipeline → מותאמים רק על מה שמועבר ל-fit (נתוני האימון)
    return Pipeline([("prep", pp.build_transformer()), ("reg", LinearRegression())])


def train(X_train, y_train) -> tuple[Pipeline, np.ndarray]:
    model = build_model().fit(X_train, y_train)
    cv = cross_val_score(build_model(), X_train, y_train, cv=5, scoring="r2")
    return model, cv


def metrics(y_true, y_pred) -> dict:
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def coefficients(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["reg"].coef_
    return (pd.DataFrame({"feature": names, "coef": coefs})
            .assign(abs=lambda d: d.coef.abs())
            .sort_values("abs", ascending=True).drop(columns="abs"))


def contributions(model: Pipeline, row: pd.DataFrame) -> pd.Series:
    """כמה כל משתנה מוסיף/מוריד מהמחיר הבסיסי (מקדם × ערך מנורמל)."""
    prep, reg = model.named_steps["prep"], model.named_steps["reg"]
    z = prep.transform(row).iloc[0]
    return pd.Series(reg.coef_ * z.values, index=z.index)


def comparables(clean: pd.DataFrame, house: dict, k: int = 15) -> pd.DataFrame:
    """הדירות הדומות ביותר בנתונים (שטח, חדרים, רחצה, גיל) — לבדיקת סבירות."""
    cols = ["sqft_living", "bedrooms", "bathrooms", "age"]
    d = clean[cols]
    dist = (((d - pd.Series(house)[cols]) / d.std()) ** 2).sum(axis=1) ** 0.5
    return clean.loc[dist.nsmallest(k).index]
