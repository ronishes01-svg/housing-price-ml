import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import preprocessing as pp

st.set_page_config(page_title="ניבוי מחירי דירות", page_icon="🏠", layout="wide")

# CSS דרך st.html — st.markdown שובר בלוק <style> עם שורות ריקות
st.html("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { direction: rtl; }
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], h1, h2, h3, p, li { text-align: right; }
[data-testid="stMetric"] { background: #f9f9f7; border: 1px solid #e1e0d9; border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
.ltr { direction: ltr; unicode-bidi: isolate; }
</style>
""")

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"

STEPS = ["1 · הכרת הנתונים", "2 · פריפרוססינג", "3 · חלוקה", "4 · אימון",
         "5 · הערכה", "6 · פרדיקציה"]
READY = 2  # כמה שלבים כבר בנויים


@st.cache_data
def get_raw():
    return pp.load_raw()


@st.cache_data
def get_clean():
    return pp.clean(get_raw())


def fmt_usd(x):
    return f"${x:,.0f}"


def base_layout(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(color=INK2, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    return fig


st.title("🏠 ניבוי מחירי דירות — רגרסיה לינארית")
st.caption("תרגיל למידת מכונה · חלק א · כל שלב בתהליך בלשונית משלו")

step = st.radio("שלב", STEPS, horizontal=True, label_visibility="collapsed", key="step")
idx = STEPS.index(step) + 1
raw = get_raw()

if idx > READY:
    st.info("השלב הזה ייבנה בהמשך — מתקדמים שלב אחרי שלב.")
    st.stop()

# ---------------------------------------------------------------- שלב 1
if idx == 1:
    st.header("הכרת הנתונים")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("דירות", f"{len(raw):,}")
    c2.metric("עמודות", raw.shape[1])
    c3.metric("מחיר חציוני", fmt_usd(raw.price.median()))
    c4.metric("ערכים חסרים", int(raw.isna().sum().sum()))

    st.subheader("דוגמה מהנתונים")
    st.dataframe(raw.head(10), width="stretch")

    st.subheader("קשר כל משתנה למחיר (קורלציה)")
    corr = raw.corr(numeric_only=True)["price"].drop("price").sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h", marker_color=BLUE,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="קורלציה עם המחיר")
    st.plotly_chart(base_layout(fig, 400), width="stretch")
    st.caption("שטח המגורים הוא המשתנה הקשור ביותר למחיר. גודל המגרש, המצב ושנת הבנייה כמעט לא קשורים אליו ישירות.")

# ---------------------------------------------------------------- שלב 2
if idx == 2:
    clean, rep = get_clean()
    st.header("פריפרוססינג — הכנת הנתונים למודל")

    c1, c2, c3 = st.columns(3)
    c1.metric("שורות לפני", f"{rep['rows_before']:,}")
    c2.metric("שורות אחרי", f"{rep['rows_after']:,}",
              delta=f"-{rep['rows_before'] - rep['rows_after']}", delta_color="off")
    c3.metric("משתנים למודל", f"{len(pp.FEATURES)} → {len(pp.build_transformer().fit(*pp.split_xy(clean)).get_feature_names_out())}")

    # --- 2.1
    st.subheader("2.1 ערכים חסרים וכפילויות")
    a, b = st.columns(2)
    a.metric("ערכים חסרים", rep["missing"])
    b.metric("שורות כפולות", rep["duplicates"])
    st.write("הנתונים נקיים. בכל זאת יש בתהליך **SimpleImputer** (כמו במחברת 01), "
             "כדי שדירה חדשה עם שדה חסר לא תפיל את התחזית.")

    # --- 2.2
    st.subheader("2.2 חריגים — שיטת IQR (פי 1.5)")
    st.dataframe(pp.outlier_summary(raw, ["price", "sqft_living", "sqft_lot", "bedrooms", "bathrooms"]),
                 hide_index=True, width="stretch")
    lo, hi = rep["price_bounds"]
    st.write(f"**ההחלטה:** מסירים רק חריגי מחיר — {rep['price_outliers']} דירות מעל "
             f"{fmt_usd(hi)}. בשאר העמודות החריגים הם בתים אמיתיים (מגרש גדול, בית גדול), "
             "והסרה של כולם הייתה מוחקת כ-15% מהנתונים.")

    kept = raw.price[raw.price.between(lo, hi)]
    removed = raw.price[~raw.price.between(lo, hi)]
    fig = go.Figure()
    fig.add_histogram(x=kept, name="נשארות", marker_color=BLUE, xbins=dict(size=50_000),
                      marker_line=dict(color="#fcfcfb", width=2),
                      hovertemplate="$%{x:,.0f}: %{y} דירות<extra>נשארות</extra>")
    fig.add_histogram(x=removed, name="מוסרות (חריגות)", marker_color=ORANGE, xbins=dict(size=50_000),
                      marker_line=dict(color="#fcfcfb", width=2),
                      hovertemplate="$%{x:,.0f}: %{y} דירות<extra>מוסרות</extra>")
    fig.add_vline(x=hi, line=dict(color=INK2, width=1, dash="dot"),
                  annotation_text=f"גבול עליון {fmt_usd(hi)}", annotation_position="top right")
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_xaxes(title="מחיר ($)", tickformat="$,.0s")
    fig.update_yaxes(title="מספר דירות")
    st.plotly_chart(base_layout(fig), width="stretch")

    # --- 2.3
    st.subheader("2.3 הנדסת משתנים")
    st.markdown(
        f"- **הוסר `sqft_above`** — בכל {rep['rows_after']:,} השורות מתקיים "
        "`sqft_living = sqft_above + sqft_basement`. אותו מידע פעמיים מבלבל את הרגרסיה (מולטיקוליניאריות).\n"
        f"- **`yr_built` הוחלף ב-`age`** = {pp.REFERENCE_YEAR} פחות שנת הבנייה. "
        "המקדם יתפרש כ\"כמה משנה כל שנת גיל\".")

    # --- 2.4
    st.subheader("2.4 קידוד משתנים קטגוריאליים — One-Hot")
    st.write("`view` (0–4) ו-`condition` (1–5) הם ציונים, לא כמויות: הקפיצה מ-0 ל-1 בנוף "
             "לא בהכרח שווה לקפיצה מ-3 ל-4. One-Hot נותן לכל ציון השפעה משלו. "
             "הקטגוריה הראשונה (`drop='first'`) משמשת בסיס להשוואה, כדי למנוע כפילות.")
    sample = clean[pp.CATEGORICAL].head(6)
    enc = pd.get_dummies(sample.astype("category").apply(
        lambda s: s.cat.set_categories(sorted(clean[s.name].unique()))),
        drop_first=True).astype(int)
    l, r = st.columns([1, 3])
    l.caption("לפני")
    l.dataframe(sample, hide_index=True, width="stretch")
    r.caption("אחרי")
    r.dataframe(enc, hide_index=True, width="stretch")
    st.caption("בבדיקה מקדימה One-Hot שיפר את R² על נתוני המבחן מ-0.569 ל-0.584 לעומת השארת הציונים כמספרים.")

    # --- 2.5
    st.subheader("2.5 נרמול — StandardScaler")
    st.write("כל משתנה מספרי מועבר לממוצע 0 וסטיית תקן 1. זה לא משנה את התחזיות של רגרסיה לינארית, "
             "אבל מאפשר להשוות מקדמים: מקדם גדול = משתנה משפיע יותר.")
    num = clean[pp.NUMERIC]
    scaled = (num - num.mean()) / num.std(ddof=0)
    st.dataframe(pd.DataFrame({
        "ממוצע לפני": num.mean().round(1), "ס\"ת לפני": num.std(ddof=0).round(1),
        "ממוצע אחרי": scaled.mean().round(2).abs(), "ס\"ת אחרי": scaled.std(ddof=0).round(2),
    }), width="stretch")

    st.warning("**חשוב:** הקידוד והנרמול *מותאמים רק על נתוני האימון* (בשלב 3). "
               "כאן הם מוצגים על כל הנתונים רק להמחשה. התאמה על כל הנתונים הייתה מדליפה "
               "מידע מנתוני המבחן ומייפה את הציון.")

    st.subheader("סיכום — הנתונים שנכנסים לשלב הבא")
    st.dataframe(clean.head(10), width="stretch")
    st.download_button("הורדת הנתונים הנקיים (CSV)", clean.to_csv(index=False).encode("utf-8-sig"),
                       "housing_clean.csv", "text/csv")
