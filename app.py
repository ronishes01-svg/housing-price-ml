import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import model as ml
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
READY = 6  # כמה שלבים כבר בנויים


@st.cache_data
def get_raw():
    return pp.load_raw()


@st.cache_data
def get_clean():
    return pp.clean(get_raw())


@st.cache_resource
def get_model():
    clean, _ = get_clean()
    X_train, X_test, y_train, y_test = ml.split(clean)
    model, cv = ml.train(X_train, y_train)
    return model, cv, X_train, X_test, y_train, y_test


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

# ---------------------------------------------------------------- שלבים 3–6
LABELS = {
    "bedrooms": "חדרי שינה", "bathrooms": "חדרי רחצה", "sqft_living": "שטח מגורים",
    "sqft_lot": "שטח מגרש", "floors": "קומות", "waterfront": "חזית למים",
    "sqft_basement": "שטח מרתף", "age": "גיל הבניין",
    **{f"view_{i}": f"נוף {i} (מול 0)" for i in range(1, 5)},
    **{f"condition_{i}": f"מצב {i} (מול 1)" for i in range(2, 6)},
}

if idx >= 3:
    clean, rep = get_clean()
    model, cv, X_train, X_test, y_train, y_test = get_model()
    pred_train, pred_test = model.predict(X_train), model.predict(X_test)
    m_train, m_test = ml.metrics(y_train, pred_train), ml.metrics(y_test, pred_test)

# ---------------------------------------------------------------- שלב 3
if idx == 3:
    st.header("חלוקה לנתוני אימון ומבחן")
    c1, c2, c3 = st.columns(3)
    c1.metric("סה\"כ דירות", f"{len(clean):,}")
    c2.metric("אימון (80%)", f"{len(X_train):,}")
    c3.metric("מבחן (20%)", f"{len(X_test):,}")
    st.write(f"`train_test_split(test_size={ml.TEST_SIZE}, random_state={ml.RANDOM_STATE})` — "
             "החלוקה אקראית, ו-`random_state` קבוע מבטיח שבכל הרצה יוצאת אותה חלוקה.")
    st.markdown(
        "- **נתוני אימון** — מהם המודל לומד את המקדמים. גם הקידוד והנרמול מותאמים רק עליהם.\n"
        "- **נתוני מבחן** — מוסתרים מהמודל עד הסוף. עליהם מודדים את איכות המודל, "
        "כי הם מדמים דירות חדשות שהמודל לא ראה.")

    st.subheader("האם שתי הקבוצות דומות?")
    st.write("חלוקה טובה שומרת על התפלגות דומה בשתי הקבוצות — אחרת המבחן לא מייצג.")
    comp = pd.DataFrame({
        "אימון": [y_train.median(), y_train.mean(), X_train.sqft_living.median(), X_train.bedrooms.mean()],
        "מבחן": [y_test.median(), y_test.mean(), X_test.sqft_living.median(), X_test.bedrooms.mean()],
    }, index=["מחיר חציוני", "מחיר ממוצע", "שטח חציוני (sqft)", "חדרי שינה (ממוצע)"]).round(1)
    st.dataframe(comp, width="stretch")

    fig = go.Figure()
    fig.add_histogram(x=y_train, name="אימון", marker_color=BLUE, histnorm="percent",
                      xbins=dict(size=50_000), opacity=0.75,
                      hovertemplate="$%{x:,.0f}: %{y:.1f}%<extra>אימון</extra>")
    fig.add_histogram(x=y_test, name="מבחן", marker_color=ORANGE, histnorm="percent",
                      xbins=dict(size=50_000), opacity=0.75,
                      hovertemplate="$%{x:,.0f}: %{y:.1f}%<extra>מבחן</extra>")
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title="מחיר ($)", tickformat="$,.0s")
    fig.update_yaxes(title="% מהדירות בקבוצה")
    st.plotly_chart(base_layout(fig), width="stretch")
    st.caption("שתי ההתפלגויות כמעט חופפות — החלוקה מייצגת.")

# ---------------------------------------------------------------- שלב 4
if idx == 4:
    st.header("אימון — רגרסיה לינארית")
    st.write("המודל מחפש משוואה מהצורה **מחיר = b₀ + b₁·x₁ + b₂·x₂ + …** "
             "שממזערת את סכום ריבועי הטעויות על נתוני האימון.")
    st.code("model = Pipeline([('prep', transformer), ('reg', LinearRegression())])\n"
            "model.fit(X_train, y_train)", language="python")

    reg = model.named_steps["reg"]
    c1, c2, c3 = st.columns(3)
    c1.metric("b₀ — מחיר בסיס", fmt_usd(reg.intercept_))
    c2.metric("משתנים במודל", len(reg.coef_))
    c3.metric("R² באימות צולב (5 קיפולים)", f"{cv.mean():.3f}", delta=f"±{cv.std():.3f}", delta_color="off")
    st.caption("מחיר הבסיס = דירה ממוצעת בכל המשתנים המספריים, עם נוף 0 ומצב 1.")

    st.subheader("המקדמים — כמה כל משתנה מזיז את המחיר")
    coef = ml.coefficients(model)
    coef["label"] = coef.feature.map(LABELS)
    colors = [BLUE if c > 0 else ORANGE for c in coef.coef]
    fig = go.Figure(go.Bar(
        x=coef.coef, y=coef.label, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    fig.update_xaxes(title="השפעה על המחיר ($)", tickformat="$,.0s")
    st.plotly_chart(base_layout(fig, 520), width="stretch")
    st.markdown(
        "- **כחול** = מעלה את המחיר, **כתום** = מוריד.\n"
        "- משתנים מספריים: ההשפעה של **סטיית תקן אחת** (למשל כ-775 sqft בשטח המגורים). "
        "בזכות הנרמול אפשר להשוות ביניהם ישירות.\n"
        "- משתני One-Hot: ההפרש מול קטגוריית הבסיס (נוף 0 / מצב 1).")
    top = coef.iloc[-1]
    st.info(f"המשתנה המשפיע ביותר: **{top.label}** ({fmt_usd(top.coef)}).")

# ---------------------------------------------------------------- שלב 5
if idx == 5:
    st.header("הערכת ביצועי המודל — על נתוני המבחן")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m_test['R²']:.3f}")
    c2.metric("MAE — טעות ממוצעת", fmt_usd(m_test["MAE"]))
    c3.metric("RMSE", fmt_usd(m_test["RMSE"]))
    c4.metric("MAPE — טעות באחוזים", f"{m_test['MAPE']:.1%}")

    baseline = (y_test - y_train.mean()).abs().mean()
    st.markdown(
        f"- **R² = {m_test['R²']:.2f}** — המודל מסביר כ-{m_test['R²']:.0%} מהשונות במחירים.\n"
        f"- **MAE** — בממוצע התחזית רחוקה {fmt_usd(m_test['MAE'])} מהמחיר האמיתי.\n"
        f"- **RMSE** גדול מ-MAE כי הוא מעניש טעויות גדולות יותר — יש כמה דירות שהמודל מפספס בהרבה.\n"
        f"- **מול מודל נאיבי** (לנחש תמיד את המחיר הממוצע): טעות ממוצעת של {fmt_usd(baseline)}. "
        f"המודל שלנו משפר אותה ב-{1 - m_test['MAE'] / baseline:.0%}.")

    st.subheader("אימון מול מבחן — האם יש התאמת יתר?")
    tbl = pd.DataFrame({"אימון": m_train, "מבחן": m_test}).T
    st.dataframe(tbl.style.format({"R²": "{:.3f}", "MAE": "${:,.0f}", "RMSE": "${:,.0f}", "MAPE": "{:.1%}"}),
                 width="stretch")
    gap = m_train["R²"] - m_test["R²"]
    verdict = "אין סימן להתאמת יתר — המודל מכליל היטב לדירות חדשות." if gap < 0.05 else \
        "יש פער מסוים — ייתכן שהמודל לומד רעש מנתוני האימון."
    st.write(f"הפער ב-R² הוא {gap:+.3f}. {verdict}")

    st.subheader("תחזית מול מחיר אמיתי")
    lim = [0, max(y_test.max(), pred_test.max()) * 1.05]
    fig = go.Figure()
    fig.add_scatter(x=y_test, y=pred_test, mode="markers", name="דירות מבחן",
                    marker=dict(color=BLUE, size=8, opacity=0.55, line=dict(color="#fcfcfb", width=1)),
                    hovertemplate="אמיתי $%{x:,.0f}<br>תחזית $%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=lim, y=lim, mode="lines", name="תחזית מושלמת",
                    line=dict(color=INK2, width=2, dash="dot"), hoverinfo="skip")
    fig.update_xaxes(title="מחיר אמיתי ($)", tickformat="$,.0s", range=lim)
    fig.update_yaxes(title="תחזית המודל ($)", tickformat="$,.0s", range=lim)
    st.plotly_chart(base_layout(fig, 460), width="stretch")
    st.caption("נקודה על הקו = תחזית מדויקת. בדירות היקרות המודל נוטה להעריך בחסר.")

    st.subheader("מה מגביל את המודל")
    st.markdown(
        "- **אין מיקום** — שכונה ומיקוד הם הגורם החזק ביותר במחירי דיור, ואין אותם בנתונים.\n"
        "- **קשרים לא לינאריים** — ההשפעה של שטח נוסף לא קבועה בכל גודל דירה.\n"
        "- **טווח** — המודל אומן על דירות עד כ-1.13 מיליון דולר; מעבר לזה הוא לא אמין.")

# ---------------------------------------------------------------- שלב 6
if idx == 6:
    st.header("פרדיקציה — דירה חדשה")
    st.write("ערכי ברירת המחדל הם דירה מומצאת ריאליסטית: 3 חדרים, 2 חדרי רחצה, כ-170 מ\"ר, נבנתה ב-1985. שנו כדי לבדוק.")

    with st.form("house"):
        a, b, c = st.columns(3)
        bedrooms = a.number_input("חדרי שינה", 1, 8, 3)
        bathrooms = b.number_input("חדרי רחצה", 1.0, 6.0, 2.0, 0.25)
        floors = c.selectbox("קומות", [1.0, 1.5, 2.0, 2.5, 3.0], index=0)
        sqft_living = a.number_input("שטח מגורים (sqft)", 500, 8000, 1800, 50)
        sqft_basement = b.number_input("מתוכו מרתף (sqft)", 0, 2600, 0, 50)
        sqft_lot = c.number_input("שטח מגרש (sqft)", 600, 100_000, 7500, 100)
        yr_built = a.number_input("שנת בנייה", 1900, 2015, 1985)
        view = b.select_slider("נוף (0 גרוע – 4 מצוין)", [0, 1, 2, 3, 4], 0)
        condition = c.select_slider("מצב (1 גרוע – 5 מצוין)", [1, 2, 3, 4, 5], 3)
        waterfront = st.checkbox("חזית למים")
        submitted = st.form_submit_button("חשב מחיר", type="primary")

    house = dict(bedrooms=bedrooms, bathrooms=bathrooms, sqft_living=sqft_living,
                 sqft_lot=sqft_lot, floors=floors, waterfront=int(waterfront),
                 sqft_basement=min(sqft_basement, sqft_living), age=pp.REFERENCE_YEAR - yr_built,
                 view=view, condition=condition)
    row = pd.DataFrame([house])[pp.FEATURES]
    price = float(model.predict(row)[0])
    resid = y_test - pred_test
    lo, hi = price + resid.quantile(0.1), price + resid.quantile(0.9)

    c1, c2 = st.columns([1, 2])
    c1.metric("מחיר חזוי", fmt_usd(price))
    c2.metric("טווח סביר (80% מהמקרים)", f"{fmt_usd(max(lo, 0))} – {fmt_usd(hi)}".replace("$", "\\$"))
    st.caption(f"≈ {sqft_living / 10.764:,.0f} מ\"ר · הטווח מחושב מהטעויות של המודל על נתוני המבחן.")

    st.subheader("האם התוצאה סבירה?")
    comps = ml.comparables(clean, house)
    med = comps.price.median()
    diff = price / med - 1
    ok = abs(diff) <= 0.25
    st.write(f"15 הדירות הדומות ביותר בנתונים (שטח, חדרים, רחצה וגיל) נמכרו במחיר חציוני של "
             f"**{fmt_usd(med)}**. התחזית {'גבוהה' if diff > 0 else 'נמוכה'} ממנו ב-{abs(diff):.0%}.")
    (st.success if ok else st.warning)(
        "✅ התחזית סבירה — קרובה למחירי דירות דומות." if ok else
        "⚠️ התחזית רחוקה ממחירי דירות דומות — כדאי לבדוק אם הדירה חריגה (נוף, חזית למים, גודל מגרש).")
    if price > rep["price_bounds"][1]:
        st.warning("המחיר החזוי מעל הטווח שעליו אומן המודל (~1.13 מיליון דולר), ולכן פחות אמין.")
    with st.expander("הדירות הדומות"):
        st.dataframe(comps[["price", "sqft_living", "bedrooms", "bathrooms", "age", "view", "condition"]],
                     hide_index=True, width="stretch")

    st.subheader("ממה מורכב המחיר")
    contrib = ml.contributions(model, row)
    contrib = contrib[contrib.abs() > 1].sort_values()
    base = model.named_steps["reg"].intercept_
    fig = go.Figure(go.Bar(
        x=contrib.values, y=[LABELS.get(i, i) for i in contrib.index], orientation="h",
        marker_color=[BLUE if v > 0 else ORANGE for v in contrib.values],
        hovertemplate="%{y}: %{x:+$,.0f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    fig.update_xaxes(title=f"תוספת/הפחתה ממחיר הבסיס {fmt_usd(base)}", tickformat="$,.0s")
    st.plotly_chart(base_layout(fig, 380), width="stretch")
    st.caption("מחיר הבסיס + סכום כל העמודות = המחיר החזוי.")
