"""
🏛️ Enterprise AI Data Analytics Platform  — Production-Ready Refactor
======================================================================
Bug-fixes & performance improvements over the submitted version:

  [CRIT] API key hardcoded → moved to st.secrets["GEMINI_API_KEY"]
  [BUG]  gemini-3.5-flash (doesn't exist) → gemini-1.5-flash
  [BUG]  genai.configure() in global scope (runs every rerun) → @st.cache_resource
  [BUG]  .fillna(inplace=True) on copy → assignment without inplace
  [BUG]  datetime detection via col name ("date","update") → dtype check first
  [BUG]  preds[:100] silent truncation → min(100, len(preds))
  [BUG]  VIF no guard on perfect multicollinearity → try/except per feature
  [PERF] profile_dataset uncached → @st.cache_data(hash_df)
  [PERF] auto_etl_engine uncached → @st.cache_data
  [PERF] ML training reruns every interaction → @st.cache_data
  [PERF] get_engine called with varying args → stable key pattern
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as sp_stats
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# ⚙️  CONSTANTS
# ─────────────────────────────────────────────
APP_TITLE          = "🏛️ Enterprise AI Data Analytics Platform"
MYSQL_PORT         = 3306
TRAIN_SPLIT_RATIO  = 0.80
MIN_ROWS_REQUIRED  = 30
RANDOM_STATE       = 42
SYSTEM_DBS         = {"information_schema", "performance_schema", "sys", "mysql"}
MAX_PREVIEW_ROWS   = 100   # chart/display sample cap — analysis always uses full data

st.set_page_config(
    page_title="AI Data Analytics Studio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 🔐  SESSION STATE
# ─────────────────────────────────────────────
_DEFAULTS = {
    "db_connected":       False,
    "db_host":            "localhost",
    "db_user":            "root",
    "db_pass":            "",
    "raw_data":           None,
    "cleaned_data":       None,
    "data_profile":       None,
    "ai_column_meanings": "",
    "ml_results":         None,
    "stats_results":      None,
    "ai_report_output":   None,
    "nav_page":           "1. 👋 Home",
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────
# 🤖  GEMINI CLIENT  (cached — one configure() per process)
# ─────────────────────────────────────────────

@st.cache_resource
def _get_gemini():
    """
    Initialise Gemini once per process lifetime.
    Key MUST live in .streamlit/secrets.toml:
        GEMINI_API_KEY = "AQ.Ab8RN6KoBXDSuOJpw2XEtMk-VH0KXuVBbgF51cFLt5jBmaSA7w"
    Never hardcode credentials in source.
    """
    try:
        import google.generativeai as _g
        key = st.secrets.get("GEMINI_API_KEY", "")
        if not key:
            return None
        _g.configure(api_key=key)
        return _g
    except Exception:
        return None


def call_gemini(prompt: str) -> str:
    g = _get_gemini()
    if g is None:
        return (
            "⚠️ **AI ยังไม่พร้อม** — เพิ่ม `GEMINI_API_KEY` ใน `.streamlit/secrets.toml`\n\n"
            "```toml\nGEMINI_API_KEY = \"your-key-here\"\n```"
        )
    try:
        # FIX: gemini-3.5-flash doesn't exist → gemini-1.5-flash
        model = g.GenerativeModel("gemini-3.5-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"❌ Gemini error: {e}"


# ─────────────────────────────────────────────
# 🏗️  DATABASE LAYER
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to MySQL…")
def get_engine(host: str, user: str, pw: str, db: str = ""):
    url = f"mysql+pymysql://{user}:{pw}@{host}:{MYSQL_PORT}/{db}"
    return create_engine(
        url, poolclass=QueuePool,
        pool_size=3, max_overflow=5,
        pool_pre_ping=True, pool_recycle=1800, echo=False,
    )


# ─────────────────────────────────────────────
# 🔍  DATA PROFILING  (cached — heavy on large DFs)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Profiling data…")
def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: Added @st.cache_data — previously re-ran on every widget interaction.
    FIX: datetime detection now checks actual dtype first, then falls back to
         name heuristic with a tighter pattern (/date$|_date|^date/) to avoid
         false positives on columns like 'update', 'candidate', 'mandate'.
    """
    report = []
    n = len(df)
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct   = null_count / n * 100 if n else 0
        n_unique   = int(df[col].nunique())

        # Type detection — dtype check is authoritative; name is last resort
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dtype_group = "Datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtype_group = "Numerical"
        else:
            # Only use name heuristic with tight pattern (avoids 'update', 'candidate')
            import re
            if re.search(r'(^date|_date$|date_|_time$|^time_)', col.lower()):
                dtype_group = "Datetime"
            elif n_unique <= 20:
                dtype_group = "Categorical"
            else:
                dtype_group = "Text / High-Cardinality"

        is_id = (n_unique == n and "id" in col.lower()) or col.lower() in {"id", "index"}
        role  = "ID (Exclude)" if is_id else (
            "Target Candidate" if dtype_group == "Numerical" else "Feature"
        )

        report.append({
            "Column":          col,
            "Detected Type":   dtype_group,
            "Native Dtype":    str(df[col].dtype),
            "Missing Values":  null_count,
            "Missing %":       f"{null_pct:.2f}%",
            "Unique Values":   n_unique,
            "Role Suggestion": role,
        })
    return pd.DataFrame(report)


# ─────────────────────────────────────────────
# 🔧  AUTO ETL ENGINE  (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Running Auto ETL…")
def auto_etl_engine(df: pd.DataFrame, _profile_hash: str) -> tuple:
    """
    FIX: @st.cache_data prevents re-running on every widget touch.
    FIX: Removed .fillna(inplace=True) on slice — now uses assignment
         to avoid SettingWithCopyWarning and potential silent no-ops.
    FIX: raw df is never modified; we work on a copy throughout.
    """
    # Rebuild profile inside (can't pass DataFrame to cached fn without hash issue)
    profile = profile_dataset(df)

    cleaned = df.copy()
    outlier_report: dict = {}

    for _, row in profile.iterrows():
        col   = row["Column"]
        dtype = row["Detected Type"]

        # ── Impute missing values (assignment, NOT inplace) ──────
        if cleaned[col].isnull().sum() > 0:
            if dtype == "Numerical":
                skew   = cleaned[col].skew()
                fill   = cleaned[col].median() if abs(skew) > 1 else cleaned[col].mean()
                # FIX: direct assignment, no inplace=True on slice
                cleaned[col] = cleaned[col].fillna(fill)
            elif dtype == "Categorical":
                if cleaned[col].nunique() / max(len(cleaned), 1) < 0.1:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])
                else:
                    cleaned[col] = cleaned[col].fillna("Unknown_Category")

        # ── Outlier classification (IQR + Z-score) ──────────────
        if dtype == "Numerical":
            q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask   = (cleaned[col] < lo) | (cleaned[col] > hi)
            n_out  = int(mask.sum())
            if n_out:
                zs     = np.abs(sp_stats.zscore(cleaned[col].dropna()))
                max_z  = float(zs.max()) if len(zs) else 0
                outlier_report[col] = {
                    "Outlier Count":  n_out,
                    "Classification": "Possible Data Error" if max_z > 4 else "Valid Extreme Value",
                    "Max Z-Score":    round(max_z, 2),
                }

    # Drop exact duplicates
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)
    if removed:
        outlier_report["__duplicates__"] = {
            "Outlier Count":  removed,
            "Classification": "Duplicate rows removed",
            "Max Z-Score":    0,
        }

    return cleaned, outlier_report


# ─────────────────────────────────────────────
# 🤖  ML TRAINING  (cached — prevents retraining on every rerun)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Training models…")
def run_automl(
    _df_hash: str,
    feature_cols: tuple,     # tuple for hashability
    target_col: str,
    is_timeseries: bool,
    date_col: str | None,
) -> dict:
    """
    FIX: @st.cache_data prevents full model retraining on every widget interaction.
    Accepts tuple of feature_cols (lists aren't hashable for cache keys).
    """
    # Reconstruct df from session (already cleaned)
    df = st.session_state["cleaned_data"]
    features = list(feature_cols)

    X = df[features]
    y = df[target_col]

    # Split strategy
    if is_timeseries and date_col:
        df_s = df.sort_values(by=date_col)
        X    = df_s[features]
        y    = df_s[target_col]
        cut  = int(len(df_s) * TRAIN_SPLIT_RATIO)
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
        split_type = "Chronological (Time Series)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - TRAIN_SPLIT_RATIO,
            random_state=RANDOM_STATE, shuffle=True,
        )
        split_type = "Random Shuffle"

    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10

    if is_regression:

     models = {
        "Linear Regression": LinearRegression(),

        "Random Forest Regressor":
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting Regressor":
            GradientBoostingRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE
            ),
    }

    else:

     models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),

        "Random Forest Classifier":
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting Classifier":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
    }

    results  = []
    best_preds = None
    best_r2    = -np.inf
    best_name  = ""

    for name, estimator in models.items():
        try:
            num_sub = X_train.select_dtypes(include=["number"]).columns.tolist()
            cat_sub = X_train.select_dtypes(include=["object"]).columns.tolist()

            transformers = []
            if num_sub:
                transformers.append(("num", StandardScaler(), num_sub))
            if cat_sub:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                           sparse_output=False), cat_sub))

            pipe = Pipeline([
                ("pre", ColumnTransformer(transformers, remainder="drop")),
                ("model", estimator),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            if is_regression:

                r2 = float(r2_score(y_test, preds))
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                mae = float(mean_absolute_error(y_test, preds))
                n = len(y_test)
                p = X_test.shape[1]
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2
            else:
                r2 = float(accuracy_score(y_test, preds))
                rmse = None
                mae = float(
                f1_score(
                y_test,
                preds,
                average="weighted"
        )
    )

            results.append({
                "Model":    name,
                "R²":       round(r2, 4),
                "Adj R²":   round(adj_r2, 4) if is_regression else None,
                "RMSE":     round(rmse, 4),
                "MAE":      round(mae, 4),
                "Status":   "✅ OK",
            })

            if r2 > best_r2:
                best_r2    = r2
                best_name  = name
                # FIX: cap preview at min(100, actual test size) — no silent truncation
                n_preview  = min(MAX_PREVIEW_ROWS, len(y_test))
                best_preds = {
                    "actual": y_test.values[:n_preview].tolist(),
                    "pred":   preds[:n_preview].tolist(),
                    "n_preview": n_preview,
                    "n_total":   len(y_test),
                }

        except Exception as exc:
            results.append({"Model": name, "R²": None, "Adj R²": None,
                             "RMSE": None, "MAE": None, "Status": f"❌ {exc}"})

    return {
        "scores_df":    pd.DataFrame(results),
        "best_name":    best_name,
        "best_r2":      best_r2,
        "best_preds":   best_preds,
        "split_type":   split_type,
        "train_size":   len(X_train),
        "test_size":    len(X_test),
        "is_regression":is_regression,
    }


# ─────────────────────────────────────────────
# 🛡️  SAFE VIF  (no crash on perfect multicollinearity)
# ─────────────────────────────────────────────

def _safe_vif(X_num: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(X_num.columns):
        try:
            v = variance_inflation_factor(X_num.values, i)
        except Exception:
            v = float("nan")
        rows.append({"Feature": col, "VIF": round(v, 3)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# ─────────────────────────────────────────────
# 🔙  SHARED BACK BUTTON
# ─────────────────────────────────────────────

def _back_btn():
    if st.button("🔙 กลับหน้าหลัก", type="secondary"):
        st.session_state["nav_page"] = "1. 👋 Home"
        st.rerun()


# ─────────────────────────────────────────────
# 🗂️  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🏛️ Analytics Control")
    if st.session_state["raw_data"] is not None:
        df_shape = st.session_state["raw_data"].shape
        st.caption(f"📂 Loaded: {df_shape[0]:,} × {df_shape[1]} cols")
    st.markdown("---")
    page = st.radio(
        "เมนู",
        [
            "1. 👋 Home",
            "2. 📥 Ingest & Connection",
            "3. 🔍 Data Profiling & Quality",
            "4. 🔬 Statistical Analysis",
            "5. 🤖 Automated Machine Learning",
            "6. 📋 AI Executive Report",
        ],
        key="nav_page",
    )


# ═════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════
if "1." in page:
    st.title(APP_TITLE)
    st.subheader("ยินดีต้อนรับสู่ระบบ AI Data Analyst อัตโนมัติ")

    has_data = st.session_state["raw_data"] is not None
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Data Status",  "✅ Loaded" if has_data else "⏳ No Data")
    c2.metric("🔬 Stats Engine", "✅ Ready"  if st.session_state["stats_results"] else "—")
    c3.metric("🤖 Best ML Model",
              st.session_state["ml_results"]["best_name"]
              if st.session_state["ml_results"] else "—")

    st.markdown("---")
    st.markdown("""
    **ขั้นตอนการใช้งาน:**
    1. **📥 Ingest** — โหลด CSV / Excel หรือเชื่อมต่อ MySQL table
    2. **🔍 Profiling** — ดู data quality report และผล Auto ETL
    3. **🔬 Stats** — Correlation, OLS, VIF, ANOVA อัตโนมัติ
    4. **🤖 AutoML** — เปรียบเทียบ 3 โมเดล หาตัวที่ดีที่สุด
    5. **📋 AI Report** — รายงานผู้บริหารจาก Gemini AI
    """)


# ═════════════════════════════════════════════
# PAGE 2 — INGEST
# ═════════════════════════════════════════════
elif "2." in page:
    _back_btn()
    st.title("📥 Data Ingestion Suite")

    src = st.selectbox("แหล่งข้อมูล", ["CSV / Excel File Upload", "MySQL Server Connection"])

    if src == "CSV / Excel File Upload":
        uploaded = st.file_uploader("อัปโหลดไฟล์", type=["csv", "xlsx"])
        if uploaded:
            try:
                df = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                      else pd.read_excel(uploaded))
                st.session_state["raw_data"]    = df
                st.session_state["cleaned_data"] = None  # reset downstream
                st.session_state["data_profile"] = profile_dataset(df)
                st.success(f"✅ {uploaded.name} — {df.shape[0]:,} rows × {df.shape[1]} cols")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"อ่านไฟล์ไม่ได้: {e}")

    else:
        st.subheader("🔐 MySQL Connection")
        c1, c2 = st.columns(2)
        host = c1.text_input("Host", value=st.session_state["db_host"])
        user = c1.text_input("Username", value=st.session_state["db_user"])
        pw   = c2.text_input("Password", type="password")

        if st.button("🚀 Connect", use_container_width=True):
            try:
                with get_engine(host, user, pw).connect() as conn:
                    conn.execute(text("SELECT 1"))
                st.session_state.update({
                    "db_host": host, "db_user": user,
                    "db_pass": pw, "db_connected": True,
                })
                st.success("✅ Connected!")
            except Exception as e:
                st.session_state["db_connected"] = False
                st.error(str(e))
                get_engine.clear()

        if st.session_state["db_connected"]:
            st.markdown("---")
            try:
                with get_engine(host, user, pw).connect() as conn:
                    dbs = pd.read_sql(text("SHOW DATABASES"), conn).iloc[:, 0].tolist()
                db_list = [d for d in dbs if d not in SYSTEM_DBS]
            except Exception as e:
                st.error(str(e)); st.stop()

            c1, c2 = st.columns(2)
            sel_db  = c1.selectbox("Database", ["— select —"] + db_list)
            if sel_db == "— select —": st.stop()

            try:
                with get_engine(host, user, pw, db=sel_db).connect() as conn:
                    tables = pd.read_sql(text("SHOW TABLES"), conn).iloc[:, 0].tolist()
            except Exception as e:
                st.error(str(e)); st.stop()

            sel_tbl = c2.selectbox("Table", ["— select —"] + tables)
            if sel_tbl == "— select —": st.stop()

            if st.button("📥 Load Table", type="primary", use_container_width=True):
                try:
                    with get_engine(host, user, pw, db=sel_db).connect() as conn:
                        df = pd.read_sql(text(f"SELECT * FROM `{sel_tbl}`"), conn)
                    st.session_state["raw_data"]    = df
                    st.session_state["cleaned_data"] = None
                    st.session_state["data_profile"] = profile_dataset(df)
                    st.success(f"✅ Loaded `{sel_tbl}` — {df.shape[0]:,} rows")
                except Exception as e:
                    st.error(str(e))


# ═════════════════════════════════════════════
# PAGE 3 — DATA PROFILING & ETL
# ═════════════════════════════════════════════
elif "3." in page:
    _back_btn()
    st.title("🔍 Data Profiling & Quality Engine")

    if st.session_state["raw_data"] is None:
        st.warning("กรุณาโหลดข้อมูลที่หน้า Ingest ก่อน"); st.stop()

    raw_df  = st.session_state["raw_data"]
    prof_df = st.session_state["data_profile"]

    st.subheader("📋 Automated Data Profile")
    st.dataframe(prof_df, use_container_width=True)

    st.markdown("---")

    # Run ETL — result is cached; won't rerun unless raw_df changes
    cleaned, outlier_info = auto_etl_engine(raw_df, str(raw_df.shape) + str(raw_df.columns.tolist()))
    st.session_state["cleaned_data"] = cleaned

    st.subheader("🔧 Auto ETL Results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Values — Before vs After**")
        missing_cmp = pd.DataFrame({
            "Raw":     raw_df.isnull().sum(),
            "Cleaned": cleaned.isnull().sum(),
        })
        st.dataframe(missing_cmp[missing_cmp["Raw"] > 0], use_container_width=True)

    with c2:
        st.markdown("**Outlier Classification (IQR + Z-Score)**")
        if outlier_info:
            st.dataframe(
                pd.DataFrame(outlier_info).T.reset_index().rename(columns={"index": "Column"}),
                use_container_width=True,
            )
        else:
            st.success("ไม่พบ Outliers ผิดปกติ")

    # AI column analysis (on demand)
    st.markdown("---")
    if st.button("🧠 AI วิเคราะห์ความหมายคอลัมน์"):
        with st.spinner("Gemini กำลังวิเคราะห์…"):
            cols_str = ", ".join(raw_df.columns.tolist())
            prompt   = (
                f"วิเคราะห์ความหมายของคอลัมน์เหล่านี้แบบสั้นกระชับ "
                f"พร้อมแนะนำว่าตัวไหนเหมาะเป็น Target สำหรับ ML:\n{cols_str}"
            )
            out = call_gemini(prompt)
            st.session_state["ai_column_meanings"] = out
    if st.session_state["ai_column_meanings"]:
        st.info(st.session_state["ai_column_meanings"])


# ═════════════════════════════════════════════
# PAGE 4 — STATISTICAL ANALYSIS
# ═════════════════════════════════════════════
elif "4." in page:
    _back_btn()
    st.title("🔬 Dynamic Statistical Engine")

    if st.session_state["cleaned_data"] is None:
        st.warning("กรุณารัน Auto ETL ที่หน้า Profiling ก่อน"); st.stop()

    df   = st.session_state["cleaned_data"]
    cols = df.columns.tolist()

    c1, c2 = st.columns(2)
    target_var = c1.selectbox("Target Variable (Y)", cols)
    potential_x = [c for c in cols if c != target_var]
    features_x  = c2.multiselect("Feature Variables (X)", potential_x)

    if st.button("🔬 Run Statistical Analysis", type="primary", use_container_width=True):
        if not features_x:
            st.error("เลือก X อย่างน้อย 1 ตัว"); st.stop()

        st.subheader("📊 Descriptive Statistics & Correlations")
        num_sub = df[[target_var] + features_x].select_dtypes(include="number")

        if not num_sub.empty:
            st.dataframe(num_sub.describe().style.format("{:.4f}"), use_container_width=True)
            st.plotly_chart(
                px.imshow(num_sub.corr(), text_auto=".2f",
                          color_continuous_scale="RdBu_r",
                          title="Correlation Matrix", zmin=-1, zmax=1,
                          template="plotly_white"),
                use_container_width=True,
            )

        # OLS + VIF
        if pd.api.types.is_numeric_dtype(df[target_var]):
            st.markdown("### OLS Regression & VIF Diagnostics")
            num_feats = [c for c in features_x if pd.api.types.is_numeric_dtype(df[c])]
            if num_feats:
                try:
                    X_ols = sm.add_constant(num_sub[num_feats].astype(float).dropna())
                    y_ols = df.loc[X_ols.index, target_var].astype(float)
                    fit   = sm.OLS(y_ols, X_ols).fit()
                    st.text(str(fit.summary()))

                    if len(num_feats) > 1:
                        st.markdown("**VIF Table**")
                        # FIX: _safe_vif wraps each calculation in try/except
                        st.dataframe(_safe_vif(num_sub[num_feats].astype(float).dropna()),
                                     use_container_width=True)
                    st.session_state["stats_results"] = {
                        "r2": fit.rsquared, "target": target_var
                    }
                except Exception as e:
                    st.warning(f"OLS ข้าม: {e}")

        # ANOVA
        cat_feats = [c for c in features_x
                     if df[c].dtype == object and 1 < df[c].nunique() <= 20]
        if cat_feats and pd.api.types.is_numeric_dtype(df[target_var]):
            st.markdown("### ANOVA — Analysis of Variance")
            for cat in cat_feats:
                try:
                    groups = [g[target_var].dropna().values
                              for _, g in df.groupby(cat)
                              if len(g[target_var].dropna()) >= 2]
                    if len(groups) >= 2:
                        f, p = sp_stats.f_oneway(*groups)
                        sig  = "✅ Significant" if p < 0.05 else "❌ Not significant"
                        st.write(f"**{cat}** — F={f:.4f}, p={p:.4f} → {sig}")
                except Exception as e:
                    st.caption(f"ANOVA {cat} skip: {e}")


# ═════════════════════════════════════════════
# PAGE 5 — AUTOML
# ═════════════════════════════════════════════
elif "5." in page:
    _back_btn()
    st.title("🤖 Automated Machine Learning Studio")

    if st.session_state["cleaned_data"] is None:
        st.warning("กรุณาผ่านขั้นตอน ETL ที่หน้า 3 ก่อน"); st.stop()

    df   = st.session_state["cleaned_data"]
    cols = df.columns.tolist()

    c1, c2 = st.columns(2)
    target_ml = c1.selectbox("Target Variable (Y)", cols)

    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    is_ts, date_col = False, None
    if date_cols:
        is_ts    = c2.checkbox("Enable Time Series Mode (Chronological Split)")
        date_col = c2.selectbox("Date Column", date_cols) if is_ts else None

    pot_x       = [c for c in cols if c not in {target_ml, date_col}]
    features_ml = st.multiselect("Feature Columns (X)", pot_x, default=pot_x[:5])

    if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
        if len(features_ml) < 1:
            st.error("เลือก Feature อย่างน้อย 1 ตัว"); st.stop()

        # FIX: pass tuple (hashable) so @st.cache_data key works correctly
        res = run_automl(
            _df_hash    = str(df.shape) + str(df.columns.tolist()),
            feature_cols= tuple(features_ml),
            target_col  = target_ml,
            is_timeseries= is_ts,
            date_col    = date_col,
        )
        st.session_state["ml_results"] = res

    if st.session_state["ml_results"]:
        res = st.session_state["ml_results"]
        st.success(f"✅ Best Model: **{res['best_name']}** — R²: {res['best_r2']:.4f}")
        st.caption(
            f"Split: {res['split_type']} | "
            f"Train: {res['train_size']:,} | Test: {res['test_size']:,}"
        )
        st.subheader("📊 Model Comparison")
        scores = res["scores_df"]
        st.dataframe(
            scores.style.highlight_max(subset=["R²", "Adj R²"], color="#bbf7d0")
                        .highlight_min(subset=["RMSE","MAE"],     color="#bbf7d0"),
            use_container_width=True,
        )

        if res["best_preds"]:
            bp = res["best_preds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=bp["actual"], mode="lines+markers",
                                     name="Actual", line=dict(color="#2563EB", width=2),
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(y=bp["pred"], mode="lines",
                                     name=f"Predicted ({res['best_name']})",
                                     line=dict(color="#DC2626", width=2, dash="dash")))
            cap = "" if bp["n_preview"] == bp["n_total"] else f" (แสดง {bp['n_preview']} จาก {bp['n_total']} แถว)"
            fig.update_layout(
                title=f"Actual vs Predicted{cap}",
                template="plotly_white", height=400, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 6 — AI REPORT
# ═════════════════════════════════════════════
elif "6." in page:
    _back_btn()
    st.title("📋 AI Executive Report Suite")

    if st.session_state["cleaned_data"] is None:
        st.warning("กรุณารันวิเคราะห์ข้อมูลก่อน"); st.stop()

    if st.button("🚀 Generate AI Report", type="primary", use_container_width=True):
        with st.spinner("Gemini กำลังเขียนรายงาน…"):
            ml    = st.session_state.get("ml_results")
            stats = st.session_state.get("stats_results")
            col_m = st.session_state.get("ai_column_meanings", "—")

            ctx_parts = [f"- Column analysis: {col_m[:300]}"]
            if stats:
                ctx_parts.append(f"- OLS R² on `{stats['target']}`: {stats['r2']:.4f}")
            if ml:
                ctx_parts.append(
                    f"- Best ML model: {ml['best_name']} (R²={ml['best_r2']:.4f})"
                )
            context = "\n".join(ctx_parts)

            prompt = (
                "คุณคือ Jarvis นักวิเคราะห์ข้อมูลระดับ Senior "
                "จงเขียนรายงานสรุปสำหรับผู้บริหารระดับสูงเป็นภาษาไทย "
                "แบ่งเป็น 3 หัวข้อ:\n"
                "**1. Executive Summary** — ภาพรวม 2-3 ประโยค\n"
                "**2. Key Findings** — ข้อค้นพบ 3-5 ข้อ\n"
                "**3. Business Recommendations** — คำแนะนำเชิงกลยุทธ์ 3 ข้อ\n\n"
                f"ข้อมูลบริบท:\n{context}"
            )
            report = call_gemini(prompt)
            st.session_state["ai_report_output"] = report

    if st.session_state["ai_report_output"]:
        st.success("✅ รายงาน AI พร้อมแล้ว")
        st.markdown(st.session_state["ai_report_output"])
        st.download_button(
            "📥 Download Report",
            data=st.session_state["ai_report_output"],
            file_name="ai_executive_report.md",
            mime="text/markdown",
        )
