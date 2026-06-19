"""
============================
A production-grade Streamlit application for MySQL-connected
time-series forecasting with full statistical analysis.

Architecture: Modular, cached, secure, memory-safe.
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — prevents thread issues

# ─────────────────────────────────────────────
# ⚙️  CONSTANTS  (single source of truth)
# ─────────────────────────────────────────────
APP_TITLE          = "🚀 Professional BI & ML Studio"
MYSQL_PORT         = 3306
PREVIEW_LIMIT      = 500
TRAIN_SPLIT_RATIO  = 0.80
IQR_FENCE          = 1.5
LAG_DAYS           = [1, 7]
MIN_ROWS_REQUIRED  = 30          # minimum rows needed after cleaning
RANDOM_STATE       = 42
SYSTEM_DBS         = {"information_schema", "performance_schema", "sys", "mysql"}

MODEL_REGISTRY: dict = {
    "Random Forest":     lambda: RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "Linear Regression": lambda: LinearRegression(),
    "K-Neighbors":         lambda: KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF Kernel)":    lambda: SVR(kernel="rbf", C=100, epsilon=0.1),
    "Decision Tree":     lambda: DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
}

# Models that benefit from feature scaling
SCALE_REQUIRED = {"K-Neighbors", "SVR (RBF Kernel)"}

# ─────────────────────────────────────────────
# 🌐  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Data Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# 🔐  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────
_defaults = {
    "logged_in": False,
    "db_host":   "",
    "db_user":   "",
    "db_pass":   "",          # kept only in session memory, NOT in URL strings
    "results_df": None,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────
# 🏗️  DATABASE LAYER  (cached, pooled, injection-safe)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to database…")
def get_engine(host: str, user: str, pw: str, db: str = "") -> "sqlalchemy.engine.Engine":
    """
    Creates and caches a SQLAlchemy engine with connection pooling.
    Credentials are passed as parameters — never embedded in a URL stored
    in session state or logs.
    """
    url = f"mysql+pymysql://{user}:{pw}@{host}:{MYSQL_PORT}/{db}"
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=3,
        max_overflow=5,
        pool_pre_ping=True,       # auto-reconnect on stale connections
        pool_recycle=1800,        # recycle connections every 30 min
        echo=False,
    )


def _safe_identifier(name: str) -> str:
    """
    Validates that a column/table name contains only safe characters.
    Prevents SQL injection through identifier names.
    Raises ValueError on violation.
    """
    import re
    if not re.match(r'^[\w\u0E00-\u0E7F ]+$', name):
        raise ValueError(f"Unsafe identifier detected: '{name}'")
    return name


@st.cache_data(show_spinner="Fetching preview data…", ttl=300)
def fetch_preview(_engine_id: int, table: str, db: str, host: str, user: str, pw: str) -> pd.DataFrame:
    """Fetches a safe preview of a table (LIMIT applied server-side)."""
    safe_table = _safe_identifier(table)
    engine = get_engine(host, user, pw, db)
    with engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM `{safe_table}` LIMIT :lim"),
            conn,
            params={"lim": PREVIEW_LIMIT},
        )


@st.cache_data(show_spinner="Loading full dataset…", ttl=300)
def fetch_full_data(
    _engine_id: int,
    table: str,
    columns: tuple,       # tuple for hashability
    db: str,
    host: str,
    user: str,
    pw: str,
) -> pd.DataFrame:
    """Fetches selected columns from a table using parameterised-safe identifiers."""
    safe_table = _safe_identifier(table)
    safe_cols  = [_safe_identifier(c) for c in columns]
    col_str    = ", ".join(f"`{c}`" for c in safe_cols)
    engine     = get_engine(host, user, pw, db)
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT {col_str} FROM `{safe_table}`"), conn)


# ─────────────────────────────────────────────────────────────────
# 🔧  DATA ENGINEERING PIPELINE
# ─────────────────────────────────────────────────────────────────

def remove_outliers_iqr(series: pd.Series, fence: float = IQR_FENCE) -> pd.Series:
    """Returns a boolean mask: True = keep row."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = Q3 - Q1
    return (series >= Q1 - fence * iqr) & (series <= Q3 + fence * iqr)


def engineer_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    extra_x: list[str],
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline:
    1. Parse & index on date column
    2. Remove numeric outliers in target AFTER coercion (correct order)
    3. Resample to daily
    4. Add calendar & lag features
    Returns cleaned daily DataFrame or raises ValueError with a descriptive message.
    """
    df = df.copy()

    # --- Coerce target to numeric first, then remove outliers ---
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[remove_outliers_iqr(df[target_col].dropna()
                                 .reindex(df.index)
                                 .fillna(df[target_col]))]  # safe reindex

    # --- Parse date ---
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    if df.empty:
        raise ValueError("No valid rows remain after date parsing. Check your date column.")

    # --- Resample to daily ---
    agg_rules = {target_col: "sum"}
    for c in extra_x:
        if c in df.columns:
            agg_rules[c] = "mean"
    daily = df.resample("D").agg(agg_rules).dropna(subset=[target_col])

    # --- Calendar features ---
    daily["month"]      = daily.index.month
    daily["day_of_week"] = daily.index.dayofweek
    daily["week"]       = daily.index.isocalendar().week.astype(int)

    # --- Lag features ---
    for lag in LAG_DAYS:
        daily[f"lag_{lag}"] = daily[target_col].shift(lag)

    # --- Rolling mean (trend signal) ---
    daily["rolling_7"] = daily[target_col].shift(1).rolling(7, min_periods=3).mean()

    daily = daily.dropna()

    if len(daily) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Only {len(daily)} rows remain after feature engineering. "
            f"Need at least {MIN_ROWS_REQUIRED}. Try a larger dataset or fewer lag features."
        )

    return daily


# ─────────────────────────────────────────────────────────────────
# 🤖  MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────

def build_model_pipeline(model_type: str) -> "sklearn.pipeline.Pipeline":
    """
    Wraps the selected model in a sklearn Pipeline.
    Adds StandardScaler for distance-based models (KNN, SVR).
    """
    estimator = MODEL_REGISTRY[model_type]()
    steps = []
    if model_type in SCALE_REQUIRED:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_evaluate(
    daily_df: pd.DataFrame,
    target_col: str,
    extra_x: list[str],
    model_type: str,
) -> dict:
    """
    Trains selected model on temporal train/test split.
    Returns a results dict with predictions, metrics, and fitted model.
    """
    base_features  = [f"lag_{lag}" for lag in LAG_DAYS] + ["month", "day_of_week", "rolling_7"]
    valid_extra    = [c for c in extra_x if c in daily_df.columns]
    feature_cols   = base_features + valid_extra

    X = daily_df[feature_cols]
    y = daily_df[target_col]

    split = int(len(X) * TRAIN_SPLIT_RATIO)
    if split < 5 or (len(X) - split) < 5:
        raise ValueError(
            f"Dataset too small for a reliable split. "
            f"Train: {split} rows, Test: {len(X)-split} rows. Need at least 5 each."
        )

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = build_model_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "R² Score":  r2_score(y_test, y_pred),
        "RMSE":      np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":       mean_absolute_error(y_test, y_pred),
        "MAPE (%)":  np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100,
    }

    # Feature importances (tree-based models only)
    importances = None
    final_estimator = pipeline.named_steps["model"]
    if hasattr(final_estimator, "feature_importances_"):
        importances = pd.Series(
            final_estimator.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

    # OLS summary for Linear Regression — gives p-values & CIs
    ols_summary = None
    if model_type == "Linear Regression":
        X_ols = sm.add_constant(X_train.astype(float))
        ols_model = sm.OLS(y_train.astype(float), X_ols).fit()
        ols_summary = ols_model.summary()

    return {
        "y_test":       y_test,
        "y_pred":       y_pred,
        "metrics":       metrics,
        "feature_cols": feature_cols,
        "importances":  importances,
        "ols_summary":  ols_summary,
        "pipeline":     pipeline,
        "X_train_size": len(X_train),
        "X_test_size":  len(X_test),
    }


# ─────────────────────────────────────────────────────────────────
# 📊  VISUALISATION HELPERS  (Plotly — no memory leaks)
# ─────────────────────────────────────────────────────────────────

def plot_forecast(y_test: pd.Series, y_pred: np.ndarray, model_type: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        name="Actual", mode="lines+markers",
        marker=dict(size=4),
        line=dict(color="#2563EB", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=y_test.index, y=y_pred,
        name=f"Predicted ({model_type})", mode="lines",
        line=dict(color="#DC2626", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {model_type}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white",
        height=420,
    )
    return fig


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    residuals = y_test.values - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.index, y=residuals,
        mode="lines+markers",
        marker=dict(size=4, color="#7C3AED"),
        line=dict(color="#7C3AED"),
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Residual Plot (Actual − Predicted)",
        xaxis_title="Date", yaxis_title="Residual",
        template="plotly_white", height=300,
    )
    return fig


def plot_feature_importance(importances: pd.Series) -> go.Figure:
    fig = px.bar(
        importances.reset_index(),
        x="index", y=importances.name or 0,
        color=importances.values,
        color_continuous_scale="Blues",
        labels={"index": "Feature", importances.name or 0: "Importance"},
        title="Feature Importance",
    )
    fig.update_layout(template="plotly_white", height=350, showlegend=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix",
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


# ─────────────────────────────────────────────────────────────────
# 🔐  LOGIN PAGE
# ─────────────────────────────────────────────────────────────────

def render_login() -> None:
    st.title("🔐 MySQL Database Login")
    st.markdown("Connect to your MySQL server to begin analysis.")

    with st.form("login_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        host = col1.text_input("Host", value="localhost", placeholder="e.g. 127.0.0.1")
        port = col2.text_input("Port", value=str(MYSQL_PORT))
        user = st.text_input("Username", value="root")
        pw   = st.text_input("Password", type="password")
        submitted = st.form_submit_button("🚀 Connect", use_container_width=True)

    if submitted:
        if not host or not user:
            st.error("Host and Username are required.")
            return
        try:
            engine = get_engine(host, user, pw, db="")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))         # lightweight ping
            # Store credentials separately — never concatenated into URL in state
            st.session_state.update({
                "logged_in": True,
                "db_host":   host,
                "db_user":   user,
                "db_pass":   pw,
            })
            st.success("✅ Connected!")
            st.rerun()
        except Exception as exc:
            st.error(f"❌ Connection failed: {exc}")
            # Clear any cached bad engine
            get_engine.clear()


# ─────────────────────────────────────────────────────────────────
# 📊  MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────

def render_dashboard() -> None:
    host = st.session_state["db_host"]
    user = st.session_state["db_user"]
    pw   = st.session_state["db_pass"]

    st.title(APP_TITLE)

    # ── Sidebar: Logout ──────────────────────────────────────────
    with st.sidebar:
        if st.button("🚪 Logout", use_container_width=True):
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            get_engine.clear()
            fetch_preview.clear()
            fetch_full_data.clear()
            st.rerun()

        # ── Step 1: Data Source ──────────────────────────────────
        st.header("📂 1. Data Source")
        try:
            meta_engine = get_engine(host, user, pw, db="")
            with meta_engine.connect() as conn:
                raw_dbs = pd.read_sql(text("SHOW DATABASES"), conn)
            db_list = [d for d in raw_dbs.iloc[:, 0] if d not in SYSTEM_DBS]
        except Exception as exc:
            st.error(f"Failed to list databases: {exc}")
            st.stop()

        selected_db = st.selectbox("Database", ["— select —"] + db_list, key="sel_db")
        if selected_db == "— select —":
            st.info("Select a database to continue.")
            st.stop()

        try:
            db_engine = get_engine(host, user, pw, db=selected_db)
            with db_engine.connect() as conn:
                table_list = pd.read_sql(text("SHOW TABLES"), conn).iloc[:, 0].tolist()
        except Exception as exc:
            st.error(f"Failed to list tables: {exc}")
            st.stop()

        selected_table = st.selectbox("Table", ["— select —"] + table_list, key="sel_table")
        if selected_table == "— select —":
            st.info("Select a table to continue.")
            st.stop()

        # Preview to get columns
        try:
            sample = fetch_preview(
                id(db_engine), selected_table, selected_db, host, user, pw
            )
        except Exception as exc:
            st.error(f"Failed to preview table: {exc}")
            st.stop()

        all_cols = sample.columns.tolist()

        # ── Step 2: Model Selection ──────────────────────────────
        st.markdown("---")
        st.header("🤖 2. Model")
        model_type = st.selectbox("Algorithm", list(MODEL_REGISTRY.keys()), key="model_type")
        if model_type in SCALE_REQUIRED:
            st.caption("ℹ️ Feature scaling applied automatically.")

        # ── Step 3: Variable Configuration ──────────────────────
        st.markdown("---")
        st.header("🎯 3. Variables")
        date_col = st.selectbox("Date Column (X index)", all_cols, key="date_col")
        target_y = st.selectbox("Target Variable (Y)", all_cols, key="target_y")

        potential_x = [c for c in all_cols if c not in {date_col, target_y}]

        # >>> [FIX] ตัดเงื่อนไขแยกของ Linear Regression ออก เพื่อให้เลือกได้ทีละหลายตัวแปรเหมือนกันหมด <<<
        extra_x = st.multiselect("Additional Features (X)", potential_x, key="extra_x")

        st.markdown("---")
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    # ── Tabs ─────────────────────────────────────────────────────
    tab_preview, tab_analysis = st.tabs(["📄 Data Preview", "📈 Analysis & Forecast"])

    # ── Preview Tab ──────────────────────────────────────────────
    with tab_preview:
        st.subheader(f"Preview — `{selected_table}` (first {PREVIEW_LIMIT} rows)")
        st.dataframe(sample, use_container_width=True)
        st.caption(f"Showing up to {PREVIEW_LIMIT} rows. Full dataset loaded during analysis.")

    # ── Analysis Tab ─────────────────────────────────────────────
    with tab_analysis:
        if not run_btn:
            st.info("Configure settings in the sidebar and click **▶ Run Analysis**.")
            st.stop()

        # ── Data Loading ─────────────────────────────────────────
        with st.spinner("Loading dataset…"):
            try:
                use_cols = tuple(dict.fromkeys([date_col, target_y] + extra_x))  # dedup, preserve order
                raw_df = fetch_full_data(
                    id(db_engine), selected_table, use_cols,
                    selected_db, host, user, pw
                )
            except ValueError as exc:
                st.error(f"⚠️ Identifier validation failed: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Data loading error: {exc}")
                st.stop()

        # ── Feature Engineering ───────────────────────────────────
        with st.spinner("Engineering features…"):
            try:
                daily_df = engineer_features(raw_df, date_col, target_y, list(extra_x))
            except ValueError as exc:
                st.warning(f"⚠️ {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error during feature engineering: {exc}")
                st.stop()

        # ── Model Training ────────────────────────────────────────
        with st.spinner(f"Training {model_type}…"):
            try:
                results = train_evaluate(daily_df, target_y, list(extra_x), model_type)
            except ValueError as exc:
                st.warning(f"⚠️ {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Model training failed: {exc}")
                st.stop()

        # ── Store results for download (persists across rerenders) ─
        res_df = pd.DataFrame(
            {"Actual": results["y_test"], "Predicted": results["y_pred"]},
            index=results["y_test"].index,
        )
        st.session_state["results_df"] = res_df

        # ═══════════════════════════════════════════
        # 📊 SECTION 1: Model Performance Metrics
        # ═══════════════════════════════════════════
        st.success(f"✅ Analysis complete using **{model_type}**")
        st.subheader("📊 Model Performance")

        metric_cols = st.columns(len(results["metrics"]))
        icons = {"R² Score": "🎯", "RMSE": "📉", "MAE": "📏", "MAPE (%)": "📊"}
        for col, (name, val) in zip(metric_cols, results["metrics"].items()):
            col.metric(f"{icons.get(name,'')} {name}", f"{val:,.4f}")

        st.caption(
            f"Train size: **{results['X_train_size']}** rows  |  "
            f"Test size: **{results['X_test_size']}** rows  |  "
            f"Split: {int(TRAIN_SPLIT_RATIO*100)}/{int((1-TRAIN_SPLIT_RATIO)*100)}"
        )

        # ═══════════════════════════════════════════
        # 📈 SECTION 2: Forecast Chart + Residuals
        # ═══════════════════════════════════════════
        st.subheader("📈 Forecast")
        st.plotly_chart(
            plot_forecast(results["y_test"], results["y_pred"], model_type),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_residuals(results["y_test"], results["y_pred"]),
            use_container_width=True,
        )

        # ═══════════════════════════════════════════
        # 🔬 SECTION 3: Statistical Analysis
        # ═══════════════════════════════════════════
        st.markdown("---")
        st.subheader("🔬 Statistical Analysis")

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("**Descriptive Statistics**")
            analysis_cols = [target_y] + [c for c in extra_x if c in daily_df.columns]
            st.dataframe(daily_df[analysis_cols].describe().style.format("{:.4f}"), use_container_width=True)

        with stat_col2:
            if len(analysis_cols) > 1:
                st.markdown("**Correlation Heatmap**")
                st.plotly_chart(
                    plot_correlation_heatmap(daily_df[analysis_cols]),
                    use_container_width=True,
                )
            else:
                st.info("Add additional X features to see correlations.")

        # ── OLS Summary (Linear Regression only) ─────────────────
        if results["ols_summary"] is not None:
            st.markdown("**OLS Regression Summary (statsmodels)**")
            st.text(str(results["ols_summary"]))

        # ── ANOVA ────────────────────────────────────────────────
        st.markdown("**ANOVA — Monthly Variance Analysis**")
        try:
            anova_data = daily_df[[target_y, "month"]].copy()
            # Guard: need at least 2 months with >1 observation each
            month_counts = anova_data.groupby("month").size()
            valid_months = month_counts[month_counts > 1]
            if len(valid_months) < 2:
                st.warning("Not enough data across months for ANOVA (need ≥2 months with >1 observation each).")
            else:
                anova_data = anova_data[anova_data["month"].isin(valid_months.index)]
                safe_target = target_y.replace("'", "")
                anova_data = anova_data.rename(columns={target_y: safe_target})
                formula = f"Q('{safe_target}') ~ C(month)"
                anova_res   = ols(formula, data=anova_data).fit()
                anova_table = sm.stats.anova_lm(anova_res, typ=2)
                st.dataframe(anova_table.style.format("{:.4f}"), use_container_width=True)
                p_val = anova_table["PR(>F)"].iloc[0]
                significance = "✅ Statistically significant (p < 0.05)" if p_val < 0.05 else "❌ Not significant (p ≥ 0.05)"
                st.write(f"**P-Value:** `{p_val:.4f}` — {significance}")
        except Exception as exc:
            st.warning(f"ANOVA could not be computed: {exc}")

        # ── Feature Importance ────────────────────────────────────
        if results["importances"] is not None:
            st.markdown("---")
            st.subheader("💡 Feature Importance")
            st.plotly_chart(
                plot_feature_importance(results["importances"]),
                use_container_width=True,
            )

        # ═══════════════════════════════════════════
        # 📥 SECTION 4: Download (outside try block)
        # ═══════════════════════════════════════════
        st.markdown("---")
        if st.session_state["results_df"] is not None:
            csv_bytes = st.session_state["results_df"].to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_bytes,
                file_name=f"forecast_{selected_table}_{model_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────
# 🚀  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    if not st.session_state["logged_in"]:
        render_login()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
