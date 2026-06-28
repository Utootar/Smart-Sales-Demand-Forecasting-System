"""
 AI Data Analytics Platform  — AI Agent & Smart ETL Edition (2026)
======================================================================
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
MAX_PREVIEW_ROWS   = 100   

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
    "ai_agent_suggestions": None,
    "ai_etl_code":        "",
    "nav_page":           "1. 👋 Home",
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────
# 🤖  GEMINI CLIENT  (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def _get_gemini():
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
        # ตัวแรงของมึง รันผ่านฉลุยจัดไปยาวๆ มึงตอง!
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
# 🔍  DATA PROFILING  (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Profiling data…")
def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    report = []
    n = len(df)
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct   = null_count / n * 100 if n else 0
        n_unique   = int(df[col].nunique())

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dtype_group = "Datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtype_group = "Numerical"
        else:
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
# 🔧  STANDARD ETL ENGINE  (Baseline cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Running Standard ETL…")
def auto_etl_engine(df: pd.DataFrame, _profile_hash: str) -> tuple:
    profile = profile_dataset(df)
    cleaned = df.copy()
    outlier_report: dict = {}

    for _, row in profile.iterrows():
        col   = row["Column"]
        dtype = row["Detected Type"]

        if dtype == "Datetime":
            cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')

        if cleaned[col].isnull().sum() > 0:
            if dtype == "Numerical":
                skew = cleaned[col].skew()
                fill = cleaned[col].median() if abs(skew) > 1 else cleaned[col].mean()
                cleaned[col] = cleaned[col].fillna(fill)
            elif dtype == "Categorical":
                if cleaned[col].nunique() / max(len(cleaned), 1) < 0.1:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])
                else:
                    cleaned[col] = cleaned[col].fillna("Unknown_Category")

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
            "5. 🤖 AI Data Agent Workspace",
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
    c3.metric("🤖 AI Agent Status", "🔥 Active & Smart" if has_data else "—")

    st.markdown("---")
    st.markdown("""
    **ขั้นตอนการใช้งาน:**
    1. **📥 Ingest** — โหลด CSV / Excel หรือเชื่อมต่อ MySQL table
    2. **🔍 Profiling** — เลือกโหมดการคลีนข้อมูลระหว่างสูตรสถิติพื้นฐาน หรือให้ AI เขียนสคริปต์ทำ Smart ETL 
    3. **🔬 Stats** — Correlation, OLS, VIF, ANOVA อัตโนมัติ
    4. **🤖 AI Agent Workspace** — มอบหมายให้ AI วิเคราะห์ข้อมูล ออกแบบสถิติ และพล็อตกราฟแบบไดนามิกตามจริง
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
                st.session_state["cleaned_data"] = None
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
# PAGE 3 — DATA PROFILING & 🔥 AI SMART ETL
# ═════════════════════════════════════════════
elif "3." in page:
    _back_btn()
    st.title("🔍 Data Profiling & Smart ETL Pipeline")

    if st.session_state["raw_data"] is None:
        st.warning("กรุณาโหลดข้อมูลที่หน้า Ingest ก่อน"); st.stop()

    raw_df  = st.session_state["raw_data"]
    prof_df = st.session_state["data_profile"]

    st.subheader("📋 Automated Data Profile")
    st.dataframe(prof_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🔧 Data Cleansing & ETL Configuration")
    
    # ทางเลือกทำ ETL ระหว่างสูตรคณิตสถิติดั้งเดิม กับ ดึงพลัง AI เข้ามาช่วยเขียนโค้ดล้างตรงตามเนื้อผ้าจริง
    etl_mode = st.radio(
        "เลือกสถาปัตยกรรมการคลีนข้อมูล (ETL Pipeline Mode)",
        ["สถิติพื้นฐานตามสเกล (Standard Statistical Imputation)", "🔥 ปลดล็อก AI Agent ออกแบบสคริปต์คลีนข้อมูลตามจริง (AI-Driven Smart ETL)"]
    )

    if "สถิติพื้นฐาน" in etl_mode:
        cleaned, outlier_info = auto_etl_engine(raw_df, str(raw_df.shape) + str(raw_df.columns.tolist()))
        st.session_state["cleaned_data"] = cleaned
        st.success("✅ ระบบใช้ Pipeline สถิติแบบดั้งเดิมคำนวณเติมค่าเฉลี่ยสำเร็จ")
    else:
        st.info("โหมด AI Smart ETL: ระบบจะส่งโครงสร้างคอลัมน์ไปให้ AI วิเคราะห์เชิงตรรกะ เพื่อเขียนโค้ดกลับมาล้างข้อมูลและเติมค่าว่างให้สมเหตุสมผลที่สุด")
        
        if st.button("⚡ สั่ง AI Agent เริ่มวิเคราะห์และเขียนสคริปต์ทำ Smart ETL", type="primary", use_container_width=True):
            with st.spinner("AI กำลังแกะตรรกะชุดข้อมูลและร่างสคริปต์ล้างข้อมูลความเร็วสูง..."):
                schema_context = f"Columns: {raw_df.columns.tolist()}\nNull Counts:\n{raw_df.isnull().sum().to_string()}\nHead Sample:\n{raw_df.head(3).to_string()}"
                
                etl_prompt = (
                    f"คุณคือ Senior Data Engineer ผู้เชี่ยวชาญด้านการทำ Data Cleansing และ Advanced ETL Pipeline\n"
                    f"นี่คือชุดข้อมูลดิบที่มีปัญหาช่องว่างหรือสิ่งสกปรก: {schema_context}\n\n"
                    f"หน้าที่ของคุณคือเขียนโค้ด Python เพื่อล้างข้อมูลบนตัวแปร `df` (มีอยู่แล้วในระบบ ห้ามโหลดใหม่) โดยมีเงื่อนไขตามหลักการจัดการสถิติระดับสากล:\n"
                    f"1. การทำ Imputation (เติมค่าว่าง): ห้ามใช้ค่า Mean/Median โง่ๆ หยอดใส่ทุกคอลัมน์ดื้อๆ แต่ให้วิเคราะห์เชิงตรรกะมนุษย์ (เช่น หากเป็นข้อมูลฟาร์ม/โรคพืช/หรือโรงพยาบาล ให้ประเมินค่าที่เหมาะสมตามความสัมพันธ์ของแถวและคอลัมน์ข้างเคียง หรือเติมหมวดหมู่ตามความน่าจะเป็นเชิงลึก)\n"
                    f"2. จัดการกับ Text Inconsistency: สั่งแก้ปัญหาข้อมูลสตริงที่สะกดเพี้ยน คลีนช่องว่างส่วนเกิน คลีนพิมพ์เล็กใหญ่ให้สม่ำเสมอ\n"
                    f"3. แปลงคอลัมน์เวลา: หากเจอคอลัมน์ไหนส่อแววเป็น วันที่/เวลา ให้ใช้คำสั่ง `pd.to_datetime(df[col], errors='coerce')` เสมอ เพื่อส่งต่อไปรันในโหมด Time Series หน้า 5 ได้\n"
                    f"4. ส่งกลับผลลัพธ์มาเฉพาะบล็อกโค้ดในรูปแบบเครื่องหมาย ```python ... ``` เท่านั้น ห้ามเขียนคำอธิบายเชิงพรรณนาเด็ดขาด โค้ดของมึงจะถูกนำไปใช้ในคำสั่ง exec() ทันที!"
                )
                
                ai_etl_raw = call_gemini(etl_prompt)
                
                # ทำการ Parser ตัดแต่งเอาเฉพาะสคริปต์โค้ดบริสุทธิ์
                if "```python" in ai_etl_raw:
                    st.session_state["ai_etl_code"] = ai_etl_raw.split("```python")[1].split("```")[0]
                elif "```" in ai_etl_raw:
                    st.session_state["ai_etl_code"] = ai_etl_raw.split("```")[1].split("```")[0]
                else:
                    st.session_state["ai_etl_code"] = ai_etl_raw

        if st.session_state["ai_etl_code"]:
            # สร้าง Environment ในการรันสคริปต์ที่ AI ส่งมาสดๆ
            df_working = raw_df.copy()
            execution_scope = {
                "pd": pd,
                "np": np,
                "df": df_working
            }
            try:
                exec(st.session_state["ai_etl_code"], globals(), execution_scope)
                # ดึงผลลัพธ์ DataFrame ที่ถูกชำระล้างโดยสคริปต์ AI เรียบร้อยแล้วกลับมาเซฟลงระบบหลัก
                st.session_state["cleaned_data"] = execution_scope["df"]
                st.success("🎯 AI Smart ETL ดำเนินการล้างข้อมูลและแก้ไขตรรกะในหน่วยความจำเรียบร้อย!")
                with st.expander("🛠️ เปิดดูสคริปต์ล้างข้อมูล (Advanced ETL Pipeline) ที่ AI เขียนขึ้นมา"):
                    st.code(st.session_state["ai_etl_code"], language="python")
            except Exception as etl_err:
                st.error(f"❌ สคริปต์ ETL ของ AI เกิดข้อผิดพลาดทางเทคนิค: {etl_err}")
                st.info("ระบบทำการสลับกลับไปใช้โครงสร้าง Pipeline สถิติพื้นฐานเพื่อเซฟตี้โมเดลไม่ให้แครช")
                cleaned, _ = auto_etl_engine(raw_df, str(raw_df.shape) + str(raw_df.columns.tolist()))
                st.session_state["cleaned_data"] = cleaned

    # แสดงผลการเปรียบเทียบบนหน้าจอให้เห็นความแตกต่างชัดเจน
    if st.session_state["cleaned_data"] is not None:
        cleaned_df = st.session_state["cleaned_data"]
        st.subheader("📊 ตารางเปรียบเทียบ Missing Values (ก่อนล้าง VS หลังล้าง)")
        missing_cmp = pd.DataFrame({
            "ข้อมูลดิบก่อนล้าง (Raw)": raw_df.isnull().sum(),
            "ข้อมูลที่ผ่านสมองกลล้างแล้ว (Cleaned)": cleaned_df.isnull().sum(),
        })
        st.dataframe(missing_cmp[missing_cmp["ข้อมูลดิบก่อนล้าง (Raw)"] > 0], use_container_width=True)

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
        st.warning("กรุณารันกระบวนการล้างข้อมูลที่หน้า Profiling & Smart ETL ก่อน"); st.stop()

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
                        st.dataframe(_safe_vif(num_sub[num_feats].astype(float).dropna()),
                                     use_container_width=True)
                    st.session_state["stats_results"] = {
                        "r2": fit.rsquared, "target": target_var
                    }
                except Exception as e:
                    st.warning(f"OLS ข้าม: {e}")

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
# PAGE 5 — AI DATA AGENT WORKSPACE (Code Interpreter)
# ═════════════════════════════════════════════
elif "5." in page:
    _back_btn()
    st.title("🤖 AI Data Agent & Advanced Analytics Workspace")

    if st.session_state["cleaned_data"] is None:
        st.warning("กรุณาผ่านขั้นตอนการทำความสะอาดข้อมูลที่หน้า 3 ก่อนนะครับ"); st.stop()

    df = st.session_state["cleaned_data"]
    
    st.subheader("🕵️‍♂️ AI Data Assessment Strategy")
    st.write("กดปุ่มด้านล่างเพื่อให้ AI สแกนเนื้อข้อมูลจริง เพื่อออกแบบกระบวนการวิเคราะห์ทางสถิติและ Machine Learning ที่ถูกต้องแม่นยำที่สุด")

    if st.button("🔍 สั่ง AI สแกนข้อมูลและวางแผนวิเคราะห์เชิงลึก", type="primary", use_container_width=True):
        with st.spinner("AI กำลังอ่านโครงสร้างดาต้าเบสของมึงเพื่อคำนวณหน้างาน..."):
            schema_info = f"Columns: {df.columns.tolist()}\nDtypes:\n{df.dtypes.to_string()}\nSample Head:\n{df.head(3).to_string()}"
            strategy_prompt = (
                f"คุณคือนักสถิติประยุกต์และ AI Agent อัจฉริยะขั้นสูง นี่คือข้อมูลจริงของผู้ใช้งาน:\n{schema_info}\n\n"
                f"จงวิเคราะห์ข้อเท็จจริงของชุดข้อมูลนี้อย่างตรงไปตรงมา และแนะนำแนวทางการวิเคราะห์เชิงลึกที่เหมาะสม "
                f"โดยเขียนหัวข้อข้อเสนอแนะออกมาเป็นข้อๆ 3 ข้อที่ระบุชื่อตัวแปรจริงในดาต้าเซ็ตให้ชัดเจน (เช่น หากเจอข้อมูลโรคพืชหรือข้อมูลเกษตร ให้เสนอการทดสอบสถิติหรือโมเดลที่สอดคล้องกับตัวแปรนั้นจริงๆ ไม่ใช่การรันสูตรโมเดลทั่วไปแบบสุ่มสี่สุ่มห้า)"
            )
            st.session_state["ai_agent_suggestions"] = call_gemini(strategy_prompt)

    if st.session_state["ai_agent_suggestions"]:
        st.info(st.session_state["ai_agent_suggestions"])

    st.markdown("---")
    st.subheader("💻 Live AI Code Interpreter Console")
    st.write("พิมพ์สั่งงานภาษาไทย หรือบอกเป้าหมายวิเคราะห์ที่มึงต้องการได้เลยเว้ยตอง เดี๋ยว AI จะเขียนโค้ดสถิติ/พล็อตกราฟมารันสดให้ดูบนหน้าจอนี้ทันที!")

    user_instruction = st.text_area(
        "ระบุคำสั่งวิเคราะห์ที่มึงต้องการให้ทำ (เช่น: 'ช่วยพล็อตกราฟแท่งกระจายตัวแปรกลุ่ม และรันสถิติ Chi-Square หรือโมเดลที่แมทช์ที่สุดเพื่อดูผลกระทบต่อโรคพืช')",
        placeholder="พิมพ์โจทย์คณิตศาสตร์/สถิติ/กราฟ ที่มึงอยากเห็นตรงนี้เลย..."
    )

    if st.button("⚡ สั่ง AI ประมวลผลและสร้างชิ้นงานสด", type="secondary", use_container_width=True):
        if not user_instruction:
            st.error("มึงพิมพ์คำสั่งบอกกูก่อนดิตอง 55"); st.stop()
            
        with st.spinner("Gemini กำลังเขียนโค้ดประมวลผลสถิติสด..."):
            schema_context = f"Columns: {df.columns.tolist()}\nDtypes:\n{df.dtypes.to_string()}\nHead:\n{df.head(2).to_string()}"
            code_prompt = (
                f"คุณคือสุดยอด AI Code Interpreter หน้าที่ของคุณคือเขียนโค้ด Python เพื่อพล็อตกราฟและคำนวณสถิติลงบนหน้าจอ Streamlit ของผู้ใช้\n"
                f"นี่คือชุดข้อมูลจริงที่มีอยู่: {schema_context}\n"
                f"คำสั่งวิเคราะห์ที่ผู้ใช้ต้องการ: '{user_instruction}'\n\n"
                f"⚠️ กฎเหล็กในการเขียนโค้ด:\n"
                f"1. ตัวแปร DataFrame หลักในระบบมีชื่อว่า `df` (มีอยู่แล้ว ห้ามโหลดไฟล์ใหม่หรือจำลองข้อมูลใหม่เด็ดขาด)\n"
                f"2. คุณสามารถใช้คลังคำสั่งได้แค่: `st`, `pd`, `np`, `px`, `go`, `sp_stats` (scipy.stats) และ `sm` (statsmodels)\n"
                f"3. จงเขียนกระบวนการจัดการข้อมูล (เช่น การแปลงประเภทตัวแปร การเข้ารหัส One-Hot คัดกรองคอลัมน์ที่ไม่เกี่ยวเช่น ID ออก) และการเลือกใช้สูตรโมเดลทางคณิตศาสตร์ให้สอดคล้องและถูกต้องตามคำสั่งของผู้ใช้อย่างสมบูรณ์แบบร้อยเปอร์เซ็นต์\n"
                f"4. ใช้ `st.plotly_chart(fig, use_container_width=True)` ในการแสดงผลกราฟ\n"
                f"5. ส่งกลับผลลัพธ์มาเฉพาะบล็อกโค้ดในรูปแบบเครื่องหมาย ```python ... ``` เท่านั้น ห้ามเขียนอธิบายเรื่องอื่นนอกกรอบโค้ดเด็ดขาด!\n"
                f"6. ทุกครั้งที่ใช้ `fig.update_layout()` มึงจำเป็นต้องระบุชื่อแกนและชื่อหัวเรื่องให้ครบถ้วนเสมอ โดยดึงชื่อตัวแปรจริงมาใส่ เช่น `title='การทำนายผลลัพธ์ของเป้าหมาย ตัวแปร Y'`, `xaxis_title='แกนข้อมูล X (ระบุชื่อตัวแปรตามจริง)'`, `yaxis_title='แกนข้อมูล Y (ระบุชื่อตัวแปรตามจริง)'` ห้ามปล่อยให้แกนแสดงผลแค่ตัวเลขดัชนีเปล่าๆ โล่งๆ เป็นอันขาด!"
            )
            
            ai_raw_code = call_gemini(code_prompt)
            
            cleaned_code = ""
            if "```python" in ai_raw_code:
                cleaned_code = ai_raw_code.split("```python")[1].split("```")[0]
            elif "```" in ai_raw_code:
                cleaned_code = ai_raw_code.split("```")[1].split("```")[0]
            else:
                cleaned_code = ai_raw_code

            st.markdown("### 📊 Live Analysis Output")
            
            execution_scope = {
                "st": st,
                "pd": pd,
                "np": np,
                "px": px,
                "go": go,
                "sp_stats": sp_stats,
                "sm": sm,
                "df": df
            }
            
            try:
                exec(cleaned_code, globals(), execution_scope)
                st.success("🎯 AI Agent รันคำนวณคณิตศาสตร์และสร้างชิ้นงานสำเร็จ!")
                with st.expander("🛠️ เปิดดู Source Code หลังบ้านที่ AI เขียนขึ้นมาสดๆ"):
                    st.code(cleaned_code, language="python")
            except Exception as runtime_err:
                st.error(f"❌ โค้ดสถิติแครชเนื่องจาก: {runtime_err}")
                st.markdown("ลองเปลี่ยนคำสั่งระบุเจาะจงชื่อคอลัมน์ให้ชัดขึ้นดูมึง")
                with st.expander("ดูโค้ดที่มีปัญหา"):
                    st.code(cleaned_code, language="python")


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
            stats = st.session_state.get("stats_results")
            col_m = st.session_state.get("ai_column_meanings", "—")

            ctx_parts = [f"- Column analysis: {col_m[:300]}"]
            if stats:
                ctx_parts.append(f"- OLS R² on `{stats['target']}`: {stats['r2']:.4f}")
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
