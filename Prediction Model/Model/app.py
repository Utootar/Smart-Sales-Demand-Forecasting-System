import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Professional ML Data Studio", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================
# 🔐 หน้าจอ Login
# ==========================================
if not st.session_state['logged_in']:
    st.title("🔐 เข้าสู่ระบบฐานข้อมูล (MySQL)")
    with st.form("login_form"):
        host = st.text_input("Host Address", value="localhost")
        user = st.text_input("Username", value="root")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("🚀 เชื่อมต่อ Server"):
            try:
                test_url = f"mysql+pymysql://{user}:{pw}@{host}:3306/"
                engine = create_engine(test_url)
                with engine.connect() as conn: pass 
                st.session_state['logged_in'] = True
                st.session_state['base_url'] = test_url
                st.rerun()
            except Exception as e: st.error(f"❌ เชื่อมต่อไม่ได้: {e}")

# ==========================================
# 📊 หน้าจอ Dashboard หลัก
# ==========================================
else:
    st.title("🚀 Professional BI & ML Studio")
    if st.sidebar.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # --- 🎯 STEP 1: เลือกแหล่งข้อมูล ---
    st.sidebar.header("📂 1. แหล่งข้อมูล")
    base_engine = create_engine(st.session_state['base_url'])
    try:
        db_list = [d for d in pd.read_sql("SHOW DATABASES", base_engine).iloc[:, 0].tolist() 
                   if d not in ['information_schema', 'performance_schema', 'sys', 'mysql']]
        selected_db = st.sidebar.selectbox("เลือก Database", ["โปรดเลือก"] + db_list)
    except: selected_db = "โปรดเลือก"

    if selected_db != "โปรดเลือก":
        db_engine = create_engine(f"{st.session_state['base_url']}{selected_db}")
        table_list = pd.read_sql("SHOW TABLES", db_engine).iloc[:, 0].tolist()
        selected_table = st.sidebar.selectbox("เลือกตาราง", ["โปรดเลือก"] + table_list)

        if selected_table != "โปรดเลือก":
            sample = pd.read_sql(f"SELECT * FROM `{selected_table}` LIMIT 500", db_engine)
            all_cols = sample.columns.tolist()

            tab_preview, tab_analysis = st.tabs(["📄 ดูข้อมูลดิบ (Preview)", "📈 วิเคราะห์และพยากรณ์"])

            with tab_preview:
                st.subheader(f"📋 ตัวอย่างข้อมูลจากตาราง: {selected_table}")
                st.dataframe(sample)
                st.info(f"แสดงตัวอย่าง 500 แถวแรก")

            with tab_analysis:
                # --- 🎯 STEP 2: เลือกโมเดลก่อน ---
                st.sidebar.markdown("---")
                st.sidebar.header(" 2. ตั้งค่าโมเดล")
                model_type = st.sidebar.selectbox("เลือกโมเดลสถิติ", 
                    ["Random Forest", "Gradient Boosting", "Linear Regression", "K-Neighbors", "SVR", "Decision Tree"])
                
                # --- 🎯 STEP 3: ตั้งค่าตัวแปร (Dynamic ตามโมเดล) ---
                st.sidebar.markdown("---")
                st.sidebar.header("🎯 3. ตั้งค่าตัวแปร (X & Y)")
                date_col = st.sidebar.selectbox("ตัวแปรต้น X (คอลัมน์วันที่)", all_cols)
                target_y = st.sidebar.selectbox("ตัวแปรตาม (Y)", all_cols)
                
                potential_x = [c for c in all_cols if c not in [date_col, target_y]]
                
                if model_type == "Linear Regression":
                    single_x = st.sidebar.selectbox("➕ เลือก X 1 ตัว (Simple Linear)", ["ไม่มี"] + potential_x)
                    extra_x = [single_x] if single_x != "ไม่มี" else []
                else:
                    extra_x = st.sidebar.multiselect("➕ เลือก X หลายตัว (Multiple Features)", potential_x)

                if st.sidebar.button(" เริ่มการวิเคราะห์", type="primary"):
                    try:
                        # 1. เตรียมข้อมูล
                        use_cols = [date_col, target_y] + [c for c in extra_x if c not in [date_col, target_y]]
                        query = f"SELECT `{ '`,`'.join(use_cols) }` FROM `{selected_table}`"
                        df = pd.read_sql(query, db_engine)
                        
                        # 2. IQR (ล้าง Outlier)
                        Q1, Q3 = df[target_y].quantile(0.25), df[target_y].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[target_y] >= Q1 - 1.5*IQR) & (df[target_y] <= Q3 + 1.5*IQR)]
                        
                        df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
                        df = df.dropna(subset=[date_col]).set_index(date_col)
                        
                        # 3. รวมกลุ่มรายวัน
                        agg_rules = {target_y: 'sum'}
                        for c in extra_x: agg_rules[c] = 'mean'
                        daily_df = df.resample('D').agg(agg_rules).dropna()
                        
                        # 4. Features (Lag)
                        daily_df['month'] = daily_df.index.month
                        daily_df['lag_1'] = daily_df[target_y].shift(1)
                        daily_df['lag_7'] = daily_df[target_y].shift(7)
                        daily_df = daily_df.dropna()

                        features_list = ['lag_1', 'lag_7', 'month'] + extra_x
                        X, y = daily_df[features_list], daily_df[target_y]
                        split = int(len(X) * 0.8)
                        X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

                        # 5. เลือกโมเดลจากคลังแสง
                        if model_type == "Random Forest": model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Gradient Boosting": model = GradientBoostingRegressor(random_state=42)
                        elif model_type == "K-Neighbors": model = KNeighborsRegressor(n_neighbors=5)
                        elif model_type == "SVR": model = SVR(kernel='rbf', C=100)
                        elif model_type == "Decision Tree": model = DecisionTreeRegressor(random_state=42)
                        else: model = LinearRegression()

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # 6. แสดงผล & Metrics
                        st.success(f"วิเคราะห์สำเร็จด้วย {model_type}!")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🎯 R2 Score", f"{r2_score(y_test, y_pred):.4f}")
                        c2.metric("📉 MSE", f"{mean_squared_error(y_test, y_pred):,.2f}")
                        c3.metric("📏 MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")

                        # กราฟเปรียบเทียบ
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(y_test.index, y_test.values, label='Actual', marker='o', alpha=0.7)
                        ax.plot(y_test.index, y_pred, label=f'Predict ({model_type})', linestyle='--', color='red')
                        ax.set_title(f"Forecast results using {model_type}")
                        ax.legend(); st.pyplot(fig)

                        # ==========================================
                        # 🔬 Statistical Analysis Section
                        # ==========================================
                        st.markdown("---")
                        st.header("🔬 Statistical Analysis")
                        
                        col_stat1, col_stat2 = st.columns(2)
                        
                        with col_stat1:
                            st.subheader("📝 Descriptive Statistics")
                            st.dataframe(daily_df[[target_y] + extra_x].describe())

                        with col_stat2:
                            st.subheader("🔗 Correlation Matrix")
                            if len([target_y] + extra_x) > 1:
                                st.write(daily_df[[target_y] + extra_x].corr())
                            else:
                                st.info("เลือก X เพิ่มเพื่อดูค่า Correlation")

                        # ANOVA Table
                        st.subheader("🧐 ANOVA Table (Monthly Analysis)")
                        try:
                            # ใช้สูตร Target ~ C(month) เพื่อดูความต่างของแต่ละเดือน
                            formula = f"Q('{target_y}') ~ C(month)"
                            anova_res = ols(formula, data=daily_df).fit()
                            anova_table = sm.stats.anova_lm(anova_res, typ=2)
                            st.dataframe(anova_table)
                            p_val = anova_table['PR(>F)'].iloc[0]
                            st.write(f"**P-Value:** {p_val:.4f} ({'มีนัยสำคัญ' if p_val < 0.05 else 'ไม่มีนัยสำคัญ'})")
                        except: st.warning("ไม่สามารถคำนวณ ANOVA ได้ (ข้อมูลอาจไม่เพียงพอในแต่ละกลุ่มเดือน)")

                        # Feature Importance (สำหรับโมเดลสาย Tree)
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("💡 Feature Importance")
                            imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
                            st.bar_chart(imp)

                        # ปุ่มดาวน์โหลด
                        res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
                        st.download_button("📥 Download Results (CSV)", res_df.to_csv().encode('utf-8'), "results.csv", "text/csv")

                    except Exception as e: st.error(f"❌ เกิดข้อผิดพลาด: {e}")