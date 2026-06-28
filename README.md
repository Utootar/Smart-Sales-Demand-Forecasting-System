 AI Data Analytics Platform (Demo)
An AI-powered analytics platform that automatically transforms raw business data into actionable insights.

 Built With:
-  Python, Pandas, NumPy
- **Database Management:** MySQL Server, SQLAlchemy Connection Pool
- **Statistical Analysis:** Statsmodels (OLS Summary, VIF), Scipy (ANOVA, Chi-Square)
- **Machine Learning Studio:** Scikit-Learn (Pipelines, HistGradientBoosting, Random Forest Parallel Processing)
- **Generative AI & Data Agent:** Gemini API (Live Code Interpreter Runtime)

---

 Project Overview
This project simulates a real-world Data Analyst / ML Engineer workflow. The system automatically:
1. **Data Ingestion** — Connects to structured data sources safely using an offline architecture.
2. **Data Profiling** — Evaluates dataset quality and structures native dtypes.
3. **Automated ETL Engine** — Performs smart missing value imputation and outlier classification.
4. **Dynamic Statistical Engine** — Runs diagnostic statistics and multi-collinearity checks.
5. **AI Data Agent Workspace** — Acts as a **Live Code Interpreter** that generates Python code and plots dynamic statistical charts on-the-fly based on real data context.
6. **Generative AI Reporting** — Compiles full executive insights and business recommendations.

---

 Supported Ingestion Formats
- Local CSV Files
- Microsoft Excel Spreadsheets
- MySQL Production Databases (Offline Secure Pipeline)

 Capabilities & Features
- **Database Connection Pool:** Secure multi-tenant database connection via SQLAlchemy without hardcoded credentials.
- **Raw Data Preservation:** Safe operational pipeline where original data is preserved and never mutated during profiling or ETL.
- **Automated Data Profiling:** Automatically detects authoritative data types, missing values, duplicate records, and suggests feature/target candidates.
- **Live AI Code Interpreter Engine:** Accepts natural language instructions, generates data visualization/statistical testing code, enforces explicit axes-label safeguards, and executes within a secure runtime scope (`exec()`).

---

Automated Data Profiling & Quality
The platform runs automated checks on dataset columns to detect and report:
- Data Type Classifications: Datetime, Numerical, Categorical, and High-Cardinality Text.
- Missing Value Handling: Automatically switches between Mean/Median imputation depending on statistical skewness.
- Outlier Classification: Combined IQR and Z-Score analysis to label extreme valid values vs potential data errors.
- Exact Duplicate Records: Instant removal statistics to keep downstream analytics clean.

---
Example:
  
<img width="755" height="407" alt="image" src="https://github.com/user-attachments/assets/68129bfe-5b23-4100-84a7-0fe9b257efa5" />

<img width="806" height="385" alt="image" src="https://github.com/user-attachments/assets/bfe87350-d95b-4ae4-a5fc-57b569e2934b" />


<img width="953" height="434" alt="image" src="https://github.com/user-attachments/assets/ed8c44f0-fc4c-451d-92e8-299cc2c169fa" />
