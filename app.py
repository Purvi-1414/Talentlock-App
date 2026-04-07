import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Talentlock Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# LOAD DATA & MODELS
# -----------------------------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    model_columns = pickle.load(open("columns.pkl", "rb"))
    return model, model_columns

model, model_columns = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("talentlock_cleaned.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})
    df['Attrition_Label'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    return df

df = load_data()
salary_col = "Monthly_Income" if "Monthly_Income" in df.columns else "MonthlyIncome"

# -----------------------------
# PREMIUM UI (CSS) -> LIGHT THEME
# -----------------------------
st.markdown("""
<style>
/* App Background */
.stApp {
    background-color: #f0f9ff; /* Soft light blue */
    color: #2c3e50;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e0f2fe; /* Slightly darker pastel blue */
    border-right: 1px solid #bae6fd;
}

/* Titles and Headers */
h1, h2, h3, h4, h5, h6 {
    color: #1e293b !important;
}

/* Sidebar Input Styling (Dropdowns & Sliders) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border: 2px solid #3b82f6 !important; /* Blue border highlight */
    border-radius: 6px;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}
section[data-testid="stSidebar"] .stSlider label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* Metrics and Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    text-align: left;
    border: 1px solid #e2e8f0;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 16px;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700;
}

/* Radio buttons in sidebar */
div.row-widget.stRadio > div {
    gap: 2px;
}

/* Buttons */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px;
    font-size: 15px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
}

/* Selectbox general */
.stSelectbox label {
    color: #334155 !important;
}

hr {
    border-color: #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.markdown('<h2><span style="color: #1e293b; font-weight: bold;">Talentlock App</span></h2>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Dashboard", "Prediction", "EDA", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Accuracy:** 81.3%")

# -----------------------------
# SESSION STATE & FILTERS
# -----------------------------
if "dept_key" not in st.session_state:
    st.session_state.dept_key = "All"
if "gender_key" not in st.session_state:
    st.session_state.gender_key = "All"
if "salary_key" not in st.session_state:
    st.session_state.salary_key = (int(df[salary_col].min()), int(df[salary_col].max()))
if "age_key" not in st.session_state:
    st.session_state.age_key = (int(df["Age"].min()), int(df["Age"].max()))

def reset_filters():
    st.session_state.dept_key = "All"
    st.session_state.gender_key = "All"
    st.session_state.salary_key = (int(df[salary_col].min()), int(df[salary_col].max()))
    st.session_state.age_key = (int(df["Age"].min()), int(df["Age"].max()))

# Show filters mainly in EDA
if page == "EDA":
    # Filters in Sidebar
    st.sidebar.markdown('<p style="font-size: 16px; font-weight: bold; color: #1e293b;">Filters</p>', unsafe_allow_html=True)
    
    st.sidebar.selectbox("Department", ["All"] + list(df["Department"].unique()), key="dept_key")
    st.sidebar.selectbox("Gender", ["All"] + list(df["Gender"].unique()), key="gender_key")
    
    st.sidebar.slider(
        "Salary Range",
        int(df[salary_col].min()), int(df[salary_col].max()),
        key="salary_key"
    )
    
    st.sidebar.slider(
        "Age Range",
        int(df["Age"].min()), int(df["Age"].max()),
        key="age_key"
    )
    
    st.sidebar.button("🔄 Reset Filters", on_click=reset_filters)

# Apply filters purely based on session_state keys
filtered_df = df.copy()
if getattr(st.session_state, "dept_key", "All") != "All":
    filtered_df = filtered_df[filtered_df["Department"] == st.session_state.dept_key]
if getattr(st.session_state, "gender_key", "All") != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == st.session_state.gender_key]

# Ensure we have tuple defaults ready
current_salary = getattr(st.session_state, "salary_key", (int(df[salary_col].min()), int(df[salary_col].max())))
current_age = getattr(st.session_state, "age_key", (int(df["Age"].min()), int(df["Age"].max())))

filtered_df = filtered_df[
    (filtered_df[salary_col].between(current_salary[0], current_salary[1])) &
    (filtered_df["Age"].between(current_age[0], current_age[1]))
]

# -----------------------------
# PAGE LOGIC
# -----------------------------
if page == "Dashboard":
    st.markdown("<h1>Talentlock Insight Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(filtered_df))
    # We display rate and format smartly
    attr_rate = (filtered_df['Attrition'].mean()*100) if not filtered_df.empty else 0
    col2.metric("Attrition Rate", f"{attr_rate:.2f}%")
    avg_salary = filtered_df[salary_col].mean() if not filtered_df.empty else 0
    col3.metric("Average Salary", f"${avg_salary:,.2f}")
    
    st.markdown("<hr style='margin: 30px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3>Key Attrition Insights</h3>", unsafe_allow_html=True)
    
    colA, colB, colC = st.columns(3)
    female_rate = (filtered_df[filtered_df["Gender"]=="Female"]["Attrition"].mean()*100) if len(filtered_df[filtered_df["Gender"]=="Female"]) > 0 else 0
    male_rate = (filtered_df[filtered_df["Gender"]=="Male"]["Attrition"].mean()*100) if len(filtered_df[filtered_df["Gender"]=="Male"]) > 0 else 0
    high_attr_dept = filtered_df.groupby("Department")["Attrition"].mean().idxmax() if not filtered_df.empty else "N/A"
    high_attr_rate = (filtered_df.groupby("Department")["Attrition"].mean().max() * 100) if not filtered_df.empty else 0
    
    # Light theme cards
    colA.markdown(f'''
    <div style="background-color:#e0f2fe; padding: 20px; border-radius: 10px; height: 130px; border: 1px solid #bae6fd;">
        <p style="color:#0284c7 !important; margin-top:0; font-weight: bold; font-size: 16px;">Female Attrition Rate {female_rate:.1f}%</p>
        <p style="color:#334155; font-size:14px;">Female employees show a specific pattern in turnover probabilities.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    colB.markdown(f'''
    <div style="background-color:#dcfce7; padding: 20px; border-radius: 10px; height: 130px; border: 1px solid #bbf7d0;">
        <p style="color:#16a34a !important; margin-top:0; font-weight: bold; font-size: 16px;">Male Attrition Rate {male_rate:.1f}%</p>
        <p style="color:#334155; font-size:14px;">Male employees tend to have different retention metrics comparatively.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    colC.markdown(f'''
    <div style="background-color:#fee2e2; padding: 20px; border-radius: 10px; height: 130px; border: 1px solid #fecaca;">
        <p style="color:#dc2626 !important; margin-top:0; font-weight: bold; font-size: 16px;">Dept Risk: {high_attr_dept}</p>
        <p style="color:#334155; font-size:14px;">The {high_attr_dept} department faces the highest {high_attr_rate:.1f}% attrition.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk Indicator
    if attr_rate > 25:
        st.error(f"⚠️ Overall High Attrition Rate: {attr_rate:.2f}%")
    elif attr_rate > 15:
        st.warning(f"⚠️ Overall Moderate Attrition Rate: {attr_rate:.2f}%")
    else:
        st.success(f"✅ Overall Low Attrition Rate: {attr_rate:.2f}%")

    st.subheader("💰 Salary Comparison")
    left_avg = filtered_df[filtered_df["Attrition"]==1][salary_col].mean() if len(filtered_df[filtered_df["Attrition"]==1])>0 else 0
    stay_avg = filtered_df[filtered_df["Attrition"]==0][salary_col].mean() if len(filtered_df[filtered_df["Attrition"]==0])>0 else 0
    
    colx, coly = st.columns(2)
    colx.metric("Avg Salary (Left)", f"${left_avg:,.0f}")
    coly.metric("Avg Salary (Stayed)", f"${stay_avg:,.0f}")

    # Insights & Recommendations Moved to Dashboard
    st.markdown("---")
    st.header("💡 Insights & Recommendations")
    
    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.subheader("📌 Key Insights")
        st.write("- **Salary** is a major driver of retention.")
        st.write("- Younger employees show a higher attrition trend.")
        st.write("- Certain departments experience significantly higher turnover due to stress and workload.")
    
    with col_ins2:
        st.subheader("🛠️ Recommendations")
        st.write("- Focus retention programs on high-attrition departments.")
        st.write("- Implement competitive salary structures aligned with industry standards.")
        st.write("- Engage younger employees through mentorship and fast-tracked growth opportunities.")

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Matplotlib styling for light theme
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": "#f8f9fa",
        "axes.facecolor": "#ffffff",
        "grid.color": "#e2e8f0",
        "text.color": "#1e293b",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569"
    })
    
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            filtered_df['Attrition_Label'].value_counts().plot(kind='bar', color=['#22c55e','#ef4444'], ax=ax)
        st.pyplot(fig)

    with col5:
        st.subheader("Attrition by Department")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            sns.countplot(x="Department", hue="Attrition_Label", data=filtered_df, ax=ax2, palette=["#22c55e", "#ef4444"])
            ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    col6, col7 = st.columns(2)
    with col6:
        st.subheader("Salary Distribution")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            sns.histplot(filtered_df[salary_col], bins=20, kde=True, color='#3b82f6', ax=ax3)
        st.pyplot(fig3)

    with col7:
        st.subheader("Salary vs Attrition")
        fig4, ax4 = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            sns.boxplot(x="Attrition_Label", y=salary_col, data=filtered_df, palette=["#22c55e", "#ef4444"], ax=ax4)
        st.pyplot(fig4)

    col8, col9 = st.columns(2)
    with col8:
        st.subheader("Age Distribution")
        fig5, ax5 = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            sns.histplot(filtered_df["Age"], bins=20, kde=True, color='#3b82f6', ax=ax5)
        st.pyplot(fig5)

    with col9:
        st.subheader("Attrition by Gender")
        fig6, ax6 = plt.subplots(figsize=(6,4))
        if not filtered_df.empty:
            sns.countplot(x="Gender", hue="Attrition_Label", data=filtered_df, palette=["#22c55e", "#ef4444"], ax=ax6)
        st.pyplot(fig6)

    st.subheader("🏆 Top 4 High Attrition Job Roles")
    if not filtered_df.empty:
        top_roles = (
            filtered_df.groupby("Job_Role")["Attrition"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        st.bar_chart(top_roles)

elif page == "Prediction":
    st.title("🔮 Employee Attrition Prediction")
    st.info("💡 Adjust key employee metrics to predict their likelihood of leaving the company.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        age_input = st.number_input("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
        salary_input = st.number_input("Salary", int(df[salary_col].min()), int(df[salary_col].max()), int(df[salary_col].median()))
        years_comp_input = st.number_input("Years at Company", int(df["Years_at_Company"].min()), int(df["Years_at_Company"].max()), int(df["Years_at_Company"].median()))

    with col_p2:
        gender_input = st.selectbox("Gender", df["Gender"].unique())
        dept_input = st.selectbox("Department", df["Department"].unique())
        wl_bal_input = st.slider("Work Life Balance", int(df["Work_Life_Balance"].min()), int(df["Work_Life_Balance"].max()), int(df["Work_Life_Balance"].median()))

    with col_p3:
        overtime_input = st.selectbox("Overtime", df["Overtime"].unique())
        role_input = st.selectbox("Job Role", df["Job_Role"].unique())
        job_sat_input = st.slider("Job Satisfaction", int(df["Job_Satisfaction"].min()), int(df["Job_Satisfaction"].max()), int(df["Job_Satisfaction"].median()))

    with st.expander("⚙️ Advanced Employee Metrics"):
        st.markdown("Adjust these advanced metrics for a more precise prediction.")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            job_level_input = st.number_input("Job Level", int(df["Job_Level"].min()), int(df["Job_Level"].max()), int(df["Job_Level"].median()))
            yrs_curr_role_input = st.number_input("Years in Current Role", int(df["Years_in_Current_Role"].min()), int(df["Years_in_Current_Role"].max()), int(df["Years_in_Current_Role"].median()))
            yrs_since_prom_input = st.number_input("Years Since Last Prom", int(df["Years_Since_Last_Promotion"].min()), int(df["Years_Since_Last_Promotion"].max()), int(df["Years_Since_Last_Promotion"].median()))
            num_comp_worked_input = st.number_input("Num Companies Worked", int(df["Number_of_Companies_Worked"].min()), int(df["Number_of_Companies_Worked"].max()), int(df["Number_of_Companies_Worked"].median()))
            
        with col_a2:
            perf_rating_input = st.slider("Performance Rating", int(df["Performance_Rating"].min()), int(df["Performance_Rating"].max()), int(df["Performance_Rating"].median()))
            train_hours_input = st.slider("Training Hours Last Year", int(df["Training_Hours_Last_Year"].min()), int(df["Training_Hours_Last_Year"].max()), int(df["Training_Hours_Last_Year"].median()))
            proj_count_input = st.slider("Project Count", int(df["Project_Count"].min()), int(df["Project_Count"].max()), int(df["Project_Count"].median()))
            absent_input = st.number_input("Absenteeism (Days)", int(df["Absenteeism"].min()), int(df["Absenteeism"].max()), int(df["Absenteeism"].median()))
            
        with col_a3:
            we_sat_input = st.slider("Work Env Satisfaction", int(df["Work_Environment_Satisfaction"].min()), int(df["Work_Environment_Satisfaction"].max()), int(df["Work_Environment_Satisfaction"].median()))
            rel_mgr_input = st.slider("Relation with Manager", int(df["Relationship_with_Manager"].min()), int(df["Relationship_with_Manager"].max()), int(df["Relationship_with_Manager"].median()))
            job_inv_input = st.slider("Job Involvement", int(df["Job_Involvement"].min()), int(df["Job_Involvement"].max()), int(df["Job_Involvement"].median()))


    if st.button("Predict Attrition"):
        input_dict = {}
        for col in model_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_dict[col] = df[col].median()
                else:
                    input_dict[col] = df[col].mode()[0]
            else:
                input_dict[col] = 0
                
        input_dict["Age"] = age_input
        input_dict[salary_col] = salary_input
        input_dict["Gender"] = gender_input
        input_dict["Department"] = dept_input
        input_dict["Years_at_Company"] = years_comp_input
        input_dict["Job_Satisfaction"] = job_sat_input
        input_dict["Job_Role"] = role_input
        input_dict["Work_Life_Balance"] = wl_bal_input
        input_dict["Overtime"] = overtime_input
        input_dict["Job_Level"] = job_level_input
        input_dict["Years_in_Current_Role"] = yrs_curr_role_input
        input_dict["Years_Since_Last_Promotion"] = yrs_since_prom_input
        input_dict["Performance_Rating"] = perf_rating_input
        input_dict["Training_Hours_Last_Year"] = train_hours_input
        input_dict["Project_Count"] = proj_count_input
        input_dict["Absenteeism"] = absent_input
        input_dict["Work_Environment_Satisfaction"] = we_sat_input
        input_dict["Relationship_with_Manager"] = rel_mgr_input
        input_dict["Job_Involvement"] = job_inv_input
        input_dict["Number_of_Companies_Worked"] = num_comp_worked_input
        
        # Calculate engineered features explicitly
        input_dict["Experience_Ratio"] = years_comp_input / age_input if age_input > 0 else 0
        input_dict["Income_Per_Year"] = salary_input / (years_comp_input + 1)
        input_dict["Stability"] = years_comp_input / (num_comp_worked_input + 1)
    
        
        input_df = pd.DataFrame([input_dict])
        
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col])
                if input_df[col][0] in le.classes_:
                    input_df[col] = le.transform(input_df[col])
                else:
                    input_df[col] = 0

        input_df = input_df[model_columns]
        prob = model.predict_proba(input_df)[0][1]

        # Use beautiful cards for prediction results
        if prob >= 0.35:
            st.error(f"🚨 High Risk of Attrition ({prob*100:.2f}%)")
        elif prob >= 0.28:
            st.warning(f"⚠️ Moderate Risk of Attrition ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Attrition ({prob*100:.2f}%)")

elif page == "About":
    st.title("About This Project")
    
    st.markdown("## Talentlock Attrition Prediction System")
    st.markdown("This project is an end-to-end Machine Learning application developed to predict employee attrition probability based on historical employee data.TalentLock is an Employee Attrition Analysis project aimed at understanding the factors that lead employees to leave an organization. Employee attrition is a major challenge for companies as it affects productivity, cost, and overall business performance. This project focuses on analyzing employee data to identify patterns and key reasons behind attrition, helping organizations make better HR decisions.")
    
    st.markdown("---")
    
    st.markdown("## Project Objective")
    st.markdown("The goal of this project is to analyze employee attributes such as job role, salary, demographics, and work conditions to identify factors driving turnover and predict future attrition.")

    st.markdown("---")
    
    st.markdown("## Application Features")
    st.markdown("""
    - **Dashboard**: High-level key performance metrics regarding employee turn-over.
    - **EDA**: Extensive interactive charts to find patterns and correlations in HR data.
    - **Prediction Engine**: A Machine Learning tab allowing real-time forecasting of individual employee attrition risk. 
    """)

    st.markdown("---")
    
    st.markdown("## Underlying Data Overview")
    tab1, tab2 = st.tabs(["🧹 Cleaned Data", "📁 Raw Data"])
    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
    with tab2:
        try:
            raw_df = pd.read_csv("employee_attrition_dataset.csv")
            st.dataframe(raw_df.head(100), use_container_width=True)
        except:
            st.warning("Raw dataset not found")