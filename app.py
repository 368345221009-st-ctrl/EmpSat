import streamlit as st
import pandas as pd
import joblib

# ---------- โหลดโมเดล ----------
MODEL_PATH = "ibm_hr_attrition_tree.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# โมเดลนี้ (DecisionTreeClassifier) ทำนาย Attrition (ลาออก: Yes/No)
# ใช้ฟีเจอร์ทั้งหมด 50 ตัว ตามลำดับใน model.feature_names_in_
# ค่าคงที่ตามชุดข้อมูล IBM HR ต้นฉบับ: EmployeeCount=1, StandardHours=80
EMPLOYEE_COUNT = 1
STANDARD_HOURS = 80

# ---------- หน้าเว็บ ----------
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 IBM HR — Employee Attrition Predictor")
st.write("กรอกข้อมูลพนักงานด้านล่าง แล้วกดปุ่มทำนายผลว่ามีแนวโน้มลาออกหรือไม่")

# ---------- Tab จัดกลุ่ม input ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 ข้อมูลส่วนตัว", "💼 งานและเงินเดือน", "😊 ความพึงพอใจ", "🗂️ ข้อมูลหมวดหมู่"]
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("อายุ (Age)", 18, 60, 35)
        education = st.selectbox(
            "ระดับการศึกษา (Education)", options=[1, 2, 3, 4, 5],
            index=2, help="1=ต่ำกว่าปริญญาตรี ... 5=ปริญญาเอก"
        )
        distance_from_home = st.slider("ระยะทางจากบ้านถึงที่ทำงาน (กม.)", 1, 30, 5)
    with col2:
        marital_status = st.selectbox("สถานภาพสมรส", ["Divorced", "Married", "Single"])
        gender = st.radio("เพศ", ["Female", "Male"], horizontal=True)
        employee_number = st.number_input("รหัสพนักงาน (EmployeeNumber)", min_value=1, value=1)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox(
            "แผนก (Department)",
            ["Human Resources", "Research & Development", "Sales"]
        )
        job_role = st.selectbox(
            "ตำแหน่งงาน (JobRole)",
            ["Healthcare Representative", "Human Resources", "Laboratory Technician",
             "Manager", "Manufacturing Director", "Research Director",
             "Research Scientist", "Sales Executive", "Sales Representative"]
        )
        job_level = st.selectbox("ระดับตำแหน่ง (JobLevel)", [1, 2, 3, 4, 5], index=0)
        education_field = st.selectbox(
            "สาขาที่จบการศึกษา (EducationField)",
            ["Human Resources", "Life Sciences", "Marketing", "Medical",
             "Other", "Technical Degree"]
        )
        business_travel = st.selectbox(
            "การเดินทางไปทำงาน (BusinessTravel)",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        over_time = st.radio("ทำงานล่วงเวลา (OverTime)", ["No", "Yes"], horizontal=True)
    with col2:
        monthly_income = st.number_input("เงินเดือน (MonthlyIncome)", min_value=1000, max_value=50000, value=5000, step=100)
        daily_rate = st.number_input("อัตราค่าแรงรายวัน (DailyRate)", min_value=100, max_value=1500, value=800)
        hourly_rate = st.number_input("อัตราค่าแรงรายชั่วโมง (HourlyRate)", min_value=30, max_value=100, value=65)
        monthly_rate = st.number_input("อัตราค่าแรงรายเดือน (MonthlyRate)", min_value=2000, max_value=27000, value=14000)
        percent_salary_hike = st.slider("เปอร์เซ็นต์เงินเดือนที่ปรับขึ้น (%)", 11, 25, 15)
        stock_option_level = st.selectbox("ระดับหุ้นพนักงาน (StockOptionLevel)", [0, 1, 2, 3], index=0)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        environment_satisfaction = st.select_slider("ความพึงพอใจต่อสภาพแวดล้อม (1-4)", [1, 2, 3, 4], value=3)
        job_satisfaction = st.select_slider("ความพึงพอใจต่องาน (1-4)", [1, 2, 3, 4], value=3)
    with col2:
        relationship_satisfaction = st.select_slider("ความพึงพอใจด้านความสัมพันธ์ (1-4)", [1, 2, 3, 4], value=3)
        work_life_balance = st.select_slider("สมดุลชีวิตและการทำงาน (1-4)", [1, 2, 3, 4], value=3)
    job_involvement = st.select_slider("ความมีส่วนร่วมกับงาน (JobInvolvement, 1-4)", [1, 2, 3, 4], value=3)
    performance_rating = st.select_slider("ผลการประเมินงาน (PerformanceRating, 1-4)", [1, 2, 3, 4], value=3)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        num_companies_worked = st.slider("จำนวนบริษัทที่เคยทำงานมา", 0, 9, 1)
        total_working_years = st.slider("จำนวนปีที่ทำงานมาทั้งหมด", 0, 40, 8)
        years_at_company = st.slider("จำนวนปีที่อยู่บริษัทนี้", 0, 40, 5)
        training_times_last_year = st.slider("จำนวนครั้งที่อบรมในปีที่แล้ว", 0, 6, 2)
    with col2:
        years_in_current_role = st.slider("จำนวนปีในตำแหน่งปัจจุบัน", 0, 18, 3)
        years_since_last_promotion = st.slider("จำนวนปีตั้งแต่เลื่อนตำแหน่งครั้งล่าสุด", 0, 15, 1)
        years_with_curr_manager = st.slider("จำนวนปีที่ทำงานกับหัวหน้าคนปัจจุบัน", 0, 17, 3)

# TotalSatisfaction = ผลรวมของคะแนนความพึงพอใจทั้ง 4 ด้าน (ฟีเจอร์ engineered ตามชุดข้อมูลนี้)
total_satisfaction = (
    environment_satisfaction + job_satisfaction + relationship_satisfaction + work_life_balance
)

# ---------- สร้าง dict ของฟีเจอร์ทั้งหมด (one-hot ตรงกับตอนเทรน, drop_first) ----------
row = {
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance_from_home,
    "Education": education,
    "EmployeeCount": EMPLOYEE_COUNT,
    "EmployeeNumber": employee_number,
    "EnvironmentSatisfaction": environment_satisfaction,
    "HourlyRate": hourly_rate,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobSatisfaction": job_satisfaction,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": monthly_rate,
    "NumCompaniesWorked": num_companies_worked,
    "PercentSalaryHike": percent_salary_hike,
    "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StandardHours": STANDARD_HOURS,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": training_times_last_year,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "YearsWithCurrManager": years_with_curr_manager,
    "TotalSatisfaction": total_satisfaction,
    "Department_Human Resources": int(department == "Human Resources"),
    "Department_Research & Development": int(department == "Research & Development"),
    "Department_Sales": int(department == "Sales"),
    "EducationField_Human Resources": int(education_field == "Human Resources"),
    "EducationField_Life Sciences": int(education_field == "Life Sciences"),
    "EducationField_Marketing": int(education_field == "Marketing"),
    "EducationField_Medical": int(education_field == "Medical"),
    "EducationField_Other": int(education_field == "Other"),
    "EducationField_Technical Degree": int(education_field == "Technical Degree"),
    "BusinessTravel_Travel_Frequently": int(business_travel == "Travel_Frequently"),
    "BusinessTravel_Travel_Rarely": int(business_travel == "Travel_Rarely"),
    "Gender_Male": int(gender == "Male"),
    "JobRole_Human Resources": int(job_role == "Human Resources"),
    "JobRole_Laboratory Technician": int(job_role == "Laboratory Technician"),
    "JobRole_Manager": int(job_role == "Manager"),
    "JobRole_Manufacturing Director": int(job_role == "Manufacturing Director"),
    "JobRole_Research Director": int(job_role == "Research Director"),
    "JobRole_Research Scientist": int(job_role == "Research Scientist"),
    "JobRole_Sales Executive": int(job_role == "Sales Executive"),
    "JobRole_Sales Representative": int(job_role == "Sales Representative"),
    "MaritalStatus_Married": int(marital_status == "Married"),
    "MaritalStatus_Single": int(marital_status == "Single"),
    "OverTime_Yes": int(over_time == "Yes"),
}

# จัดคอลัมน์ให้ตรงลำดับกับตอนเทรนโมเดลเป๊ะ ๆ
input_df = pd.DataFrame([row])[model.feature_names_in_]

with st.expander("🔎 ดูข้อมูลที่จะส่งให้โมเดล"):
    st.dataframe(input_df, use_container_width=True)

# ---------- ทำนายผล ----------
st.divider()
if st.button("🔮 ทำนายผล", type="primary"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    leave_prob = proba[classes.index("Yes")] * 100

    if prediction == "Yes":
        st.error(f"⚠️ ทำนายว่า: **มีแนวโน้มลาออก (Attrition = Yes)** — ความน่าจะเป็น {leave_prob:.1f}%")
    else:
        st.success(f"✅ ทำนายว่า: **ไม่มีแนวโน้มลาออก (Attrition = No)** — ความน่าจะเป็นลาออกเพียง {leave_prob:.1f}%")

    st.progress(leave_prob / 100)

# ---------- Footer ----------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "สมาชิก น.ส.กมวรรณ จันทร์ผึ้ง001&nbsp;&nbsp;"
    "น.ส.ชลดา อิศรเสนา ณ อยุธยา009&nbsp;&nbsp;"
    "น.ส.ณัฐพร เผือกผ่อง016"
    "</p>",
    unsafe_allow_html=True,
)
