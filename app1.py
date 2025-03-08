import streamlit as st
import pandas as pd
import joblib
import shap
from streamlit_shap import st_shap

if 'language' not in st.session_state:
    st.session_state.language = 'en'  

translations = {
    'en': {
        'title': '🩺 ICU Multidrug-Resistant Bacteria Clinical Early Warning System',
        'subheader': 'A Predictive Analytics Platform for Prevention and Control of Antimicrobial Resistance in Critical Care Units',
        'ltcf': 'Longterm Care Facility Residency (0=No, 1=Yes)',
        'prior_bed': 'Prior Bed with MDRO Patients (0=No, 1=Yes)',
        'cvc': 'Central Venous Catheter (0=No, 1=Yes)',
        'surgery': 'Recent Surgery (0=No, 1=Yes)',
        'arterial': 'Arterial Tube Days',
        'predict_button': 'Predict Infection Risk',
        'prediction': 'Prediction',
        'probability': 'Probability',
        'high_risk': 'High Risk',
        'low_risk': 'Low Risk',
        'model_explanation': 'Model Explanation (SHAP force plot)',
        'error': 'Error',
        'developed_by': 'Developed by: Gu Genying | Affiliated Mingji Hospital of Nanjing Medical University',
        'contact': 'Contact: +86-137-7073-0245 | © 2025 Clinical Prediction Systems',
        'language_selector': 'Language / 语言'
    },
    'zh': {
        'title': '🩺 ICU多重耐药菌临床早期预警系统',
        'subheader': '重症监护单元抗菌素耐药性预防和控制的预测分析平台',
        'ltcf': '长期护理机构居住 (0=否, 1=是)',
        'prior_bed': '先床位居住MDRO患者 (0=否, 1=是)',
        'cvc': '中心静脉置管 (0=否, 1=是)',
        'surgery': '感染/定值前手术 (0=否, 1=是)',
        'arterial': '动脉导管留置天数',
        'predict_button': '预测MDRO感染风险',
        'prediction': '预测结果',
        'probability': '概率',
        'high_risk': '高MDRO感染风险',
        'low_risk': '低MDRO感染风险',
        'model_explanation': '模型解释（SHAP力图）',
        'error': '错误',
        'developed_by': '开发者: 顾艮莹 | 南京医科大学附属明基医院',
        'contact': '联系方式: +86-137-7073-0245 | © 2025 临床预测系统',
        'language_selector': 'Language / 语言'
    }
}

lang = st.sidebar.selectbox(
    translations[st.session_state.language]['language_selector'],
    options=['en', 'zh'],
    format_func=lambda x: 'English' if x == 'en' else '中文',
    index=0 if st.session_state.language == 'en' else 1
)

if lang != st.session_state.language:
    st.session_state.language = lang
    st.rerun()

lang = st.session_state.language
t = translations[lang]

st.markdown("""
<style>
.title {color: #2E86C1; font-size: 40px; font-weight: bold;}
.subheader {color: #3498DB; font-size: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<p class="title">{t["title"]}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subheader">{t["subheader"]}</p>', unsafe_allow_html=True)

Longtermcare_facility_residency = st.selectbox(t["ltcf"], ("0", "1"))
Prior_bed_housed_MDRO_patients = st.selectbox(t["prior_bed"], ("0", "1"))
CVC = st.selectbox(t["cvc"], ("0", "1"))
surgery = st.selectbox(t["surgery"], ("0", "1"))
arterial_Tubedays = st.number_input(t["arterial"], min_value=0, value=0)

# Prediction button
if st.button(t["predict_button"]):
    try:
        pipeline = joblib.load("xgb_model.pkl")
        
        # Display pipeline steps for debugging
        st.write("Pipeline Steps:", pipeline.named_steps.keys())
        
        xgb_model = pipeline.named_steps['xgb']
        
        input_data = pd.DataFrame({
            "Longtermcare_facility_residency": [int(Longtermcare_facility_residency)],
            "Prior_bed_housed_MDRO_patients": [int(Prior_bed_housed_MDRO_patients)],
            "CVC": [int(CVC)],
            "surgery": [int(surgery)],
            "arterial_Tubedays": [arterial_Tubedays]
        })
        
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        risk_level = t["high_risk"] if prediction == 1 else t["low_risk"]
        
        st.success(f"{t['prediction']}: {risk_level} ({t['probability']}: {probability:.2%})")
        
        # Process data for SHAP explanation
        if len(pipeline.named_steps) > 1:
            processed_data = pipeline[:-1].transform(input_data)
            feature_names = pipeline[:-1].get_feature_names_out()
        else:
            processed_data = input_data.values
            feature_names = input_data.columns.tolist()
        
        # SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(processed_data)
        
        # Display SHAP explanation
        st.subheader(t["model_explanation"])
        shap_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=processed_data,
            feature_names=feature_names,
            matplotlib=False  # 使用HTML组件
        )
        st.markdown("""
        <style>
        .shap-container {
            max-width: 100%;
            overflow-x: auto;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)

        # 在容器中显示图表
        with st.markdown('<div class="shap-container">', unsafe_allow_html=True):
            st_shap(
                shap_plot,
                height=400,  # 适当降低高度
                width=1200   # 设置稍大于常规屏幕宽度的尺寸
            )
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"{t['error']}: {str(e)}")
        raise e

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; font-size: 12px; color: #707B7C;">
<p>{t['developed_by']}</p>
<p>{t['contact']}</p>
</div>
""", unsafe_allow_html=True)
