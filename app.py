import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(page_title="Blood Risk Predictor", page_icon="🩸", layout="wide")

# ------------------- CUSTOM STYLING -------------------
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #DC143C; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💉 Blood Risk Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered prediction system for blood donation risk assessment</p>', unsafe_allow_html=True)

# ------------------- LOAD MODEL & SCALER -------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("lr_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'lr_model.pkl' and 'scaler.pkl' are in the directory.")
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

# ------------------- FEATURES -------------------
scaled_features = ["Recency", "Frequency", "Monetary", "Time"]
all_features = [
    "Recency", "Frequency", "Monetary", "Time",
    "Feature5", "Feature6", "Feature7", "Feature8",
    "Feature9", "Feature10"
]

# ------------------- SIDEBAR INPUT -------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/blood-donation.png", width=100)
st.sidebar.title("📋 Patient Input Panel")
st.sidebar.markdown("---")

# File Upload Section
st.sidebar.subheader("📤 Upload Previous Report")
uploaded_file = st.sidebar.file_uploader(
    "Upload report to extract data",
    type=['pdf', 'csv', 'xlsx', 'xls', 'txt', 'json'],
    help="Upload a previously generated report (PDF, CSV, Excel, TXT, JSON) to auto-fill the form"
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            from PyPDF2 import PdfReader
            
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            st.sidebar.success("✅ PDF uploaded successfully!")
            
            with st.sidebar.expander("📄 View Extracted Text", expanded=False):
                st.text_area("Extracted Content", text, height=200)
            
            # Try to parse data from PDF
            try:
                import re
                recency_match = re.search(r'Recency:\s*(\d+(?:\.\d+)?)', text)
                frequency_match = re.search(r'Frequency:\s*(\d+(?:\.\d+)?)', text)
                monetary_match = re.search(r'Monetary:\s*(\d+(?:\.\d+)?)', text)
                time_match = re.search(r'Time:\s*(\d+(?:\.\d+)?)', text)
                
                if all([recency_match, frequency_match, monetary_match, time_match]):
                    st.session_state.uploaded_recency = float(recency_match.group(1))
                    st.session_state.uploaded_frequency = float(frequency_match.group(1))
                    st.session_state.uploaded_monetary = float(monetary_match.group(1))
                    st.session_state.uploaded_time = float(time_match.group(1))
                    st.sidebar.info("📊 Data extracted and loaded into form!")
            except:
                st.sidebar.warning("⚠️ Could not auto-parse data. Please enter manually.")
        
        elif file_type == 'csv':
            df_upload = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ CSV uploaded successfully!")
            
            with st.sidebar.expander("📄 View CSV Data", expanded=False):
                st.dataframe(df_upload.head())
            
            # Try to extract data from CSV
            if all(col in df_upload.columns for col in ['Recency', 'Frequency', 'Monetary', 'Time']):
                st.session_state.uploaded_recency = float(df_upload['Recency'].iloc[0])
                st.session_state.uploaded_frequency = float(df_upload['Frequency'].iloc[0])
                st.session_state.uploaded_monetary = float(df_upload['Monetary'].iloc[0])
                st.session_state.uploaded_time = float(df_upload['Time'].iloc[0])
                st.sidebar.info("📊 Data extracted and loaded into form!")
            else:
                st.sidebar.warning("⚠️ CSV must contain: Recency, Frequency, Monetary, Time columns")
        
        elif file_type in ['xlsx', 'xls']:
            df_upload = pd.read_excel(uploaded_file)
            st.sidebar.success("✅ Excel file uploaded successfully!")
            
            with st.sidebar.expander("📄 View Excel Data", expanded=False):
                st.dataframe(df_upload.head())
            
            # Try to extract data from Excel
            if all(col in df_upload.columns for col in ['Recency', 'Frequency', 'Monetary', 'Time']):
                st.session_state.uploaded_recency = float(df_upload['Recency'].iloc[0])
                st.session_state.uploaded_frequency = float(df_upload['Frequency'].iloc[0])
                st.session_state.uploaded_monetary = float(df_upload['Monetary'].iloc[0])
                st.session_state.uploaded_time = float(df_upload['Time'].iloc[0])
                st.sidebar.info("📊 Data extracted and loaded into form!")
            else:
                st.sidebar.warning("⚠️ Excel must contain: Recency, Frequency, Monetary, Time columns")
        
        elif file_type == 'txt':
            text = uploaded_file.read().decode('utf-8')
            st.sidebar.success("✅ Text file uploaded successfully!")
            
            with st.sidebar.expander("📄 View Text Content", expanded=False):
                st.text_area("File Content", text, height=200)
            
            # Try to parse data from text
            try:
                import re
                recency_match = re.search(r'Recency:\s*(\d+(?:\.\d+)?)', text)
                frequency_match = re.search(r'Frequency:\s*(\d+(?:\.\d+)?)', text)
                monetary_match = re.search(r'Monetary:\s*(\d+(?:\.\d+)?)', text)
                time_match = re.search(r'Time:\s*(\d+(?:\.\d+)?)', text)
                
                if all([recency_match, frequency_match, monetary_match, time_match]):
                    st.session_state.uploaded_recency = float(recency_match.group(1))
                    st.session_state.uploaded_frequency = float(frequency_match.group(1))
                    st.session_state.uploaded_monetary = float(monetary_match.group(1))
                    st.session_state.uploaded_time = float(time_match.group(1))
                    st.sidebar.info("📊 Data extracted and loaded into form!")
            except:
                st.sidebar.warning("⚠️ Could not auto-parse data. Please enter manually.")
        
        elif file_type == 'json':
            import json
            json_data = json.load(uploaded_file)
            st.sidebar.success("✅ JSON file uploaded successfully!")
            
            with st.sidebar.expander("📄 View JSON Data", expanded=False):
                st.json(json_data)
            
            # Try to extract data from JSON
            if all(key in json_data for key in ['Recency', 'Frequency', 'Monetary', 'Time']):
                st.session_state.uploaded_recency = float(json_data['Recency'])
                st.session_state.uploaded_frequency = float(json_data['Frequency'])
                st.session_state.uploaded_monetary = float(json_data['Monetary'])
                st.session_state.uploaded_time = float(json_data['Time'])
                st.sidebar.info("📊 Data extracted and loaded into form!")
            else:
                st.sidebar.warning("⚠️ JSON must contain: Recency, Frequency, Monetary, Time keys")
                
    except ImportError as e:
        st.sidebar.error(f"❌ Required library not installed: {str(e)}")
        st.sidebar.info("Install with: pip install PyPDF2 openpyxl")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {str(e)}")

st.sidebar.markdown("---")

def get_user_input():
    # Check if data was loaded from PDF
    default_recency = st.session_state.get('uploaded_recency', 10)
    default_frequency = st.session_state.get('uploaded_frequency', 5)
    default_monetary = st.session_state.get('uploaded_monetary', 50)
    default_time = st.session_state.get('uploaded_time', 36)
    
    with st.sidebar.expander("🩸 Donor Information", expanded=True):
        st.markdown("**Primary Donation Metrics**")
        
        Recency = st.slider(
            "Days since last donation",
            min_value=0, 
            max_value=100, 
            value=int(default_recency),
            help="Number of days elapsed since the patient's most recent blood donation"
        )
        
        Frequency = st.slider(
            "Total donation count",
            min_value=0, 
            max_value=50, 
            value=int(default_frequency),
            help="Total number of times the patient has donated blood in their history"
        )
        
        Monetary = st.slider(
            "Blood volume donated (units)",
            min_value=0, 
            max_value=500, 
            value=int(default_monetary),
            help="Total volume of blood donated, measured in standard units (typically 450-500ml per unit)"
        )
        
        Time = st.slider(
            "Active donor period (months)",
            min_value=0, 
            max_value=120, 
            value=int(default_time),
            help="Duration in months that the patient has been an active blood donor"
        )
    
    with st.sidebar.expander("🔧 Advanced Options", expanded=False):
        st.markdown("**Additional Features** (Optional)")
        st.caption("These features are automatically set to default values. Adjust only if you have specific data.")
        
        col1, col2 = st.columns(2)
        with col1:
            Feature5 = st.number_input("Feature 5", value=0.0, step=0.1)
            Feature6 = st.number_input("Feature 6", value=0.0, step=0.1)
            Feature7 = st.number_input("Feature 7", value=0.0, step=0.1)
        with col2:
            Feature8 = st.number_input("Feature 8", value=0.0, step=0.1)
            Feature9 = st.number_input("Feature 9", value=0.0, step=0.1)
            Feature10 = st.number_input("Feature 10", value=0.0, step=0.1)
    
    data = {
        "Recency": Recency,
        "Frequency": Frequency,
        "Monetary": Monetary,
        "Time": Time,
        "Feature5": Feature5,
        "Feature6": Feature6,
        "Feature7": Feature7,
        "Feature8": Feature8,
        "Feature9": Feature9,
        "Feature10": Feature10
    }
    
    df = pd.DataFrame([data], columns=all_features)
    return df

input_df = get_user_input()

# Add prediction button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Analyze Risk", type="primary", use_container_width=True)

# ------------------- SCALING -------------------
scaled_part = scaler.transform(input_df[scaled_features])
scaled_df = pd.DataFrame(scaled_part, columns=scaled_features)

for f in all_features:
    if f not in scaled_features:
        scaled_df[f] = input_df[f].values

final_input = scaled_df[all_features]

# ------------------- PREDICTION -------------------
prediction = model.predict(final_input)[0]
prob = model.predict_proba(final_input)[0][1]

# ------------------- MAIN CONTENT -------------------
if predict_button or 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = True

tab1, tab2, tab3 = st.tabs(["📊 Prediction Results", "📈 Feature Analysis", "📄 Patient Report"])

with tab1:
    st.markdown("---")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("🎯 Risk Assessment")
        if prediction == 1:
            st.error("⚠️ **HIGH RISK DETECTED**")
            st.warning("⚕️ Recommendation: Consult with a healthcare professional before next donation")
        else:
            st.success("✅ **LOW RISK**")
            st.info("👍 Patient is suitable for blood donation")
    
    with col2:
        st.subheader("📊 Risk Probability")
        st.metric(
            label="Probability Score",
            value=f"{prob*100:.1f}%",
            delta=f"{(prob-0.5)*100:.1f}% from threshold" if prob > 0.5 else f"{(0.5-prob)*100:.1f}% below threshold",
            delta_color="inverse" if prediction == 1 else "normal"
        )
        
        # Progress bar
        st.progress(prob, text=f"Risk Level: {prob*100:.1f}%")
    
    with col3:
        st.subheader("🔔 Status")
        if prob >= 0.75:
            st.error("Critical")
        elif prob >= 0.5:
            st.warning("Elevated")
        elif prob >= 0.25:
            st.info("Moderate")
        else:
            st.success("Normal")
    
    st.markdown("---")
    
    # Input Summary
    with st.expander("📋 View Input Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Days Since Last Donation", f"{input_df['Recency'].values[0]} days")
            st.metric("Total Donations", f"{input_df['Frequency'].values[0]} times")
        with col2:
            st.metric("Blood Volume Donated", f"{input_df['Monetary'].values[0]} units")
            st.metric("Active Donor Period", f"{input_df['Time'].values[0]} months")

with tab2:
    st.subheader("📈 Feature Importance Analysis")
    
    # Get feature coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#DC143C' if x > 0 else '#2E8B57' for x in feature_importance['Coefficient']]
            ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient Value', fontsize=12)
            ax.set_title('Model Feature Coefficients', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Feature Impact Legend**")
            st.markdown("🔴 **Red bars**: Increase risk")
            st.markdown("🟢 **Green bars**: Decrease risk")
            st.markdown("---")
            st.markdown("**Top 3 Important Features**")
            for idx, row in feature_importance.head(3).iterrows():
                st.metric(
                    row['Feature'],
                    f"{row['Coefficient']:.3f}",
                    delta="High impact"
                )
    
    st.markdown("---")
    
    # Patient profile visualization
    with st.expander("🧪 Patient Profile Visualization", expanded=True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Recency vs Frequency
        axes[0, 0].scatter(input_df['Recency'], input_df['Frequency'], 
                          s=200, c='red' if prediction == 1 else 'green', alpha=0.6, edgecolors='black')
        axes[0, 0].set_xlabel('Recency (days)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Recency vs Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time vs Monetary
        axes[0, 1].scatter(input_df['Time'], input_df['Monetary'], 
                          s=200, c='red' if prediction == 1 else 'green', alpha=0.6, edgecolors='black')
        axes[0, 1].set_xlabel('Time (months)')
        axes[0, 1].set_ylabel('Monetary (units)')
        axes[0, 1].set_title('Active Period vs Blood Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Risk probability gauge
        axes[1, 0].barh(['Risk Probability'], [prob], color='red' if prediction == 1 else 'green', alpha=0.6)
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_title('Risk Probability Score')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Feature values
        main_features = ['Recency', 'Frequency', 'Monetary', 'Time']
        values = [input_df[f].values[0] for f in main_features]
        axes[1, 1].bar(main_features, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Input Feature Values')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("📄 Patient Report Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Report Summary**")
        st.write(f"**Patient ID:** {np.random.randint(10000, 99999)}")
        st.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Risk Level:** {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
        st.write(f"**Risk Probability:** {prob*100:.2f}%")
        
        st.markdown("---")
        st.markdown("**Input Parameters:**")
        report_df = input_df[scaled_features].copy()
        st.dataframe(report_df, use_container_width=True)
    
    with col2:
        st.markdown("**📥 Download Options**")
        
        # Generate PDF
        def create_pdf():
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            
            try:
                # Header
                p.setFont("Helvetica-Bold", 20)
                p.drawString(100, height - 50, "Blood Risk Prediction Report")
                
                # Date
                p.setFont("Helvetica", 12)
                p.drawString(100, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Results
                p.setFont("Helvetica-Bold", 14)
                p.drawString(100, height - 120, "Risk Assessment Results:")
                p.setFont("Helvetica", 12)
                p.drawString(100, height - 145, f"Risk Level: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
                p.drawString(100, height - 165, f"Risk Probability: {prob*100:.2f}%")
                
                # Input data
                p.setFont("Helvetica-Bold", 14)
                p.drawString(100, height - 205, "Input Parameters:")
                p.setFont("Helvetica", 11)
                y_pos = height - 230
                for feature in scaled_features:
                    p.drawString(120, y_pos, f"{feature}: {input_df[feature].values[0]}")
                    y_pos -= 20
                
                # Recommendation
                p.setFont("Helvetica-Bold", 14)
                p.drawString(100, height - 350, "Recommendation:")
                p.setFont("Helvetica", 11)
                if prediction == 1:
                    p.drawString(120, height - 375, "Consult healthcare professional before next donation")
                else:
                    p.drawString(120, height - 375, "Patient is suitable for blood donation")
                
                # Footer - Use regular Helvetica instead of Italic
                p.setFont("Helvetica", 9)
                p.drawString(100, 50, "This report is generated by AI and should be reviewed by medical professionals.")
                
            except Exception as e:
                # Fallback to basic fonts if there's any issue
                p.setFont("Helvetica", 12)
                p.drawString(100, height - 50, "Blood Risk Prediction Report")
                p.drawString(100, height - 80, f"Risk: {'HIGH' if prediction == 1 else 'LOW'}")
                p.drawString(100, height - 100, f"Probability: {prob*100:.2f}%")
            
            p.save()
            buffer.seek(0)
            return buffer
        
        pdf_buffer = create_pdf()
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"blood_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        # Text report option
        report_text = f"""Blood Risk Prediction Report
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Risk Assessment Results:
------------------------
Risk Level: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}
Risk Probability: {prob*100:.2f}%

Input Parameters:
-----------------
Recency: {input_df['Recency'].values[0]} days
Frequency: {input_df['Frequency'].values[0]} times
Monetary: {input_df['Monetary'].values[0]} units
Time: {input_df['Time'].values[0]} months

Recommendation:
---------------
{'Consult healthcare professional before next donation' if prediction == 1 else 'Patient is suitable for blood donation'}

========================================
This report is generated by AI and should be reviewed by medical professionals.
"""
        
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"blood_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.info("💡 Download the complete analysis report in PDF or TXT format")
    
    st.markdown("---")
    
    # Upload and compare section
    st.subheader("📤 Upload & Compare Previous Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Historical Report**")
        compare_file = st.file_uploader(
            "Upload a previous report for comparison",
            type=['pdf', 'csv', 'xlsx', 'xls', 'txt', 'json'],
            key="compare_upload",
            help="Compare current analysis with a previous report (PDF, CSV, Excel, TXT, JSON)"
        )
        
        if compare_file is not None:
            file_type = compare_file.name.split('.')[-1].lower()
            
            try:
                prev_prob = None
                prev_risk = None
                
                if file_type == 'pdf':
                    from PyPDF2 import PdfReader
                    
                    pdf_reader = PdfReader(compare_file)
                    compare_text = ""
                    for page in pdf_reader.pages:
                        compare_text += page.extract_text()
                    
                    st.success("✅ Historical PDF report loaded!")
                    
                    # Extract previous risk data
                    import re
                    prev_prob_match = re.search(r'Risk Probability:\s*(\d+(?:\.\d+)?)%', compare_text)
                    prev_risk_match = re.search(r'Risk Level:\s*(HIGH RISK|LOW RISK)', compare_text)
                    
                    if prev_prob_match and prev_risk_match:
                        prev_prob = float(prev_prob_match.group(1)) / 100
                        prev_risk = prev_risk_match.group(1)
                
                elif file_type == 'csv':
                    df_compare = pd.read_csv(compare_file)
                    st.success("✅ Historical CSV report loaded!")
                    
                    if 'Risk_Probability' in df_compare.columns:
                        prev_prob = float(df_compare['Risk_Probability'].iloc[0])
                        if prev_prob > 1:  # If stored as percentage
                            prev_prob = prev_prob / 100
                        prev_risk = 'HIGH RISK' if prev_prob >= 0.5 else 'LOW RISK'
                    elif 'Probability' in df_compare.columns:
                        prev_prob = float(df_compare['Probability'].iloc[0])
                        if prev_prob > 1:
                            prev_prob = prev_prob / 100
                        prev_risk = 'HIGH RISK' if prev_prob >= 0.5 else 'LOW RISK'
                
                elif file_type in ['xlsx', 'xls']:
                    df_compare = pd.read_excel(compare_file)
                    st.success("✅ Historical Excel report loaded!")
                    
                    if 'Risk_Probability' in df_compare.columns:
                        prev_prob = float(df_compare['Risk_Probability'].iloc[0])
                        if prev_prob > 1:
                            prev_prob = prev_prob / 100
                        prev_risk = 'HIGH RISK' if prev_prob >= 0.5 else 'LOW RISK'
                    elif 'Probability' in df_compare.columns:
                        prev_prob = float(df_compare['Probability'].iloc[0])
                        if prev_prob > 1:
                            prev_prob = prev_prob / 100
                        prev_risk = 'HIGH RISK' if prev_prob >= 0.5 else 'LOW RISK'
                
                elif file_type == 'txt':
                    compare_text = compare_file.read().decode('utf-8')
                    st.success("✅ Historical text report loaded!")
                    
                    import re
                    prev_prob_match = re.search(r'Risk Probability:\s*(\d+(?:\.\d+)?)%', compare_text)
                    prev_risk_match = re.search(r'Risk Level:\s*(HIGH RISK|LOW RISK)', compare_text)
                    
                    if prev_prob_match and prev_risk_match:
                        prev_prob = float(prev_prob_match.group(1)) / 100
                        prev_risk = prev_risk_match.group(1)
                
                elif file_type == 'json':
                    import json
                    json_compare = json.load(compare_file)
                    st.success("✅ Historical JSON report loaded!")
                    
                    if 'Risk_Probability' in json_compare:
                        prev_prob = float(json_compare['Risk_Probability'])
                        if prev_prob > 1:
                            prev_prob = prev_prob / 100
                        prev_risk = json_compare.get('Risk_Level', 'HIGH RISK' if prev_prob >= 0.5 else 'LOW RISK')
                
                if prev_prob is not None and prev_risk is not None:
                    st.session_state.prev_prob = prev_prob
                    st.session_state.prev_risk = prev_risk
                else:
                    st.warning("⚠️ Could not extract risk data from file. Please check the format.")
                    
            except ImportError:
                st.error("❌ PyPDF2 or openpyxl not installed. Install with: pip install PyPDF2 openpyxl")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        if 'prev_prob' in st.session_state and 'prev_risk' in st.session_state:
            st.markdown("**Comparison Results**")
            
            prev_prob = st.session_state.prev_prob
            prev_risk = st.session_state.prev_risk
            
            # Show comparison metrics
            prob_change = (prob - prev_prob) * 100
            
            st.metric(
                "Current Risk Probability",
                f"{prob*100:.1f}%",
                delta=f"{prob_change:.1f}%",
                delta_color="inverse"
            )
            
            st.metric(
                "Previous Risk Probability",
                f"{prev_prob*100:.1f}%"
            )
            
            # Risk trend
            if prob > prev_prob:
                st.warning("⚠️ Risk has increased since last assessment")
            elif prob < prev_prob:
                st.success("✅ Risk has decreased since last assessment")
            else:
                st.info("➡️ Risk level remains unchanged")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            assessments = ['Previous', 'Current']
            probabilities = [prev_prob * 100, prob * 100]
            colors = ['#FFA07A' if prev_prob > 0.5 else '#90EE90', 
                     '#DC143C' if prob > 0.5 else '#2E8B57']
            
            bars = ax.bar(assessments, probabilities, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Risk Probability (%)', fontsize=12)
            ax.set_title('Risk Comparison: Previous vs Current', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Risk Threshold (50%)')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("📊 Upload a previous report to see comparison")

# ------------------- FOOTER -------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🏥 **Model Type:** Logistic Regression")
with col2:
    st.caption("🔬 **Accuracy:** Training-dependent")
with col3:
    st.caption("⚡ **Version:** 2.0")

st.caption("Developed using Streamlit + Scikit-learn | © 2025 Blood Risk Prediction System")

# Add info section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### How It Works
    This application uses a trained Logistic Regression model to predict blood donation risk based on patient history.
    
    **Key Features:**
    - 🎯 Real-time risk assessment
    - 📊 Visual probability indicators
    - 📈 Feature importance analysis
    - 📄 Downloadable PDF reports
    
    **Model Inputs:**
    - **Recency:** Days since last donation
    - **Frequency:** Total number of donations
    - **Monetary:** Total blood volume donated
    - **Time:** Duration as active donor
    
    **Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.
    """)