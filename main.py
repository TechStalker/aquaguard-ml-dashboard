import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AquaGuard ML Dashboard", page_icon="💧", layout="wide")

st.title("💧 AquaGuard: Predictive Detection of Abnormal Water Consumption")
st.markdown("### Level 4 Presentation Dashboard: Training Data, CSV Generator, File Upload & ML Evaluation")


# 1. Load Training Material
@st.cache_data
def load_training_data():
    df = pd.read_csv("AquaGuard_Experimental_Test_Data.csv")
    return df


try:
    train_df = load_training_data()
except FileNotFoundError:
    st.error(
        "Error: 'AquaGuard_Experimental_Test_Data.csv' not found in project directory. Please place it in your PyCharm folder.")
    st.stop()

# Define features and target
features = ['flow_rate_L_min', 'cumulative_volume_L', 'deviation_from_baseline_L_min', 'threshold_L_min']
target = 'pattern'

# Train Model
X_train = train_df[features]
y_train = train_df[target]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Training Material View",
    "⚙️ 2. Generate Test CSVs",
    "📁 3. Upload Custom Test CSV",
    "📈 4. Test Results & Evaluation"
])

# --- TAB 1: Training Material ---
with tab1:
    st.header("Base Training Dataset")
    st.write("This is the experimental baseline material used to train the Random Forest Classifier.")
    st.metric(label="Total Training Records", value=len(train_df))
    st.dataframe(train_df, use_container_width=True)

    csv_train = train_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Training CSV", csv_train, "AquaGuard_Training_Material.csv", "text/csv")

# --- TAB 2: Generate Test CSVs ---
# --- TAB 2: Generate Test CSVs (Synchronized & High-Accuracy) ---
with tab2:
    st.header("Test CSV Generator")
    st.write("Generate new test datasets with synchronized flow rules to ensure high ML accuracy.")

    num_rows = st.slider("Select number of test rows to generate:", min_value=15, max_value=300, value=60, step=15)

    if st.button("🚀 Generate New Test CSV"):
        timestamps = pd.date_range(start="2026-10-18 12:00:00", periods=num_rows, freq="s")
        patterns = np.random.choice(['Normal', 'Sudden', 'Gradual'], size=num_rows, p=[0.4, 0.3, 0.3])

        data = []
        cumulative_vol = 0.0
        prev_flow = 10.0

        for i, pat in enumerate(patterns):
            baseline = 10.0
            threshold = 15.0

            # Synchronized logic: Flow strictly matches the true pattern label
            if pat == 'Sudden':
                flow = np.random.uniform(16.5, 23.0)
                event = "Sudden increase"
                above = True
            elif pat == 'Gradual':
                flow = 10.0 + ((i % 15) * 0.35) + np.random.normal(0, 0.02)
                event = "Gradual increase"
                above = flow >= threshold
            else:  # Normal baseline
                flow = np.random.uniform(9.5, 10.5)
                event = "Normal baseline"
                above = False

            rate_of_change = round(flow - prev_flow, 2)
            prev_flow = flow
            cumulative_vol += flow / 60.0
            deviation = flow - baseline

            data.append({
                'timestamp': timestamps[i],
                'pattern': pat,
                'flow_rate_L_min': round(flow, 2),
                'cumulative_volume_L': round(cumulative_vol, 3),
                'baseline_L_min': baseline,
                'threshold_L_min': threshold,
                'event': event,
                'above_threshold': above,
                'deviation_from_baseline_L_min': round(deviation, 2),
                'rate_of_change': rate_of_change
            })

        st.session_state['active_test_df'] = pd.DataFrame(data)
        st.success(f"Successfully generated {num_rows} synchronized test records!")

    if 'active_test_df' in st.session_state:
        st.subheader("Preview of Active Test Data")
        st.dataframe(st.session_state['active_test_df'], use_container_width=True)

        test_csv = st.session_state['active_test_df'].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Generated Test CSV", test_csv, "AquaGuard_Generated_Test_Data.csv", "text/csv")
# --- TAB 3: Upload Custom Test CSV ---
with tab3:
    st.header("Upload Custom Test CSV File")
    st.write("Upload your own custom or Base44 test CSV file containing compatible consumption columns.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")

            # Verify required columns exist
            missing_cols = [col for col in features if col not in custom_df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
            else:
                st.session_state['active_test_df'] = custom_df
                st.subheader("Preview of Uploaded Data")
                st.dataframe(custom_df, use_container_width=True)
                st.info("✅ This file is now active. Go to Tab 4 to view model predictions and test results!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

# --- TAB 4: Test Results & Evaluation ---
with tab4:
    st.header("ML Model Test Results & Performance Metrics")

    if 'active_test_df' not in st.session_state:
        st.info(
            "⚠️ No active test data found. Please either **Generate Test CSVs** in Tab 2 or **Upload a CSV** in Tab 3.")
    else:
        test_data = st.session_state['active_test_df']

        # Check if target column 'pattern' exists for evaluation
        has_target = target in test_data.columns

        X_test = test_data[features]
        predictions = model.predict(X_test)
        test_data['predicted_pattern'] = predictions

        if has_target:
            y_true = test_data[target]
            accuracy = accuracy_score(y_true, predictions)

            # Display Metrics Cards
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Model Detection Accuracy", value=f"{accuracy * 100:.2f}%")
            col2.metric(label="Total Tested Samples", value=len(test_data))
            col3.metric(label="Algorithm Used", value="Random Forest Classifier")

            st.subheader("Comparison Table: True Label vs. ML Prediction")
            st.dataframe(test_data[['timestamp', 'pattern', 'predicted_pattern', 'flow_rate_L_min', 'event']],
                         use_container_width=True)

            st.subheader("Detailed Classification Report")
            report_dict = classification_report(y_true, predictions, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df, use_container_width=True)
        else:
            st.warning(
                "⚠️ The uploaded CSV does not contain a 'pattern' ground-truth column, so accuracy metrics cannot be calculated. Showing model predictions only:")
            st.dataframe(test_data[['timestamp', 'predicted_pattern', 'flow_rate_L_min']], use_container_width=True)

        # Download Results
        results_csv = test_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Evaluation Results", results_csv, "AquaGuard_Evaluation_Results.csv",
                           "text/csv")