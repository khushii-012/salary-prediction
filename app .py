import streamlit as st
import pandas as pd
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title('💰 Salary Prediction App')
st.write('Predict the salary based on various job attributes.')

# --- Load the compressed model ---
@st.cache_resource
def load_model(model_path):
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model_path = 'random_forest_regressor_model_compressed.pkl.gz'
model = load_model(model_path)

# --- Helper function for encoding categorical features ---
# In a production scenario, you would save the fitted LabelEncoders for each column
# For this example, we'll recreate them using the unique values from the original dataframe.
# This assumes the categories seen during training are the only ones expected.

# Recreating the original dataframe structure and unique values for encoders
# (This is a simplified approach, a robust solution would save the encoders)

# Get unique values from the original df for Label Encoding consistency
# NOTE: This requires access to the original training data's unique values for each categorical column.
# For a real application, you'd save these or the fitted LabelEncoder objects.

# Let's use the unique values that were present in the training data to initialize new encoders
# (This is a simplification. In reality, you'd save the fitted encoders.)

def get_fitted_label_encoders(df_original):
    encoders = {}
    for col in df_original.columns:
        if df_original[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(df_original[col].astype(str).unique())
            encoders[col] = le
    return encoders

# Check if df_encoded (from previous notebook steps) is available
if 'df_encoded' in globals():
    # Re-construct unique values from the original `df` before encoding
    # This is a bit tricky since we only have `df_encoded` and `df` in the kernel state.
    # Ideally, we should have saved the `LabelEncoder` objects themselves.

    # For this demonstration, we'll use the original `df` to fit new encoders.
    # This assumes that `df` in the global scope still holds the pre-encoded data.
    try:
        original_df_for_encoders = pd.read_csv('/content/Salary_Dataset_DataScienceLovers (1).csv')
        original_df_for_encoders['Company Name'].fillna(original_df_for_encoders['Company Name'].mode()[0], inplace=True)
        label_encoders = get_fitted_label_encoders(original_df_for_encoders)
    except Exception as e:
        st.error(f"Error initializing label encoders: {e}. Please ensure the original CSV is accessible or encoders are saved.")
        st.stop()
else:
    st.warning("Original DataFrame (df) not found in global scope. Cannot reliably initialize LabelEncoders for new inputs. Prediction might be inaccurate.")
    # Fallback for demonstration if df is not available (less robust)
    label_encoders = {
        'Company Name': LabelEncoder().fit(['Sasken', 'Advanced Millennium Technologies', 'Unacademy', 'SnapBizz Cloudtech', 'Appoids Tech Solutions', 'Tata Consultancy Services']), # Example unique values
        'Job Title': LabelEncoder().fit(['Android Developer', 'Web Developer', 'Full Stack Web Developer', 'Associate Web Developer']), # Example unique values
        'Location': LabelEncoder().fit(['Bangalore', 'Chennai', 'Hyderabad', 'New Delhi', 'Mumbai', 'Pune']), # Example unique values
        'Employment Status': LabelEncoder().fit(['Full Time', 'Intern']), # Example unique values
        'Job Roles': LabelEncoder().fit(['Android', 'Web', 'Full Stack', 'Associate'])
    }

# --- User Input Features ---
st.header('Job Details')

rating = st.slider('Rating (out of 5)', 1.0, 5.0, 3.5, 0.1)
company_name_input = st.text_input('Company Name', 'Tata Consultancy Services')
job_title_input = st.text_input('Job Title', 'Software Engineer')
salaries_reported = st.number_input('Salaries Reported', 1, 100, 5)
location_input = st.selectbox('Location', ['Bangalore', 'Chennai', 'Hyderabad', 'New Delhi', 'Mumbai', 'Pune', 'Kolkata', 'Ahmedabad'])
employment_status_input = st.selectbox('Employment Status', ['Full Time', 'Intern'])
job_roles_input = st.text_input('Job Roles (e.g., Android, Web, Data Science)', 'Data Science')

# --- Preprocess User Input ---
def preprocess_input(rating, company_name, job_title, salaries_reported, location, employment_status, job_roles, label_encoders):
    data = {
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': job_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }
    input_df = pd.DataFrame([data])

    # Apply Label Encoding using the *fitted* encoders
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            # Handle potential new categories by assigning an unknown label (or error)
            # For simplicity, if a new category is seen, we'll try to transform,
            # which will raise an error if not handled. For robustness, map to a default value.
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError:
                # Handle unseen labels by mapping them to -1 or a specific unknown category index
                st.warning(f"Unseen category '{input_df[col].iloc[0]}' in '{col}'. Using a default/unknown value.")
                input_df[col] = -1 # Or a more appropriate fallback value

    return input_df


# --- Make Prediction ---
if st.button('Predict Salary'):
    processed_input = preprocess_input(
        rating, company_name_input, job_title_input,
        salaries_reported, location_input, employment_status_input,
        job_roles_input, label_encoders
    )

    # Ensure the order of columns matches X_train
    # This is crucial! The model expects features in the exact order it was trained on.
    # Assuming the training features were: 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'
    # The order of columns in X_train was: 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'
    expected_columns = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
    processed_input = processed_input[expected_columns]

    prediction = model.predict(processed_input)
    st.success(f'Predicted Salary: ₹{prediction[0]:,.2f}')

# --- Instructions/Footer ---
st.markdown("""
---
*This app uses a Random Forest Regressor model trained on the provided salary dataset.*
""")
