#%%
# 1. Importing Libraries and Backend Logic

import pandas as pd
import numpy as np
import re
from ipywidgets import VBox, HBox, Button, Label, IntSlider, FloatSlider, Dropdown, Output, Layout
from IPython.display import display, HTML, clear_output

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Data Loading and Preprocessing ---
def load_and_prepare_data():
    """Loads and preprocesses the Telco Customer Churn dataset."""
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Use direct assignment to avoid FutureWarning
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)
    return df

# --- Model Training ---
def train_model(df):
    """Trains a RandomForestClassifier model and returns the fitted pipeline."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X, y)
    return model_pipeline

# Run the data preparation and model training once
df = load_and_prepare_data()
model = train_model(df)
print("Model trained and ready for interactive prediction.")

# --- Define option lists for widgets to avoid NameError ---
contract_options = df['Contract'].unique().tolist()
internet_options = df['InternetService'].unique().tolist()
gender_options = df['gender'].unique().tolist()
#%%
# 2. Interactive Frontend with Form and Prediction Logic

# Create the form widgets
tenure_slider = IntSlider(
    min=0, max=72, step=1, value=12,
    description='Tenure (months):',
    layout=Layout(width='90%')
)
monthly_charges_slider = FloatSlider(
    min=18.0, max=118.0, step=0.05, value=50.0,
    description='Monthly Charges ($):',
    layout=Layout(width='90%')
)
contract_dropdown = Dropdown(
    options=contract_options,
    value='Month-to-month',
    description='Contract:',
    layout=Layout(width='90%')
)
internet_service_dropdown = Dropdown(
    options=internet_options,
    value='Fiber optic',
    description='Internet Service:',
    layout=Layout(width='90%')
)
gender_dropdown = Dropdown(
    options=gender_options,
    value='Male',
    description='Gender:',
    layout=Layout(width='90%')
)

# Create the prediction button
predict_button = Button(
    description="Predict Churn",
    button_style='success',
    layout=Layout(width='90%', margin='20px 0 0 0')
)

# Output widget to display the prediction result
result_output = Output()

# The function that runs when the button is clicked
def on_button_clicked(b):
    with result_output:
        clear_output(wait=True)

        tenure = tenure_slider.value
        monthly_charges = monthly_charges_slider.value
        contract = contract_dropdown.value
        internet_service = internet_service_dropdown.value
        gender = gender_dropdown.value

        # Create a DataFrame for the prediction
        input_data = {
            'gender': [gender], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'],
            'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'],
            'InternetService': [internet_service],
            'OnlineSecurity': ['No internet service' if internet_service == 'No' else 'No'],
            'OnlineBackup': ['No internet service' if internet_service == 'No' else 'No'],
            'DeviceProtection': ['No internet service' if internet_service == 'No' else 'No'],
            'TechSupport': ['No internet service' if internet_service == 'No' else 'No'],
            'StreamingTV': ['No internet service' if internet_service == 'No' else 'No'],
            'StreamingMovies': ['No internet service' if internet_service == 'No' else 'No'],
            'Contract': [contract],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [tenure * monthly_charges]
        }
        input_df = pd.DataFrame(input_data)

        prediction_proba = model.predict_proba(input_df)[0]
        churn_probability = prediction_proba[1]

        if churn_probability > 0.5:
            color = "#dc2626" # red-600
            prediction_text = "High Churn Risk"
        else:
            color = "#16a34a" # green-600
            prediction_text = "Low Churn Risk"

        html_output = HTML(f"""
            <div style="border: 2px solid {color}; border-radius: 8px; padding: 16px; margin-top: 16px; background-color: #f9f9f9;">
                <h3 style="font-weight: bold; font-size: 1.25rem; color: {color};">Prediction: {prediction_text}</h3>
                <p style="margin-top: 8px; font-size: 1rem;">
                    Probability of Churn: <strong>{churn_probability:.2f}</strong>
                </p>
            </div>
        """)
        display(html_output)

# Link the button click to the prediction function
predict_button.on_click(on_button_clicked)

# Combine all widgets into a vertical box and display them
form_widgets = VBox([
    Label("AI Customer Churn Prediction Dashboard", layout=Layout(font_weight='bold', font_size='24px', margin='0 0 10px 0')),
    HBox([VBox([tenure_slider, monthly_charges_slider]), VBox([contract_dropdown, internet_service_dropdown, gender_dropdown])]),
    HBox([predict_button], layout=Layout(justify_content='center')),
    result_output
])

display(form_widgets)
#%%
