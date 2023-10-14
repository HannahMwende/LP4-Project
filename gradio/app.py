import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Define prediction function
def make_prediction(
    gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges):


    # Make a dataframe from input data
    input_data = pd.DataFrame({
        'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
        'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
        'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })

    # Load already saved pipeline and make predictions
    with open("preprocessor.joblib", "rb") as p:
        preprocessor = joblib.load(p)
        input_data = preprocessor.transform(input_data)
        # You may need to update this part to drop the relevant columns for your model
        input_data = input_data.drop(['encode__PaperlessBilling_No', 'encode__MultipleLines_No', 'encode__InternetService_Fiber optic', 'encode__StreamingMovies_No internet service', 'encode__InternetService_No', 'encode__OnlineBackup_No internet service', 'encode__StreamingTV_No internet service'], axis=1)

    # Load already saved pipeline and make predictions
    with open("rf_model.joblib", "rb") as f:
        model = joblib.load(f)
        predt = model.predict(input_data)

    
    # Return prediction
    if np.any(predt == 1):
        return 'Customer Will Churn'
    else:
        return 'Customer Will Not Churn'


# Create the input components for Gradio with labels
gender_input = gr.Dropdown(choices=['Female', 'Male'], label='Select gender')
SeniorCitizen_input = gr.Dropdown(choices=['Yes', 'No'], label='Is the customer a senior citizen?')
Partner_input = gr.Dropdown(choices=['Yes', 'No'], label='Has the customer a partner?')
Dependents_input = gr.Dropdown(choices=['Yes', 'No'], label='Does the customer have dependents?')
tenure_input = gr.Number(label='Number of months the customer has stayed with the company')
PhoneService_input = gr.Dropdown(choices=['Yes', 'No'], label='Does the customer have phone service?')
MultipleLines_input = gr.Dropdown(choices=['No phone service', 'No', 'Yes'], label='Does the customer have multiple phone lines?')
InternetService_input = gr.Dropdown(choices=['DSL', 'Fiber optic', 'No'], label='Type of Internet service')
OnlineSecurity_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label='Does the customer have online security?')
OnlineBackup_input = gr.Dropdown(choices=['Yes', 'No', 'No internet service'], label='Does the customer have online backup?')
DeviceProtection_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label='Does the customer have device protection?')
TechSupport_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label='Does the customer have tech support?')
StreamingTV_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label='Does the customer have streaming TV?')
StreamingMovies_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'], label='Does the customer have streaming movies?')
Contract_input = gr.Dropdown(choices=['Month-to-month', 'One year', 'Two year'], label='Type of contract')
PaperlessBilling_input = gr.Dropdown(choices=['Yes', 'No'], label='Is the customer using paperless billing?')
PaymentMethod_input = gr.Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='Payment method')
MonthlyCharges_input = gr.Number(label='Monthly charges')
TotalCharges_input = gr.Number(label='Total charges')

output = gr.Textbox(label='Prediction')

# Set the title of the Gradio app
app = gr.Interface(fn=make_prediction, inputs=[
    gender_input, SeniorCitizen_input, Partner_input, Dependents_input, tenure_input,
    PhoneService_input, MultipleLines_input, InternetService_input, OnlineSecurity_input,
    OnlineBackup_input, DeviceProtection_input, TechSupport_input, StreamingTV_input,
    StreamingMovies_input, Contract_input, PaperlessBilling_input, PaymentMethod_input,
    MonthlyCharges_input, TotalCharges_input], outputs=output, title='Customer Churn Prediction App')

app.launch(share=True, debug=True)
