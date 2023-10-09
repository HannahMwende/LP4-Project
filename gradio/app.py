import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Define prediction function
def make_prediction(
    gender='Female', SeniorCitizen='No', Partner='No', Dependents='No', tenure=0, PhoneService='Yes',
    MultipleLines=None, InternetService='DSL', OnlineSecurity=None, OnlineBackup=None,
    DeviceProtection=None, TechSupport=None, StreamingTV=None, StreamingMovies=None,
    Contract='One year', PaperlessBilling='Yes', PaymentMethod='Electronic check',
    MonthlyCharges=10, TotalCharges=None):

    # Check if any input values are None (indicating user did not make a selection) and set default values
    
    if MultipleLines is None:
        MultipleLines = 'No'  
    if OnlineSecurity is None:
        OnlineSecurity = 'No'  
    if OnlineBackup is None:
        OnlineBackup = 'No'  
    if DeviceProtection is None:
        DeviceProtection = 'No'  
    if TechSupport is None:
        TechSupport = 'No'  
    if StreamingTV is None:
        StreamingTV = 'No'  
    if StreamingMovies is None:
        StreamingMovies = 'No'  
    if TotalCharges is None:
        TotalCharges = 18 

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
    return 'Customer Will Not Churn'


# Create the input components for Gradio
gender_input = gr.Dropdown(choices=['Female', 'Male'])
SeniorCitizen_input = gr.Dropdown(choices=['Yes', 'No'])
Partner_input = gr.Dropdown(choices=['Yes', 'No'])
Dependents_input = gr.Dropdown(choices=['Yes', 'No'])
tenure_input = gr.Number()
PhoneService_input = gr.Dropdown(choices=['Yes', 'No'])
MultipleLines_input = gr.Dropdown(choices=['No phone service', 'No', 'Yes'])
InternetService_input = gr.Dropdown(choices=['DSL', 'Fiber optic', 'No'])
OnlineSecurity_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'])
OnlineBackup_input = gr.Dropdown(choices=['Yes', 'No', 'No internet service'])
DeviceProtection_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'])
TechSupport_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'])
StreamingTV_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'])
StreamingMovies_input = gr.Dropdown(choices=['No', 'Yes', 'No internet service'])
Contract_input = gr.Dropdown(choices=['Month-to-month', 'One year', 'Two year'])
PaperlessBilling_input = gr.Dropdown(choices=['Yes', 'No'])
PaymentMethod_input = gr.Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges_input = gr.Number()
TotalCharges_input = gr.Number()

output = gr.Textbox(label='Prediction')

# Set the title of the Gradio app
app = gr.Interface(fn=make_prediction, inputs=[
    gender_input, SeniorCitizen_input, Partner_input, Dependents_input, tenure_input,
    PhoneService_input, MultipleLines_input, InternetService_input, OnlineSecurity_input,
    OnlineBackup_input, DeviceProtection_input, TechSupport_input, StreamingTV_input,
    StreamingMovies_input, Contract_input, PaperlessBilling_input, PaymentMethod_input,
    MonthlyCharges_input, TotalCharges_input], outputs=output, title='Customer Churn Prediction App')

app.launch(share=True, debug=True)
