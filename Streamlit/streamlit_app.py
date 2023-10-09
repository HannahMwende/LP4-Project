import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Machine Learning Modeling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

# Set the page layout to full width
st.set_page_config(layout="wide")
# Initialize df as None
df = None

st.sidebar.title("Favorita Stores")
selected_option = st.sidebar.radio("Select to Proceed", ["Data Statistics", "Visuals", "Time Series Analysis", "Forecasting"])

# Custom CSS styling for the title
st.markdown(
    """
    <style>
    .title-text {
        font-size: 28px;
        text-align: center;
        background-color: #3498db;
        color: white;
        padding: 10px 0;
        width: 100%;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title
st.markdown('<p class="title-text">Machine Learning App for Sales Prediction</p>', unsafe_allow_html=True)

# Function to load and process the data
def load_and_process_data():
    global df
    # Allow the user to upload an Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        # Check if the file is an Excel file
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Read the Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)
            # Remove null values
            df.dropna(inplace=True)
            df = df.drop(columns='Unnamed: 0')
        else:
            st.write("Please upload a valid Excel file.")

# Load and process the data
load_and_process_data()

if selected_option == "Data Statistics":
    # Rest of the code for "Data Statistics" option using df
    if df is not None:
        number_sample = st.number_input("Enter sample size to display data", min_value=5, max_value=10, step=1, value=5)
        displayed_data = df.head(number_sample)
        st.write("Sample data", displayed_data)
        st.write("Summary Statistics of float/Integer columns", df.describe())
        object_columns = df.select_dtypes(include='object').columns.tolist()
        selected_column = st.selectbox("Select column of Data Type Object to View Unique values", object_columns)
        if selected_column:
            unique_values = df[selected_column].unique()
            st.write("Unique values are", unique_values)

elif selected_option == "Visuals":
    # Rest of the code for "Visuals" option using df
    if df is not None:
        object_columns = df.select_dtypes(include='object').columns.tolist()
        selected_column = st.selectbox("Select column of Data Type Object for Visualization", object_columns)
        if selected_column:
            df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if applicable
            df_grouped = df.groupby(selected_column)['sales'].sum().head(10)
            df_grouped = df_grouped.sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.bar(df_grouped.index, df_grouped.values)
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Sales Count')
            ax.set_title(f'Top 10 Sales Count for {selected_column}')
            st.pyplot(fig)  # Pass the figure to st.pyplot()
elif selected_option == "Time Series Analysis":
    if df is not None:
        # Choose date and sales columns
        timeseriesdata = df[['sales', 'date']]
        timeseriesdata.index = timeseriesdata['date']
        # Make date the index
        timeseriesdata = timeseriesdata.resample('D').sum()  # Resample to daily sales

        # Resample the data based on user's choice
        resample_method = st.selectbox("Select a resampling method", ['M', 'Q', 'Y'])
        if resample_method:
            resampled_data = timeseriesdata.resample(resample_method).sum()

            # Plot the time series using Seaborn lineplot
            plt.figure(figsize=(15, 6))
            sns.lineplot(data=resampled_data)
            plt.ylabel('Sales')
            plt.title(f'Sales Time Series (Resampled by {resample_method})')
            st.pyplot(plt.gcf())
else:
    st.write("Please enter these inputs to predict sales. Thank you!")
    # Load the pre-trained model and preprocessor
    model = joblib.load('./xgb_model.joblib')
    preprocessor = joblib.load('./preprocessor.joblib') 

    

    # Create a layout with 2 columns for even distribution
    col1, col2 = st.columns(2)  

    # User Inputs - Number
    with col1:
        # Create a date input using st.date_input
        date = st.date_input("Enter Date")      

        # Convert the selected date to a string in the desired format (e.g., YYYY-MM-DD)
        formatted_date = date.strftime('%Y-%m-%d')      

    # User Inputs - Year
    with col2:
        family = st.selectbox("Select product family", ['CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS',
           'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I',
           'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE',
           'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER',
           'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES',
           'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS', 'PRODUCE',
           'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD', 'AUTOMOTIVE', 'BABY CARE',
           'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY']) 

    # User Inputs - On Promotion
    with col1:
        onpromotion = st.number_input("Enter Number for onpromotion", min_value=0, step=1)  


    # User Inputs - Day of the Week
    with col2:
        city = st.selectbox("Select city", ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
           'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
           'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
           'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen'])    

    # User Inputs - Product Category
    with col1:
        oil_prices = st.number_input("Enter oil price", min_value=1, step=1)    
 

    # User Inputs - Day of the Week
    with col2:
        holiday_type = st.selectbox("Select holiday type", ['Holiday', 'Additional', 'Transfer', 'Event', 'Bridge'])    

    # User Inputs - Product Category
    with col1:
        sales_lag_1 = st.number_input("Enter Number for sales lag", min_value=0, step=1)    


    # User Inputs - Day of the Week
    with col2:
        moving_average = st.number_input("Enter Number for moving average", min_value=0, step=1)    

    # Placeholder for Predicted Value   

    # Add custom spacing between columns
    st.markdown("<hr>", unsafe_allow_html=True) 



    # Predict Button
    if st.button("Predict"):
        # Prepare input data for prediction
        # Prepare input data for prediction
        # Create a DataFrame with all required columns except "sales"
        prediction_placeholder = st.empty()
        input_df = pd.DataFrame({
            "family": [family],
            "onpromotion": [onpromotion],
            "city": [city],
            "oil_prices": [oil_prices],
            "holiday_type": [holiday_type],
            "sales_lag_1": [sales_lag_1],
            "moving_average": [moving_average]
        })

        # Transform the input DataFrame using the preprocessor
        preprocessed_data = preprocessor.transform(input_df)



        # Make a prediction
        prediction = model.predict(preprocessed_data)   

         
        # Display the prediction
        prediction_placeholder.text(f"Predicted Value for sales: {prediction[0]: ,.2f}")  

        if prediction >= 0:
            prediction_placeholder.markdown(
            f'Predicted Value for sales: <span style="background-color: green; padding: 2px 5px; border-radius: 5px;">${prediction[0]:,.2f}</span>',
            unsafe_allow_html=True
        )
        else:
            prediction_placeholder.markdown(
            f'Predicted Value for sales: <span style="background-color: red; padding: 2px 5px; border-radius: 5px;">${prediction[0]:,.2f}</span>',
            unsafe_allow_html=True
        )
