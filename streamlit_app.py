import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize df as None
df = None

st.sidebar.title("Sales Forecasting Favorita Stores")
selected_option = st.sidebar.radio("Select to Proceed", ["Data Statistics", "Visuals", "Time Series Analysis", "Forecasting"])

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



