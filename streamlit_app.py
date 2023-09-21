import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Streamlit App - Load, Clean Data, and Create Dynamic Plots")

# Allow the user to upload an Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

# Process the uploaded Excel file
if uploaded_file is not None:
    # Check if the file is an Excel file
    if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        # Read the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Remove null values
        df.dropna(inplace=True)
        df=df.drop(columns='Unnamed: 0')
        df=df.sample(5)
        
        # Display the cleaned DataFrame
        st.write("### Cleaned DataFrame:")
        st.write(df)

        # Allow user to select the type of plot
        plot_type = st.selectbox("Select a plot type", ["Scatter Plot", "Bar Chart"])

        if plot_type == "Scatter Plot":
            # Allow user to select the X and Y columns for the scatter plot
            x_column = st.selectbox("Select X-axis column", df.columns)
            y_column = st.selectbox("Select Y-axis column", df.columns)
            st.write("### Scatter Plot:")
            plt.scatter(df[x_column], df[y_column])
            st.pyplot()

        elif plot_type == "Bar Chart":
            # Allow user to select the column for the bar chart
            bar_column = st.selectbox("Select a column for the bar chart", df.columns)
            st.write("### Bar Chart:")
            sns.countplot(x=bar_column, data=df)
            st.pyplot()
    else:
        st.write("Please upload a valid Excel file.")
