import streamlit as st
import pandas as pd

st.title("Streamlit App - Load Data from User")

# Allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("### Uploaded CSV file:")
    st.write(uploaded_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("### DataFrame from CSV:")
    st.write(df)
