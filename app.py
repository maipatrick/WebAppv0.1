import streamlit as st

st.title("Premium File Processing App")

st.write("Upload your files for analysis!")

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"File {uploaded_file.name} uploaded successfully!")
    
    if st.button("OK"):
        st.write("Files have been processed successfully!")


# streamlit run app.py
