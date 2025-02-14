import streamlit as st

# Display the logo
st.image("logo.png", width=200)

st.title("Premium File Processing App")

st.write("Upload your files for analysis!")

# File uploader for Excel files
uploaded_excel = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# File uploader for video files
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_excel:
    st.write(f"Excel file {uploaded_excel.name} uploaded successfully!")

if uploaded_video:
    st.write(f"Video file {uploaded_video.name} uploaded successfully!")
    st.video(uploaded_video)

# Enable the "OK" button only if both files are uploaded
if uploaded_excel and uploaded_video:
    if st.button("OK"):
        st.write("Files have been processed successfully!")