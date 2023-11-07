import easyocr as ocr  #OCR
import streamlit as st


@st.cache_resource
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 

