"""Streamlit prototype: upload a leaf photo, get the fused prediction and an advise / refer decision.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st
from PIL import Image

from inference import Ensemble

st.set_page_config(page_title="Tea leaf disease check", layout="centered")
st.title("Tea leaf disease check")
st.caption("CNN + DenseNet121 + VGG16, accuracy-weighted soft voting. Trained on teaLeafBD only; "
           "on photos from other sources accuracy drops sharply, so treat the result as a hint.")


@st.cache_resource
def get_model():
    return Ensemble("outputs")


threshold = st.sidebar.slider("Confidence threshold", 0.5, 0.99, 0.8, 0.01)
agree = st.sidebar.checkbox("Require all three models to agree", value=True)
file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, width=320)
    r = get_model().predict(img, threshold, agree)
    if r["decision"] == "advise":
        st.success(f"{r['class']} (confidence {r['confidence']:.2f})")
    else:
        st.warning(f"Not sure: best guess {r['class']} (confidence {r['confidence']:.2f}). Please ask an expert.")
    st.write("Each model:", r["members"])
    st.bar_chart(pd.Series(r["probabilities"], name="probability"))
