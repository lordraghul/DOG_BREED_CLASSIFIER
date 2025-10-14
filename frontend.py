import streamlit as st
import requests

st.title("Classification d'Images de Chiens")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file)
    with st.spinner("Prédiction en cours..."):
        response = requests.post(
            "http://localhost:8000/predict", 
            files={"file": uploaded_file.getvalue()}
        )
        if response.ok:
            pred = response.json()["prediction"]
            st.success(f"La race prédite est : {pred}")
        else:
            st.error("Erreur lors de la prédiction.")

