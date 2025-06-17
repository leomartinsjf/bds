import streamlit as st
import pandas as pd

def load_data():
    uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            reset_feature_engineering_keys()
            st.success("Dados carregados com sucesso!")
            return df
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return None
    return None
