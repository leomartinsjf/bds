# mep3.py

import streamlit as st
st.set_page_config(page_title="Aplicativo de Análises BDS", layout="wide")

import pandas as pd
import numpy as np
import io

from data_cleaning import show_preprocessing_interface
from feature_engineering import show_feature_engineering
from exploratory_analysis import show_exploratory_analysis
from model_training import (
    show_linear_regression_model,
    show_logistic_regression_model,
    show_path_analysis_model,
    show_multilevel_model
)
from model_classification_regression import show_machine_learning_page
from bayesian_analysis import show_bayesian_analysis_page
from model_multilevel import show_multilevel_model_extended

from model_multilevel_lvl3 import show_multilevel_model_lvl3_full
from model_multilevel_cross_classified import show_multilevel_model_cross
from model_l4_extended import show_l4_model 



# --- Inicialização do session_state ---
default_state = {
    "df_original": None,
    "df_processed": None,
    "last_uploaded_file_name": None,
    "df_loaded_for_processing": False,
}
for key, value in default_state.items():
    st.session_state.setdefault(key, value)

# --- Barra Lateral: Upload de Dados ---
st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV ou Excel", type=["csv", "xlsx"]
)
if uploaded_file:
    if (
        st.session_state["last_uploaded_file_name"] != uploaded_file.name
        or not st.session_state["df_loaded_for_processing"]
    ):
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state["df_original"] = df.copy()
            st.session_state["df_processed"] = df.copy()
            st.session_state["last_uploaded_file_name"] = uploaded_file.name
            st.session_state["df_loaded_for_processing"] = True
            st.sidebar.success("Arquivo carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar arquivo: {e}")
            st.session_state["df_loaded_for_processing"] = False
    else:
        st.sidebar.info("Arquivo já carregado.")
else:
    if not st.session_state["df_loaded_for_processing"]:
        st.sidebar.info("Aguardando upload de arquivo.")

# --- Barra Lateral: Navegação Principal ---
st.sidebar.header("🧭 Navegação Principal")
page = st.sidebar.radio(
    "Selecione uma seção:",
    [
        "🏠 Início",
        "🧹 Pré-processamento de Dados",
        "🧪Engenharia de Variáveis",
        "📊 Análise Exploratória",
        "📈 Modelagem Estatística",
        "🤖 Machine Learning",
        "🔬 Análise Bayesiana",
        "📚 Multinível Nível 3",
        "🔀 Multinível Não Hierárquico",
        "🔷 Modelo L4",
        "📤 Exportar"

    ]
)

# --- Cabeçalho Principal ---
st.title("📊 Aplicativo de Análises BDS")

# --- Lógica de Navegação ---
if page == "🏠 Início":
    st.header("Início")
    if st.session_state["df_original"] is not None:
        st.subheader("Visão Geral do Dataset")
        st.write("**Arquivo:**", st.session_state["last_uploaded_file_name"])
        st.write(
            f"Linhas: {st.session_state['df_original'].shape[0]}, Colunas: {st.session_state['df_original'].shape[1]}"
        )
        st.dataframe(st.session_state["df_original"].head())
    else:
        st.info("Por favor, faça o upload de um arquivo para começar.")

elif page == "🧹 Pré-processamento de Dados":
    st.header("🧹 Pré-processamento de Dados")
    if st.session_state.get("df_loaded_for_processing"):
        result = show_preprocessing_interface()
        if isinstance(result, pd.DataFrame):
            st.session_state["df_processed"] = result
        elif isinstance(result, tuple) and isinstance(result[0], pd.DataFrame):
            st.session_state["df_processed"] = result[0]
    else:
        st.warning("Nenhum dado carregado. Por favor, faça upload de um arquivo válido antes de iniciar o pré-processamento.")

elif page == "🧪 Engenharia de Variáveis":
    st.header("🧪 Engenharia de Variáveis")
    if st.session_state.get("df_loaded_for_processing"):
        df_new = show_feature_engineering()
        if isinstance(df_new, pd.DataFrame):
            st.session_state["df_processed"] = df_new
    else:
        st.warning("Nenhum dado processado disponível para engenharia de variáveis. Por favor, carregue e processe os dados primeiro.")

elif page == "📊 Análise Exploratória":
    st.header("📊 Análise Exploratória")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        show_exploratory_analysis()
    else:
        st.warning("Nenhum dado processado disponível para análise exploratória. Por favor, carregue e processe os dados primeiro.")

elif page == "📈 Modelagem Estatística":
    st.header("📈 Modelagem Estatística")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        st.subheader("Modelos de Regressão Linear")
        show_linear_regression_model()
        st.subheader("Modelos de Regressão Logística")
        show_logistic_regression_model()
        st.subheader("Path Analysis")
        show_path_analysis_model()
        st.subheader("Modelos Multinível")
        show_multilevel_model_extended()
    else:
        st.warning("Nenhum dado processado disponível para modelagem estatística. Por favor, carregue e processe os dados primeiro.")

elif page == "🤖 Machine Learning":
    st.header("🤖 Machine Learning")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        show_machine_learning_page()
    else:
        st.warning("Nenhum dado processado disponível para machine learning. Por favor, carregue e processe os dados primeiro.")

elif page == "🔬 Análise Bayesiana":
    st.header("🔬 Análise Bayesiana")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        show_bayesian_analysis_page(st.session_state["df_processed"])
    else:
        st.warning("Por favor, carregue e processe os dados antes de acessar esta seção.")


elif page == "📚 Multinível Nível 3":
    show_multilevel_model_lvl3_full()

elif page == "🔀 Multinível Não Hierárquico":
    show_multilevel_model_cross()

elif page == "🔷 Modelo L4":
    show_l4_model()

elif page == "📤 Exportar":
    st.header("📤 Exportar Dados")
    df_exp = st.session_state.get("df_processed")
    if isinstance(df_exp, pd.DataFrame) and not df_exp.empty:
        st.markdown("### Download dos dados processados")
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar CSV",
            csv,
            file_name="dados_processados.csv",
            mime="text/csv",
        )
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="Dados")
        st.download_button(
            "📥 Baixar Excel",
            towrite.getvalue(),
            file_name="dados_processados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Não há dados para exportar. Execute o pré-processamento primeiro.")
