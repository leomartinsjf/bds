#ok 27/06

import streamlit as st
st.set_page_config(page_title="BDs: ambiente integrado de análise de dados", layout="wide")

import pandas as pd
import numpy as np
import io
import zipfile
from typing import List

from data_cleaning import show_preprocessing_interface
from feature_engineering import show_feature_engineering
from exploratory_analysis import show_exploratory_analysis
from model_training import (
    show_linear_regression_model,
    show_logistic_regression_model,
    show_path_analysis_model,
)
from model_classification_regression import show_machine_learning_page
from bayesian_analysis import show_bayesian_analysis_page
from model_multilevel import show_multilevel_model_extended
from model_multilevel_lvl3 import show_multilevel_model_lvl3_full
from model_multilevel_cross_classified import show_multilevel_model_cross
from model_l4_extended import show_l4_model
from multilevel_models import show_multilevel_tabs

# --- Funções Auxiliares ---

def limpar_estado_l4():
    chaves_l4 = [
        'selected_trocas', 'selected_subjetividades',
        'selected_relacoes', 'selected_estrutura', 'l4_score_method',
        'l4_scores_calculated', 'l4x_group_test', 'l4x_test_type_radio',
        'l4x_model_y', 'l4x_n_clusters', 'l4x_cluster_method_radio',
        'l4x_cv_y', 'l4x_cv_n_folds'
    ]
    for chave in chaves_l4:
        st.session_state.pop(chave, None)

    # Remove colunas L4 do df_l4 (se existir)
    # df_processed será limpo diretamente na show_l4_page para garantir
    # que a cópia passada para show_l4_model seja sempre limpa.
    if "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df_base = st.session_state["df_processed"].copy()
        colunas_l4 = ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", "Cluster_L4"]
        df_base = df_base.drop(columns=[c for c in colunas_l4 if c in df_base.columns], errors="ignore")
        st.session_state["df_l4"] = df_base  # redefine df_l4 a partir de df_processado limpo
    else:
        st.session_state["df_l4"] = None


def load_data(uploaded_file) -> pd.DataFrame | None:
    """
    Carrega CSV ou Excel a partir do arquivo enviado e retorna um DataFrame.
    Em caso de erro, exibe mensagem de erro e retorna None.
    """
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar arquivo: {e}")
        return None


def export_buttons(df: pd.DataFrame) -> None:
    """
    Exibe o DataFrame final, os logs de pré-processamento e feature engineering
    (cada um em seu expander), e gera botões de download para CSV, Excel e ZIP.
    """
    # 1) Validação
    if df is None or df.empty:
        st.info("Não há dados para exportar. Execute o pré-processamento primeiro.")
        return

    # 2) Visualização do DataFrame final
    st.subheader("📊 DataFrame Final")
    st.dataframe(df)

    # 3) Expanders de históricos
        # … dentro de export_buttons …

    # 3) Expanders de históricos
    logs_pre = st.session_state.get("preprocessing_log", [])
    all_fe_logs = st.session_state.get("feature_engineering_logs", [])
    # Filtra só as entradas de feature engineering, excluindo as de session_state
    logs_fe = [log for log in all_fe_logs if "Atualizado session_state" not in log]

    with st.expander("📝 Histórico de Pré-processamento"):
        if logs_pre:
            for entry in logs_pre:
                st.write(entry)
        else:
            st.info("Nenhuma operação de pré-processamento registrada.")

    with st.expander("📝 Histórico de Feature Engineering"):
        if logs_fe:
            for entry in logs_fe:
                st.write(entry)
        else:
            st.info("Nenhuma operação de feature engineering registrada.")

    # 4) Construção do ZIP em memória
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        zf.writestr("dados_processados.csv", csv_bytes)

        # Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Dados")
        zf.writestr("dados_processados.xlsx", excel_buffer.getvalue())

        # Logs
        zf.writestr("preprocessing_log.txt", "\n".join(logs_pre))
        zf.writestr("feature_engineering_log.txt", "\n".join(logs_fe))

    zip_buffer.seek(0)

    # 5) Botões de download
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.download_button(
            label="📥 Baixar CSV",
            data=csv_bytes,
            file_name="dados_processados.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="📥 Baixar Excel",
            data=excel_buffer.getvalue(),
            file_name="dados_processados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col3:
        st.download_button(
            label="📥 Baixar tudo (.zip)",
            data=zip_buffer.getvalue(),
            file_name="exportacao_bds.zip",
            mime="application/zip"
        )
# 🔧 Função de reset dos campos interativos de engenharia

def reset_feature_engineering_keys():
    """Zera estados temporários de engenharia de variáveis."""
    substrings = [
        "transform_to_cat_col_select", "new_categorical_col_name_input", "map_",
        "remove_original_col_checkbox_final", "selected_cols_for_ops_multiselect",
        "select_all_for_ops_checkbox", "apply_categorical_transform_button",
        "rename_", "duplicate_",
    ]
    for key in list(st.session_state.keys()):
        if any(sub in key for sub in substrings):
            del st.session_state[key]
    for flag in ("run_feature_engineering_rerun", "feature_engineered_flag"):
        st.session_state.pop(flag, None)

# --- Inicialização do session_state ---

def _init_state():
    defaults = {
        "df_original": None,
        "df_processed": None,
        "last_uploaded_file_name": None,
        "df_loaded_for_processing": False,
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

_init_state()

# --- Barra Lateral: Upload de Dados ---

st.sidebar.header("📁 Upload de Dados")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV ou Excel", type=["csv", "xlsx"]
)
if uploaded_file:
    if (
        uploaded_file.name != st.session_state["last_uploaded_file_name"]
        or not st.session_state["df_loaded_for_processing"]
    ):
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state["df_original"] = df.copy()
            st.session_state["df_processed"] = df.copy()
            st.session_state["last_uploaded_file_name"] = uploaded_file.name
            st.session_state["df_loaded_for_processing"] = True
            reset_feature_engineering_keys()
            st.sidebar.success("Arquivo carregado com sucesso!")
    else:
        st.sidebar.info("Arquivo já carregado.")
else:
    if not st.session_state["df_loaded_for_processing"]:
        st.sidebar.info("Aguardando upload de arquivo.")

# --- Definições das Páginas ---

def show_home_page():
    #st.header("Início")
    df = st.session_state["df_original"]
    if df is not None:
        st.subheader("Visão Geral do Dataset")
        st.write("**Arquivo:**", st.session_state["last_uploaded_file_name"])
        rows, cols = df.shape
        st.write(f"Linhas: {rows}, Colunas: {cols}")
        show_all = st.checkbox("Mostrar todas as linhas do DataFrame", key="show_all_rows")
        st.dataframe(df if show_all else df.head())
    else:
        st.info("Por favor, faça o upload de um arquivo para começar.")

def show_preprocessing_page():
    st.header("🧹 Pré-processamento de Dados")
    if st.session_state.get("df_loaded_for_processing"):
        result = show_preprocessing_interface()
        if isinstance(result, pd.DataFrame):
            st.session_state["df_processed"] = result
        elif isinstance(result, tuple) and isinstance(result[0], pd.DataFrame):
            st.session_state["df_processed"] = result[0]
    else:
        st.warning("Nenhum dado carregado. Faça upload antes de pré-processar.")

def show_feature_engineering_page():
    st.header("🧪 Engenharia de Variáveis")
    if st.session_state.get("df_loaded_for_processing"):
        df_new = show_feature_engineering()
        if isinstance(df_new, pd.DataFrame):
            st.session_state["df_processed"] = df_new
    else:
        st.warning("Carregue e processe dados antes de engenharia.")

def show_exploratory_page():
    #st.header("📊 Análise Exploratória")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        show_exploratory_analysis()
    else:
        st.warning("Processar dados primeiro.")

def show_statistical_modeling_page():
    st.header("📈 Modelagem Estatística")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        st.subheader("Regressão Linear")
        show_linear_regression_model()
        st.subheader("Regressão Logística")
        show_logistic_regression_model()
        st.subheader("Path Analysis")
        show_path_analysis_model()
    else:
        st.warning("Processar dados primeiro.")

def show_ml_page():
    st.header("🤖 Machine Learning")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        show_machine_learning_page()
    else:
        st.warning("Processar dados primeiro.")

def show_bayesian_page():
    #st.header("🔬 Análise Bayesiana")
    if isinstance(st.session_state.get("df_processed"), pd.DataFrame):
        df_for_bayes = st.session_state.get("df_l4", st.session_state["df_processed"])
        show_bayesian_analysis_page(df_for_bayes)
    else:
        st.warning("Processar dados primeiro.")

def show_multilevel_2_3_page():
    show_multilevel_tabs()

def show_cross_classified_page():
    #st.header("🔀 Multinível Não Hierárquico")
    show_multilevel_model_cross()

def show_l4_page():
    # Clean L4 specific session state variables and df_l4
    limpar_estado_l4()

    if "df_processed" not in st.session_state or st.session_state["df_processed"] is None:
        st.warning("⚠️ Os dados ainda não foram carregados ou processados.")
        return

    # Garante que df_main_for_l4 não contenha as colunas L4 de execuções anteriores,
    # antes de passá-lo para show_l4_model.
    df_main_for_l4 = st.session_state["df_processed"].copy()
    l4_score_cols_to_clean = ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", "Cluster_L4"]
    for col in l4_score_cols_to_clean:
        if col in df_main_for_l4.columns:
            df_main_for_l4 = df_main_for_l4.drop(columns=[col])

    # Pass the cleaned df_main_for_l4 to the model
    show_l4_model(df_main_for_l4)

def show_export_page():
    st.header("📤 Exportar Dados")
    export_buttons(st.session_state.get("df_processed"))

# --- Estrutura de Navegação Dinâmica ---
PAGES = {
    "🏠 Início": show_home_page,
    "🧹 Pré-processamento de Dados": show_preprocessing_page,
    "🧪 Engenharia de Variáveis": show_feature_engineering_page,
    "📊 Análise Exploratória": show_exploratory_page,
    "📈 Modelagem Estatística": show_statistical_modeling_page,
    "🤖 Machine Learning": show_ml_page,
    "🔬 Análise Bayesiana": show_bayesian_page,
    "📚 Análise Multinível Níveis 2 e 3": show_multilevel_2_3_page,
    "🔀 Multinível Não Hierárquico": show_cross_classified_page,
    "🔷 Modelo L4": show_l4_page,
    "📤 Exportar": show_export_page,
}

st.sidebar.header("🧭 Navegação Principal")
selection = st.sidebar.radio("Selecione uma seção:", list(PAGES.keys()))

# --- Título e Chamada da Página Selecionada ---
st.title("📊 BDs: ambiente integrado de análise de dados")
st.write("##### Marcos Emanoel Pereira (UFBa/UFS) & Marcus Eugênio O. Lima (UFS)")
PAGES[selection]()