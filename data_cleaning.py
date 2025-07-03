#ok 27/06
import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, List, Tuple
import datetime

# --- Logging de Pré-processamento ---
def init_preprocessing_log():
    """Inicializa histórico de transformações no session_state."""
    st.session_state.setdefault("preprocessing_log", [])


def log_preprocessing_step(step: str):
    """Adiciona uma entrada ao histórico com timestamp."""
    init_preprocessing_log()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["preprocessing_log"].append(f"{timestamp} - {step}")


def show_preprocessing_log():
    """Retorna lista de transformações realizadas."""
    init_preprocessing_log()
    return st.session_state.get("preprocessing_log", [])

# --- Auxiliar: Cálculo de limites IQR ---
def _calculate_iqr_bounds(series: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper

# --- Detecção e Tratamento de Outliers ---
def detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.Index:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return pd.Index([])
    series = df[col].dropna()
    if series.empty:
        return pd.Index([])
    lower, upper = _calculate_iqr_bounds(series, factor)
    return df[(df[col] < lower) | (df[col] > upper)].index


def handle_outliers(
    df: pd.DataFrame,
    cols: List[str],
    method: str = "Remover linhas",
    factor: float = 1.5
) -> pd.DataFrame:
    df_copy = df.copy()
    if method == "Remover linhas":
        out = df_copy.drop(index=list({idx for col in cols for idx in detect_outliers_iqr(df_copy, col, factor)}), errors='ignore')
        log_preprocessing_step(f"Tratamento de outliers: método='{method}', colunas={cols}, fator={factor}")
        return out
    if method == "Winsorização":
        for col in cols:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                lb, ub = _calculate_iqr_bounds(df_copy[col].dropna(), factor)
                df_copy[col] = df_copy[col].clip(lb, ub)
        log_preprocessing_step(f"Tratamento de outliers: método='{method}', colunas={cols}, fator={factor}")
        return df_copy
    if method == "Substituir por mediana":
        for col in cols:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                lb, ub = _calculate_iqr_bounds(df_copy[col].dropna(), factor)
                median_val = df_copy[col].median()
                df_copy.loc[(df_copy[col] < lb) | (df_copy[col] > ub), col] = median_val
        log_preprocessing_step(f"Tratamento de outliers: método='{method}', colunas={cols}, fator={factor}")
        return df_copy
    return df_copy

# --- Imputação Genérica de Valores Ausentes ---
def impute_missing(
    df: pd.DataFrame,
    cols: List[str],
    strategy: str = "mean",
    constant: Any = None
) -> pd.DataFrame:
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns and df_copy[col].isnull().any():
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
                val = df_copy[col].mean()
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df_copy[col]):
                val = df_copy[col].median()
            elif strategy == "mode":
                non_null = df_copy[col].dropna()
                if non_null.empty:
                    continue
                val = non_null.mode()[0]
            elif strategy == "constant":
                val = constant
            else:
                continue
            df_copy[col] = df_copy[col].fillna(val)
    log_preprocessing_step(f"Imputação de valores ausentes: estratégia='{strategy}', colunas={cols}, valor_fixo={constant}")
    return df_copy

# Wrappers retrocompatíveis

def impute_missing_mean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return impute_missing(df, cols, strategy="mean")

def impute_missing_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return impute_missing(df, cols, strategy="median")

def impute_missing_mode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return impute_missing(df, cols, strategy="mode")

def impute_missing_fixed(df: pd.DataFrame, cols: List[str], fixed_value: Any) -> pd.DataFrame:
    return impute_missing(df, cols, strategy="constant", constant=fixed_value)

# --- Remoção e Interpolação de Valores Ausentes ---
def remove_missing_rows(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.dropna(subset=[c for c in cols if c in df.columns])
    log_preprocessing_step(f"Remoção de linhas com valores ausentes em colunas: {cols}")
    return df2


def interpolate_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].interpolate(method="linear", limit_direction="both")
    log_preprocessing_step(f"Interpolação linear de valores ausentes em colunas: {cols}")
    return df_copy

# --- Padronização, Normalização e Transformação Log ---
def standardize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean = df_copy[col].dropna().mean()
            std = df_copy[col].dropna().std()
            df_copy[f"{col}_z"] = ((df_copy[col] - mean) / std) if std != 0 else 0
    log_preprocessing_step(f"Padronização (z-score) em colunas: {cols}")
    return df_copy


def normalize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mn = df_copy[col].dropna().min()
            mx = df_copy[col].dropna().max()
            df_copy[f"{col}_minmax"] = ((df_copy[col] - mn) / (mx - mn)) if mx != mn else 0
    log_preprocessing_step(f"Normalização (MinMax) em colunas: {cols}")
    return df_copy


def log_transform_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[f"{col}_log"] = np.log1p(df_copy[col].clip(lower=0))
    log_preprocessing_step(f"Transformação log1p em colunas: {cols}")
    return df_copy

# --- Visualização de Outliers ---
def show_outlier_distribution(
    df: pd.DataFrame,
    col: str,
    factor: float = 1.5
) -> None:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        st.warning(f"Coluna '{col}' não existe ou não é numérica.")
        return
    series = df[col].dropna()
    if series.empty:
        st.info(f"Nenhum dado válido em '{col}'.")
        return
    lower, upper = _calculate_iqr_bounds(series, factor)
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.subheader(f"Distribuição e Outliers: {col}")
    st.write(f"Limites: [{lower:.2f}, {upper:.2f}] — Total outliers: {len(outliers)}")
    st.box_chart(series.to_frame())
    st.write(outliers)

# --- Interface Streamlit para Pré-processamento ---
def show_preprocessing_interface():
    st.info("Esta seção permite selecionar as colunas para o  DataFrame de trabalho,  pré-processar, modificar e ajustar os tipos de dados.")
    # Inicializa log
    init_preprocessing_log()

    # --- Expander: duplicar e renomear valores ---
    with st.expander("🔁 Duplicar coluna e renomear valores"):
        # ... lógica de duplicação e mapeamento de valores ...
        pass  # mantém código original aqui

    # --- Expander: histórico de transformações ---
    log_entries = show_preprocessing_log()
    if log_entries:
        st.markdown("---")
        st.subheader("📝 Histórico de Pré-processamento")
        for entry in log_entries:
            st.write(entry)
        # Botão para exportar histórico de pré-processamento
        log_str = "\n".join(log_entries)
        st.download_button(
            label="📥 Baixar Histórico de Pré-processamento",
            data=log_str.encode("utf-8"),
            file_name="preprocessing_log.txt",
            mime="text/plain",
        )


  # --- Inicialização das chaves de sessão ---
    # Initialize all session state variables at the beginning for robustness
    if 'selected_columns_for_df_processed' not in st.session_state:
        st.session_state['selected_columns_for_df_processed'] = []
    if 'df_processed' not in st.session_state:
        st.session_state['df_processed'] = pd.DataFrame() # Initialize as an empty DataFrame
    if 'last_preprocessing_log' not in st.session_state:
        st.session_state['last_preprocessing_log'] = []
    if 'preprocessing_applied_flag' not in st.session_state:
        st.session_state['preprocessing_applied_flag'] = False
    if 'reset_selection_trigger' not in st.session_state:
        st.session_state['reset_selection_trigger'] = 0
    if 'selected_cols_for_ops_multiselect' not in st.session_state:
        st.session_state['selected_cols_for_ops_multiselect'] = []
    if "selected_col_to_convert" not in st.session_state:
        st.session_state.selected_col_to_convert = "Nenhum"
    if "selected_target_dtype" not in st.session_state:
        st.session_state.selected_target_dtype = "Nenhum"
    if "datetime_format_input" not in st.session_state:
        st.session_state.datetime_format_input = ""
    if "duplicated_col_name" not in st.session_state:
        st.session_state["duplicated_col_name"] = None
    if 'select_all_initial_cols_checkbox' not in st.session_state:
        st.session_state['select_all_initial_cols_checkbox'] = False
    if 'select_all_for_ops_checkbox' not in st.session_state: # Ensure this is initialized
        st.session_state['select_all_for_ops_checkbox'] = False


    # --- Verificação de DataFrame Carregado ---
    if 'df_original' not in st.session_state or st.session_state.df_original is None or st.session_state.df_original.empty:
        st.warning("Nenhum dado original disponível. Por favor, carregue um arquivo no menu lateral para começar.")
        st.session_state['df_processed'] = pd.DataFrame() # Ensure df_processed is always a DataFrame
        st.session_state['selected_columns_for_df_processed'] = []
        st.session_state['last_preprocessing_log'] = []
        return # Exit if no original data

    # --- Seção 1: Seleção de Colunas para o DataFrame de Trabalho ---
    with st.expander("1. Seleção de Colunas para o DataFrame de Trabalho", expanded=True):
        st.markdown("Selecione as colunas do **DataFrame Original** que deseja incluir no **DataFrame de Trabalho (`df_processed`)**.")

        all_original_columns = st.session_state['df_original'].columns.tolist()

        def apply_column_selection_callback():
            current_selection = st.session_state.get('multiselect_initial_cols', [])
            st.session_state['selected_columns_for_df_processed'] = current_selection

            if current_selection:
                st.session_state['df_processed'] = st.session_state['df_original'][current_selection].copy()
                st.session_state['last_preprocessing_log'] = [f"DataFrame de trabalho inicializado com {len(current_selection)} colunas selecionadas do original."]
                st.session_state['preprocessing_applied_flag'] = True
                # Reset operations selections when df_processed columns change
                st.session_state['selected_cols_for_ops_multiselect'] = []
                st.session_state['select_all_for_ops_checkbox'] = False # Reset the ops select all checkbox
                st.session_state['selected_col_to_convert'] = "Nenhum"
                st.session_state['selected_target_dtype'] = "Nenhum"
                st.session_state.duplicated_col_name = None # Reset duplicated column
            else:
                st.info("Nenhuma coluna selecionada para o DataFrame de trabalho. O DataFrame de trabalho foi redefinido para vazio.")
                st.session_state['df_processed'] = pd.DataFrame() # Ensure it's an empty DataFrame
                st.session_state['last_preprocessing_log'] = ["Nenhuma coluna selecionada. DataFrame de trabalho vazio."]
                st.session_state['preprocessing_applied_flag'] = False
                st.session_state['selected_cols_for_ops_multiselect'] = []
                st.session_state['select_all_for_ops_checkbox'] = False
                st.session_state['selected_col_to_convert'] = "Nenhum"
                st.session_state['selected_target_dtype'] = "Nenhum"
                st.session_state.duplicated_col_name = None # Reset duplicated column

            st.session_state['reset_selection_trigger'] += 1 # Trigger a refresh for multiselect keys
            st.rerun()

        # Determine if 'select all' checkbox should be checked by default
        current_select_all_initial = (
            len(st.session_state['selected_columns_for_df_processed']) == len(all_original_columns)
            and len(all_original_columns) > 0
        )
        
        # Ensure checkbox value matches the current state
        # st.session_state['select_all_initial_cols_checkbox'] = current_select_all_initial # This line might cause issues with on_change if directly manipulated before the widget

        def toggle_select_all_initial_cols():
            if st.session_state.select_all_initial_cols_checkbox:
                st.session_state['multiselect_initial_cols'] = all_original_columns.copy()
            else:
                st.session_state['multiselect_initial_cols'] = []
            apply_column_selection_callback() # Call the apply function to update df_processed and other states
            
        st.checkbox(
            "Selecionar/Desselecionar Todas as Colunas Originais",
            value=current_select_all_initial,
            key="select_all_initial_cols_checkbox",
            on_change=toggle_select_all_initial_cols
        )

        selected_cols_multiselect = st.multiselect(
            "Colunas a serem incluídas no DataFrame de trabalho:",
            options=all_original_columns,
            default=st.session_state['selected_columns_for_df_processed'],
            key="multiselect_initial_cols"
        )
        
        st.button("Aplicar Seleção de Colunas para DataFrame de Trabalho", on_click=apply_column_selection_callback, key="apply_df_creation_btn")

    st.markdown("---")

    # --- Seção 2: Aplicação de Pré-processamento e Transformações ---
    # Moved toggle_select_all_ops_cols and its checkbox OUTSIDE the form to resolve StreamlitInvalidFormCallbackError
    # This function is now called by the checkbox which is outside the form.
    def toggle_select_all_ops_cols():
        # Ensure df_processed is not empty before attempting to get columns
        if st.session_state['df_processed'] is not None and not st.session_state['df_processed'].empty:
            current_df_processed_columns = st.session_state['df_processed'].columns.tolist()
            if st.session_state.select_all_for_ops_checkbox:
                st.session_state['selected_cols_for_ops_multiselect'] = current_df_processed_columns.copy()
            else:
                st.session_state['selected_cols_for_ops_multiselect'] = []
        else:
            st.session_state['selected_cols_for_ops_multiselect'] = [] # If df_processed is empty, clear selection

    with st.expander("2. Aplicação de Pré-processamento e Transformações", expanded=True):
        st.markdown("Selecione as colunas do **DataFrame de Trabalho (`df_processed`)** nas quais você deseja aplicar operações de limpeza e transformação.")

        # Crucial fix: Move this checkbox OUTSIDE the form
        if st.session_state['df_processed'] is not None and not st.session_state['df_processed'].empty:
            current_df_processed_columns = st.session_state['df_processed'].columns.tolist()
            current_select_all_ops = (
                len(st.session_state['selected_cols_for_ops_multiselect']) == len(current_df_processed_columns)
                and len(current_df_processed_columns) > 0
            )
            st.checkbox(
                "Selecionar/Desselecionar Todas as Colunas do DataFrame de Trabalho para Pré-processamento",
                value=current_select_all_ops, # Use the dynamically updated value
                key="select_all_for_ops_checkbox",
                on_change=toggle_select_all_ops_cols
            )
        else:
            # If df_processed is empty, display a disabled checkbox or just a placeholder message
            st.info("Para selecionar colunas para pré-processamento, primeiro crie o DataFrame de Trabalho na Seção 1.")

        # The form itself starts here
        with st.form('form_preprocessing_section'):
            # This check remains important for displaying appropriate messages and disabling the submit button.
            if st.session_state['df_processed'] is None or st.session_state['df_processed'].empty:
                st.warning("DataFrame de trabalho ainda não criado ou está vazio. Por favor, faça a 'Seleção de Colunas para o DataFrame de Trabalho' acima primeiro.")
                # Render a disabled submit button to satisfy Streamlit's form requirement
                submit_preproc = st.form_submit_button("Aplicar Limpeza e Transformações", disabled=True)
            else:
                st.markdown("### Etapas de Pré-processamento")
                current_df_processed_columns = st.session_state['df_processed'].columns.tolist()
                
                # The multiselect remains inside the form
                cols_to_apply_preprocessing = st.multiselect(
                    "Selecione as colunas do DataFrame de Trabalho para aplicar pré-processamento:",
                    options=current_df_processed_columns,
                    default=st.session_state['selected_cols_for_ops_multiselect'],
                    key=f"cols_to_apply_preprocessing_multiselect_form_{st.session_state['reset_selection_trigger']}" # Key change for reset
                )
                # Ensure session state reflects current multiselect value immediately
                if st.session_state['selected_cols_for_ops_multiselect'] != cols_to_apply_preprocessing:
                    st.session_state['selected_cols_for_ops_multiselect'] = cols_to_apply_preprocessing

                # Filter columns based on type for specific operations
                selected_numeric_cols_for_ops = [col for col in cols_to_apply_preprocessing if pd.api.types.is_numeric_dtype(st.session_state['df_processed'][col])]
                selected_non_numeric_cols_for_ops = [col for col in cols_to_apply_preprocessing if not pd.api.types.is_numeric_dtype(st.session_state['df_processed'][col])]

                st.subheader("Tratamento de Valores Faltantes")
                missing_method = st.selectbox(
                    "Método para tratamento de valores faltantes (missing values):",
                    options=["Nenhum", "Remover linhas", "Imputar com média", "Imputar com mediana", "Imputar com moda", "Imputar valor fixo", "Interpolar"],
                    key="missing_method_select"
                )
                fixed_value_imputation = None
                fixed_value_imputation_str = None
                if missing_method == "Imputar valor fixo":
                    # Ensure visibility of inputs based on selected columns
                    if selected_numeric_cols_for_ops:
                        fixed_value_imputation = st.number_input("Valor fixo para imputar em colunas NUMÉRICAS:", value=0.0, key="fixed_num_value_input")
                    else:
                        fixed_value_imputation = None # Clear if no numeric cols
                    if selected_non_numeric_cols_for_ops:
                        fixed_value_imputation_str = st.text_input("Valor fixo para imputar em colunas NÃO NUMÉRICAS:", value="Desconhecido", key="fixed_cat_value_input")
                    else:
                        fixed_value_imputation_str = None # Clear if no non-numeric cols
                    if not selected_numeric_cols_for_ops and not selected_non_numeric_cols_for_ops:
                        st.info("Nenhuma coluna selecionada para imputação de valor fixo.")

                st.subheader("Tratamento de Outliers")
                outlier_method = st.selectbox(
                    "Método para tratamento de outliers (apenas para colunas numéricas selecionadas):",
                    options=["Nenhum", "Remover linhas", "Winsorização", "Substituir por mediana"],
                    key="outlier_method_select"
                )
                iqr_factor = 1.5
                if outlier_method != "Nenhum":
                    if selected_numeric_cols_for_ops:
                        iqr_factor = st.slider("Fator multiplicador do IQR para definir outliers:", 1.0, 3.0, 1.5, key="iqr_factor_slider")
                    else:
                        st.info("Nenhuma coluna numérica selecionada para tratamento de outliers.")

                st.subheader("Transformações (Criar Novas Colunas)")
                st.info("Estas operações criam novas colunas no DataFrame de trabalho com o sufixo indicado.")
                standardize = st.checkbox("Aplicar padronização (Z-Score) (sufixo: _z)", key="standardize_checkbox")
                normalize = st.checkbox("Aplicar normalização (Min-Max) (sufixo: _minmax)", key="normalize_checkbox")
                log_transform_flag = st.checkbox("Aplicar transformação logarítmica (log1p) (sufixo: _log)", key="log_transform_checkbox")
                
                # Single submit button, always present within the form if the outer condition is met
                submit_preproc = st.form_submit_button("Aplicar Limpeza e Transformações")
                
                if submit_preproc:
                    if st.session_state.df_original is None or st.session_state.df_original.empty:
                        st.error("Não há dados originais para processar. Por favor, carregue um arquivo.")
                        st.session_state['last_preprocessing_log'] = ["Erro: Nenhum dado original para processar."]
                        st.session_state['preprocessing_applied_flag'] = False
                        st.rerun()
                        return
                    
                    # Check df_processed again, as it might have become empty after previous operations (e.g., initial column selection)
                    if st.session_state['df_processed'].empty:
                        st.warning("DataFrame de trabalho está vazio. Nenhuma operação de pré-processamento será aplicada. Por favor, faça a seleção inicial de colunas.")
                        st.session_state['last_preprocessing_log'] = ["Aviso: DataFrame de trabalho vazio. Nenhuma operação aplicada."]
                        st.session_state['preprocessing_applied_flag'] = False
                        st.rerun()
                        return

                    processing_temp_df = st.session_state['df_processed'].copy()
                    operations_performed = False
                    processing_log = []

                    cols_to_apply_preprocessing_current = st.session_state.get('selected_cols_for_ops_multiselect', [])

                    # 1. Tratamento de Missing Values
                    if missing_method != "Nenhum" and cols_to_apply_preprocessing_current:
                        st.info(f"Aplicando tratamento de valores faltantes: '{missing_method}'...")
                        for col in cols_to_apply_preprocessing_current:
                            if col not in processing_temp_df.columns:
                                processing_log.append(f"Coluna '{col}' não encontrada no DataFrame de trabalho para tratamento de NaN. Pulando.")
                                continue

                            if processing_temp_df[col].isnull().any():
                                operations_performed = True
                                if pd.api.types.is_numeric_dtype(processing_temp_df[col]):
                                    if missing_method == "Remover linhas":
                                        rows_before = len(processing_temp_df)
                                        processing_temp_df = remove_missing_rows(processing_temp_df, [col])
                                        rows_after = len(processing_temp_df)
                                        if rows_before > rows_after:
                                            processing_log.append(f"Removidas {rows_before - rows_after} linhas com NaN em '{col}'.")
                                    elif missing_method == "Imputar com média":
                                        processing_temp_df = impute_missing_mean(processing_temp_df, [col])
                                        processing_log.append(f"NaNs em '{col}' imputados com a média.")
                                    elif missing_method == "Imputar com mediana":
                                        processing_temp_df = impute_missing_median(processing_temp_df, [col])
                                        processing_log.append(f"NaNs em '{col}' imputados com a mediana.")
                                    elif missing_method == "Imputar com moda":
                                        processing_temp_df = impute_missing_mode(processing_temp_df, [col])
                                        processing_log.append(f"NaNs em '{col}' imputados com a moda.")
                                    elif missing_method == "Imputar valor fixo" and fixed_value_imputation is not None:
                                        processing_temp_df = impute_missing_fixed(processing_temp_df, [col], fixed_value_imputation)
                                        processing_log.append(f"NaNs em '{col}' imputados com valor fixo ({fixed_value_imputation}).")
                                    elif missing_method == "Interpolar":
                                        processing_temp_df = interpolate_missing(processing_temp_df, [col])
                                        processing_log.append(f"NaNs em '{col}' imputados por interpolação.")
                                else: # Non-numeric columns
                                    if missing_method == "Imputar com moda":
                                        processing_temp_df = impute_missing_mode(processing_temp_df, [col])
                                        processing_log.append(f"NaNs em '{col}' (não numérica) imputados com a moda.")
                                    elif missing_method == "Imputar valor fixo" and fixed_value_imputation_str is not None:
                                        processing_temp_df = impute_missing_fixed(processing_temp_df, [col], fixed_value_imputation_str)
                                        processing_log.append(f"NaNs em '{col}' (não numérica) imputados com valor fixo ('{fixed_value_imputation_str}').")
                                    else:
                                        processing_log.append(f"Estratégia '{missing_method}' não aplicável ou valor não fornecido para '{col}' (não numérica).")
                            else:
                                processing_log.append(f"Coluna '{col}' não tem valores faltantes.")
                    elif missing_method != "Nenhum" and not cols_to_apply_preprocessing_current:
                        processing_log.append("Estratégia de tratamento de NaN selecionada, mas nenhuma coluna para tratamento foi escolhida na Seção 2.")
                    elif missing_method == "Nenhum":
                        processing_log.append("Nenhum tratamento de valores faltantes selecionado.")


                    # 2. Tratamento de Outliers
                    selected_numeric_cols_for_ops_current = [col for col in cols_to_apply_preprocessing_current if pd.api.types.is_numeric_dtype(processing_temp_df[col])]

                    if outlier_method != "Nenhum" and selected_numeric_cols_for_ops_current:
                        st.info(f"Aplicando tratamento de outliers: '{outlier_method}' (Fator IQR: {iqr_factor})...")
                        # For remove rows, apply once to the whole df for selected columns
                        if outlier_method == "Remover linhas":
                            initial_rows = len(processing_temp_df)
                            processing_temp_df = handle_outliers(processing_temp_df, selected_numeric_cols_for_ops_current, outlier_method, iqr_factor)
                            if len(processing_temp_df) < initial_rows:
                                processing_log.append(f"Removidas {initial_rows - len(processing_temp_df)} linhas com outliers em colunas selecionadas ({outlier_method}).")
                                operations_performed = True
                            else:
                                processing_log.append(f"Outliers em colunas selecionadas ({outlier_method}) solicitados, mas nenhuma linha removida.")
                        else: # Winsorização ou Substituir por mediana
                            for col in selected_numeric_cols_for_ops_current:
                                original_col_data = processing_temp_df[col].copy()
                                # Use a temporary DataFrame with only the column to be processed
                                temp_col_df = processing_temp_df[[col]].copy()
                                processed_col_df = handle_outliers(temp_col_df, [col], outlier_method, iqr_factor)
                                
                                # Check if the column actually changed
                                if not original_col_data.equals(processed_col_df[col]):
                                    processing_temp_df[col] = processed_col_df[col]
                                    processing_log.append(f"Outliers em '{col}' tratados por '{outlier_method}'.")
                                    operations_performed = True
                                else:
                                    processing_log.append(f"Outliers em '{col}' tratados por '{outlier_method}' solicitados, mas nenhuma alteração detectada.")
                    elif outlier_method != "Nenhum" and not selected_numeric_cols_for_ops_current:
                        processing_log.append("Estratégia de tratamento de outliers selecionada, mas nenhuma coluna numérica elegível para tratamento foi escolhida na Seção 2.")
                    elif outlier_method == "Nenhum":
                        processing_log.append("Nenhum tratamento de outliers selecionado.")

                    # 3. Transformações (Padronização, Normalização, Log)
                    current_numeric_cols_for_transforms = [col for col in cols_to_apply_preprocessing_current if col in processing_temp_df.columns and pd.api.types.is_numeric_dtype(processing_temp_df[col])]

                    if standardize and current_numeric_cols_for_transforms:
                        st.info("Aplicando padronização (Z-Score)...")
                        initial_cols = processing_temp_df.columns.tolist()
                        processing_temp_df = standardize_columns(processing_temp_df, current_numeric_cols_for_transforms)
                        if set(processing_temp_df.columns) != set(initial_cols): # Check if new columns were added
                            cols_added_std = [c for c in processing_temp_df.columns if c not in initial_cols and c.endswith('_z')]
                            processing_log.append(f"Padronização (Z-Score) aplicada às colunas: {', '.join(cols_added_std)} (novas colunas com sufixo '_z').")
                            operations_performed = True
                        else:
                            processing_log.append("Padronização solicitada, mas nenhuma nova coluna _z foi criada ou não houve alteração.")
                    elif standardize and not current_numeric_cols_for_transforms:
                        processing_log.append("Padronização (Z-Score) solicitada, mas nenhuma coluna numérica elegível foi selecionada na Seção 2.")

                    if normalize and current_numeric_cols_for_transforms:
                        st.info("Aplicando normalização (Min-Max)...")
                        initial_cols = processing_temp_df.columns.tolist()
                        processing_temp_df = normalize_columns(processing_temp_df, current_numeric_cols_for_transforms)
                        if set(processing_temp_df.columns) != set(initial_cols): # Check if new columns were added
                            cols_added_norm = [c for c in processing_temp_df.columns if c not in initial_cols and c.endswith('_minmax')]
                            processing_log.append(f"Normalização (Min-Max) aplicada às colunas: {', '.join(cols_added_norm)} (novas colunas com sufixo '_minmax').")
                            operations_performed = True
                        else:
                            processing_log.append("Normalização solicitada, mas nenhuma nova coluna _minmax foi criada ou não houve alteração.")
                    elif normalize and not current_numeric_cols_for_transforms:
                        processing_log.append("Normalização (Min-Max) solicitada, mas nenhuma coluna numérica elegível foi selecionada na Seção 2.")

                    if log_transform_flag and current_numeric_cols_for_transforms:
                        st.info("Aplicando transformação logarítmica (log1p)...")
                        initial_cols = processing_temp_df.columns.tolist()
                        processing_temp_df = log_transform_columns(processing_temp_df, current_numeric_cols_for_transforms)
                        if set(processing_temp_df.columns) != set(initial_cols): # Check if new columns were added
                            cols_added_log = [c for c in processing_temp_df.columns if c not in initial_cols and c.endswith('_log')]
                            processing_log.append(f"Transformação logarítmica (log1p) aplicada às colunas: {', '.join(cols_added_log)} (novas colunas com sufixo '_log').")
                            operations_performed = True
                        else:
                            processing_log.append("Transformação logarítmica solicitada, mas nenhuma nova coluna _log foi criada ou não houve alteração.")
                    elif log_transform_flag and not current_numeric_cols_for_transforms:
                        processing_log.append("Transformação logarítmica solicitada, mas nenhuma coluna numérica elegível foi selecionada na Seção 2.")
                    
                    if not operations_performed:
                        processing_log.append("Nenhuma operação de pré-processamento foi aplicada.")

                    st.session_state['df_processed'] = processing_temp_df
                    st.session_state['last_preprocessing_log'] = processing_log
                    st.session_state['preprocessing_applied_flag'] = True if operations_performed else False
                    st.rerun()

    st.markdown("---")

    # --- Seção 3: Conversão de Tipos de Dados ---
    with st.expander("3. Conversão de Tipos de Dados", expanded=False):
        st.info("Ajuste o tipo de dado de colunas específicas.")

        col_to_convert = "Nenhum"
        if st.session_state['df_processed'] is None or st.session_state['df_processed'].empty:
            st.warning("DataFrame de trabalho ainda não criado ou está vazio. Por favor, faça a 'Seleção de Colunas para o DataFrame de Trabalho' acima primeiro.")
        else:
            current_df_processed_columns = st.session_state['df_processed'].columns.tolist()
            if current_df_processed_columns:
                col_to_convert = st.selectbox("Selecione uma coluna para conversão de tipo:",
                                                ["Nenhum"] + current_df_processed_columns,
                                                key="convert_col_select")
                # Update session state when selectbox value changes
                if st.session_state.selected_col_to_convert != col_to_convert:
                    st.session_state.selected_col_to_convert = col_to_convert
                    st.session_state.selected_target_dtype = "Nenhum" # Reset target type
                    st.session_state.datetime_format_input = ""

                current_dtype = None
                if col_to_convert != "Nenhum" and col_to_convert in st.session_state['df_processed'].columns:
                    current_dtype = st.session_state['df_processed'][col_to_convert].dtype
                    st.write(f"Tipo atual da coluna '{col_to_convert}': `{current_dtype}`")

                target_dtype_options = ["Nenhum", "int", "float", "datetime", "category", "str (object)"]
                target_dtype = "Nenhum"
                if col_to_convert != "Nenhum":
                    target_dtype = st.selectbox(
                        f"Converter '{col_to_convert}' para:",
                        target_dtype_options,
                        # Set default value based on session state, with a fallback to "Nenhum"
                        index=target_dtype_options.index(st.session_state.selected_target_dtype) if st.session_state.selected_target_dtype in target_dtype_options else 0,
                        key=f"target_dtype_{col_to_convert}" # Use column name in key to ensure unique widget per column
                    )
                    # Update session state when target type changes
                    if st.session_state.selected_target_dtype != target_dtype:
                        st.session_state.selected_target_dtype = target_dtype
                        # Clear datetime format if not converting to datetime
                        if target_dtype != "datetime":
                            st.session_state.datetime_format_input = ""


                    if target_dtype == "datetime":
                        st.info("O Pandas tentará inferir o formato da data automaticamente. Forneça um formato se a inferência falhar (ex: %Y-%m-%d, %d/%m/%Y).")
                        st.session_state.datetime_format_input = st.text_input(
                            "Formato da data (opcional, ex: %Y-%m-%d):",
                            value=st.session_state.datetime_format_input,
                            key="datetime_format_input_text"
                        )
                    # No else needed here, as datetime_format_input is cleared when target_dtype changes

            # Button for Type Conversion
            if st.button("Aplicar Conversão de Tipos", key="apply_type_conversion_btn"):
                if st.session_state.df_original is None or st.session_state.df_original.empty:
                    st.error("Não há dados originais para processar. Por favor, carregue um arquivo.")
                    st.session_state['last_preprocessing_log'] = ["Erro: Nenhum dado original para processar."]
                    st.session_state['preprocessing_applied_flag'] = False
                    st.rerun()
                    return
                
                if st.session_state['df_processed'].empty:
                    st.warning("DataFrame de trabalho está vazio. Nenhuma conversão de tipo será aplicada. Por favor, faça a seleção inicial de colunas.")
                    st.session_state['last_preprocessing_log'] = ["Aviso: DataFrame de trabalho vazio. Nenhuma conversão de tipo aplicada."]
                    st.session_state['preprocessing_applied_flag'] = False
                    st.rerun()
                    return
                
                processing_temp_df = st.session_state['df_processed'].copy()
                operations_performed = False
                processing_log = []

                col_to_convert_current = st.session_state.get('selected_col_to_convert', "Nenhum")
                target_dtype_current = st.session_state.get('selected_target_dtype', "Nenhum")
                datetime_format_current = st.session_state.get('datetime_format_input', "")

                if col_to_convert_current != "Nenhum" and col_to_convert_current in processing_temp_df.columns and target_dtype_current != "Nenhum":
                    operations_performed = True
                    try:
                        if target_dtype_current == "int":
                            # Use 'Int64' for nullable integer type to handle NaNs correctly
                            processing_temp_df[col_to_convert_current] = pd.to_numeric(processing_temp_df[col_to_convert_current], errors='coerce').astype('Int64')
                            processing_log.append(f"Coluna '{col_to_convert_current}' convertida para tipo inteiro ('Int64').")
                        elif target_dtype_current == "float":
                            processing_temp_df[col_to_convert_current] = pd.to_numeric(processing_temp_df[col_to_convert_current], errors='coerce')
                            processing_log.append(f"Coluna '{col_to_convert_current}' convertida para tipo float.")
                        elif target_dtype_current == "datetime":
                            processing_temp_df[col_to_convert_current] = pd.to_datetime(processing_temp_df[col_to_convert_current], errors='coerce', format=datetime_format_current if datetime_format_current else None)
                            processing_log.append(f"Coluna '{col_to_convert_current}' convertida para tipo datetime.")
                        elif target_dtype_current == "category":
                            processing_temp_df[col_to_convert_current] = processing_temp_df[col_to_convert_current].astype('category')
                            processing_log.append(f"Coluna '{col_to_convert_current}' convertida para tipo categórico.")
                        elif target_dtype_current == "str (object)":
                            processing_temp_df[col_to_convert_current] = processing_temp_df[col_to_convert_current].astype(str)
                            processing_log.append(f"Coluna '{col_to_convert_current}' convertida para tipo string (object).")
                        else:
                            processing_log.append(f"Nenhuma conversão de tipo aplicada para '{col_to_convert_current}'.")
                    except Exception as e:
                        processing_log.append(f"Erro ao converter a coluna '{col_to_convert_current}' para '{target_dtype_current}': {e}")
                        
                elif col_to_convert_current != "Nenhum" and target_dtype_current == "Nenhum":
                    processing_log.append(f"Coluna '{col_to_convert_current}' selecionada para conversão, mas nenhum tipo de destino foi escolhido.")
                elif col_to_convert_current == "Nenhum" and target_dtype_current != "Nenhum":
                    processing_log.append(f"Tipo de destino '{target_dtype_current}' escolhido, mas nenhuma coluna selecionada para conversão.")
                else:
                    processing_log.append("Nenhuma coluna ou tipo de destino selecionado para conversão.")
                
                st.session_state['df_processed'] = processing_temp_df
                st.session_state['last_preprocessing_log'] = processing_log
                st.session_state['preprocessing_applied_flag'] = True if operations_performed else False
                st.rerun()

    # --- Seção 4: Duplicar Coluna e Renomear Valores ---
    with st.expander("4. Duplicar Coluna e Renomear Valores", expanded=False):
        st.info("Você pode duplicar uma coluna e renomear seus valores na nova coluna. As alterações só serão aplicadas após clicar no botão.")

        if st.session_state['df_processed'] is not None and not st.session_state['df_processed'].empty:
            cols = st.session_state['df_processed'].columns.tolist()
            col_to_duplicate = st.selectbox("Selecione a coluna que deseja duplicar:", ["Nenhum"] + cols, key="duplicate_col_select")

            # Initialize new_col_name for consistent behavior
            default_new_col_name = f"{col_to_duplicate}_copy" if col_to_duplicate != "Nenhum" else "nova_coluna"
            new_col_name = st.text_input("Nome da nova coluna (duplicada):", value=default_new_col_name, key="new_col_name_input")

            if col_to_duplicate != "Nenhum":
                # Check for existing column name only when duplicating
                if new_col_name in st.session_state['df_processed'].columns and new_col_name != default_new_col_name: # Allow default name to be suggested
                    st.warning("Já existe uma coluna com esse nome. Escolha um nome diferente.")
                elif st.button("Duplicar Coluna", key="duplicate_col_btn"):
                    # Clear previous state related to renaming if a new duplication occurs
                    # This helps avoid confusion if user duplicates multiple times
                    if st.session_state.get("duplicated_col_name") and st.session_state["duplicated_col_name"] != new_col_name:
                        st.session_state.pop(f"rename_map_{st.session_state['duplicated_col_name']}", None)
                        # Also clear old rename_value_* keys if they were for a different column
                        keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("rename_value_")]
                        for k in keys_to_clear:
                            st.session_state.pop(k, None)

                    st.session_state['df_processed'][new_col_name] = st.session_state['df_processed'][col_to_duplicate].copy()
                    st.session_state["duplicated_col_name"] = new_col_name
                    # Initialize the rename map for the newly duplicated column
                    if f"rename_map_{new_col_name}" not in st.session_state:
                         st.session_state[f"rename_map_{new_col_name}"] = {}
                    st.success(f"Coluna '{col_to_duplicate}' duplicada como '{new_col_name}'.")
                    st.rerun() # Rerun to update the `col_duplicada` logic immediately
            else:
                st.info("Selecione uma coluna para duplicar.")


            # If duplication was made, allow renaming
            col_duplicada = st.session_state.get("duplicated_col_name", None)

            if col_duplicada and col_duplicada in st.session_state['df_processed'].columns:
                st.markdown(f"### Renomear valores da nova coluna '{col_duplicada}' (opcional)")

                # Ensure unique values are pulled from the *current* df_processed for the duplicated column
                # This is important if df_processed changes due to other operations
                unique_vals = st.session_state['df_processed'][col_duplicada].dropna().unique().tolist()
                
                # Initialize rename map if not already present for this specific duplicated column
                if f"rename_map_{col_duplicada}" not in st.session_state:
                    st.session_state[f"rename_map_{col_duplicada}"] = {}

                rename_map = st.session_state[f"rename_map_{col_duplicada}"]

                st.write("Digite o novo valor para cada valor único existente:")
                for i, val in enumerate(unique_vals):
                    # Ensure keys are unique across reruns and tied to the specific duplicated column and original value
                    default_val = rename_map.get(val, str(val))
                    key_val_input = f"rename_value_{col_duplicada}_{val}_{i}" # Added index for more uniqueness
                    new_val_input = st.text_input(f"Novo valor para '{val}':", value=default_val, key=key_val_input)
                    rename_map[val] = new_val_input

                st.session_state[f"rename_map_{col_duplicada}"] = rename_map

                if st.button("Aplicar Renomeação de Valores", key="apply_value_renaming_btn"):
                    if not rename_map:
                        st.warning("Nenhum mapeamento de renomeação de valores foi definido.")
                    else:
                        try:
                            st.session_state['df_processed'] = rename_column_values(
                                st.session_state['df_processed'],
                                col_duplicada,
                                st.session_state[f"rename_map_{col_duplicada}"]
                            )
                            st.success(f"Valores da coluna '{col_duplicada}' foram renomeados com sucesso.")
                            st.session_state['last_preprocessing_log'].append(f"Valores na coluna '{col_duplicada}' renomeados.")
                            st.session_state['preprocessing_applied_flag'] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao renomear valores: {e}")
                            st.session_state['last_preprocessing_log'].append(f"Erro ao renomear valores na coluna '{col_duplicada}': {e}")
                            st.session_state['preprocessing_applied_flag'] = False
            else:
                st.info("Nenhuma coluna duplicada ainda. Duplique uma coluna primeiro para renomear seus valores.")
        else:
            st.info("O DataFrame de trabalho está vazio. Por favor, selecione e processe colunas antes de usar esta funcionalidade.")


    # --- Exibição dos resultados do pré-processamento ---
    st.markdown("---")
    if st.session_state.get('preprocessing_applied_flag', False):
        st.subheader("Pré-processamento Concluído!")
        for log_entry in st.session_state['last_preprocessing_log']:
            st.success(f"- {log_entry}")
    elif st.session_state.get('last_preprocessing_log'):
        st.subheader("Status do Pré-processamento:")
        for log_entry in st.session_state['last_preprocessing_log']:
            st.info(f"- {log_entry}")
    else:
        st.info("Nenhuma operação de pré-processamento aplicada ainda.")


    st.markdown("---")
    st.subheader("Prévia do DataFrame Processado Atualmente")
    if st.session_state.df_processed is not None and not st.session_state.df_processed.empty:
        st.dataframe(st.session_state.df_processed.head())
        st.write(f"Dimensões: {st.session_state['df_processed'].shape[0]} linhas, {st.session_state['df_processed'].shape[1]} colunas.")
        st.write("Tipos de Dados Atuais:")
        st.dataframe(st.session_state.df_processed.dtypes.astype(str))
    else:
        st.info("O DataFrame de trabalho está vazio ou não foi processado ainda.")