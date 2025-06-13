import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime # Importar datetime para timestamps

def col_exists(df, col_name):
    return col_name in df.columns

def show_col_preview(df, col_names):
    if isinstance(col_names, str):
        col_names = [col_names]
    st.write("Prévia das novas colunas:")
    st.dataframe(df[col_names].head())

def apply_op(series, op, val):
    if op == "==":
        return series == val
    elif op == "!=":
        return series != val
    elif op == ">":
        return series > val
    elif op == ">=":
        return series >= val
    elif op == "<":
        return series < val
    elif op == "<=":
        return series <= val
    else:
        return series == val

def convert_val(dtype, val):
    if pd.api.types.is_numeric_dtype(dtype):
        try:
            v = float(val)
            if v == int(v): v = int(v)
            return v
        except Exception:
            return val
    elif pd.api.types.is_bool_dtype(dtype):
        return str(val).lower() == 'true'
    else:
        return val

def show_feature_engineering():
    if st.session_state['df_processed'] is None or st.session_state['df_processed'].empty:
        st.warning("⚠️ Dados não carregados ou pré-processados. Por favor, complete as etapas anteriores.")
        return False

    if 'run_feature_engineering_rerun' not in st.session_state:
        st.session_state['run_feature_engineering_rerun'] = False

    # Inicializa a lista de logs se ainda não existir na session_state
    if 'feature_engineering_logs' not in st.session_state:
        st.session_state['feature_engineering_logs'] = []

    df_current_session_copy = st.session_state['df_processed'].copy()
    st.markdown("---")

    # PAINEL DE REMOÇÃO DE MÚLTIPLAS COLUNAS
    with st.expander("🧹 Remover múltiplas colunas"):
        if len(df_current_session_copy.columns) > 0:
            cols_to_remove = st.multiselect(
                "Selecione as colunas a remover",
                options=df_current_session_copy.columns.tolist(),
                key="fe_remove_multiselect"
            )
            if cols_to_remove:
                if st.button("Remover selecionadas", key="fe_remove_cols_button"):
                    df_current_session_copy.drop(columns=cols_to_remove, inplace=True)
                    st.session_state['df_processed'] = df_current_session_copy
                    log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Colunas removidas: {', '.join(cols_to_remove)}."
                    st.session_state['feature_engineering_logs'].append(log_message)
                    st.success(f"Colunas removidas: {', '.join(cols_to_remove)}")
                    st.dataframe(df_current_session_copy.head())
                    st.rerun()
        else:
            st.info("O DataFrame não possui colunas para remoção.")

    st.markdown("---")

    # Atualiza listas após remoção
    df_current_session_copy = st.session_state['df_processed'].copy()
    num_cols = df_current_session_copy.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_current_session_copy.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = df_current_session_copy.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    all_cols = num_cols + cat_cols + date_cols

    key_prefix = "fe_"
    feature_engineered_flag = False

    # 1. Combinar Variáveis Numéricas
    with st.expander("➕ Combinar Variáveis Numéricas"):
        selected_combine = st.multiselect("Variáveis a combinar (numéricas)", num_cols, key=key_prefix + "combine_vars_select")
        operation = st.selectbox("Operação", ["Soma", "Média"], key=key_prefix + "combine_operation_select")
        new_var_name_combine = st.text_input("Nome da nova variável combinada", value="nova_combinacao", key=key_prefix + "new_combine_var_name_input")
        if st.button("Criar variável combinada", key=key_prefix + "create_combo_button"):
            if not selected_combine or not new_var_name_combine:
                st.warning("Selecione variáveis e forneça um nome.")
            elif col_exists(df_current_session_copy, new_var_name_combine):
                st.error(f"O nome '{new_var_name_combine}' já existe.")
            else:
                try:
                    if operation == "Soma":
                        df_current_session_copy[new_var_name_combine] = df_current_session_copy[selected_combine].sum(axis=1)
                    else:
                        df_current_session_copy[new_var_name_combine] = df_current_session_copy[selected_combine].mean(axis=1)
                    st.session_state.df_processed = df_current_session_copy
                    log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Variável '{new_var_name_combine}' criada pela combinação de {', '.join(selected_combine)} usando '{operation}'."
                    st.session_state['feature_engineering_logs'].append(log_message)
                    st.success(f"Variável '{new_var_name_combine}' criada.")
                    show_col_preview(df_current_session_copy, new_var_name_combine)
                    feature_engineered_flag = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.markdown("---")

    # 2. Criar Variáveis Dummies
    with st.expander("🏷️ Criar Variáveis Dummies"):
        if cat_cols:
            selected_cat_dummy = st.selectbox("Variável categórica", cat_cols, key=key_prefix + "catdummy_select")
            drop_first_dummy = st.checkbox("Remover primeira categoria (drop_first)", value=True, key=key_prefix + "dropfirst_checkbox")
            if st.button("Criar dummies", key=key_prefix + "createdummies_button"):
                if selected_cat_dummy:
                    try:
                        dummies = pd.get_dummies(df_current_session_copy[selected_cat_dummy], prefix=selected_cat_dummy, drop_first=drop_first_dummy, dtype=int)
                        existing_dummy_cols = [col for col in dummies.columns if col_exists(df_current_session_copy, col)]
                        if existing_dummy_cols:
                            st.error(f"Colunas dummy já existem: {', '.join(existing_dummy_cols)}.")
                        else:
                            df_current_session_copy = pd.concat([df_current_session_copy, dummies], axis=1)
                            st.session_state["df_processed"] = df_current_session_copy
                            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dummies criadas para a variável '{selected_cat_dummy}'. Nova(s) coluna(s): {', '.join(dummies.columns.tolist())}."
                            st.session_state['feature_engineering_logs'].append(log_message)
                            st.success(f"Dummies para '{selected_cat_dummy}' criadas.")
                            show_col_preview(df_current_session_copy, dummies.columns.tolist())
                            feature_engineered_flag = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao criar dummies: {e}")
                else:
                    st.warning("Selecione uma variável.")
        else:
            st.info("Nenhuma variável categórica disponível.")

    st.markdown("---")

    # 3. Criar Variável Binária (com operadores)
    with st.expander("🔀 Criar Variável Binária"):
        if all_cols:
            bin_var = st.selectbox("Variável de referência", all_cols, key=key_prefix + "binvar_select")
            if bin_var in df_current_session_copy.columns:
                unique_vals = df_current_session_copy[bin_var].dropna().unique().tolist()
                if len(unique_vals) > 0:
                    op = st.selectbox("Operação de comparação", ["==", "!=", ">", ">=", "<", "<="], key=key_prefix + "bin_op")
                    val_pos = st.selectbox("Valor para comparação", unique_vals, key=key_prefix + "binpos_select")
                    bin_name_create = st.text_input("Nome da nova variável binária", key=key_prefix + "binnamebc_input")
                    if st.button("Criar variável binária", key=key_prefix + "create_bin_button"):
                        if bin_name_create:
                            if col_exists(df_current_session_copy, bin_name_create):
                                st.error(f"O nome '{bin_name_create}' já existe.")
                            else:
                                try:
                                    col_data = df_current_session_copy[bin_var]
                                    val_compare = convert_val(col_data.dtype, val_pos)
                                    mask = apply_op(col_data, op, val_compare)
                                    df_current_session_copy[bin_name_create] = mask.astype(int)
                                    st.session_state["df_processed"] = df_current_session_copy
                                    log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Variável binária '{bin_name_create}' criada a partir de '{bin_var}' com condição '{op} {val_pos}'."
                                    st.session_state['feature_engineering_logs'].append(log_message)
                                    st.success(f"Variável '{bin_name_create}' criada.")
                                    show_col_preview(df_current_session_copy, bin_name_create)
                                    feature_engineered_flag = True
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erro ao criar variável binária: {e}")
                        else:
                            st.warning("Forneça um nome.")
                else:
                    st.info("A variável não tem valores disponíveis para criar uma binária.")
            else:
                st.info("A variável de referência selecionada não está mais presente no DataFrame.")
        else:
            st.info("Nenhuma variável adequada para criar binária.")

    st.markdown("---")

    # 4. Criar variável filtrada por valor (condição única) (com operadores)
    with st.expander("📌 Criar variável filtrada por valor (condição única)"):
        if all_cols:
            filter_col = st.selectbox("Variável para filtrar", all_cols, key=key_prefix + "filtercol_select")
            op = st.selectbox("Operação de comparação", ["==", "!=", ">", ">=", "<", "<="], key=key_prefix + "filter_op")
            filter_value_single = st.text_input("Valor para comparar (ex: 'Sim', 1.0)", value="", key=key_prefix + "filterval_single_input")
            new_filtered_name_single = st.text_input("Nome da nova variável filtrada", key=key_prefix + "filternewvar_single_input")
            if st.button("Criar variável filtrada (única)", key=key_prefix + "create_filtered_single_button"):
                if filter_col and filter_value_single and new_filtered_name_single:
                    if col_exists(df_current_session_copy, new_filtered_name_single):
                        st.error(f"O nome '{new_filtered_name_single}' já existe.")
                    else:
                        try:
                            col_data = df_current_session_copy[filter_col]
                            val_compare = convert_val(col_data.dtype, filter_value_single)
                            mask = apply_op(col_data, op, val_compare)
                            df_current_session_copy[new_filtered_name_single] = np.where(mask, col_data, np.nan)
                            st.session_state["df_processed"] = df_current_session_copy
                            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Variável '{new_filtered_name_single}' criada por filtragem de '{filter_col}' com condição '{op} {val_compare}'."
                            st.session_state['feature_engineering_logs'].append(log_message)
                            st.success(f"Variável '{new_filtered_name_single}' criada com base em {filter_col} {op} {val_compare}.")
                            show_col_preview(df_current_session_copy, new_filtered_name_single)
                            feature_engineered_flag = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ocorreu um erro ao aplicar a filtragem: {e}")
                else:
                    st.warning("Preencha todos os campos para a filtragem única.")
        else:
            st.info("Nenhuma variável disponível para filtragem única.")

    st.markdown("---")

    # 5. Filtrar e Criar Variável com Até Três Condições (cada uma com operador)
    with st.expander("🔬 Filtrar e Criar Variável com Até Três Condições"):
        st.subheader("Criar Variável Baseada em Múltiplas Condições")
        st.info("Cria uma nova variável que preserva os valores de uma coluna de referência somente quando uma, duas ou até três condições (comparadores) são atendidas.")
        if all_cols:
            col_ref_cond_multi = st.selectbox("Variável de Referência (valores a preservar):", all_cols, key=key_prefix + "ref_col_cond_multi_select")

            st.markdown("##### Primeira Condição")
            col_cond1_multi = st.selectbox("Coluna para a 1ª Condição:", all_cols, key=key_prefix + "col_cond1_multi_select")
            op_cond1 = st.selectbox("Operação de comparação 1ª condição", ["==", "!=", ">", ">=", "<", "<="], key=key_prefix + "op_cond1")
            value_cond1_multi = st.text_input("Valor da 1ª Condição:", value="", key=key_prefix + "val_cond1_multi_select")

            st.markdown("##### Segunda Condição (Opcional)")
            add_cond2_multi = st.checkbox("Adicionar Segunda Condição?", key=key_prefix + "add_cond2_multi_checkbox")
            col_cond2_multi = None
            op_cond2 = None
            value_cond2_multi = ""
            if add_cond2_multi:
                col_cond2_multi = st.selectbox("Coluna para a 2ª Condição:", all_cols, key=key_prefix + "col_cond2_multi_select")
                op_cond2 = st.selectbox("Operação de comparação 2ª condição", ["==", "!=", ">", ">=", "<", "<="], key=key_prefix + "op_cond2")
                value_cond2_multi = st.text_input("Valor da 2ª Condição:", value="", key=key_prefix + "val_cond2_multi_select")

            st.markdown("##### Terceira Condição (Opcional)")
            add_cond3_multi = st.checkbox("Adicionar Terceira Condição?", key=key_prefix + "add_cond3_multi_checkbox")
            col_cond3_multi = None
            op_cond3 = None
            value_cond3_multi = ""
            if add_cond3_multi:
                col_cond3_multi = st.selectbox("Coluna para a 3ª Condição:", all_cols, key=key_prefix + "col_cond3_multi_select")
                op_cond3 = st.selectbox("Operação de comparação 3ª condição", ["==", "!=", ">", ">=", "<", "<="], key=key_prefix + "op_cond3")
                value_cond3_multi = st.text_input("Valor da 3ª Condição:", value="", key=key_prefix + "val_cond3_multi_select")

            new_filtered_name_multi_level = st.text_input(
                "Nome da Nova Variável Filtrada:",
                value=f"{col_ref_cond_multi}_filtered_multi" if col_ref_cond_multi else "",
                key=key_prefix + "new_name_multi_level_input"
            )

            if st.button("Aplicar Filtragem de Múltiplas Condições", key=key_prefix + "apply_multi_level_filter_button"):
                if not col_ref_cond_multi or not col_cond1_multi or value_cond1_multi == "" or not new_filtered_name_multi_level:
                    st.warning("Preencha a variável de referência, a primeira condição e o nome da nova variável.")
                elif add_cond2_multi and (not col_cond2_multi or value_cond2_multi == ""):
                    st.warning("Se a segunda condição estiver marcada, selecione a coluna e o valor para ela.")
                elif add_cond3_multi and (not col_cond3_multi or value_cond3_multi == ""):
                    st.warning("Se a terceira condição estiver marcada, selecione a coluna e o valor para ela.")
                elif col_exists(df_current_session_copy, new_filtered_name_multi_level):
                    st.error(f"O nome '{new_filtered_name_multi_level}' já existe.")
                else:
                    try:
                        v1 = convert_val(df_current_session_copy[col_cond1_multi].dtype, value_cond1_multi)
                        final_condition = apply_op(df_current_session_copy[col_cond1_multi], op_cond1, v1)
                        condition_description = f"'{col_cond1_multi}' {op_cond1} '{value_cond1_multi}'"
                        if add_cond2_multi:
                            v2 = convert_val(df_current_session_copy[col_cond2_multi].dtype, value_cond2_multi)
                            final_condition = final_condition & apply_op(df_current_session_copy[col_cond2_multi], op_cond2, v2)
                            condition_description += f" AND '{col_cond2_multi}' {op_cond2} '{value_cond2_multi}'"
                        if add_cond3_multi:
                            v3 = convert_val(df_current_session_copy[col_cond3_multi].dtype, value_cond3_multi)
                            final_condition = final_condition & apply_op(df_current_session_copy[col_cond3_multi], op_cond3, v3)
                            condition_description += f" AND '{col_cond3_multi}' {op_cond3} '{value_cond3_multi}'"
                        df_current_session_copy[new_filtered_name_multi_level] = np.where(
                            final_condition,
                            df_current_session_copy[col_ref_cond_multi],
                            np.nan
                        )
                        st.session_state["df_processed"] = df_current_session_copy
                        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Variável '{new_filtered_name_multi_level}' criada por filtragem de '{col_ref_cond_multi}' com múltiplas condições: {condition_description}."
                        st.session_state['feature_engineering_logs'].append(log_message)
                        st.success(f"Variável '{new_filtered_name_multi_level}' criada com base em múltiplas condições.")
                        show_col_preview(df_current_session_copy, new_filtered_name_multi_level)
                        feature_engineered_flag = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ocorreu um erro ao aplicar a filtragem: {e}")
        else:
            st.info("Nenhuma variável disponível para filtragem condicional de múltiplos níveis.")

    st.markdown("---")

    # 6. Transformações Matemáticas
    with st.expander("🧮 Transformações Matemáticas"):
        if num_cols:
            math_var = st.selectbox("Variável numérica", num_cols, key=key_prefix + "mathvar_select")
            transform_type = st.selectbox("Tipo de transformação", ["Log", "Quadrado", "Raiz quadrada", "Z-score"], key=key_prefix + "transform_type_select")
            if st.button("Aplicar transformação", key=key_prefix + "applytransform_button"):
                try:
                    new_math_col_name = f"{math_var}_{transform_type.lower().replace(' ', '_')}"
                    if col_exists(df_current_session_copy, new_math_col_name):
                        st.error(f"O nome '{new_math_col_name}' já existe.")
                    else:
                        if transform_type == "Log":
                            if (df_current_session_copy[math_var] < 0).any():
                                st.error("Log requer valores não-negativos.")
                            else:
                                df_current_session_copy[new_math_col_name] = np.log1p(df_current_session_copy[math_var].clip(lower=0))
                        elif transform_type == "Quadrado":
                            df_current_session_copy[new_math_col_name] = df_current_session_copy[math_var] ** 2
                        elif transform_type == "Raiz quadrada":
                            if (df_current_session_copy[math_var] < 0).any():
                                st.error("Raiz quadrada requer valores não-negativos.")
                            else:
                                df_current_session_copy[new_math_col_name] = np.sqrt(df_current_session_copy[math_var].clip(lower=0))
                        elif transform_type == "Z-score":
                            std = df_current_session_copy[math_var].std()
                            df_current_session_copy[new_math_col_name] = 0 if std == 0 else (df_current_session_copy[math_var] - df_current_session_copy[math_var].mean()) / std
                        st.session_state["df_processed"] = df_current_session_copy
                        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Transformação '{transform_type}' aplicada na variável '{math_var}'. Nova coluna: '{new_math_col_name}'."
                        st.session_state['feature_engineering_logs'].append(log_message)
                        st.success(f"Transformação '{transform_type}' aplicada. Nova coluna: '{new_math_col_name}'.")
                        show_col_preview(df_current_session_copy, new_math_col_name)
                        feature_engineered_flag = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
        else:
            st.info("Nenhuma variável numérica disponível.")

    st.markdown("---")

    # 7. Inverter Escala de Variáveis Likert
    with st.expander("🔁 Inverter Escala de Variáveis Likert"):
        if num_cols:
            likert_var = st.selectbox("Variável Likert a inverter", num_cols, key=key_prefix + "likertvar_select")
            max_val = st.number_input("Valor máximo da escala Likert", min_value=1, value=5, key=key_prefix + "likertmax_input")
            new_name_likert = st.text_input("Nome da variável invertida", value=f"{likert_var}_inv", key=key_prefix + "likertname_input")
            if st.button("Inverter variável Likert", key=key_prefix + "invertlikert_button"):
                if col_exists(df_current_session_copy, new_name_likert):
                    st.error(f"O nome '{new_name_likert}' já existe.")
                else:
                    try:
                        df_current_session_copy[new_name_likert] = max_val + 1 - df_current_session_copy[likert_var]
                        st.session_state["df_processed"] = df_current_session_copy
                        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Escala da variável Likert '{likert_var}' invertida para '{new_name_likert}' (Max Val: {max_val})."
                        st.session_state['feature_engineering_logs'].append(log_message)
                        st.success(f"Variável '{new_name_likert}' criada.")
                        show_col_preview(df_current_session_copy, new_name_likert)
                        feature_engineered_flag = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao inverter variável Likert: {e}")
        else:
            st.info("Nenhuma variável numérica disponível.")

    st.markdown("---")

    # 8. Interações entre Variáveis
    with st.expander("✖️ Criar Interações entre Variáveis"):
        interaction_vars = st.multiselect("Selecione duas variáveis numéricas", num_cols, key=key_prefix + "interactvars_multiselect")
        if st.button("Criar interação", key=key_prefix + "create_interaction_button"):
            if len(interaction_vars) == 2:
                new_interaction_name = f"{interaction_vars[0]}_x_{interaction_vars[1]}"
                if col_exists(df_current_session_copy, new_interaction_name):
                    st.error(f"A coluna '{new_interaction_name}' já existe.")
                else:
                    try:
                        df_current_session_copy[new_interaction_name] = df_current_session_copy[interaction_vars[0]] * df_current_session_copy[interaction_vars[1]]
                        st.session_state["df_processed"] = df_current_session_copy
                        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Interação criada entre '{interaction_vars[0]}' e '{interaction_vars[1]}'. Nova coluna: '{new_interaction_name}'."
                        st.session_state['feature_engineering_logs'].append(log_message)
                        st.success(f"Interação '{new_interaction_name}' criada.")
                        show_col_preview(df_current_session_copy, new_interaction_name)
                        feature_engineered_flag = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao criar interação: {e}")
            else:
                st.warning("Selecione exatamente duas variáveis.")

    st.markdown("---")

    # 9. Discretização de Variáveis (pd.qcut)
    with st.expander("📊 Discretizar Variável Numérica (Quantis)"):
        if num_cols:
            var_to_bin_qcut = st.selectbox("Variável a discretizar", num_cols, key=key_prefix + "discretizevar_qcut_select")
            bins_qcut = st.number_input("Número de bins", min_value=2, max_value=20, value=5, key=key_prefix + "bins_qcut_input")
            new_bin_name_qcut = st.text_input("Nome da variável discretizada", value=f"{var_to_bin_qcut}_binned_qcut", key=key_prefix + "binnamed_qcut_input")
            if st.button("Discretizar variável (Quantis)", key=key_prefix + "apply_discretize_qcut_button"):
                if col_exists(df_current_session_copy, new_bin_name_qcut):
                    st.error(f"O nome '{new_bin_name_qcut}' já existe.")
                else:
                    try:
                        temp_series_no_nan = df_current_session_copy[var_to_bin_qcut].dropna()
                        if not temp_series_no_nan.empty:
                            binned_data = pd.qcut(temp_series_no_nan, q=int(bins_qcut), duplicates='drop')
                            df_current_session_copy.loc[binned_data.index, new_bin_name_qcut] = binned_data
                            st.session_state["df_processed"] = df_current_session_copy
                            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Variável '{var_to_bin_qcut}' discretizada em {int(bins_qcut)} quantis. Nova coluna: '{new_bin_name_qcut}'."
                            st.session_state['feature_engineering_logs'].append(log_message)
                            st.success(f"Variável '{new_bin_name_qcut}' criada por quantis.")
                            show_col_preview(df_current_session_copy, new_bin_name_qcut)
                            feature_engineered_flag = True
                            st.rerun()
                        else:
                            st.warning(f"Coluna '{var_to_bin_qcut}' contém apenas valores nulos ou insuficientes para discretização por quantis.")
                    except Exception as e:
                        st.error(f"Erro ao discretizar por quantis: {e}")
        else:
            st.info("Nenhuma variável numérica disponível.")

    st.markdown("---")

    # 10. Redução de Dimensionalidade - PCA
    with st.expander("📉 Redução de Dimensionalidade (PCA)"):
        if num_cols:
            pca_vars = st.multiselect("Variáveis para PCA", num_cols, key=key_prefix + "pcavars_multiselect")
            n_components_pca = 0
            if len(pca_vars) == 0:
                st.info("Selecione pelo menos uma variável para PCA.")
                n_components_pca = 0
            elif len(pca_vars) == 1:
                st.info(f"Apenas uma variável selecionada ('{pca_vars[0]}'). O número de componentes será 1.")
                n_components_pca = 1
            else:
                max_pca_components = len(pca_vars)
                n_components_pca = st.slider("Número de componentes", 1, max_pca_components, value=min(2, max_pca_components), key=key_prefix + "pca_comp_slider")
            pca_var_name_base = st.text_input("Nome base para componentes PCA", value="PCA_Comp", key=key_prefix + "pcavarname_input")
            if st.button("Aplicar PCA", key=key_prefix + "apply_pca_button"):
                if n_components_pca == 0:
                    st.warning("Nenhuma variável selecionada ou número de componentes inválido para PCA.")
                elif len(pca_vars) < n_components_pca:
                    st.warning("O número de componentes não pode ser maior que o número de variáveis selecionadas.")
                else:
                    try:
                        df_pca_input = df_current_session_copy[pca_vars].dropna()
                        if df_pca_input.empty:
                            st.error("Não há dados completos (sem NaNs) nas colunas selecionadas para PCA. Por favor, trate os valores ausentes primeiro.")
                        else:
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(df_pca_input)
                            pca = PCA(n_components=n_components_pca)
                            components = pca.fit_transform(scaled_data)
                            existing_pca_comp_cols = []
                            for i in range(n_components_pca):
                                comp_name = f"{pca_var_name_base}_comp{i+1}"
                                if col_exists(df_current_session_copy, comp_name):
                                    existing_pca_comp_cols.append(comp_name)
                            if existing_pca_comp_cols:
                                st.error(f"Algumas colunas de componentes PCA já existem: {', '.join(existing_pca_comp_cols)}.")
                            else:
                                created_cols = []
                                for i in range(n_components_pca):
                                    comp_name = f"{pca_var_name_base}_comp{i+1}"
                                    df_current_session_copy.loc[df_pca_input.index, comp_name] = components[:, i]
                                    created_cols.append(comp_name)
                                st.session_state["df_processed"] = df_current_session_copy
                                log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PCA aplicado nas variáveis {', '.join(pca_vars)}. Criado(s) {n_components_pca} componente(s): {', '.join(created_cols)}."
                                st.session_state['feature_engineering_logs'].append(log_message)
                                st.success(f"PCA aplicado e {n_components_pca} componentes criados.")
                                st.write(f"Variância Explicada por Componente: {pca.explained_variance_ratio_}")
                                st.write(f"Variância Total Explicada: {pca.explained_variance_ratio_.sum():.2f}")
                                show_col_preview(df_current_session_copy, created_cols)
                                feature_engineered_flag = True
                                st.rerun()
                    except Exception as e:
                        st.error(f"Erro na PCA: {e}")
        else:
            st.info("Nenhuma variável numérica disponível.")

    st.markdown("---")

    # 11. Transformar Numéricas/Binárias em Categóricas Nomeadas
    with st.expander("🏷️ Transformar Numéricas/Binárias em Categóricas Nomeadas"):
        st.subheader("Criar Variável Categórica a partir de Numérica/Binária")
        st.info("Permite converter colunas numéricas (incluindo binárias 0/1) com poucos valores únicos em novas colunas categóricas com nomes personalizados para cada valor.")
        candidate_cols_for_naming = []
        for col in df_current_session_copy.columns:
            if pd.api.types.is_numeric_dtype(df_current_session_copy[col]):
                if df_current_session_copy[col].nunique() <= 10 or (df_current_session_copy[col].isin([0, 1]).all() and df_current_session_copy[col].nunique() <= 2):
                    candidate_cols_for_naming.append(col)
            elif pd.api.types.is_bool_dtype(df_current_session_copy[col]):
                candidate_cols_for_naming.append(col)
        if not candidate_cols_for_naming:
            st.info("Nenhuma coluna numérica ou binária adequada encontrada para transformar em categórica nomeada (espera-se 0/1 ou poucos valores únicos).")
        else:
            selected_col_for_naming = st.selectbox("Selecione a coluna para transformar:", options=candidate_cols_for_naming, key=key_prefix + "transform_to_cat_col_select")
            if selected_col_for_naming:
                st.write(f"Valores únicos na coluna '{selected_col_for_naming}': {df_current_session_copy[selected_col_for_naming].dropna().unique().tolist()}")
                unique_values_to_map = df_current_session_copy[selected_col_for_naming].dropna().unique().tolist()
                unique_values_to_map.sort()
                st.markdown("#### Mapeamento de Valores para Nova Categoria")
                mapping = {}
                new_col_name_for_cat = st.text_input("Nome da Nova Coluna Categórica:", value=f"{selected_col_for_naming}_cat", key=key_prefix + "new_categorical_col_name_input")
                if new_col_name_for_cat and col_exists(df_current_session_copy, new_col_name_for_cat):
                    st.warning(f"O nome '{new_col_name_for_cat}' já existe.")
                cols_map = st.columns(2)
                for i, val in enumerate(unique_values_to_map):
                    with cols_map[i % 2]:
                        new_category_name = st.text_input(f"Mapear '{val}' para:", key=f"{key_prefix}map_{selected_col_for_naming}_{str(val).replace('.', '_').replace('-', '_')}")
                        if new_category_name:
                            mapping[val] = new_category_name
                if st.button("Aplicar Transformação Categórica", key=key_prefix + "apply_categorical_transform_button"):
                    if not new_col_name_for_cat:
                        st.error("Por favor, forneça um nome para a nova coluna categórica.")
                    elif col_exists(df_current_session_copy, new_col_name_for_cat):
                        st.error(f"O nome '{new_col_name_for_cat}' já existe.")
                    elif len(mapping) != len(unique_values_to_map) or any(not v for v in mapping.values()):
                        st.error("Forneça um nome categórico para *todos* os valores únicos da coluna selecionada.")
                    else:
                        try:
                            df_current_session_copy[new_col_name_for_cat] = df_current_session_copy[selected_col_for_naming].map(mapping).astype('category')
                            st.session_state.df_processed = df_current_session_copy
                            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coluna '{selected_col_for_naming}' transformada para a nova coluna categórica '{new_col_name_for_cat}' com mapeamento {mapping}."
                            st.session_state['feature_engineering_logs'].append(log_message)
                            st.success(f"Coluna '{selected_col_for_naming}' transformada para a nova coluna categórica '{new_col_name_for_cat}' com sucesso!")
                            show_col_preview(df_current_session_copy, new_col_name_for_cat)
                            if st.checkbox(f"Remover a coluna original '{selected_col_for_naming}' após a transformação?", key=key_prefix + "remove_original_col_checkbox_final"):
                                df_current_session_copy.drop(columns=[selected_col_for_naming], inplace=True)
                                log_message_remove = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coluna original '{selected_col_for_naming}' removida após transformação categórica."
                                st.session_state['feature_engineering_logs'].append(log_message_remove)
                                st.session_state.df_processed = df_current_session_copy
                                st.info(f"Coluna original '{selected_col_for_naming}' removida.")
                            feature_engineered_flag = True
                            st.session_state['run_feature_engineering_rerun'] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ocorreu um erro ao aplicar a transformação: {e}")

    st.markdown("---")

    # 12. Extrair Componentes Temporais de Colunas de Data/Hora
    with st.expander("⏰ Extrair Componentes Temporais"):
        st.info("Crie novas colunas numéricas a partir de componentes de data/hora (Ano, Mês, Dia, Hora, Minuto, Segundo).")
        if date_cols:
            selected_date_col = st.selectbox("Selecione uma coluna de data/hora:", options=date_cols, key=key_prefix + "date_col_select")
            if selected_date_col:
                temporal_features = st.multiselect("Selecione os componentes a extrair:", options=["Ano", "Mês", "Dia", "Dia da Semana", "Hora", "Minuto", "Segundo"], key=key_prefix + "temporal_features_multiselect")
                if st.button("Extrair Componentes Temporais", key=key_prefix + "extract_temporal_button"):
                    if not temporal_features:
                        st.warning("Selecione pelo menos um componente temporal para extrair.")
                    else:
                        any_new_col_created = False
                        created_temp_cols = []
                        try:
                            df_current_session_copy[selected_date_col] = pd.to_datetime(df_current_session_copy[selected_date_col], errors='coerce')
                            for feature in temporal_features:
                                new_col_name = f"{selected_date_col}_{feature.lower().replace(' ', '_')}"
                                if col_exists(df_current_session_copy, new_col_name):
                                    st.info(f"Coluna '{new_col_name}' já existe e será ignorada.")
                                    continue
                                if feature == "Ano":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.year
                                elif feature == "Mês":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.month
                                elif feature == "Dia":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.day
                                elif feature == "Dia da Semana":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.dayofweek
                                elif feature == "Hora":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.hour
                                elif feature == "Minuto":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.minute
                                elif feature == "Segundo":
                                    df_current_session_copy[new_col_name] = df_current_session_copy[selected_date_col].dt.second
                                any_new_col_created = True
                                created_temp_cols.append(new_col_name)
                            if any_new_col_created:
                                st.session_state.df_processed = df_current_session_copy
                                log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Componentes temporais {', '.join(temporal_features)} extraídos de '{selected_date_col}'. Nova(s) coluna(s): {', '.join(created_temp_cols)}."
                                st.session_state['feature_engineering_logs'].append(log_message)
                                st.success(f"Componentes temporais extraídos de '{selected_date_col}'.")
                                show_col_preview(df_current_session_copy, created_temp_cols)
                                feature_engineered_flag = True
                                st.rerun()
                            else:
                                st.info("Nenhuma nova coluna temporal foi criada (verifique as seleções ou se já existem).")
                        except Exception as e:
                            st.error(f"Erro ao transformar variáveis temporais: {e}")
            else:
                st.info("Selecione uma coluna de data/hora.")
        else:
            st.info("Nenhuma coluna de data/hora disponível.")

    st.markdown("---")

    # PAINEL DE LOGS DE OPERAÇÕES
    with st.expander("📝 Histórico de Operações de Feature Engineering"):
        if st.session_state['feature_engineering_logs']:
            # Botão para salvar o log
            log_content = "\n".join(st.session_state['feature_engineering_logs'])
            st.download_button(
                label="Salvar Log (.txt)",
                data=log_content,
                file_name=f"feature_engineering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_feature_engineering_log"
            )
            st.markdown("---") # Separador para o botão
            for log in reversed(st.session_state['feature_engineering_logs']): # Mostrar os mais recentes primeiro
                st.write(log)
        else:
            st.info("Nenhuma operação de engenharia de fatores registrada ainda.")

    st.markdown("---")

    st.subheader("Prévia do DataFrame Processado Atualmente")
    st.dataframe(st.session_state.df_processed.head())
    st.write(f"Dimensões: {st.session_state['df_processed'].shape[0]} linhas, {st.session_state['df_processed'].shape[1]} colunas.")

    st.info("As alterações são salvas automaticamente no DataFrame da sessão após cada aplicação bem-sucedida.")

    if feature_engineered_flag:
        st.info("As mudanças foram aplicadas. O DataFrame foi atualizado. Você pode continuar a engenharia de fatores ou prosseguir para a próxima etapa.")
    if st.session_state['run_feature_engineering_rerun']:
        st.session_state['run_feature_engineering_rerun'] = False

    return True