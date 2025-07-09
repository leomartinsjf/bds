# ok 07/07  sem subtab

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import f_oneway, kruskal
import plotly.express as px
import plotly.graph_objects as go

# Importa a função para o teste post-hoc de Tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from io import StringIO
shap.initjs()

# Modify this line: Add 'df' as an argument
def show_l4_model(df_main):
    st.subheader("🔷 Modelo L4 Estendido - Realismo Crítico")
    st.markdown("Este módulo permite explorar quatro dimensões estruturadas e aplicar análises explicativas, comparativas e preditivas.")

    # Generate a fingerprint of df_main to detect changes in the input data
    df_main_fingerprint = str(df_main.shape) + "|".join(sorted(df_main.columns.tolist()))

    # 🔁 Gerenciamento de df_l4 e l4_scores_calculated na session_state
    # Colunas esperadas para os escores L4
    l4_score_cols = ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]

    # Verifica se df_l4 existe E se ele contém todas as colunas de escores L4
    l4_columns_present_in_session_state_df = all(col in st.session_state.get("df_l4", pd.DataFrame()).columns for col in l4_score_cols)

    # Check if the input df_main has changed from the one that initialized df_l4
    df_main_has_changed = False
    if "df_main_fingerprint" in st.session_state:
        if st.session_state["df_main_fingerprint"] != df_main_fingerprint:
            df_main_has_changed = True
            st.sidebar.write("DEBUG: df_main_fingerprint changed. Resetting df_l4.") # Debug message
    else: # If fingerprint not in session_state, it's a first run, so it's "changed" conceptually
        df_main_has_changed = True


    # Se df_l4 não estiver no session_state OU se as colunas L4 estiverem faltando nele OU se df_main mudou: reinicializa.
    if "df_l4" not in st.session_state or not l4_columns_present_in_session_state_df or df_main_has_changed:
        st.sidebar.write("DEBUG: Reinitializing df_l4 and L4 scores.") # Debug message
        st.session_state["df_l4"] = df_main.copy()
        st.session_state['l4_scores_calculated'] = False
        # Inicializa variáveis selecionadas como listas vazias
        st.session_state['selected_trocas'] = []
        st.session_state['selected_subjetividades'] = []
        st.session_state['selected_relacoes'] = []
        st.session_state['selected_estrutura'] = []
        st.session_state['l4_score_method'] = "Média"
        # Store the current df_main's fingerprint
        st.session_state["df_main_fingerprint"] = df_main_fingerprint
    
    # --- NOVO: Re-avalia 'l4_scores_calculated' com base no conteúdo real de df_l4 o mais cedo possível ---
    # Isso garante que o flag reflita o conteúdo REAL de df_l4 ANTES de qualquer lógica que
    # possa (re)inicializar colunas para NaN com base num flag desatualizado.
    current_df_in_session_state = st.session_state["df_l4"]
    if all(col in current_df_in_session_state.columns for col in l4_score_cols):
        if not current_df_in_session_state[l4_score_cols].isnull().all().all():
            st.session_state['l4_scores_calculated'] = True
        else: # Se as colunas existirem mas forem TODAS NaN, o flag deve ser False
            st.session_state['l4_scores_calculated'] = False
    else: # Se as colunas L4 nem existirem, o flag deve ser False
        st.session_state['l4_scores_calculated'] = False

    # --- DEBUG START (Após re-avaliação inicial de l4_scores_calculated e estado do df_l4) ---
    st.sidebar.write(f"DEBUG (Após Re-avaliação): l4_scores_calculated = {st.session_state.get('l4_scores_calculated', 'N/A')}")
    if "df_l4" in st.session_state and all(col in st.session_state["df_l4"].columns for col in l4_score_cols):
        l4_cols_are_all_nan_initial = st.session_state["df_l4"][l4_score_cols].isnull().all().all()
        st.sidebar.write(f"DEBUG (Após Re-avaliação): L4 columns ALL NaN (in st.session_state['df_l4']) = {l4_cols_are_all_nan_initial}")
    else:
        st.sidebar.write("DEBUG (Após Re-avaliação): Colunas L4 não prontas em st.session_state['df_l4'].")
    # --- DEBUG END ---

    # Sempre carrega df do session_state para consistência.
    df_current_session = st.session_state["df_l4"].copy() 

    # Garante que as colunas de escores L4 existam. 
    # REMOVIDO: A lógica que re-nanava se l4_scores_calculated fosse False.
    # Agora, se o flag for False, é porque a re-avaliação no início confirmou que as colunas são NaN.
    for col in l4_score_cols:
        if col not in df_current_session.columns:
            df_current_session[col] = np.nan # Apenas adiciona se estiver faltando completamente

    # Adiciona um debug para ver o estado de df_current_session após o loop de garantia de colunas.
    if all(col in df_current_session.columns for col in l4_score_cols):
        l4_cols_are_all_nan_after_loop = df_current_session[l4_score_cols].isnull().all().all()
        st.sidebar.write(f"DEBUG (Após Loop Garante Colunas): L4 columns ALL NaN (in df_current_session) = {l4_cols_are_all_nan_after_loop}")
    else:
        st.sidebar.write("DEBUG (Após Loop Garante Colunas): Colunas L4 ainda não prontas em df_current_session.")

    st.session_state["df_l4"] = df_current_session.copy() 

    # Garante que outras chaves do session_state sejam inicializadas após df_l4 ser tratado
    for key in ["selected_trocas", "selected_subjetividades", "selected_relacoes", "selected_estrutura"]:
        if key not in st.session_state:
            st.session_state[key] = []
    if 'l4_score_method' not in st.session_state:
        st.session_state['l4_score_method'] = "Média"


    # 🧩 Pega todas as colunas disponíveis para seleção (sem colunas calculadas)
    all_cols = df_current_session.columns.tolist() # Use df_current_session aqui

    # 🧠 Inicializa as abas do módulo
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Escores L4", "🧪 Testes Estatísticos", "📈 Modelagem",
        "🧬 Clusters", "🔁 Validação", "📌 Interpretação"
    ])

    with tab1:
        st.header("📊 Escores por Dimensão")
        st.info("Para selecionar múltiplas variáveis em qualquer caixa, **clique nos itens desejados dentro do menu suspenso**. O menu permanecerá aberto até você clicar fora dele ou pressionar 'Enter'.")

        # Adiciona um botão de reset das seleções
        if st.button("🔄 Resetar Seleções", key="reset_l4_selections_button"):
            st.session_state['selected_trocas'] = []
            st.session_state['selected_subjetividades'] = []
            st.session_state['selected_relacoes'] = []
            st.session_state['selected_estrutura'] = []
            st.session_state['l4_score_method'] = "Média"
            st.session_state['l4_scores_calculated'] = False # Assegura que o cálculo será refeito
            
            # --- NOVO DEBUG: Verifique o estado da sessão imediatamente antes do rerun ---
            st.sidebar.write("DEBUG: Botão 'Resetar Seleções' clicado.")
            st.sidebar.write(f"DEBUG: st.session_state['selected_trocas'] ANTES rerun: {st.session_state['selected_trocas']}")
            st.sidebar.write(f"DEBUG: st.session_state['l4_scores_calculated'] ANTES rerun: {st.session_state['l4_scores_calculated']}")
            # -------------------------------------------------------------------------
            
            st.rerun() # Força o Streamlit a redesenhar a página

        # Recupera valores padrão da session_state para persistência dos widgets
        default_trocas = st.session_state['selected_trocas']
        default_subjetividades = st.session_state['selected_subjetividades']
        default_relacoes = st.session_state['selected_relacoes']
        default_estrutura = st.session_state['selected_estrutura']
        default_method = st.session_state['l4_score_method']

        # --- NOVO DEBUG: Verifique o valor 'default' sendo passado para o multiselect ---
        st.sidebar.write(f"DEBUG: default_trocas para multiselect: {default_trocas}")
        # -------------------------------------------------------------------------------

        # Aplica os valores default e chaves únicas
        selected_trocas = st.multiselect("🔁 Trocas materiais", options=all_cols, default=default_trocas, key="l4x_trocas_select")
        selected_subjetividades = st.multiselect("🌌 Subjetividades", options=all_cols, default=default_subjetividades, key="l4x_subjetividades_select")
        selected_relacoes = st.multiselect("🤝 Relações interpessoais", options=all_cols, default=default_relacoes, key="l4x_relacoes_select")
        selected_estrutura = st.multiselect("🏛️ Estrutura / Instituições", options=all_cols, default=default_estrutura, key="l4x_estrutura_select")

        method_options = ["Média", "PCA (1º componente)"]
        method_index = method_options.index(default_method) # Encontra o índice do valor padrão

        method = st.radio("Como calcular os escores por dimensão?", method_options, horizontal=True, index=method_index, key="l4x_score_method_radio_tab1")

        # Salva as seleções (atuais) para persistência na próxima execução
        st.session_state['selected_trocas'] = selected_trocas
        st.session_state['selected_subjetividades'] = selected_subjetividades
        st.session_state['selected_relacoes'] = selected_relacoes
        st.session_state['selected_estrutura'] = selected_estrutura
        st.session_state['l4_score_method'] = method

        def compute_score(vars_list, current_df, label):
            if not vars_list: # Adiciona verificação para lista vazia para evitar erro
                st.warning(f"Nenhuma variável selecionada para a dimensão {label}.")
                return pd.Series(np.nan, index=current_df.index)

            # Garante que as variáveis selecionadas existam no dataframe e não sejam todas NaN
            valid_vars = [v for v in vars_list if v in current_df.columns]
            if not valid_vars:
                st.warning(f"Variáveis selecionadas para {label} não encontradas no DataFrame.")
                return pd.Series(np.nan, index=current_df.index)

            df_temp = current_df[valid_vars]

            if method == "Média":
                # Adição: Verifica se todas as variáveis selecionadas para a dimensão são NaN
                if df_temp.isnull().all().all():
                    st.warning(f"Todas as variáveis selecionadas para '{label}' contêm apenas valores nulos. Escores serão NaN.")
                    return pd.Series(np.nan, index=current_df.index)
                return df_temp.mean(axis=1)
            elif method == "PCA (1º componente)":
                df_pca = df_temp.dropna()
                # Verifica se há dados suficientes para PCA
                if df_pca.empty or df_pca.shape[1] == 0:
                    st.warning(f"Não há dados válidos para PCA na dimensão {label}. Retornando NaN.")
                    return pd.Series(np.nan, index=current_df.index)
                if df_pca.shape[1] == 1: # PCA com 1 componente em 1 variável é a própria variável
                    return df_pca.iloc[:, 0]
                try:
                    pca = PCA(n_components=1)
                    scores = pca.fit_transform(df_pca)
                    return pd.Series(scores.flatten(), index=df_pca.index)
                except ValueError as e:
                    st.error(f"Erro ao calcular PCA para {label}: {e}. Verifique se há variância suficiente nos dados selecionados.")
                    return pd.Series(np.nan, index=current_df.index)

        # Adiciona um botão explícito para calcular os escores
        if st.button("Calcular Escores L4", key="calculate_l4_scores_button"):
            # Check if all selected variable lists are non-empty
            if all(len(grupo) > 0 for grupo in [selected_trocas, selected_subjetividades, selected_relacoes, selected_estrutura]):
                # Create a copy of the dataframe from session_state to modify
                df_to_update = st.session_state["df_l4"].copy()

                df_to_update["L4_Trocas"] = compute_score(selected_trocas, df_to_update, "L4_Trocas")
                df_to_update["L4_Subjetividades"] = compute_score(selected_subjetividades, df_to_update, "L4_Subjetividades")
                df_to_update["L4_Relacoes"] = compute_score(selected_relacoes, df_to_update, "L4_Relacoes")
                df_to_update["L4_Estrutura"] = compute_score(selected_estrutura, df_to_update, "L4_Estrutura")

                # Salva o DataFrame ATUALIZADO de volta no session_state
                st.session_state["df_l4"] = df_to_update.copy()
                # Check if any of the L4 score columns actually have non-NaN values
                if not st.session_state['df_l4'][l4_score_cols].isnull().all().all():
                    st.session_state['l4_scores_calculated'] = True # Marca que os escores foram calculados
                    st.success("✅ Escores calculados com sucesso.")
                else:
                    st.session_state['l4_scores_calculated'] = False
                    # Alterado para st.error para maior visibilidade
                    st.error("❌ Os escores foram calculados, mas resultaram em **todos os valores NaN**. Por favor, verifique se as variáveis selecionadas contêm dados numéricos válidos e sem muitos valores ausentes.")

            else:
                st.warning("⚠️ Selecione ao menos uma variável em cada dimensão para calcular os escores L4.")
                # Se não houver variáveis selecionadas, certifique-se de que as colunas L4 não existam ou estejam preenchidas com NaN
                df_to_update = st.session_state["df_l4"].copy()
                for col_name in l4_score_cols:
                    if col_name in df_to_update.columns:
                        df_to_update[col_name] = np.nan
                st.session_state["df_l4"] = df_to_update.copy() # Garante que o estado seja consistente
                st.session_state['l4_scores_calculated'] = False

        # Exibe o dataframe apenas se os escores já foram calculados ou se a página foi recarregada e eles estão no df_l4
        # Garante que as colunas de escore existam antes de tentar acessá-las.
        # Use o flag para exibição
        if st.session_state.get('l4_scores_calculated', False) and \
           all(col in st.session_state['df_l4'].columns for col in l4_score_cols):
            st.dataframe(st.session_state['df_l4'][l4_score_cols].head())
        else:
            st.info("Pressione 'Calcular Escores L4' para ver um preview dos escores.")


    # A partir daqui, todas as abas devem usar a versão mais recente de st.session_state["df_l4"]
    # Re-obtém o df para as próximas abas, garantindo que seja a versão mais atualizada após tab1.
    df = st.session_state["df_l4"].copy()

    # Verificação de pré-requisito para as outras abas
    # This flag should now accurately reflect if there are *non-null* L4 scores

    # --- DEBUG START (Antes da definição de l4_scores_exist) ---
    st.sidebar.write(f"DEBUG (Pre-l4_scores_exist): l4_scores_calculated = {st.session_state.get('l4_scores_calculated', 'N/A')}")
    if all(col in df.columns for col in l4_score_cols):
        l4_cols_are_all_nan_pre_check = df[l4_score_cols].isnull().all().all()
        st.sidebar.write(f"DEBUG (Pre-l4_scores_exist): L4 columns ALL NaN (in local df) = {l4_cols_are_all_nan_pre_check}")
    else:
        st.sidebar.write("DEBUG (Pre-l4_scores_exist): Colunas L4 não prontas no local df.")
    # --- DEBUG END ---

    l4_scores_exist = st.session_state.get('l4_scores_calculated', False) and \
                      all(col in df.columns for col in l4_score_cols) and \
                      not df[l4_score_cols].isnull().all().all()


    with tab2:
        st.header("🧪 Testes Estatísticos por Grupo")

        if not l4_scores_exist:
            st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro.")
        else:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if cat_cols:
                group_col = st.selectbox("Selecione a variável de agrupamento:", cat_cols, key="l4x_group_test")
                test_type = st.radio("Tipo de teste estatístico:", ["ANOVA", "Kruskal-Wallis"], horizontal=True, key="l4x_test_type_radio")

                resultado = []
                for col in l4_score_cols:
                    # Filtra apenas os grupos que têm dados para a dimensão específica
                    # Garante que os dados sejam numéricos e não tenham NaN para o teste
                    data_for_test = df[[col, group_col]].dropna()

                    if data_for_test.empty:
                        stat, p, interpretacao = np.nan, np.nan, "Dados insuficientes para o teste."
                        resultado.append({
                            "Dimensão": col,
                            "Estatística": stat,
                            "p-valor": p,
                            "Interpretação": interpretacao
                        })
                        continue # Pula para a próxima dimensão

                    grupos_validos = [data_for_test[col][data_for_test[group_col] == g].values for g in data_for_test[group_col].unique()]

                    # Ensure there are at least two valid groups for the statistical test
                    grupos_validos = [g for g in grupos_validos if len(g) > 0 and not np.all(np.isnan(g))] # Filter out empty or all-NaN groups
                    if len(grupos_validos) < 2:
                        stat, p, interpretacao = np.nan, np.nan, "Poucos grupos com dados válidos para o teste."
                    else:
                        try:
                            stat, p = f_oneway(*grupos_validos) if test_type == "ANOVA" else kruskal(*grupos_validos)
                            interpretacao = "✅ Diferença significativa (p < 0.05)" if p < 0.05 else "🔸 Sem diferença significativa (p >= 0.05)"
                        except Exception as e:
                            stat, p, interpretacao = np.nan, np.nan, f"Erro: {e}"

                    resultado.append({
                        "Dimensão": col,
                        "Estatística": stat,
                        "p-valor": p,
                        "Interpretação": interpretacao,
                        "P_Value_Raw": p # Guarda o p-valor bruto para a lógica do post-hoc
                    })

                result_df = pd.DataFrame(resultado)
                # Formata o p-valor para 3 casas decimais ANTES de exibir
                result_df['p-valor'] = result_df['p-valor'].round(3)

                st.dataframe(result_df.drop(columns=["P_Value_Raw"])) # Remove a coluna bruta para exibição
                st.download_button("📥 Baixar Tabela de Resultados", result_df.drop(columns=["P_Value_Raw"]).to_csv(index=False).encode("utf-8"),
                                   file_name="l4_testes_estatisticos.csv", mime="text/csv")

                # ADIÇÃO DO TESTE POST HOC
                if test_type == "ANOVA":
                    st.subheader("Pós-Hoc de Tukey (para ANOVA significativa)")
                    post_hoc_performed_any_dim = False # Flag para verificar se ALGUM post-hoc foi realizado
                    for index, row in result_df.iterrows():
                        # Use o P_Value_Raw para a condição, pois ele não está arredondado
                        if row["P_Value_Raw"] is not np.nan and row["P_Value_Raw"] < 0.05: # Apenas se a ANOVA for significativa e p-valor não for NaN
                            col_dim = row["Dimensão"]
                            st.write(f"--- **Resultados Pós-Hoc para: {col_dim}** ---")

                            # Filtra os dados para a dimensão e a coluna de agrupamento
                            data_for_tukey = df[[col_dim, group_col]].dropna()

                            # O Tukey requer pelo menos dois grupos únicos
                            unique_groups = data_for_tukey[group_col].unique()
                            if not data_for_tukey.empty and len(unique_groups) >= 2:
                                try:
                                    # Usa o Tukey HSD
                                    tukey_results = pairwise_tukeyhsd(endog=data_for_tukey[col_dim],
                                                                    groups=data_for_tukey[group_col],
                                                                    alpha=0.05)

                                    # CÓDIGO MELHORADO PARA EXIBIR A TABELA PÓS-HOC
                                    # Acessa os dados da tabela interna do resultado e cria um DataFrame
                                    results_data = tukey_results._results_table.data
                                    header = results_data[0]
                                    data_rows = results_data[1:]
                                    tukey_df = pd.DataFrame(data_rows, columns=header)

                                    # Formata colunas numéricas para 4 casas decimais e a coluna 'reject'
                                    numeric_cols_to_format = ['meandiff', 'lower', 'upper', 'p-adj']
                                    for col_name in numeric_cols_to_format:
                                        if col_name in tukey_df.columns:
                                            tukey_df[col_name] = pd.to_numeric(tukey_df[col_name], errors='coerce').round(4)

                                    if 'reject' in tukey_df.columns:
                                        tukey_df['reject'] = tukey_df['reject'].map({True: 'Sim', False: 'Não', 'True': 'Sim', 'False': 'Não'}, na_action='ignore')

                                    st.dataframe(tukey_df)

                                    post_hoc_performed_any_dim = True # Marca que um post-hoc foi realizado
                                except Exception as e:
                                    st.warning(f"Não foi possível realizar o teste Pós-Hoc de Tukey para {col_dim}: {e}")
                            else:
                                st.info(f"Dados insuficientes para Pós-Hoc de Tukey para {col_dim} (mínimo 2 grupos com dados válidos).")

                    # CORREÇÃO AQUI: A mensagem final agora reflete se *algum* teste post-hoc foi feito.
                    if not post_hoc_performed_any_dim:
                        st.info("Nenhum teste Pós-Hoc de Tukey foi realizado, pois nenhuma ANOVA foi significativa (p >= 0.05), ou dados insuficientes/problemas de cálculo.")

            else:
                st.warning("⚠️ Não há variáveis categóricas para agrupamento.")

    with tab3:
        st.header("📈 Modelagem com os Escores do L4")

        if not l4_scores_exist:
            st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro.")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in l4_score_cols + ["Cluster_L4"]]

            if numeric_cols:
                y_var = st.selectbox("Selecione a variável dependente:", numeric_cols, key="l4x_model_y")
                Xy = df[l4_score_cols + [y_var]].dropna()

                if Xy.empty:
                    st.warning("⚠️ Não há dados suficientes para construir o modelo com as variáveis selecionadas. Verifique valores nulos.")
                else:
                    X = Xy[l4_score_cols]
                    y = Xy[y_var]

                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)

                    coef_df = pd.DataFrame({
                        "Variável": ["Intercepto"] + list(X.columns),
                        "Coeficiente": [model.intercept_] + list(model.coef_)
                    })

                    st.write(f"**R²:** {r2:.4f}")
                    st.dataframe(coef_df)
                    st.download_button("📥 Baixar Coeficientes", coef_df.to_csv(index=False).encode("utf-8"),
                                       file_name="l4_modelo_coeficientes.csv", mime="text/csv")
            else:
                st.warning("⚠️ Não há variáveis numéricas disponíveis para modelagem (além dos escores L4).")

    with tab4:
        st.header("🧬 Agrupamento (Clusters) com os Escores")

        if not l4_scores_exist:
            st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro.")
        else:
            X_cluster = df[l4_score_cols].dropna()

            if X_cluster.empty:
                st.warning("⚠️ Não há dados válidos para aplicar o agrupamento. Verifique valores nulos nos escores L4.")
            else:
                min_samples = 2
                n_clusters_max = min(10, len(X_cluster))

                if n_clusters_max < min_samples:
                    st.warning(f"⚠️ Não há dados suficientes ({len(X_cluster)} amostras) para formar clusters. Mínimo necessário: {min_samples} amostras.")
                else:
                    n_clusters = st.slider("Número de Clusters", min_value=min_samples, max_value=n_clusters_max, value=min(3, n_clusters_max), key="l4x_n_clusters")
                    cluster_method = st.radio("Método de agrupamento:", ["KMeans", "Agglomerative"], horizontal=True, key="l4x_cluster_method_radio")

                    if n_clusters < 2:
                        st.warning("⚠️ Selecione pelo menos 2 clusters.")
                    elif len(X_cluster) < n_clusters:
                         st.warning(f"⚠️ Número de amostras ({len(X_cluster)}) é insuficiente para {n_clusters} clusters. Reduza o número de clusters.")
                    else:
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') if cluster_method == "KMeans" else AgglomerativeClustering(n_clusters=n_clusters)
                        cluster_labels = model.fit_predict(X_cluster)

                        # FIX for Pandas FutureWarning (Line 333)
                        # Ensure 'Cluster_L4' column exists before assignment
                        if "Cluster_L4" not in df.columns:
                            df["Cluster_L4"] = np.nan # Initialize if it doesn't exist

                        df.loc[X_cluster.index, "Cluster_L4"] = pd.Series(cluster_labels.astype(str), index=X_cluster.index, dtype='object')
                        # Original: df.loc[X_cluster.index, "Cluster_L4"] = cluster_labels.astype(str)

                        st.session_state["df_l4"] = df.copy()

                        df_grouped = df.groupby("Cluster_L4")[l4_score_cols].mean().reset_index()
                        fig = go.Figure()

                        for _, row in df_grouped.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[row["L4_Trocas"], row["L4_Subjetividades"], row["L4_Relacoes"], row["L4_Estrutura"], row["L4_Trocas"]],
                                theta=["Trocas", "Subjetividades", "Relações", "Estrutura", "Trocas"],
                                fill='toself',
                                name=f"Cluster {row['Cluster_L4']}"
                            ))

                        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Radar dos Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button("📥 Baixar Dados com Clusters", df.to_csv(index=False).encode("utf-8"),
                                           file_name="l4_clusters.csv", mime="text/csv")

    with tab5:
        st.header("🔁 Validação Cruzada")

        if not l4_scores_exist:
            st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro.")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_options = [col for col in numeric_cols if col not in l4_score_cols + ["Cluster_L4"]]

            if y_options:
                y_var = st.selectbox("Variável dependente para predição:", y_options, key="l4x_cv_y")
                n_folds = st.slider("Número de Folds (K)", min_value=2, max_value=10, value=5, key="l4x_cv_n_folds")

                Xy = df[l4_score_cols + [y_var]].dropna()

                if Xy.empty:
                    st.warning("⚠️ Não há dados suficientes para realizar a validação cruzada. Verifique valores nulos.")
                else:
                    X = Xy[l4_score_cols]
                    y = Xy[y_var]

                    if len(X) < n_folds:
                        st.warning(f"⚠️ O número de amostras ({len(X)}) é menor que o número de folds ({n_folds}). Reduza o número de folds ou forneça mais dados.")
                    else:
                        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        r2_scores, mae_scores = [], []

                        for train_index, test_index in kf.split(X):
                            model = LinearRegression().fit(X.iloc[train_index], y.iloc[train_index])
                            y_pred = model.predict(X.iloc[test_index])
                            r2_scores.append(r2_score(y.iloc[test_index], y_pred))
                            mae_scores.append(mean_absolute_error(y.iloc[test_index], y_pred))

                        results_df = pd.DataFrame({"Fold": list(range(1, n_folds+1)), "R²": r2_scores, "MAE": mae_scores})
                        st.dataframe(results_df)
                        fig = px.line(results_df, x="Fold", y=["R²", "MAE"], markers=True, title="Performance por Fold")
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button("📥 Baixar Resultados da Validação", results_df.to_csv(index=False).encode("utf-8"),
                                           file_name="l4_validacao_cruzada.csv", mime="text/csv")

    with tab6:
        st.header("📌 Interpretação e Classificação Avançada")
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📉 Importância dos Escores", "🧠 Classificação", "🔍 Perfil dos Clusters", "📐 Regressão e Variância"
        ])

        with subtab1:
            st.subheader("📉 Importância das Variáveis nas Dimensões L4")
            method = st.session_state.get('l4_score_method', 'Média')

            if method == "Média":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                score_cols = l4_score_cols
                available_vars = [col for col in numeric_cols if col not in score_cols]

                # Initialize with a default value if not already in session_state
                if "selected_importance_vars" not in st.session_state:
                    st.session_state["selected_importance_vars"] = []

                selected_vars = st.multiselect(
                    "Selecione as variáveis para estimar a importância:",
                    options=available_vars,
                    default=st.session_state["selected_importance_vars"],
                    key="l4x_importance_vars_select" # Adicionei uma key explícita para evitar avisos futuros
                )
                st.session_state["selected_importance_vars"] = selected_vars # Salva a seleção atual

                if selected_vars:
                    # Check if L4 scores exist before attempting correlation
                    if not l4_scores_exist: # This is the line triggering the error
                        st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro para ver a importância das variáveis.")
                    else:
                        for dim in score_cols:
                            if dim in df.columns:
                                st.write(f"### {dim}")
                                # Ensure both the selected variables and the dimension score are not all NaN for correlation
                                temp_df_corr = df[selected_vars + [dim]].dropna()
                                if not temp_df_corr.empty and len(temp_df_corr) > 1:
                                    # Ensure there's variance in the dimension column for correlation
                                    if temp_df_corr[dim].var() > 0:
                                        corr = temp_df_corr[selected_vars].corrwith(temp_df_corr[dim]).dropna().sort_values(key=abs, ascending=False)
                                        if not corr.empty:
                                            fig_corr = px.bar(corr, x=corr.index, y=corr.values, title=f"Correlação com {dim}")
                                            st.plotly_chart(fig_corr, use_container_width=True, key=f"plot_corr_{dim}")
                                        else:
                                            st.info(f"Sem correlações válidas para {dim} com as variáveis selecionadas.")
                                    else:
                                        st.info(f"Variância zero para {dim}. Não é possível calcular a correlação.")
                                else:
                                    st.info(f"Dados insuficientes para calcular a correlação para {dim}.")
                else:
                    st.info("🔎 Selecione ao menos uma variável para análise de importância.")
            else: # PCA method
                st.info("🔎 Para o método PCA, a importância das variáveis é dada pelos pesos dos componentes principais. Esta funcionalidade será incluída em futuras versões.")

        with subtab2:
            st.subheader("🧠 Classificação com Escores L4")
            target_options = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if target_options:
                target = st.selectbox("Selecione a variável categórica para prever:", target_options, key="l4x_classification_target")
            else:
                st.warning("⚠️ Não há variáveis categóricas para classificação.")
                target = None

            if target:
                model_type = st.radio("Modelo:", ["Random Forest", "Logistic Regression"], horizontal=True, key="l4x_classification_model_type")

                required_cols = l4_score_cols
                if all(col in df.columns for col in required_cols + [target]) and l4_scores_exist: # Check l4_scores_exist
                    df_model = df[required_cols + [target]].dropna()
                else:
                    st.warning("⚠️ Os escores L4 não foram calculados ou a variável alvo não está presente. Calcule os escores na aba '📊 Escores L4' primeiro.")
                    df_model = pd.DataFrame() # Ensure df_model is empty to prevent further errors

                if df_model.empty:
                    st.warning("⚠️ Não há dados suficientes para classificação com as variáveis selecionadas. Verifique valores nulos.")
                else:
                    X = df_model[l4_score_cols]
                    y = df_model[target]

                    if len(y.unique()) < 2:
                        st.warning("⚠️ A variável alvo selecionada tem menos de 2 classes únicas. Não é possível realizar a classificação.")
                    else:
                        # Handle cases where X or y might become empty after stratification if a class is too small
                        try:
                            # Corrected typo: train_test_split instead of train_train_split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
                        except ValueError as e:
                            st.warning(f"Não foi possível dividir os dados para treinamento e teste. Erro: {e}. Isso pode ocorrer se alguma classe tiver apenas uma amostra. Considere usar mais dados ou uma variável alvo diferente.")
                            return # Exit the tab logic

                        if model_type == "Random Forest":
                            model = RandomForestClassifier(random_state=42)
                        else:
                            model = LogisticRegression(max_iter=1000)

                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        except Exception as e:
                            st.warning(f"Erro ao treinar ou prever com o modelo: {e}. Verifique os dados e a configuração do modelo.")
                            return

                        st.markdown("### 📝 Relatório de Classificação")
                        report_dict = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose().round(3)
                        st.dataframe(report_df, use_container_width=True)
                        st.download_button("📥 Baixar Relatório de Classificação", report_df.to_csv().encode("utf-8"), file_name="l4_relatorio_classificacao.csv")

                        st.markdown("### 🔀 Matriz de Confusão")
                        cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=model.classes_, columns=model.classes_)
                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="blues", title="Matriz de Confusão")
                        st.plotly_chart(fig_cm, use_container_width=True)

                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            importance_df = pd.DataFrame({
                                "Dimensão L4": X.columns,
                                "Importância": importances
                            }).sort_values("Importância", ascending=False)
                            fig_imp = px.bar(importance_df, x="Dimensão L4", y="Importância", title="Importância das Dimensões (Random Forest)")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        elif hasattr(model, 'coef_'):
                            # For Logistic Regression, coefficients can be interpreted as importance (magnitude)
                            # For multi-class, coef_ is (n_classes, n_features)
                            if model.coef_.ndim > 1 and model.coef_.shape[0] > 1: # Multi-class Logistic Regression
                                st.markdown("#### Coeficientes do Modelo (Regressão Logística)")
                                for i, class_name in enumerate(model.classes_):
                                    st.write(f"**Classe: {class_name}**")
                                    coef_class_df = pd.DataFrame({
                                        "Dimensão L4": X.columns,
                                        "Coeficiente": model.coef_[i]
                                    }).sort_values("Coeficiente", key=abs, ascending=False) # Sort by absolute value
                                    st.dataframe(coef_class_df)
                            else: # Binary Logistic Regression
                                importance_df = pd.DataFrame({
                                    "Dimensão L4": X.columns,
                                    "Coeficiente": model.coef_[0]
                                }).sort_values("Coeficiente", key=abs, ascending=False)
                                fig_imp = px.bar(importance_df, x="Dimensão L4", y="Coeficiente", title="Coeficientes do Modelo (Regressão Logística)")
                                st.plotly_chart(fig_imp, use_container_width=True)


                        if len(y.unique()) == 2:
                            st.markdown("### 🔵 Curva ROC e AUC")
                            y_proba = model.predict_proba(X_test)[:, 1]
                            # Map target unique values to 0 and 1 for roc_curve
                            # Ensure mapping is consistent if y_test contains only one class after splitting (unlikely with stratify)
                            unique_y_test = y_test.unique()
                            if len(unique_y_test) == 2:
                                target_mapping = {unique_y_test[0]: 0, unique_y_test[1]: 1}
                                fpr, tpr, _ = roc_curve(y_test.map(target_mapping), y_proba)
                                roc_auc = auc(fpr, tpr)
                                fig_roc = px.area(x=fpr, y=tpr, title=f"Curva ROC (AUC={roc_auc:.2f})",
                                                 labels=dict(x="FPR", y="TPR"), width=600, height=400)
                                fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                                st.plotly_chart(fig_roc)
                            else:
                                st.info("Não foi possível gerar a Curva ROC: a variável alvo de teste não contém duas classes distintas.")
                        else:
                            st.info("Curva ROC e AUC são aplicáveis apenas para classificação binária.")


                        st.markdown("### 🔁 Validação Cruzada (Acurácia Média)")
                        try:
                            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            scores = []
                            # Check if split generates any folds (can be an issue with very small datasets or imbalanced classes)
                            if len(list(skf.split(X, y))) > 0:
                                for train_idx, test_idx in skf.split(X, y):
                                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                                    scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))
                                st.write(f"**Acurácia média (5 folds)**: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                            else:
                                st.warning("Não foi possível realizar a validação cruzada estratificada. Verifique o tamanho do seu dataset e a distribuição das classes.")
                        except ValueError as e:
                            st.warning(f"Erro na validação cruzada: {e}. Isso pode ocorrer se o número de amostras por classe for menor que n_splits. Tente um número menor de folds ou mais dados.")


                        st.markdown("### 🔎 Explicabilidade com SHAP")
                        try:
                            # Seleciona o explainer com base no tipo de modelo
                            if model_type == "Random Forest":
                                explainer = shap.TreeExplainer(model)
                            else: # Logistic Regression
                                # Para modelos lineares como Regressão Logística, LinearExplainer é eficiente
                                # X_train é usado para inferir a estrutura e escala dos dados.
                                explainer = shap.LinearExplainer(model, X_train)

                            shap_values = explainer.shap_values(X_test)

                            # Garante que X_test tenha os nomes das colunas corretos para o plot SHAP
                            X_test_display = X_test.copy()
                            X_test_display.columns = l4_score_cols

                            # Logic for handling shap_values for multi-class and binary
                            if isinstance(shap_values, list): # Expected for multi-class TreeExplainer or LinearExplainer
                                class_names = list(model.classes_)
                                num_classes = len(class_names)

                                if num_classes > 0: # Ensure there are classes to select
                                    selected_class_idx_ui = st.selectbox(
                                        "Selecione a classe para explicação SHAP:",
                                        list(range(num_classes)),
                                        format_func=lambda i: f"{i} - {class_names[i]}",
                                        key="shap_class_select_list" # Unique key
                                    )

                                    st.markdown(f"#### 🔍 SHAP para classe: {class_names[selected_class_idx_ui]}")
                                    # Ensure shap_values[selected_class_idx_ui] is not None or empty
                                    if shap_values[selected_class_idx_ui] is not None and shap_values[selected_class_idx_ui].size > 0:
                                        fig_shap = shap.summary_plot(shap_values[selected_class_idx_ui], X_test_display, show=False)
                                        fig = plt.gcf()
                                        st.pyplot(fig)
                                        plt.clf()
                                    else:
                                        st.info(f"Nenhum valor SHAP gerado para a classe {class_names[selected_class_idx_ui]}.")
                                else:
                                    st.info("Nenhuma classe disponível para explicação SHAP.")
                            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3: # If it's a 3D numpy array (n_samples, n_features, n_classes)
                                # This path might be less common but included for robustness if explainer outputs this
                                class_names = list(model.classes_)
                                num_classes = len(class_names)

                                if num_classes > 0:
                                    if shap_values.shape[2] != num_classes:
                                        st.warning(f"A dimensão das classes nos valores SHAP ({shap_values.shape[2]}) não corresponde ao número de classes do modelo ({num_classes}). Isso pode causar problemas.")

                                    selected_class_idx_ui = st.selectbox(
                                        "Selecione a classe para explicação SHAP:",
                                        list(range(num_classes)),
                                        format_func=lambda i: f"{i} - {class_names[i]}",
                                        key="shap_class_select_3d_array" # Unique key
                                    )
                                    st.markdown(f"#### 🔍 SHAP para classe: {class_names[selected_class_idx_ui]}")
                                    if shap_values[:, :, selected_class_idx_ui] is not None and shap_values[:, :, selected_class_idx_ui].size > 0:
                                        fig_shap = shap.summary_plot(shap_values[:, :, selected_class_idx_ui], X_test_display, show=False)
                                        fig = plt.gcf()
                                        st.pyplot(fig)
                                        plt.clf()
                                    else:
                                        st.info(f"Nenhum valor SHAP gerado para a classe {class_names[selected_class_idx_ui]}.")
                                else:
                                    st.info("Nenhuma classe disponível para explicação SHAP.")

                            else: # Binary classification or simple regression output (shap_values is 2D array or 1D array)
                                if shap_values is not None and shap_values.size > 0:
                                    fig_shap = shap.summary_plot(shap_values, X_test_display, show=False)
                                    fig = plt.gcf()
                                    st.pyplot(fig)
                                    plt.clf()
                                else:
                                    st.info("Nenhum valor SHAP gerado.")

                        except Exception as e:
                            st.warning(f"Não foi possível gerar explicação SHAP: {e}. Por favor, verifique a seleção do modelo e os dados.")
                            st.write(f"Detalhes do erro: {e}")
                            st.write(f"Shape de X_test: {X_test.shape}")
                            if 'shap_values' in locals():
                                if isinstance(shap_values, list):
                                    st.write(f"Número de arrays de valores SHAP: {len(shap_values)}")
                                    if len(shap_values) > 0:
                                        st.write(f"Shape do primeiro array de valores SHAP: {shap_values[0].shape}")
                                elif hasattr(shap_values, 'ndim'):
                                    st.write(f"Shape de shap_values: {shap_values.shape}")
                                else:
                                    st.write(f"Tipo de shap_values: {type(shap_values)}")
            else:
                st.info("Nenhuma variável categórica disponível para classificação ou dados insuficientes.")

        with subtab3:
            st.subheader("🔍 Perfis Típicos dos Clusters")
            # Check if Cluster_L4 exists and has non-null values
            if "Cluster_L4" in df.columns and not df["Cluster_L4"].isnull().all():
                cluster_summary = df.groupby("Cluster_L4")[l4_score_cols].mean()
                if not cluster_summary.empty:
                    st.dataframe(cluster_summary)

                    profile = cluster_summary.apply(lambda x: ["↑" if v > df[vname].mean() else "↓" for v, vname in zip(x, cluster_summary.columns)], axis=1)
                    profile_df = pd.DataFrame(profile.tolist(), columns=cluster_summary.columns, index=cluster_summary.index)
                    st.markdown("**Sinais Típicos por Cluster (↑ acima da média, ↓ abaixo):**")
                    st.dataframe(profile_df)
                    st.download_button("📥 Baixar Perfis", profile_df.to_csv(index=False).encode("utf-8"), file_name="l4_cluster_perfis.csv")
                else:
                    st.warning("⚠️ O sumário dos clusters está vazio. Verifique os dados de agrupamento.")
            else:
                st.warning("⚠️ Execute o agrupamento na aba 'Clusters' para gerar os perfis típicos.")

        with subtab4:
            st.subheader("📐 Regressão e Variância dos Escores")
            numeric_cols_for_regression = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out L4 scores and Cluster_L4 if they are already in the dataframe as numeric
            numeric_cols_for_regression = [col for col in numeric_cols_for_regression if col not in l4_score_cols + ["Cluster_L4"]]

            if numeric_cols_for_regression:
                y_numeric = st.selectbox("Selecione a variável dependente (numérica):", numeric_cols_for_regression, key="l4x_regression_y_var")

                if y_numeric:
                    if not l4_scores_exist:
                        st.warning("⚠️ Escores L4 não calculados ou incompletos. Calcule-os na aba 'Escores L4' primeiro para a regressão.")
                    else:
                        result_list = []
                        for col in l4_score_cols:
                            temp_df = df[[col, y_numeric]].dropna()
                            if not temp_df.empty and len(temp_df) > 1: # Need at least 2 points for regression
                                # Check for variance in the independent variable
                                if temp_df[col].var() > 0:
                                    X_ = temp_df[[col]]
                                    y_ = temp_df[y_numeric]
                                    model = LinearRegression().fit(X_, y_)
                                    slope = model.coef_[0]
                                    intercept = model.intercept_
                                    r = np.corrcoef(temp_df[col], temp_df[y_numeric])[0, 1]
                                    result_list.append({"Escore": col, "Slope": slope, "Intercepto": intercept, "Correlação": r})
                                else:
                                    result_list.append({"Escore": col, "Slope": np.nan, "Intercepto": np.nan, "Correlação": np.nan, "Info": "Variância zero no escore L4."})
                            else:
                                result_list.append({"Escore": col, "Slope": np.nan, "Intercepto": np.nan, "Correlação": np.nan, "Info": "Dados insuficientes para regressão."})


                        reg_df = pd.DataFrame(result_list).round(3)
                        st.dataframe(reg_df)

                        st.markdown("### 🔍 Dispersão com Reta de Regressão")
                        for row in reg_df.itertuples():
                            if pd.notna(row.Slope): # Only plot if regression was successful (Slope is not NaN)
                                fig = px.scatter(df, x=row.Escore, y=y_numeric, trendline="ols", title=f"{row.Escore} vs {y_numeric}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif hasattr(row, "Info"): # Display specific info if available
                                st.info(f"Dados insuficientes para plotar Regressão de {row.Escore} vs {y_numeric} ({row.Info})")
                            else:
                                st.info(f"Dados insuficientes para plotar Regressão de {row.Escore} vs {y_numeric}")
                else:
                    st.info("Selecione uma variável numérica para a regressão.")
            else:
                st.warning("Não há variáveis numéricas disponíveis para regressão (além dos escores L4 e clusters).")


            # Check if Cluster_L4 exists and has non-null values for variance analysis
            if "Cluster_L4" in df.columns and not df["Cluster_L4"].isnull().all():
                st.markdown("### 🧬 Variância dos Escores por Cluster")
                for col in l4_score_cols:
                    try:
                        # Ensure there are enough valid data points for variance calculation within each cluster
                        data_for_var = df[[col, "Cluster_L4"]].dropna()
                        if not data_for_var.empty and data_for_var["Cluster_L4"].nunique() > 1:
                            # Filter out clusters with only one data point as variance will be NaN
                            valid_clusters = data_for_var.groupby("Cluster_L4").filter(lambda x: len(x) > 1)
                            if not valid_clusters.empty:
                                grupo_var = valid_clusters.groupby("Cluster_L4")[col].var()
                                total_var = data_for_var[col].var()
                                if total_var > 0: # Avoid division by zero
                                    pct_explicada = grupo_var.mean() / total_var * 100
                                    st.write(f"**{col}**: variância média entre clusters = {grupo_var.mean():.2f}, variância total = {total_var:.2f}, explicação ≈ {pct_explicada:.1f}%)")

                                    fig_box = px.box(df, x="Cluster_L4", y=col, points="all", title=f"Boxplot de {col} por Cluster")
                                    st.plotly_chart(fig_box, use_container_width=True)
                                else:
                                    st.info(f"Variância zero para {col}. Não é possível calcular a porcentagem explicada.")
                            else:
                                st.info(f"Nenhum cluster com dados suficientes (mais de 1 amostra) para calcular a variância para {col}.")
                        else:
                            st.info(f"Dados insuficientes ou apenas um cluster para calcular a variância por cluster para {col}.")
                    except Exception as e:
                        st.warning(f"Não foi possível calcular a variância para {col}: {e}")
            else:
                st.info("Execute o agrupamento na aba 'Clusters' para analisar a variância dos escores por cluster.")