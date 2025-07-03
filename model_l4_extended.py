# model_l4_extended.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import f_oneway, kruskal
import plotly.express as px
import plotly.graph_objects as go

# Importa a função para o teste post-hoc de Tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def show_l4_model():
    st.subheader("🔷 Modelo L4 Estendido - Realismo Crítico")
    st.markdown("Este módulo permite explorar quatro dimensões estruturadas e aplicar análises explicativas, comparativas e preditivas.")

    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.warning("⚠️ Nenhum dado processado encontrado. Por favor, carregue e trate os dados na aba de pré-processamento.")
        return

    # Inicializa 'df_l4' a partir de 'df_processed' SE AINDA NÃO EXISTIR
    if 'df_l4' not in st.session_state or st.session_state['df_l4'] is None:
        st.session_state['df_l4'] = st.session_state['df_processed'].copy()
    
    # Sempre trabalhe com uma CÓPIA do df_l4 da session_state
    df = st.session_state['df_l4'].copy()

    # Define all_cols AQUI para que ela inclua TODAS as colunas atuais do df_processed, para as seleções.
    # Isso garante que as opções de seleção não sejam influenciadas pelas novas colunas L4 ou clusters.
    all_cols = st.session_state['df_processed'].columns.tolist() 

    # Inicializa o flag para saber se os escores L4 foram calculados.
    if 'l4_scores_calculated' not in st.session_state:
        st.session_state['l4_scores_calculated'] = False

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Escores L4", "🧪 Testes Estatísticos", "📈 Modelagem", "🧬 Clusters", "🔁 Validação"
    ])

    with tab1:
        st.header("📊 Escores por Dimensão")
        st.info("Para selecionar múltiplas variáveis em qualquer caixa, **clique nos itens desejados dentro do menu suspenso**. O menu permanecerá aberto até você clicar fora dele ou pressionar 'Enter'.")

        # Recupera valores padrão da session_state para persistência dos widgets
        default_trocas = st.session_state['selected_trocas'] if 'selected_trocas' in st.session_state else []
        default_subjetividades = st.session_state['selected_subjetividades'] if 'selected_subjetividades' in st.session_state else []
        default_relacoes = st.session_state['selected_relacoes'] if 'selected_relacoes' in st.session_state else []
        default_estrutura = st.session_state['selected_estrutura'] if 'selected_estrutura' in st.session_state else []
        
        # Garante que default_method sempre seja um valor válido para o st.radio
        if 'l4_score_method' not in st.session_state or st.session_state['l4_score_method'] not in ["Média", "PCA (1º componente)"]:
            default_method = "Média" # Define um valor padrão seguro
            st.session_state['l4_score_method'] = "Média" # E o salva na session_state
        else:
            default_method = st.session_state['l4_score_method']

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
            if all(len(grupo) > 0 for grupo in [selected_trocas, selected_subjetividades, selected_relacoes, selected_estrutura]):
                df["L4_Trocas"] = compute_score(selected_trocas, df, "L4_Trocas")
                df["L4_Subjetividades"] = compute_score(selected_subjetividades, df, "L4_Subjetividades")
                df["L4_Relacoes"] = compute_score(selected_relacoes, df, "L4_Relacoes")
                df["L4_Estrutura"] = compute_score(selected_estrutura, df, "L4_Estrutura")

                # Salva o DataFrame ATUALIZADO de volta no session_state
                st.session_state["df_l4"] = df.copy() 
                st.session_state['l4_scores_calculated'] = True # Marca que os escores foram calculados
                st.success("✅ Escores calculados com sucesso.")
            else:
                st.warning("⚠️ Selecione ao menos uma variável em cada dimensão para calcular os escores L4.")
                # Se não houver variáveis selecionadas, certifique-se de que as colunas L4 não existam ou estejam preenchidas com NaN
                for col_name in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]:
                    if col_name in df.columns:
                        df[col_name] = np.nan
                st.session_state["df_l4"] = df.copy() # Garante que o estado seja consistente
                st.session_state['l4_scores_calculated'] = False

        # Exibe o dataframe apenas se os escores já foram calculados ou se a página foi recarregada e eles estão no df_l4
        # Garante que as colunas de escore existam antes de tentar acessá-las.
        if 'df_l4' in st.session_state and all(col in st.session_state['df_l4'].columns for col in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]):
             # Verifica se há pelo menos um valor não-NaN nas colunas L4
            if not st.session_state['df_l4'][["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].isnull().all().all():
                st.dataframe(st.session_state['df_l4'][["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].head())


    # A partir daqui, todas as abas devem usar a versão mais recente de st.session_state["df_l4"]
    # Re-obtém o df para as próximas abas, garantindo que seja a versão mais atualizada após tab1.
    df = st.session_state["df_l4"].copy()

    # Verificação de pré-requisito para as outras abas
    l4_scores_exist = all(col in df.columns for col in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]) \
                      and not df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].isnull().all().all()


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
                for col in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]:
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
            numeric_cols = [col for col in numeric_cols if col not in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", "Cluster_L4"]]

            if numeric_cols:
                y_var = st.selectbox("Selecione a variável dependente:", numeric_cols, key="l4x_model_y")
                Xy = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", y_var]].dropna()
                
                if Xy.empty:
                    st.warning("⚠️ Não há dados suficientes para construir o modelo com as variáveis selecionadas. Verifique valores nulos.")
                else:
                    X = Xy[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]
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
            X_cluster = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].dropna()

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
                        
                        df.loc[X_cluster.index, "Cluster_L4"] = cluster_labels.astype(str)

                        st.session_state["df_l4"] = df.copy() 

                        df_grouped = df.groupby("Cluster_L4")[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].mean().reset_index()
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
            y_options = [col for col in numeric_cols if col not in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", "Cluster_L4"]]

            if y_options:
                y_var = st.selectbox("Variável dependente para predição:", y_options, key="l4x_cv_y")
                n_folds = st.slider("Número de Folds (K)", min_value=2, max_value=10, value=5, key="l4x_cv_n_folds")

                Xy = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", y_var]].dropna()
                
                if Xy.empty:
                    st.warning("⚠️ Não há dados suficientes para realizar a validação cruzada. Verifique valores nulos.")
                else:
                    X = Xy[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]
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