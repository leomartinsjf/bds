
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

def show_l4_model():
    st.subheader("🔷 Modelo L4 Estendido - Realismo Crítico")
    st.markdown("Este módulo permite explorar quatro dimensões estruturadas e aplicar análises explicativas, comparativas e preditivas.")

    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.warning("⚠️ Nenhum dado processado encontrado. Por favor, carregue e trate os dados na aba de pré-processamento.")
        return

    df = st.session_state['df_processed'].copy()
    all_cols = df.columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Escores L4", "🧪 Testes Estatísticos", "📈 Modelagem", "🧬 Clusters", "🔁 Validação"
    ])

    with tab1:
        st.header("📊 Escores por Dimensão")

        selected_trocas = st.multiselect("🔁 Trocas materiais", options=all_cols, key="l4x_trocas")
        selected_subjetividades = st.multiselect("🌌 Subjetividades", options=all_cols, key="l4x_subjetividades")
        selected_relacoes = st.multiselect("🤝 Relações interpessoais", options=all_cols, key="l4x_relacoes")
        selected_estrutura = st.multiselect("🏛️ Estrutura / Instituições", options=all_cols, key="l4x_estrutura")

        method = st.radio("Como calcular os escores por dimensão?", ["Média", "PCA (1º componente)"], horizontal=True)

        def compute_score(vars_list, df, label):
            if method == "Média":
                return df[vars_list].mean(axis=1)
            elif method == "PCA (1º componente)":
                df_pca = df[vars_list].dropna()
                pca = PCA(n_components=1)
                scores = pca.fit_transform(df_pca)
                return pd.Series(scores.flatten(), index=df_pca.index)


        if all(len(grupo) > 0 for grupo in [selected_trocas, selected_subjetividades, selected_relacoes, selected_estrutura]):
            df["L4_Trocas"] = compute_score(selected_trocas, df, "L4_Trocas")
            df["L4_Subjetividades"] = compute_score(selected_subjetividades, df, "L4_Subjetividades")
            df["L4_Relacoes"] = compute_score(selected_relacoes, df, "L4_Relacoes")
            df["L4_Estrutura"] = compute_score(selected_estrutura, df, "L4_Estrutura")

            st.success("✅ Escores calculados com sucesso.")
            st.dataframe(df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].head())
            st.session_state["df_l4"] = df
        else:
            st.warning("⚠️ Selecione ao menos uma variável em cada dimensão para continuar.")

    with tab2:
        st.header("🧪 Testes Estatísticos por Grupo")

        if "df_l4" in st.session_state:
            df = st.session_state["df_l4"]
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if cat_cols:
                group_col = st.selectbox("Selecione a variável de agrupamento:", cat_cols, key="l4x_group_test")
                test_type = st.radio("Tipo de teste estatístico:", ["ANOVA", "Kruskal-Wallis"], horizontal=True)

                resultado = []
                for col in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]:
                    grupos = [grupo[1][col].dropna().values for grupo in df.groupby(group_col)]
                    try:
                        stat, p = f_oneway(*grupos) if test_type == "ANOVA" else kruskal(*grupos)
                        interpretacao = "✅ Diferença significativa" if p < 0.05 else "🔸 Sem diferença significativa"
                    except Exception as e:
                        stat, p, interpretacao = np.nan, np.nan, f"Erro: {e}"

                    resultado.append({
                        "Dimensão": col,
                        "Estatística": stat,
                        "p-valor": p,
                        "Interpretação": interpretacao
                    })

                result_df = pd.DataFrame(resultado)
                st.dataframe(result_df)
                st.download_button("📥 Baixar Tabela de Resultados", result_df.to_csv(index=False).encode("utf-8"),
                                   file_name="l4_testes_estatisticos.csv", mime="text/csv")

    with tab3:
        st.header("📈 Modelagem com os Escores do L4")

        if "df_l4" in st.session_state:
            df = st.session_state["df_l4"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]

            if numeric_cols:
                y_var = st.selectbox("Selecione a variável dependente:", numeric_cols, key="l4x_model_y")
                Xy = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", y_var]].dropna()
                X = Xy[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]
                y = Xy[y_var]

                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                pseudo_r2 = r2  # fallback se _residues não disponível

                coef_df = pd.DataFrame({
                    "Variável": ["Intercepto"] + list(X.columns),
                    "Coeficiente": [model.intercept_] + list(model.coef_)
                })

                st.write(f"**R²:** {r2:.4f}")
                st.write(f"**Pseudo R²:** {pseudo_r2:.4f}")
                st.dataframe(coef_df)
                st.download_button("📥 Baixar Coeficientes", coef_df.to_csv(index=False).encode("utf-8"),
                                   file_name="l4_modelo_coeficientes.csv", mime="text/csv")

    with tab4:
        st.header("🧬 Agrupamento (Clusters) com os Escores")

        if "df_l4" in st.session_state:
            df = st.session_state["df_l4"]
            X_cluster = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].dropna()
            n_clusters = st.slider("Número de Clusters", min_value=2, max_value=10, value=3)
            cluster_method = st.radio("Método de agrupamento:", ["KMeans", "Agglomerative"], horizontal=True)

            model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') if cluster_method == "KMeans" else AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X_cluster)
            df_clustered = df.loc[X_cluster.index].copy()
            df_clustered["Cluster_L4"] = cluster_labels.astype(str)

            df_grouped = df_clustered.groupby("Cluster_L4")[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]].mean().reset_index()
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
            st.download_button("📥 Baixar Dados com Clusters", df_clustered.to_csv(index=False).encode("utf-8"),
                               file_name="l4_clusters.csv", mime="text/csv")

    with tab5:
        st.header("🔁 Validação Cruzada")

        if "df_l4" in st.session_state:
            df = st.session_state["df_l4"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_options = [col for col in numeric_cols if col not in ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]

            if y_options:
                y_var = st.selectbox("Variável dependente para predição:", y_options, key="l4x_cv_y")
                n_folds = st.slider("Número de Folds (K)", min_value=2, max_value=10, value=5)

                Xy = df[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura", y_var]].dropna()
                X = Xy[["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]]
                y = Xy[y_var]

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
