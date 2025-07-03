import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols, glm, mixedlm
import re
import networkx as nx
from model_multilevel_lvl3 import show_multilevel_model_lvl3_full


def show_multilevel_model():
    # Garante que os campos comecem sempre limpos, apenas uma vez
    if "mlm_reset" not in st.session_state:
        st.session_state["mlm_dependent"] = ""
        st.session_state["mlm_group"] = ""
        st.session_state["mlm_reset"] = True
    st.subheader("🌳 Análise Multinível (Modelos Lineares Mistos)")
    st.info("Utilize a Análise Multinível para modelar dados com estrutura hierárquica ou aninhada (ex: estudantes em escolas, pacientes em hospitais).")
    st.info("Esta análise permite que os efeitos das variáveis variem entre os diferentes grupos.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    with st.expander("Configurar e Executar Análise Multinível", expanded=False):
        
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.markdown("#### Configuração do Modelo Multinível")

        dependent_var = st.selectbox(
            "Escolha a variável contínua a ser prevista:",
            options=[""] + numeric_cols,
            index=0
        )
        if dependent_var == "":
            dependent_var = None

        group_var = st.selectbox(
            "Escolha a variável categórica que define os grupos (nível superior):",
            options=[""] + categorical_cols,
            index=0
        )
        if group_var == "":
            group_var = None

        options_fixed_effects = [col for col in all_cols if col != dependent_var and col != group_var]
        fixed_effects_vars = st.multiselect("Escolha as variáveis preditoras (efeitos fixos):", options=options_fixed_effects, default=[], key="mlm_fixed_effects")

        random_effects_options = ["Intercepto Aleatório"] + [col for col in fixed_effects_vars if col in numeric_cols]
        selected_random_effects = st.multiselect(
            "Escolha os efeitos aleatórios:",
            options=random_effects_options,
            default=["Intercepto Aleatório"],
            key="mlm_random_effects"
        )

        if not dependent_var or not group_var or not selected_random_effects:
            st.info("Por favor, selecione a variável dependente, a de agrupamento e ao menos um efeito aleatório.")
            return

        if st.button("Executar Análise Multinível", key="run_mlm_model"):
            st.session_state["reset_multilevel_form"] = True
            with st.spinner("Treinando Modelo Multinível..."):
                try:
                    model_vars = [dependent_var, group_var] + fixed_effects_vars
                    for re_var in selected_random_effects:
                        if re_var != "Intercepto Aleatório" and re_var not in model_vars:
                            model_vars.append(re_var)

                    df_model = df[model_vars].dropna()
                    if not pd.api.types.is_categorical_dtype(df_model[group_var]):
                        df_model[group_var] = df_model[group_var].astype('category')

                    fixed_part = " + ".join(fixed_effects_vars) if fixed_effects_vars else "1"

                    random_part = []
                    if "Intercepto Aleatório" in selected_random_effects:
                        random_part.append("1")
                    for var in selected_random_effects:
                        if var != "Intercepto Aleatório" and var in numeric_cols:
                            random_part.append(var)
                    re_formula = "~ " + " + ".join(random_part)

                    formula = f"{dependent_var} ~ {fixed_part}"
                    model = mixedlm(formula=formula, data=df_model, re_formula=re_formula, groups=df_model[group_var])
                    results = model.fit()

                    st.markdown("---")
                    st.subheader("Resultados do Modelo Multinível")
                    st.write("#### Sumário Completo do Modelo (Statsmodels MixedLM)")
                    st.code(results.summary().as_text())

                    st.subheader("Variância e Desvio Padrão dos Componentes do Modelo")

                    if hasattr(results, 'vc_params') and results.vc_params is not None and not results.vc_params.empty:
                        st.write("##### Variância e Desvio Padrão dos Efeitos Aleatórios por Componente")
                        vc_df = results.vc_params.to_frame(name='Variância Estimada')
                        vc_df['Desvio Padrão Estimado'] = np.sqrt(vc_df['Variância Estimada'])
                        st.dataframe(vc_df)
                    else:
                        st.info("Não foi possível extrair a variância de efeitos aleatórios explicitamente.")

                    st.write("##### Variância e Desvio Padrão do Termo de Erro (Residual)")
                    st.write(f"Variância Residual (Scale): `{results.scale:.4f}`")
                    st.write(f"Desvio Padrão Residual (Scale): `{np.sqrt(results.scale):.4f}`")

                    st.subheader("Visualização dos Efeitos Aleatórios Estimados por Grupo")
                    random_effects_df = pd.DataFrame(results.random_effects).T

                    if not random_effects_df.empty:
                        st.write("Tabela de Efeitos Aleatórios Estimados (Primeiras 5 Linhas):")
                        st.dataframe(random_effects_df.head())

                        for effect_name in random_effects_df.columns:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                            sns.histplot(random_effects_df[effect_name], kde=True, ax=axes[0])
                            axes[0].set_title(f"Distribuição do Efeito Aleatório: {effect_name}")
                            sns.boxplot(x=random_effects_df[effect_name], ax=axes[1])
                            axes[1].set_title(f"Box Plot do Efeito Aleatório: {effect_name}")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.info("Nenhum efeito aleatório estimado para visualizar ou o modelo não convergiu adequadamente.")

                except Exception as e:
                    st.error(f"Erro ao executar a análise multinível: {e}")




def show_multilevel_tabs():
    st.subheader("📚 Modelos Multiníveis hierárquicos")
    tab1, tab2 = st.tabs(["Modelo Multinível", "Modelo Multinível de 3 Níveis"])

    with tab1:
        show_multilevel_model()
        

    with tab2:
        show_multilevel_model_lvl3_full()