# model_multilevel.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
from datetime import datetime
import tempfile

# --- Função principal ---
def show_multilevel_model_extended():
    st.subheader("🌳 Análise Multinível Estendida (MixedLM com ICC, Checagem e Exportação)")
    st.info("Este módulo realiza análise multinível (MixedLM) com cálculo do ICC, checagem de variáveis, visualização dos efeitos aleatórios e exportação dos resultados em CSV e PDF.")
    
    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    df = st.session_state.df_processed.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()

    with st.expander("⚙️ Configurar o Modelo Multinível", expanded=True):
        # 1. Variável dependente
        dependent_var = st.selectbox("Selecione a variável dependente (contínua):", numeric_cols, key="mlm_ext_dependent")
        st.success(f"Variável dependente: {dependent_var}")

        # 2. Variável de agrupamento
        group_var = st.selectbox("Selecione a variável de agrupamento (nível 2):", categorical_cols, key="mlm_ext_group")
        st.success(f"Variável de agrupamento: {group_var}")

        # 3. Efeitos fixos
        options_fixed = [col for col in all_cols if col != dependent_var and col != group_var]
        fixed_effects_vars = st.multiselect("Selecione as variáveis de efeitos fixos:", options_fixed, key="mlm_ext_fixed_effects")
        st.success(f"Efeitos fixos: {', '.join(fixed_effects_vars) if fixed_effects_vars else 'Nenhum selecionado'}")

        # 4. Slope aleatório (UM por vez, seguro)
        slope_var = st.selectbox(
            "Selecione UMA variável para slope aleatório (ou deixe em branco para usar só intercepto aleatório):",
            [""] + [col for col in fixed_effects_vars if col in numeric_cols],
            key="mlm_ext_slope"
        )
        if slope_var == "":
            st.success("Efeitos aleatórios: apenas Intercepto Aleatório")
        else:
            st.success(f"Efeitos aleatórios: Intercepto Aleatório + Slope Aleatório para {slope_var}")


    # --- Checagem automática antes de rodar o modelo ---
    with st.expander("🔍 Checagem Automática das Variáveis", expanded=True):
        st.info("Antes de rodar o modelo, verifique se as variáveis selecionadas são adequadas.")
        
        # Número de grupos
        num_groups = df[group_var].nunique()
        st.write(f"**Número de grupos em `{group_var}`:** {num_groups}")
        if num_groups < 5:
            st.warning("⚠️ Poucos grupos! Modelos multinível geralmente requerem pelo menos 5-6 grupos.")

        # Tamanho dos grupos
        group_counts = df[group_var].value_counts()
        st.write("**Tamanho dos grupos:**")
        st.dataframe(group_counts)

        if group_counts.min() < 5:
            st.warning("⚠️ Existem grupos com menos de 5 observações — risco elevado de Singular matrix.")

        # Variância intra-grupo para cada efeito fixo (somente se for variável numérica)
        for var in fixed_effects_vars:
            if pd.api.types.is_numeric_dtype(df[var]):
                var_intra_group = df.groupby(group_var)[var].var()
                st.write(f"**Variância intra-grupo para `{var}`:**")
                st.dataframe(var_intra_group)

                if var_intra_group.min() < 1e-5:
                    st.warning(f"⚠️ A variável `{var}` apresenta variância intra-grupo muito baixa — risco de Singular matrix.")
            else:
                st.info(f"ℹ️ Variável `{var}` não é numérica — checagem de variância intra-grupo não aplicável.")



        # Correlação com o intercepto
    intercept_means = df.groupby(group_var)[dependent_var].mean()
    for var in fixed_effects_vars:
        if pd.api.types.is_numeric_dtype(df[var]):
            var_means = df.groupby(group_var)[var].mean()
            if len(intercept_means) == len(var_means):
                corr = intercept_means.corr(var_means)
                st.write(f"**Correlação entre média de `{var}` e intercepto dos grupos:** {corr:.4f}")

                if abs(corr) > 0.9:
                    st.warning(f"⚠️ Alta correlação entre `{var}` e intercepto dos grupos — risco de colinearidade!")
        else:
            st.info(f"ℹ️ Variável `{var}` não é numérica — checagem de correlação não aplicável.")


    # --- Botão para rodar o modelo ---
    if st.button("🚀 Executar Análise Multinível", key="run_mlm_ext_model"):
        with st.spinner("Treinando modelo..."):
            try:
                model_vars = [dependent_var, group_var] + fixed_effects_vars
                if slope_var != "" and slope_var not in model_vars:
                    model_vars.append(slope_var)

                df_model = df[model_vars].dropna()

                if not pd.api.types.is_categorical_dtype(df_model[group_var]):
                    df_model[group_var] = df_model[group_var].astype('category')

                fixed_part = " + ".join(fixed_effects_vars) if fixed_effects_vars else "1"
                re_formula_str = "~ 1" if slope_var == "" else f"~ 1 + {slope_var}"

                formula = f"{dependent_var} ~ {fixed_part}"

                model = mixedlm(
                    formula=formula,
                    data=df_model,
                    re_formula=re_formula_str,
                    groups=df_model[group_var]
                )

                results = model.fit()

                st.success("✅ Modelo treinado com sucesso!")

                st.markdown("---")
                st.subheader("📋 Sumário do Modelo")
                st.code(results.summary().as_text())

                # --- Cálculo do ICC ---
                st.subheader("📈 Cálculo do ICC (Intraclass Correlation Coefficient)")

                try:
                    var_random_intercept = results.cov_re.iloc[0, 0]
                    st.write(f"Variância do Intercepto Aleatório: `{var_random_intercept:.4f}`")
                except:
                    var_random_intercept = 0.0
                    st.warning("⚠️ Não foi possível calcular a variância do intercepto aleatório.")

                var_residual = results.scale
                st.write(f"Variância Residual: `{var_residual:.4f}`")

                icc = var_random_intercept / (var_random_intercept + var_residual) if (var_random_intercept + var_residual) > 0 else 0.0
                st.success(f"**ICC Calculado:** `{icc:.4f}`")

                # Gráfico do ICC
                fig_icc = go.Figure(go.Pie(
                    labels=['Entre Grupos (Intercepto Aleatório)', 'Dentro dos Grupos (Residual)'],
                    values=[var_random_intercept, var_residual],
                    hole=0.4
                ))
                fig_icc.update_layout(title_text="🔍 ICC - Proporção da Variância por Nível")
                st.plotly_chart(fig_icc)

                st.info("O ICC representa a proporção da variância total que se deve às diferenças entre os grupos.")

                # --- Efeitos Aleatórios ---
                st.subheader("🎲 Efeitos Aleatórios por Grupo")

                random_effects_df = pd.DataFrame(results.random_effects).T
                st.dataframe(random_effects_df)

                # Gráfico por efeito
                for effect_name in random_effects_df.columns:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    sns.histplot(random_effects_df[effect_name], kde=True, ax=axes[0])
                    axes[0].set_title(f"Distribuição do Efeito Aleatório: {effect_name}")
                    axes[0].set_xlabel(f"Efeito Aleatório ({effect_name})")
                    axes[0].set_ylabel("Frequência")

                    sns.boxplot(x=random_effects_df[effect_name], ax=axes[1])
                    axes[1].set_title(f"Boxplot do Efeito Aleatório: {effect_name}")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.info(f"O gráfico de '{effect_name}' mostra a variação dos efeitos aleatórios entre os grupos.")



                # --- Exportação dos Resultados ---

                st.subheader("💾 Exportação dos Resultados")

                # Botão de exportação CSV
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                csv_filename = f"resultados_multinivel_{timestamp}.csv"

                export_df = pd.DataFrame({
                    'Variável': ['ICC', 'Var(Intercepto Aleatório)', 'Var(Residual)'],
                    'Valor': [icc, var_random_intercept, var_residual]
                })

                st.download_button(
                    label="📥 Baixar Resultados como CSV",
                    data=export_df.to_csv(index=False).encode('utf-8'),
                    file_name=csv_filename,
                    mime='text/csv'
                )

                # --- Exportação em PDF ---
                st.subheader("📄 Exportação em PDF")

                pdf_filename = f"resultados_multinivel_{timestamp}.pdf"
                pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt="Resultados da Análise Multinível", ln=True, align='C')
                pdf.ln(10)

                pdf.cell(200, 10, txt=f"Data/Hora: {timestamp}", ln=True, align='L')
                pdf.ln(10)

                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="Sumário ICC", ln=True, align='L')
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt=f"ICC: {icc:.4f}", ln=True, align='L')
                pdf.cell(200, 10, txt=f"Var(Intercepto Aleatório): {var_random_intercept:.4f}", ln=True, align='L')
                pdf.cell(200, 10, txt=f"Var(Residual): {var_residual:.4f}", ln=True, align='L')

                pdf.ln(10)
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(200, 10, txt="Sumário do Modelo", ln=True, align='L')
                pdf.set_font("Arial", size=10)

                model_summary_lines = results.summary().as_text().split('\n')
                for line in model_summary_lines:
                    pdf.cell(200, 5, txt=line[:100], ln=True, align='L')

                pdf.output(pdf_path)

                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Baixar PDF com Resultados",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime='application/pdf'
                    )

                st.success("✅ Exportações geradas com sucesso!")

            except Exception as e:
                st.error(f"Erro ao executar a análise multinível: {e}")
                st.info("Verifique se as variáveis selecionadas são apropriadas e se não há valores NaN nas variáveis.")

# --- FIM DA FUNÇÃO show_multilevel_model_extended() ---
