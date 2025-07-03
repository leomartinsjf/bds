
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile
import os


def reset_multilevel_cross_state():
    keys_to_reset = [
        "cross_dep_var", "cross_indep_vars", "cross_use_l4",
        "cross_group_1", "cross_group_2"
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]


def show_multilevel_model_cross():
    st.subheader("🔀 Modelo Multinível Não Hierárquico (Cross-Classificado) — Versão Completa")

    df = st.session_state.get("df_l4")
    if df is None or df.empty:
        df = st.session_state.get("df_processed")
    if df is None or df.empty:
        st.warning("⚠️ Nenhum dado disponível. Carregue dados no pré-processamento ou calcule os escores L4.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.markdown("### 🔎 Selecione as variáveis")

    if st.button("🔄 Limpar Seleções"):
        reset_multilevel_cross_state()
        st.rerun()

    dep_var = st.selectbox("📌 Variável dependente", [""] + numeric_cols, key="cross_dep_var", index=0)
    use_l4 = st.checkbox("🔷 Usar escores L4 como variáveis independentes", value=False, key="cross_use_l4")

    if use_l4:
        l4_vars = ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]
        indep_vars = [v for v in l4_vars if v in df.columns]
        if len(indep_vars) < 4:
            st.error("⚠️ Escores L4 incompletos. Calcule-os na aba correspondente.")
            return
        st.info(f"Usando escores L4: {', '.join(indep_vars)}")
    else:
        indep_vars = st.multiselect("📈 Variáveis independentes (nível 1)", numeric_cols, key="cross_indep_vars")

    group_1 = st.selectbox("🏫 Agrupamento 1 (ex: escola)", [""] + cat_cols, key="cross_group_1", index=0)
    group_2 = st.selectbox("🌍 Agrupamento 2 (ex: município)", [""] + cat_cols, key="cross_group_2", index=0)

    if st.button("📈 Rodar Modelo Multinível Não Hierárquico"):
        if "" in [dep_var, group_1, group_2] or not indep_vars:
            st.error("⚠️ Selecione todas as variáveis obrigatórias antes de rodar o modelo.")
            return

        try:
            df_model = df[[dep_var] + indep_vars + [group_1, group_2]].dropna()
            df_model[group_1] = df_model[group_1].astype("category")
            df_model[group_2] = df_model[group_2].astype("category")

            formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
            st.code(f"Fórmula: {formula}")

            md = smf.mixedlm(formula,
                             df_model,
                             groups=df_model[group_1],
                             re_formula="1",
                             vc_formula={group_2: f"0 + C({group_2})"})

            mdf = md.fit(reml=True)
            st.success("✅ Modelo ajustado com sucesso.")

            st.markdown("### 📋 Sumário do Modelo")
            st.text(mdf.summary())

            var_g1 = mdf.cov_re.iloc[0, 0]
            var_g2 = mdf.vcomp[0] if hasattr(mdf, "vcomp") and len(mdf.vcomp) > 0 else 0
            var_res = mdf.scale
            var_total = var_g1 + var_g2 + var_res

            icc_1 = var_g1 / var_total
            icc_2 = var_g2 / var_total
            icc_res = var_res / var_total

            st.markdown("### 🧮 ICCs por Agrupamento")
            st.write(f"**Variância {group_1}**: {var_g1:.4f}")
            st.write(f"**Variância {group_2}**: {var_g2:.4f}")
            st.write(f"**Variância Residual**: {var_res:.4f}")
            st.write(f"**ICC {group_1}**: {icc_1:.4f}")
            st.write(f"**ICC {group_2}**: {icc_2:.4f}")
            st.write(f"**Erro intra-indivíduo**: {icc_res:.4f}")

            df_icc = pd.DataFrame({
                "Componente": [group_1, group_2, "Residual"],
                "Valor": [icc_1, icc_2, icc_res]
            })

            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(df_icc, x="Componente", y="Valor", text="Valor", color="Componente")
                fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig_bar.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = go.Figure(data=[go.Pie(labels=df_icc["Componente"], values=df_icc["Valor"],
                                                 textinfo='label+percent', hole=.3)])
                fig_pie.update_layout(title_text="Proporção da Variância Total")
                st.plotly_chart(fig_pie, use_container_width=True)

            report = io.StringIO()
            report.write("# Relatório do Modelo Multinível Não Hierárquico\n\n")
            report.write(f"## Fórmula\n`{formula}`\n\n")
            report.write(f"## Agrupamentos Cruzados\n- {group_1}\n- {group_2}\n\n")
            report.write("## ICCs e Variâncias\n")
            report.write(f"- Variância {group_1}: {var_g1:.4f}\n")
            report.write(f"- Variância {group_2}: {var_g2:.4f}\n")
            report.write(f"- Residual: {var_res:.4f}\n")
            report.write(f"- ICC {group_1}: {icc_1:.4f}\n")
            report.write(f"- ICC {group_2}: {icc_2:.4f}\n")
            report.write(f"- ICC Residual: {icc_res:.4f}\n\n")
            report.write("## Sumário do Modelo\n" + mdf.summary().as_text())

            st.download_button("📥 Baixar Relatório (.txt)",
                               data=report.getvalue(),
                               file_name="relatorio_cross_classificado.txt",
                               mime="text/plain")

            zip_path = "/tmp/modelo_cross_classificado.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                with open("/tmp/summary_cross.txt", "w") as f:
                    f.write(mdf.summary().as_text())
                zipf.write("/tmp/summary_cross.txt", arcname="summary_cross.txt")

                df_icc.to_csv("/tmp/icc_cross.csv", index=False)
                zipf.write("/tmp/icc_cross.csv", arcname="icc_cross.csv")

                if st.session_state.get("df_l4") is not None:
                    st.session_state["df_l4"].to_csv("/tmp/escores_l4_cross.csv", index=False)
                    zipf.write("/tmp/escores_l4_cross.csv", arcname="escores_l4_cross.csv")

            with open(zip_path, "rb") as f:
                st.download_button("📥 Baixar ZIP com todos os resultados",
                                   data=f,
                                   file_name="modelo_cross_classificado_resultados.zip",
                                   mime="application/zip")

        except Exception as e:
            st.error(f"Erro ao ajustar o modelo: {e}")
