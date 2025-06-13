# model_multilevel_lvl3_full.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import io
from scipy import stats

def show_multilevel_model_lvl3_full():
    st.subheader("📚 Modelo Multinível Nível 3 — Versão Completa")
    st.markdown("Este módulo permite ajustar modelos multiníveis com até três níveis, com ICCs, comparações e relatórios integrados.")

    # 🔍 Carregar dados
    df = st.session_state.get("df_l4")
    if df is None or df.empty:
        df = st.session_state.get("df_processed")

    if df is None or df.empty:
        st.warning("⚠️ Nenhum dado disponível. Carregue dados no pré-processamento ou calcule os escores L4.")
        return

    # 🔧 Seleção de variáveis
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.markdown("### 🔎 Selecione as variáveis")

    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox("📌 Variável dependente", numeric_cols, key="mlm_full_dep")
        indep_vars = st.multiselect("📈 Variáveis independentes (nível 1)", numeric_cols, key="mlm_full_indep")
    with col2:
        group_lvl2 = st.selectbox("🏫 Agrupamento Nível 2 (ex: escola)", cat_cols, key="mlm_full_lvl2")
        group_lvl3 = st.selectbox("🌍 Agrupamento Nível 3 (ex: município)", cat_cols, key="mlm_full_lvl3")

    # Armazenar seleções para uso nos blocos seguintes
    st.session_state["mlm3_config"] = {
        "dep_var": dep_var,
        "indep_vars": indep_vars,
        "group_lvl2": group_lvl2,
        "group_lvl3": group_lvl3,
        "df_model": df
    }

    st.markdown("### ⚙️ Ajustar Modelo de 3 Níveis")

    if st.button("📈 Rodar Modelo Multinível de Nível 3"):
        cfg = st.session_state["mlm3_config"]
        df_model = cfg["df_model"]
        dep_var = cfg["dep_var"]
        indep_vars = cfg["indep_vars"]
        group_lvl2 = cfg["group_lvl2"]
        group_lvl3 = cfg["group_lvl3"]

        if not dep_var or not indep_vars or not group_lvl2 or not group_lvl3:
            st.warning("⚠️ Selecione todas as variáveis para continuar.")
            return

        try:
            df_clean = df_model[[dep_var] + indep_vars + [group_lvl2, group_lvl3]].dropna()
            df_clean[group_lvl2] = df_clean[group_lvl2].astype("category")
            df_clean[group_lvl3] = df_clean[group_lvl3].astype("category")

            formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
            re_formula = "1"

            st.code(f"Formula: {formula}")

            md3 = smf.mixedlm(formula,
                              data=df_clean,
                              groups=df_clean[group_lvl3],
                              re_formula=re_formula,
                              vc_formula={group_lvl2: f"0 + C({group_lvl2})"})

            mdf3 = md3.fit(reml=True)
            st.success("✅ Modelo de 3 níveis ajustado com sucesso.")

            st.markdown("### 📋 Sumário do Modelo (Nível 3)")
            st.text(mdf3.summary())

            # Calcular ICCs
            var_lvl3 = mdf3.cov_re.iloc[0, 0]
            var_lvl2 = mdf3.vcomp[0] if hasattr(mdf3, "vcomp") and len(mdf3.vcomp) > 0 else 0
            var_resid = mdf3.scale
            var_total = var_lvl3 + var_lvl2 + var_resid

            icc_3 = var_lvl3 / var_total
            icc_2 = var_lvl2 / var_total
            icc_resid = var_resid / var_total

            st.markdown("### 🧮 ICCs por Nível")
            st.write(f"**Variância Nível 3 ({group_lvl3})**: `{var_lvl3:.4f}`")
            st.write(f"**Variância Nível 2 ({group_lvl2})**: `{var_lvl2:.4f}`")
            st.write(f"**Variância Residual (nível 1)**: `{var_resid:.4f}`")

            st.write(f"**ICC Nível 3**: `{icc_3:.4f}`")
            st.write(f"**ICC Nível 2**: `{icc_2:.4f}`")
            st.write(f"**Erro intra-indivíduo**: `{icc_resid:.4f}`")

            # Armazenar modelo e métricas
            st.session_state["mlm3_model"] = mdf3
            st.session_state["mlm3_variancias"] = {
                "var_3": var_lvl3, "var_2": var_lvl2, "var_res": var_resid,
                "icc_3": icc_3, "icc_2": icc_2, "icc_res": icc_resid
            }

        except Exception as e:
            st.error(f"Erro ao ajustar o modelo de 3 níveis: {e}")

    if "mlm3_variancias" in st.session_state:
        st.markdown("### 📊 Visualização das Proporções de Variância (ICCs)")

        var_data = st.session_state["mlm3_variancias"]
        df_icc = pd.DataFrame({
            "Componente": ["Nível 3", "Nível 2", "Residual"],
            "Valor": [var_data["icc_3"], var_data["icc_2"], var_data["icc_res"]]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📉 Gráfico de Barras")
            import plotly.express as px
            fig_bar = px.bar(
                df_icc, x="Componente", y="Valor", text="Valor",
                color="Componente", title="Distribuição das Variâncias (ICC)"
            )
            fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_bar.update_layout(yaxis_tickformat=".0%", uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("#### 🥧 Gráfico de Pizza")
            import plotly.graph_objects as go
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_icc["Componente"],
                values=df_icc["Valor"],
                textinfo='label+percent',
                hole=.3
            )])
            fig_pie.update_layout(title_text="Proporção da Variância Total por Nível")
            st.plotly_chart(fig_pie, use_container_width=True)


        st.markdown("### 🔄 Comparação com Modelo de 2 Níveis")

        try:
            # Modelo de 2 níveis: ignora o agrupamento de nível 3
            md2 = smf.mixedlm(formula,
                              data=df_clean,
                              groups=df_clean[group_lvl2],
                              re_formula="1")

            mdf2 = md2.fit(reml=True)
            st.success("✅ Modelo de 2 níveis ajustado.")

            # Comparar métricas
            aic3, bic3, llf3 = mdf3.aic, mdf3.bic, mdf3.llf
            aic2, bic2, llf2 = mdf2.aic, mdf2.bic, mdf2.llf

            st.markdown("#### 📊 AIC / BIC / LogLik")
            st.write(f"**Modelo 3 níveis**: AIC = `{aic3:.2f}`, BIC = `{bic3:.2f}`, LL = `{llf3:.2f}`")
            st.write(f"**Modelo 2 níveis**: AIC = `{aic2:.2f}`, BIC = `{bic2:.2f}`, LL = `{llf2:.2f}`")

            # Likelihood Ratio Test
            LRT_stat = 2 * (llf3 - llf2)
            p_value = 1 - stats.chi2.cdf(LRT_stat, df=1)

            st.markdown("#### 🧪 Likelihood Ratio Test (LRT)")
            st.write(f"**LRT**: `{LRT_stat:.4f}`, **p-valor**: `{p_value:.4f}`")
            if p_value < 0.05:
                st.success("✅ Diferença significativa — o modelo de 3 níveis é melhor.")
            else:
                st.warning("🔸 Sem diferença significativa — o nível 3 pode não ser necessário.")

        except Exception as e:
            st.error(f"Erro na comparação com o modelo de 2 níveis: {e}")


    st.markdown("### 📄 Gerar Relatório Consolidado")

    if "mlm3_model" in st.session_state and "mlm3_variancias" in st.session_state:
        cfg = st.session_state["mlm3_config"]
        var = st.session_state["mlm3_variancias"]
        mdf3 = st.session_state["mlm3_model"]

        report = io.StringIO()
        report.write("# Relatório do Modelo Multinível Nível 3\n\n")
        report.write("## Fórmula do Modelo\n")
        report.write(f"`{cfg['dep_var']} ~ {' + '.join(cfg['indep_vars'])}`\n\n")
        report.write("## Agrupamentos\n")
        report.write(f"- Nível 2: {cfg['group_lvl2']}\n")
        report.write(f"- Nível 3: {cfg['group_lvl3']}\n\n")

        report.write("## ICCs e Variâncias Estimadas\n")
        report.write(f"- Variância Nível 3: {var['var_3']:.4f}\n")
        report.write(f"- Variância Nível 2: {var['var_2']:.4f}\n")
        report.write(f"- Variância Residual: {var['var_res']:.4f}\n\n")
        report.write(f"- ICC Nível 3: {var['icc_3']:.4f}\n")
        report.write(f"- ICC Nível 2: {var['icc_2']:.4f}\n")
        report.write(f"- ICC Residual: {var['icc_res']:.4f}\n\n")

        report.write("## Sumário do Modelo\n")
        report.write(mdf3.summary().as_text())
        report.write("\n\n")

        if "mlm3_compare" in st.session_state:
            comp = st.session_state["mlm3_compare"]
            report.write("## Comparação com Modelo de 2 Níveis\n")
            report.write(f"- AIC 3 Níveis: {comp['aic3']:.2f} | AIC 2 Níveis: {comp['aic2']:.2f}\n")
            report.write(f"- BIC 3 Níveis: {comp['bic3']:.2f} | BIC 2 Níveis: {comp['bic2']:.2f}\n")
            report.write(f"- LRT: {comp['lrt_stat']:.4f} | p-valor: {comp['pval']:.4f}\n")
            if comp['pval'] < 0.05:
                report.write("✅ Diferença significativa. O modelo de 3 níveis é preferível.\n")
            else:
                report.write("🔸 Sem diferença significativa. O modelo de 2 níveis pode ser suficiente.\n")

        st.download_button("📥 Baixar Relatório (.txt)",
                           data=report.getvalue(),
                           file_name="relatorio_modelo_nivel3.txt",
                           mime="text/plain")


    use_l4 = st.checkbox("🔷 Usar escores L4 como variáveis independentes", value=False)

    if use_l4:
        l4_vars = ["L4_Trocas", "L4_Subjetividades", "L4_Relacoes", "L4_Estrutura"]
        l4_present = [var for var in l4_vars if var in df.columns]
        if len(l4_present) < 4:
            st.error("⚠️ Nem todos os escores L4 estão disponíveis. Calcule-os primeiro na aba 'Modelo L4'.")
            return
        indep_vars = l4_present
        st.info(f"Usando variáveis: {', '.join(l4_present)}")
    else:
        indep_vars = st.multiselect("📈 Variáveis independentes (nível 1)", numeric_cols, key="mlm_full_indep2")


    st.markdown("### 📦 Exportação Completa")

    if "mlm3_model" in st.session_state:
        import zipfile
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # Sumário
            summary_text = st.session_state["mlm3_model"].summary().as_text()
            zf.writestr("sumario_modelo_nivel3.txt", summary_text)

            # ICCs e variâncias
            var = st.session_state["mlm3_variancias"]
            icc_df = pd.DataFrame({
                "Componente": ["Variância Nível 3", "Variância Nível 2", "Variância Residual", "ICC Nível 3", "ICC Nível 2", "Erro Intra (nível 1)"],
                "Valor": [var["var_3"], var["var_2"], var["var_res"], var["icc_3"], var["icc_2"], var["icc_res"]]
            })
            zf.writestr("iccs_e_variancias.csv", icc_df.to_csv(index=False))

            # Comparação com modelo 2 níveis
            if "mlm3_compare" in st.session_state:
                comp = st.session_state["mlm3_compare"]
                comp_df = pd.DataFrame({
                    "Métrica": ["AIC", "BIC", "LogLik", "LRT", "p-valor"],
                    "Modelo 3 níveis": [comp["aic3"], comp["bic3"], comp["llf3"], comp["lrt_stat"], comp["pval"]],
                    "Modelo 2 níveis": [comp["aic2"], comp["bic2"], comp["llf2"], "", ""]
                })
                zf.writestr("comparacao_modelos.csv", comp_df.to_csv(index=False))

            # Escores L4 se existirem
            if "df_l4" in st.session_state:
                zf.writestr("escores_L4.csv", st.session_state["df_l4"].to_csv(index=False))

        st.download_button(
            label="📥 Baixar ZIP com todos os resultados",
            data=buffer.getvalue(),
            file_name="modelo_multinivel_nivel3_completo.zip",
            mime="application/zip"
        )
