#correta 16/06

import tempfile, zipfile, os
from contextlib import redirect_stdout  # Para captura de saída de info()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from scipy.stats import spearmanr, pearsonr

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # Para post-hoc Tukey

from pingouin import welch_anova, pairwise_gameshowell  # Para Games-Howell e Welch
import pingouin as pg # Importado aqui para uso direto no show_correlation_matrix_interface

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Para PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis



# --- Funções Auxiliares para cálculo de tamanho de efeito ---
# Estas funções são leves, não precisam de cache
def cohens_d(x, y):
    """Calcula o Cohen's d para duas amostras."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pool_std

def cohens_d_paired(x_pre, x_post):
    """Calcula o Cohen's d para amostras pareadas."""
    diff = x_post - x_pre
    return np.mean(diff) / np.std(diff, ddof=1)

def cohens_d_one_sample(data, pop_mean):
    """Calcula o Cohen's d para uma amostra."""
    return (np.mean(data) - pop_mean) / np.std(data, ddof=1)

def calculate_partial_eta_squared(anova_table, effect_term):
    """
    Calcula o Eta-Quadrado Parcial (ηp²) para um termo de efeito na tabela ANOVA.
    Assume que 'anova_table' é o DataFrame retornado por anova_lm.
    """
    ss_effect = anova_table.loc[effect_term, 'sum_sq']
    ss_error = anova_table.loc['Residual', 'sum_sq']
    
    # anova_lm com typ=2 ou typ=3 para SS_effect, e 'Residual' para SS_error
    if ss_effect + ss_error == 0: # Evitar divisão por zero se ambos são 0
        return 0.0
    return ss_effect / (ss_effect + ss_error)


# --- Funções de Análise Exploratória (Refatoradas para Expander e com Cache) ---

# 1. Análise de Contingência
# Aplicar cache_data pois envolve cálculo de tabelas e qui-quadrado em potencialmente grandes DataFrames
@st.cache_data(show_spinner=False) # show_spinner=False para controlar o spinner manualmente
def _perform_contingency_analysis_core(df_temp, col1, col2):
    """Função core para cálculo da tabela de contingência e qui-quadrado."""
    observed_table = pd.crosstab(df_temp[col1].fillna("Valor_Ausente"), df_temp[col2].fillna("Valor_Ausente"), dropna=False)
    
    chi2, p, dof, expected = stats.chi2_contingency(observed_table)
    
    expected_df = pd.DataFrame(expected, index=observed_table.index, columns=observed_table.columns)
    
    table_sm = Table(observed_table)
    standardized_residuals = table_sm.standardized_resids

    data_to_combine = {
        'Observado': observed_table,
        'Esperado': expected_df,
        'Res. Padronizado': standardized_residuals
    }
    combined_table = pd.concat(data_to_combine, axis=1, keys=['Observado', 'Esperado', 'Res. Padronizado'])
    combined_table = combined_table.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    # Diagnóstico automático de dominância
    row_prop = pd.crosstab(df_temp[col1], df_temp[col2], normalize='index') * 100
    max_row_share = row_prop.max(axis=1).max()

    return combined_table, chi2, p, dof, max_row_share


def show_contingency_analysis(df):
    st.subheader("Análise de Contingência (Tabelas e Gráficos)")
    st.info("Utilize esta seção para explorar a relação entre duas variáveis categóricas ou a distribuição de uma única variável categórica.")

    df_temp = df.copy() # Cópia para manipulação de NaN
    
    cat_cols = df_temp.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if not cat_cols:
        st.warning("Não há colunas categóricas ou booleanas no DataFrame para realizar análise de contingência.")
        return

    analysis_type = st.radio(
        "Selecione o tipo de análise de contingência:",
        ["Tabela de Frequência (1 Variável)", "Tabela de Contingência (2 Variáveis)"],
        key="contingency_analysis_type"
    )

    if analysis_type == "Tabela de Frequência (1 Variável)":
        col_freqs = st.multiselect(
            "Selecione uma ou mais variáveis categóricas:",
            cat_cols,
            key="contingency_col_freq_multi"
        )
        if st.button("Gerar Tabelas de Frequência", key="generate_freq_tables_multi"):
            if not col_freqs:
                st.warning("Selecione pelo menos uma variável para gerar a(s) tabela(s) de frequência.")
            for col_freq in col_freqs:
                st.write(f"### Tabela de Frequência para '{col_freq}'")
                # A função value_counts é relativamente rápida, não há necessidade de cache aqui,
                # a menos que o DataFrame seja gigantesco e a chamada seja feita repetidamente
                freq_table = df_temp[col_freq].fillna("Valor_Ausente").value_counts(dropna=False).reset_index()
                freq_table.columns = [col_freq, 'Frequência']
                freq_table['Porcentagem (%)'] = (freq_table['Frequência'] / freq_table['Frequência'].sum()) * 100
                st.dataframe(freq_table)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(
                    data=df_temp,
                    y=col_freq,
                    order=df_temp[col_freq].fillna("Valor_Ausente").value_counts(dropna=False).index,
                    ax=ax
                )
                ax.set_title(f'Distribuição de {col_freq}')
                ax.set_xlabel('Frequência')
                ax.set_ylabel(col_freq)
                st.pyplot(fig)
                plt.close(fig)

    elif analysis_type == "Tabela de Contingência (2 Variáveis)":
        col1 = st.selectbox("Selecione a primeira variável categórica:", [""] + cat_cols, index=0, key="contingency_col1")
        col2 = st.selectbox("Selecione a segunda variável categórica:", [""] + cat_cols, index=0, key="contingency_col2")

        if st.button("Gerar Tabela de Contingência e Teste Qui-Quadrado", key="generate_contingency_table"):
            if col1 and col2:
                with st.spinner("Calculando tabela de contingência e Qui-Quadrado..."):
                    try:
                        combined_table, chi2, p, dof, max_row_share = _perform_contingency_analysis_core(df_temp, col1, col2)
                        
                        st.write(f"### Análise de Contingência Completa para '{col1}' vs '{col2}'")
                        st.dataframe(combined_table.round(3))

                        if max_row_share >= 90:
                            st.warning("📌 Algumas categorias têm distribuição altamente concentrada (dominância > 90%). Pode haver associação forte entre as variáveis.")
                        elif max_row_share < 50:
                            st.info("📌 As distribuições estão relativamente equilibradas entre as categorias.")
                        else:
                            st.info("📌 Há moderada assimetria na distribuição conjunta das variáveis.")

                        st.info("Para cada célula, são exibidos: **Observado** (frequência real), **Esperado** (frequência sob independência) e **Res. Padronizado** (desvio da frequência esperada, valores |>2| indicam contribuição significativa).")

                        st.write("---") # Separador visual

                        st.write("### Resultados do Teste Qui-Quadrado")
                        st.write(f"Estatística Qui-Quadrado: {chi2:.3f}")
                        st.write(f"Valor p: {p:.3f}")
                        st.write(f"Graus de Liberdade: {dof}")
                        st.write(f"Significância (p < 0.05): {'Sim' if p < 0.05 else 'Não'}")

                        if p < 0.05:
                            st.success("Há evidências de uma associação significativa entre as duas variáveis categóricas.")
                        else:
                            st.info("Não há evidências suficientes para rejeitar a hipótese de independência entre as variáveis.")

                    except ValueError as e:
                        st.error(f"Não foi possível realizar o teste Qui-Quadrado. Erro: {e}. Isso pode ocorrer se houver células com frequência esperada muito baixa (menor que 5) ou se as colunas selecionadas são idênticas após o preenchimento de NaN.")
                    except Exception as e:
                        st.error(f"Ocorreu um erro inesperado no teste Qui-Quadrado ou no cálculo dos resíduos: {e}")

                st.write("---") # Separador visual para os gráficos

                st.write("### Gráfico de Barras Agrupado")
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.countplot(data=df_temp, x=col1, hue=col2, ax=ax, palette='viridis')
                ax.set_title(f'Contagem de {col1} por {col2}')
                ax.set_xlabel(col1)
                ax.set_ylabel('Contagem')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)

                st.write("### Gráfico de Barras Empilhado (Proporcional)")
                fig, ax = plt.subplots(figsize=(12, 7))
                (df_temp.groupby(col1)[col2].value_counts(normalize=True).unstack(fill_value=0) * 100).plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
                ax.set_title(f'Proporção de {col2} dentro de {col1}')
                ax.set_xlabel(col1)
                ax.set_ylabel('Porcentagem (%)')
                plt.xticks(rotation=45, ha='right')
                ax.legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning("Selecione ambas as variáveis para gerar a tabela de contingência.")

# 2. Análise de Correlação
@st.cache_data(show_spinner=False)
def _calculate_correlations(df_selected_cols, method):
    """Calcula a matriz de correlação para um método específico."""
    return df_selected_cols.corr(method=method)

@st.cache_data(show_spinner=False)
def _calculate_partial_correlation(df_clean, col_x, col_y, col_z):
    """Calcula a correlação parcial usando pingouin."""
    return pg.partial_corr(data=df_clean, x=col_x, y=col_y, covar=col_z)

@st.cache_data(show_spinner=False)
def _generate_pairplot(df_selected_pair_cols):
    """Gera o pairplot."""
    return sns.pairplot(df_selected_pair_cols, kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})


def show_correlation_matrix_interface(df):
    st.markdown("### Análise de Correlação Completa")
    st.info("Abaixo estão integradas as análises de Pearson, Spearman, Parcial e por Subgrupos.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas.")
        return

    st.markdown("#### Correlação Pearson + Spearman (por pares)")
    selected_corr_cols = st.multiselect(
        "Selecione variáveis numéricas:",
        options=num_cols,
        default=[],
        key="corr_combined_vars"
    )
    if selected_corr_cols and len(selected_corr_cols) >= 2:
        with st.spinner("Calculando correlações Pearson e Spearman..."):
            pearson_corr = _calculate_correlations(df[selected_corr_cols], "pearson")
            spearman_corr = _calculate_correlations(df[selected_corr_cols], "spearman")

        # --- Tabela de pares ---
        pearson_pairs = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool)).stack().reset_index()
        spearman_pairs = spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool)).stack().reset_index()

        pearson_pairs.columns = ["Var1", "Var2", "Pearson"]
        spearman_pairs.columns = ["Var1", "Var2", "Spearman"]

        paired_corrs = pd.merge(pearson_pairs, spearman_pairs, on=["Var1", "Var2"])
        st.dataframe(paired_corrs.round(2))

        # --- Heatmap aprimorado (Pearson) ---
        st.markdown("#### Heatmap de Correlação de Pearson")
        fig, ax = plt.subplots(figsize=(min(1.2 * len(selected_corr_cols), 12), 1.2 * len(selected_corr_cols)))
        sns.heatmap(
            pearson_corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            ax=ax
        )
        ax.set_title("Matriz de Correlação de Pearson", fontsize=14, pad=12)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Selecione ao menos duas variáveis para calcular correlações.")


    st.divider()

    st.markdown("#### Correlação Total vs Parcial")
    col_x = st.selectbox("Variável X", [""] + num_cols, index=0, key="partial_x")
    col_y = st.selectbox("Variável Y", [""] + num_cols, index=0, key="partial_y")
    col_z = st.selectbox("Controlar por (Z)", [""] + num_cols, index=0, key="partial_z")

    if st.button("Calcular Correlação Total e Parcial", key="calc_partial_corr"):
        if col_x and col_y and col_z and len(set([col_x, col_y, col_z])) == 3:
            df_clean = df[[col_x, col_y, col_z]].dropna()

            if df_clean.shape[0] < 3:
                st.warning("Número insuficiente de observações válidas após remoção de valores ausentes.")
            else:
                with st.spinner("Calculando correlação total e parcial..."):
                    r_total, p_total = pearsonr(df_clean[col_x], df_clean[col_y])
                    partial_result = _calculate_partial_correlation(df_clean, col_x, col_y, col_z)

                result_table = pd.DataFrame({
                    "Correlação Total (r)": [round(r_total, 3)],
                    "p-valor Total": [round(p_total, 4)],
                    "Correlação Parcial (r)": [round(partial_result["r"].iloc[0], 3)],
                    "p-valor Parcial": [round(partial_result["p-val"].iloc[0], 4)]
                })
                st.dataframe(result_table)
        else:
            st.info("Selecione três variáveis distintas para comparar correlação total e parcial.")


    st.divider()

    st.markdown("#### Correlação por Subgrupo Categórico")
    group_col = st.selectbox("Variável categórica para segmentar:", [""] + cat_cols, index=0, key="group_var")
    group_corr_cols = st.multiselect(
        "Variáveis numéricas para correlação por grupo:",
        options=num_cols,
        default=[],
        key="group_corr_vars"
    )
    if st.button("Calcular Correlação por Subgrupo", key="calc_grouped_corr"):
        if group_col and len(group_corr_cols) >= 2:
            with st.spinner(f"Calculando correlações por subgrupo para '{group_col}'..."):
                grouped_corrs = []
                for name, group in df.groupby(group_col):
                    try:
                        # Cached call for group correlation
                        corr = _calculate_correlations(group[group_corr_cols], "pearson") 
                        corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        corr = corr.stack().reset_index()
                        corr.columns = ["Var1", "Var2", f"r_{name}"]
                        grouped_corrs.append(corr)
                    except Exception as e:
                        st.warning(f"Erro no grupo '{name}': {e}")
                if grouped_corrs:
                    merged = grouped_corrs[0]
                    for other in grouped_corrs[1:]:
                        merged = pd.merge(merged, other, on=["Var1", "Var2"], how="outer")
                    st.dataframe(merged.round(2))
                else:
                    st.warning("Não foi possível calcular correlações para os grupos selecionados.")
        elif group_col:
            st.info("Selecione ao menos duas variáveis numéricas.")
        elif group_corr_cols:
            st.info("Selecione a variável categórica para segmentação.")
        else:
            st.info("Selecione a variável categórica e ao menos duas variáveis numéricas.")


    st.divider()

    st.markdown("#### Gráfico de Dispersão entre Pares")
    selected_pair_cols = st.multiselect(
        "Selecione variáveis para gráfico de pares:",
        options=num_cols,
        default=[],
        key="pairplot_vars"
    )
    if st.button("Gerar Gráfico de Pares", key="generate_pairplot_button"):
        if len(selected_pair_cols) >= 2:
            st.markdown("#### Gráfico de Dispersão com Regressão Linear (Pairplot)")
            with st.spinner("Gerando Pairplot (pode demorar para muitos dados/colunas)..."):
                fig = _generate_pairplot(df[selected_pair_cols])
                st.pyplot(fig)
                plt.close(fig.fig) # CHANGED THIS LINE: access the .fig attribute
        elif selected_pair_cols:
            st.warning("Selecione ao menos duas variáveis.")
        else:
            st.info("Selecione as variáveis para gerar o gráfico de pares.")


# 3. Análise ANOVA
@st.cache_data(show_spinner=False)
def _perform_anova_core(df_anova, dv_col, iv_cols):
    """Função core para execução da ANOVA."""
    formula_terms = [f'C({col})' for col in iv_cols]
    if len(formula_terms) > 1:
        formula = f"{dv_col} ~ {' * '.join(formula_terms)}"
    else:
        formula = f"{dv_col} ~ {formula_terms[0]}"
    
    model = ols(formula, data=df_anova).fit()
    anova_table = anova_lm(model, typ=2)
    return anova_table, formula

@st.cache_data(show_spinner=False)
def _perform_levene_test(df_anova, dv_col, iv_cols):
    """Função core para execução do Teste de Levene."""
    if len(iv_cols) > 1:
        df_anova['_combined_group_'] = df_anova[iv_cols].astype(str).agg('_'.join, axis=1)
        levene_groups = [df_anova[dv_col][df_anova['_combined_group_'] == g].dropna() for g in df_anova['_combined_group_'].unique()]
    else:
        levene_groups = [df_anova[dv_col][df_anova[iv_cols[0]] == g].dropna() for g in df_anova[iv_cols[0]].unique()]

    levene_groups = [g for g in levene_groups if not g.empty]

    if len(levene_groups) < 2:
        return None, 1.0 # Não foi possível rodar Levene, retorna p-val 1.0 (homogêneo)
    
    stat_levene, levene_p_val = stats.levene(*levene_groups)
    return stat_levene, levene_p_val

@st.cache_data(show_spinner=False)
def _perform_tukey_hsd(endog_data, groups_data):
    """Executa o teste post-hoc Tukey HSD."""
    return pairwise_tukeyhsd(endog=endog_data, groups=groups_data, alpha=0.05)

@st.cache_data(show_spinner=False)
def _perform_games_howell(df, dv_col, between_col):
    """Executa o teste post-hoc Games-Howell."""
    # This line is correct, it passes dv_col to pingouin's dv, etc.
    return pg.pairwise_gameshowell(data=df, dv=dv_col, between=between_col)
    with st.spinner("Executando Games-Howell..."):
        try:
            gameshowell_result = _perform_games_howell(
                df=df_anova_results,
                dv_col=dv_col_results, # <--- ENSURE THIS IS 'dv_col'
                between_col=selected_posthoc_factor # <--- ENSURE THIS IS 'between_col'
            )
            gameshowell_result["Significativo?"] = gameshowell_result["pval"].apply(lambda p: "✅ Sim" if p < 0.05 else "❌ Não")
            for col in ["diff", "se", "pval", "ci_low", "ci_high"]:
                if col in gameshowell_result.columns:
                    gameshowell_result[col] = gameshowell_result[col].round(3)
            st.markdown("#### Resultados do Games-Howell (com destaque para significância)")
            st.dataframe(gameshowell_result)
        except Exception as e:
            st.error(f"Erro ao executar Games-Howell: {e}")

def show_anova_analysis(df):
    st.subheader("Análise de Variância (ANOVA)")
    st.info("Utilize a ANOVA para verificar se há diferenças significativas entre as médias de grupos.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if not num_cols:
        st.warning("Não há colunas numéricas (variável dependente) no DataFrame para realizar a ANOVA.")
        return
    if not cat_cols:
        st.warning("Não há colunas categóricas (variáveis independentes) no DataFrame para realizar a ANOVA.")
        return

    # 1. Seleção da Variável Dependente
    dv_col = st.selectbox("Variável Dependente (Numérica):", [""] + num_cols, index=0, key="anova_dv_col")
    if dv_col == "":
        st.info("Selecione a variável dependente para continuar.")
        return

    # 2. Seleção do Tipo de ANOVA (Unifatorial ou Fatorial)
    anova_type = st.radio(
        "Tipo de ANOVA:",
        ["ANOVA Unifatorial (um fator)", "ANOVA Fatorial (dois ou mais fatores)"],
        key="anova_selection_type"
    )

    iv_cols = []
    if anova_type == "ANOVA Unifatorial (um fator)":
        iv_col = st.selectbox("Variável Independente (Categórica - Fator):", [""] + cat_cols, index=0, key="anova_iv_col_unifactorial")
        if iv_col == "":
            st.info("Selecione a variável independente para continuar.")
            return
        iv_cols.append(iv_col)
    else:  # ANOVA Fatorial
        iv_cols = st.multiselect("Variáveis Independentes (Categóricas - Fatores):", cat_cols, default=[], key="anova_iv_cols_factorial")
        if not iv_cols or len(iv_cols) < 2:
            st.warning("Para ANOVA Fatorial, selecione pelo menos duas variáveis independentes.")
            return

    if st.button("Executar Análise ANOVA", key="run_anova"):
        # Limpa resultados anteriores para evitar confusão se o usuário mudar as seleções
        for key in ['anova_results', 'anova_table', 'levene_p_anova', 'anova_dv_current', 'anova_ivs_current']:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state['anova_dv_current'] = dv_col
        st.session_state['anova_ivs_current'] = iv_cols

        st.write(f"### ANOVA para '{st.session_state['anova_dv_current']}' por {st.session_state['anova_ivs_current']}")

        cols_for_anova = [st.session_state['anova_dv_current']] + st.session_state['anova_ivs_current']
        df_anova = df[cols_for_anova].dropna()

        if df_anova.empty:
            st.warning("Não há dados suficientes após remover valores faltantes para realizar a ANOVA com as colunas selecionadas.")
            return

        for iv in st.session_state['anova_ivs_current']:
            unique_groups = df_anova[iv].unique()
            if len(unique_groups) < 2:
                st.warning(f"A variável independente '{iv}' deve ter pelo menos dois grupos distintos.")
                return
            if anova_type == "ANOVA Unifatorial (um fator)" and len(unique_groups) < 3:
                st.info(f"A variável independente '{iv}' tem apenas {len(unique_groups)} grupos. A ANOVA é mais apropriada para três ou mais grupos. Para dois grupos, considere um teste t.")

        st.write("#### Teste de Homogeneidade de Variâncias (Levene's Test)")
        with st.spinner("Executando Teste de Levene..."):
            stat_levene, levene_p_val = _perform_levene_test(df_anova, st.session_state['anova_dv_current'], st.session_state['anova_ivs_current'])
        
        if stat_levene is None: # Se _perform_levene_test retornou None, significa que não havia dados suficientes
            st.warning("Não há grupos suficientes com dados válidos para realizar o Teste de Levene.")
            st.info("Prosseguindo com a ANOVA sem avaliação detalhada de homogeneidade de variâncias.")
            levene_p_val = 1.0 # Assume homogeneidade para não bloquear
        else:
            st.write(f"Estatística de Levene: {stat_levene:.3f}")
            st.write(f"Valor p de Levene: {levene_p_val:.3f}")
            if levene_p_val < 0.05:
                st.warning("As variâncias entre os grupos NÃO são homogêneas (p < 0.05). Considere usar o teste de Welch ANOVA para análises unifatoriais ou esteja ciente para as análises post-hoc.")
            else:
                st.success("As variâncias entre os grupos são homogêneas (p >= 0.05).")
        
        st.session_state['levene_p_anova'] = levene_p_val

        st.write("#### Resultados da ANOVA")
        with st.spinner("Executando ANOVA..."):
            try:
                anova_table, formula = _perform_anova_core(df_anova, st.session_state['anova_dv_current'], st.session_state['anova_ivs_current'])

                anova_table_fmt = (
                    anova_table
                    .reset_index()
                    .rename(columns={
                        "index": "Termo",
                        "sum_sq": "Soma dos Quadrados",
                        "df": "GL",
                        "F": "Estatística F",
                        "PR(>F)": "Valor p"
                    })
                    .round(3)
                )

                st.markdown("#### Tabela ANOVA")
                st.dataframe(anova_table_fmt)

                st.write("#### Tamanho do Efeito (Eta-Quadrado Parcial, ηp²)")
                eta_squared_data = []
                for term in anova_table.index:
                    if term not in ['Residual', 'Intercept']:
                        eta_p2 = calculate_partial_eta_squared(anova_table, term)
                        eta_squared_data.append({'Termo': term, 'ηp²': eta_p2})
                
                if eta_squared_data:
                    eta_squared_df = pd.DataFrame(eta_squared_data)
                    st.dataframe(eta_squared_df.set_index('Termo').round(3))
                    st.info("Interpretação do ηp² (Cohen): 0.01 (pequeno), 0.06 (médio), 0.14 (grande).")
                else:
                    st.info("Nenhum termo de efeito para calcular ηp².")

                st.session_state['anova_results'] = {
                    'df_anova': df_anova,
                    'dv_col': st.session_state['anova_dv_current'],
                    'iv_cols': st.session_state['anova_ivs_current'],
                    'anova_table': anova_table,
                    'formula': formula
                }
                st.session_state['anova_table'] = anova_table

                p_val_overall_significant = False
                for term in iv_cols:
                    term_name = f"C({term})"
                    if term_name in anova_table.index and anova_table.loc[term_name, 'PR(>F)'] < 0.05:
                        p_val_overall_significant = True
                        break

                if len(iv_cols) > 1:
                    interaction_term = ":".join([f"C({col})" for col in iv_cols])
                    if interaction_term in anova_table.index and anova_table.loc[interaction_term, 'PR(>F)'] < 0.05:
                        p_val_overall_significant = True

                if p_val_overall_significant:
                    st.success("✅ Há pelo menos um efeito estatisticamente significativo (p < 0.05). Prossiga para a análise Post-Hoc.")
                else:
                    st.info("ℹ️ Nenhum dos termos da ANOVA foi estatisticamente significativo (p ≥ 0.05). Post-hoc não é necessário.")

            except Exception as e:
                st.error(f"Erro ao executar ANOVA: {e}. Verifique se as variáveis selecionadas são apropriadas e se não há problemas nos dados (ex: poucas observações por grupo, variância zero).")
                for key in ['anova_results', 'anova_table', 'levene_p_anova']:
                    if key in st.session_state:
                        del st.session_state[key]
            
    if 'anova_results' in st.session_state and 'anova_table' in st.session_state:
        st.write("---")
        st.write("#### Análise Post-Hoc (Comparações Múltiplas)")
        st.info("Esta seção permite realizar comparações post-hoc se a ANOVA geral indicar um efeito significativo.")

        df_anova_results = st.session_state['anova_results']['df_anova']
        dv_col_results = st.session_state['anova_results']['dv_col']
        iv_cols_results = st.session_state['anova_results']['iv_cols']
        anova_table_results = st.session_state['anova_table']
        levene_p_val_results = st.session_state['levene_p_anova']

        significant_main_factors = []
        for factor in iv_cols_results:
            term_name = f'C({factor})'
            if term_name in anova_table_results.index and anova_table_results.loc[term_name, 'PR(>F)'] < 0.05:
                significant_main_factors.append(factor)
        
        significant_interaction_terms = []
        if len(iv_cols_results) > 1:
            interaction_term_name = ':'.join([f'C({c})' for c in iv_cols_results])
            if interaction_term_name in anova_table_results.index and anova_table_results.loc[interaction_term_name, 'PR(>F)'] < 0.05:
                significant_interaction_terms.append("Interação: " + " x ".join(iv_cols_results))

        if not significant_main_factors and not significant_interaction_terms:
            st.info("Não há fatores principais ou termos de interação significativos na ANOVA geral para realizar análises post-hoc.")
        else:
            st.markdown("**Fatores/Interações Significativas para Análise Post-Hoc:**")
            for factor in significant_main_factors:
                st.write(f"- {factor}")
            for term in significant_interaction_terms:
                st.write(f"- {term}")

            if significant_main_factors:
                selected_posthoc_factor = st.selectbox(
                    "Selecione o fator principal para análise Post-Hoc:",
                    significant_main_factors,
                    key="posthoc_factor_select"
                )

                if selected_posthoc_factor:
                    st.write(f"Comparação Post-Hoc para: **{selected_posthoc_factor}**")
                    posthoc_method = st.selectbox(
                        "Selecione o método Post-Hoc:",
                        ["Tukey HSD (se variâncias homogêneas)", "Games-Howell (se variâncias heterogêneas)"],
                        key=f"posthoc_method_select_{selected_posthoc_factor}"
                    )

                    if st.button(f"Executar Post-Hoc para {selected_posthoc_factor}", key=f"run_posthoc_button_{selected_posthoc_factor}"):
                        if posthoc_method == "Tukey HSD (se variâncias homogêneas)":
                            if levene_p_val_results < 0.05:
                                st.warning("Tukey HSD assume homogeneidade das variâncias. O teste de Levene indicou heterogeneidade. Considere Games-Howell.")
                            with st.spinner("Executando Tukey HSD..."):
                                try:
                                    tukey_result = _perform_tukey_hsd(
                                        endog_data=df_anova_results[dv_col_results],
                                        groups_data=df_anova_results[selected_posthoc_factor]
                                    )
                                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
                                    tukey_df["Significativo?"] = tukey_df["p-adj"].apply(lambda p: "✅ Sim" if float(p) < 0.05 else "❌ Não")
                                    tukey_df[["meandiff", "p-adj", "lower", "upper"]] = tukey_df[["meandiff", "p-adj", "lower", "upper"]].astype(float).round(3)
                                    st.markdown("#### Resultados do Tukey HSD (com destaque para significância)")
                                    st.dataframe(tukey_df)
                                    fig = tukey_result.plot_simultaneous()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.error(f"Erro ao executar Tukey HSD: {e}")
                        elif posthoc_method == "Games-Howell (se variâncias heterogêneas)":
                            if levene_p_val_results >= 0.05:
                                st.warning("Games-Howell é para variâncias heterogêneas. O teste de Levene indicou homogeneidade. Considere Tukey HSD.")
                            with st.spinner("Executando Games-Howell..."):
                                try:
                                    gameshowell_result = _perform_games_howell(
                                    df=df_anova_results,
                                    dv_col=dv_col_results,
                                    between_col=selected_posthoc_factor
)
                                    gameshowell_result["Significativo?"] = gameshowell_result["pval"].apply(lambda p: "✅ Sim" if p < 0.05 else "❌ Não")
                                    for col in ["diff", "se", "pval", "ci_low", "ci_high"]:
                                        if col in gameshowell_result.columns:
                                            gameshowell_result[col] = gameshowell_result[col].round(3)
                                    st.markdown("#### Resultados do teste post hoc Games-Howell ")
                                    st.dataframe(gameshowell_result)
                                except Exception as e:
                                    st.error(f"Erro ao executar Games-Howell: {e}")
            else:
                st.info("Nenhum fator principal significativo para análise post-hoc individual.")

            if significant_interaction_terms:
                st.write("---")
                st.write("#### Interpretação de Interação")
                st.info("Quando um termo de interação é significativo, a relação de um fator com a variável dependente muda a depender dos níveis do outro fator. Deve ser interpretada com gráficos de interação.")
                st.warning("A interpretação de interações é complexa e usualmente exigir visualizações específicas.")

    st.write("---")
    st.write("#### Gráficos de Visualização dos Grupos")
    if 'anova_results' in st.session_state:
        df_anova_plot = st.session_state['anova_results']['df_anova']
        dv_col_plot = st.session_state['anova_results']['dv_col']
        iv_cols_plot = st.session_state['anova_results']['iv_cols']

        for iv_plot in iv_cols_plot:
            st.write(f"##### Gráfico de Médias de {dv_col_plot} por {iv_plot}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, errorbar='se', ax=ax, palette='viridis')
            ax.set_title(f'Média de {dv_col_plot} por {iv_plot} com Erro Padrão')
            ax.set_xlabel(iv_plot)
            ax.set_ylabel(f'Média de {dv_col_plot}')
            st.pyplot(fig)
            plt.close(fig)

            st.write(f"##### Boxplot de {dv_col_plot} por {iv_plot}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, ax=ax, palette='viridis')
            sns.stripplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, color='black', size=3, jitter=True, ax=ax)
            ax.set_title(f'Boxplot de {dv_col_plot} por {iv_plot}')
            ax.set_xlabel(iv_plot)
            ax.set_ylabel(dv_col_plot)
            st.pyplot(fig)
            plt.close(fig)
        
        if len(iv_cols_plot) == 2:
            st.write(f"##### Gráfico de Interação para {iv_cols_plot[0]} x {iv_cols_plot[1]}")
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.pointplot(data=df_anova_plot, x=iv_cols_plot[0], y=dv_col_plot, hue=iv_cols_plot[1], 
                          errorbar='se', dodge=True, ax=ax, palette='dark')
            ax.set_title(f'Gráfico de Interação de {dv_col_plot} por {iv_cols_plot[0]} e {iv_cols_plot[1]}')
            ax.set_xlabel(iv_cols_plot[0])
            ax.set_ylabel(dv_col_plot)
            ax.legend(title=iv_cols_plot[1], bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            plt.close(fig)
        elif len(iv_cols_plot) > 2:
            st.info("Para mais de dois fatores, a visualização de interação se torna complexa e não é gerada automaticamente aqui.")
    else:
        st.info("Execute a análise ANOVA para visualizar os gráficos dos grupos.")




# 4. Testes T
# Os testes T são geralmente rápidos, mas para garantir, podemos cachear as funções que chamam stats.ttest
@st.cache_data(show_spinner=False)
def _perform_one_sample_ttest(sample_data, pop_mean):
    """Executa o teste t de uma amostra."""
    stat, p = stats.ttest_1samp(sample_data, pop_mean)
    return stat, p # Return serializable values

@st.cache_data(show_spinner=False)
def _perform_independent_ttest(group1_data, group2_data, equal_var):
    """Executa o teste t independente."""
    stat, p = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
    return stat, p # Return serializable values

@st.cache_data(show_spinner=False)
def _perform_levene_independent_ttest(group1_data, group2_data):
    """Executa o teste de Levene para o teste t independente."""
    stat, p = stats.levene(group1_data, group2_data)
    return stat, p # Return serializable values (Levene's also returns a similar object)


@st.cache_data(show_spinner=False)
def _perform_paired_ttest(df_paired_col_pre, df_paired_col_post):
    """Executa o teste t pareado."""
    stat, p = stats.ttest_rel(df_paired_col_pre, df_paired_col_post)
    return stat, p # Return serializable values

# ... (rest of the code) ...

def show_t_tests(df):
    st.subheader("Testes T")
    st.info("Realize testes t para comparar médias de uma ou duas amostras.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if not num_cols:
        st.warning("Não há colunas numéricas no DataFrame para realizar testes t.")
        return

    test_type = st.radio(
        "Selecione o tipo de teste T:",
        ["Teste T de Uma Amostra", "Teste T de Duas Amostras (Independentes)", "Teste T de Amostras Pareadas"],
        key="t_test_type"
    )

    if test_type == "Teste T de Uma Amostra":
        st.markdown("#### Teste T de Uma Amostra")
        one_sample_col = st.selectbox("Selecione a variável numérica:", [""] + num_cols, index=0, key="one_sample_col")
        pop_mean = st.number_input("Média da população a ser testada (μ₀):", value=0.0, key="pop_mean")

        if one_sample_col == "":
            st.info("Selecione uma variável numérica.")
        else:
            if st.button("Executar Teste T de Uma Amostra", key="run_one_sample_t_test"):
                sample_data = df[one_sample_col].dropna()
                if sample_data.empty:
                    st.warning("A coluna selecionada não possui dados válidos para o teste de uma amostra.")
                    return
                with st.spinner("Executando Teste T de Uma Amostra..."):
                    stat, p = _perform_one_sample_ttest(sample_data, pop_mean)
                d = cohens_d_one_sample(sample_data, pop_mean)

                st.write(f"### Resultados do Teste T de Uma Amostra para '{one_sample_col}'")
                st.write(f"Média da Amostra: {sample_data.mean():.3f}")
                st.write(f"Desvio Padrão da Amostra: {sample_data.std():.3f}")
                st.write(f"Estatística T: {stat:.3f}")
                st.write(f"Valor p: {p:.3f}")
                st.write(f"Graus de Liberdade: {len(sample_data) - 1}")
                st.write(f"Cohen's d (Tamanho do Efeito): {d:.3f}")

                if p < 0.05:
                    st.success(f"A média da amostra é significativamente diferente da média da população ({pop_mean}).")
                else:
                    st.info(f"Não há diferença significativa entre a média da amostra e a média da população ({pop_mean}).")

    elif test_type == "Teste T de Duas Amostras (Independentes)":
        st.markdown("#### Teste T de Duas Amostras Independentes")
        dv_col_ind = st.selectbox("Variável Dependente (Numérica):", [""] + num_cols, index=0, key="dv_col_ind")
        group_col_ind = st.selectbox("Variável de Agrupamento (Categórica, 2 grupos):", [""] + cat_cols, index=0, key="group_col_ind")

        if dv_col_ind == "" or group_col_ind == "":
            st.info("Selecione as duas variáveis.")
        else:
            unique_groups = df[group_col_ind].dropna().unique().tolist()
            if len(unique_groups) != 2:
                st.warning("A variável de agrupamento deve ter exatamente dois valores únicos.")
                if len(unique_groups) > 2:
                    st.info(f"Variável '{group_col_ind}' tem mais de 2 grupos: {unique_groups}. Considere usar ANOVA.")
                return

            group1_name, group2_name = unique_groups
            st.write(f"Grupos detectados: '{group1_name}' e '{group2_name}'.")

            if st.button("Executar Teste T Independente", key="run_independent_t_test"):
                df_filtered = df[[dv_col_ind, group_col_ind]].dropna()
                group1_data = df_filtered[df_filtered[group_col_ind] == group1_name][dv_col_ind]
                group2_data = df_filtered[df_filtered[group_col_ind] == group2_name][dv_col_ind]

                if group1_data.empty or group2_data.empty:
                    st.warning("Um dos grupos está vazio após a remoção de valores faltantes.")
                    return

                st.write("#### Estatísticas Descritivas por Grupo")
                st.write(f"**Grupo '{group1_name}':** Média={group1_data.mean():.3f}, DP={group1_data.std():.3f}, N={len(group1_data)}")
                st.write(f"**Grupo '{group2_name}':** Média={group2_data.mean():.3f}, DP={group2_data.std():.3f}, N={len(group2_data)}")

                with st.spinner("Executando Teste de Levene..."):
                    stat_levene, p_levene = _perform_levene_independent_ttest(group1_data, group2_data)
                
                equal_var = p_levene >= 0.05
                st.write("#### Teste de Homogeneidade de Variâncias (Levene)")
                st.write(f"Estatística: {stat_levene:.3f}, Valor p: {p_levene:.3f}")
                st.success("Variâncias homogêneas.") if equal_var else st.warning("Variâncias não homogêneas (usando Welch).")

                with st.spinner("Executando Teste T Independente..."):
                    stat_t, p_t = _perform_independent_ttest(group1_data, group2_data, equal_var=equal_var)
                d = cohens_d(group1_data, group2_data)

                st.write("#### Resultado do Teste T")
                st.write(f"Estatística T: {stat_t:.3f}")
                st.write(f"Valor p: {p_t:.3f}")
                st.write(f"Cohen's d: {d:.3f}")

                if p_t < 0.05:
                    maior = group1_name if group1_data.mean() > group2_data.mean() else group2_name
                    st.success(f"Há diferença significativa: o grupo **{maior}** tem média maior.")
                else:
                    st.info("Não foi encontrada diferença significativa entre os grupos.")

    elif test_type == "Teste T de Amostras Pareadas":
        st.markdown("#### Teste T de Amostras Pareadas")
        paired_cols = [col for col in num_cols if df[col].nunique() > 1]
        col_pre = st.selectbox("Variável 'Pré':", [""] + paired_cols, index=0, key="col_pre_paired")
        col_post = st.selectbox("Variável 'Pós':", [""] + paired_cols, index=0, key="col_post_paired")

        if col_pre == "" or col_post == "":
            st.info("Selecione ambas as variáveis.")
        elif col_pre == col_post:
            st.warning("As variáveis devem ser diferentes.")
        else:
            if st.button("Executar Teste T Pareado", key="run_paired_t_test"):
                df_paired = df[[col_pre, col_post]].dropna()
                if df_paired.empty:
                    st.warning("Não há dados suficientes para o teste pareado.")
                    return
                with st.spinner("Executando Teste T Pareado..."):
                    stat, p = _perform_paired_ttest(df_paired[col_pre], df_paired[col_post])
                d = cohens_d_paired(df_paired[col_pre], df_paired[col_post])

                st.write(f"### Resultado do Teste T Pareado: '{col_pre}' vs '{col_post}'")
                st.write(f"Estatística T: {stat:.3f}")
                st.write(f"Valor p: {p:.3f}")
                st.write(f"Cohen's d: {d:.3f}")

                if p < 0.05:
                    st.success("Diferença significativa entre as condições.")
                else:
                    st.info("Não há diferença estatisticamente significativa.")

# 5. Clustering
@st.cache_data(show_spinner=False)
def _perform_kmeans_and_pca(scaled_data, num_clusters):
    """Executa K-Means e PCA para visualização."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(scaled_data)

    silhouette_avg = None
    if num_clusters > 1:
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)

    pca = None
    principal_components = None
    pca_explained_variance_ratio = None
    if scaled_data.shape[1] >= 2:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        pca_explained_variance_ratio = pca.explained_variance_ratio_

    return cluster_labels, silhouette_avg, principal_components, pca_explained_variance_ratio


def show_clustering_analysis(df):
    st.subheader("Análise de Clusters (K-Means)")
    st.info("Utilize a Análise de Clusters para identificar grupos homogêneos (segmentos) dentro dos seus dados, com base nas características selecionadas.")

    if 'df_processed' in st.session_state and not st.session_state.df_processed.empty:
        df_base_for_clustering = st.session_state.df_processed.copy()
        st.info("Utilizando o DataFrame processado da sessão para a análise de clusters.")
    else:
        df_base_for_clustering = df.copy()
        st.warning("DataFrame processado não encontrado na sessão. Utilizando o DataFrame original carregado para a análise de clusters. Por favor, certifique-se de carregar e pré-processar os dados primeiro.")

    num_cols = df_base_for_clustering.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.error("Não há colunas numéricas no DataFrame base para realizar a análise de clusters após o pré-processamento. O K-Means requer dados numéricos.")
        return

    st.markdown("### Seleção de Variáveis para Clustering")
    selected_cols_for_clustering = st.multiselect(
        "Selecione as variáveis numéricas para usar na análise de clusters:",
        num_cols,
        key="cluster_cols_select"
    )

    if not selected_cols_for_clustering:
        st.info("Por favor, selecione pelo menos duas variáveis para realizar a análise de clusters.")
        return

    df_cluster_raw = df_base_for_clustering[selected_cols_for_clustering].copy()
    initial_rows = df_cluster_raw.shape[0]
    for col in df_cluster_raw.columns:
        df_cluster_raw[col] = pd.to_numeric(df_cluster_raw[col], errors='coerce')

    df_cluster_clean = df_cluster_raw.dropna()

    if df_cluster_clean.shape[0] < initial_rows:
        st.warning(f"Foram removidas {initial_rows - df_cluster_clean.shape[0]} linhas contendo valores ausentes ou não-numéricos nas variáveis selecionadas para clustering. Restam {df_cluster_clean.shape[0]} observações válidas.")

    if df_cluster_clean.empty:
        st.error("Não há dados suficientes para clusterizar após a limpeza e remoção de valores ausentes/não-numéricos nas colunas selecionadas. Por favor, ajuste suas variáveis de seleção ou verifique a qualidade dos dados.")
        return

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster_clean)
    scaled_df = pd.DataFrame(scaled_data, columns=selected_cols_for_clustering, index=df_cluster_clean.index)

    st.markdown("### 1. Determinação do Número Ideal de Clusters (Método do Cotovelo)")
    max_k_elbow = min(15, len(scaled_df) - 1)
    if max_k_elbow < 2:
        st.warning(f"Dados insuficientes ({len(scaled_df)} observações) para realizar o método do cotovelo.")
        return

    inertias = []
    k_range_elbow = range(1, max_k_elbow + 1)

    with st.spinner("Calculando inércias para o Método do Cotovelo..."):
        for k in k_range_elbow:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(k_range_elbow, inertias, marker='o')
    ax_elbow.set_title('Método do Cotovelo para K-Means (Inércia vs. Número de Clusters)')
    ax_elbow.set_xlabel('Número de Clusters (k)')
    ax_elbow.set_ylabel('Inércia')
    ax_elbow.set_xticks(k_range_elbow)
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)

    st.markdown("### 2. Execução do K-Means e Avaliação")
    num_clusters = st.number_input(
        "Insira o número de clusters (k) para o K-Means (mínimo 2):",
        min_value=2,
        max_value=max_k_elbow,
        value=min(3, max_k_elbow),
        key="k_input"
    )

    if st.button("Executar K-Means", key="run_kmeans_button"):
        with st.spinner(f"Executando K-Means com {num_clusters} clusters e PCA para visualização..."):
            try:
                cluster_labels, silhouette_avg, principal_components, pca_explained_variance_ratio = \
                    _perform_kmeans_and_pca(scaled_data, num_clusters)

                if silhouette_avg is not None:
                    st.success(f"Silhouette Score: {silhouette_avg:.3f}")
                    st.info("O Silhouette Score varia de -1 (pior) a +1 (melhor). Valores próximos de 1 indicam clusters bem definidos e separados. Valores próximos de 0 indicam sobreposição. Valores negativos indicam atribuição incorreta.")

                st.session_state['cluster_labels_latest'] = cluster_labels
                st.session_state['cluster_features_latest'] = selected_cols_for_clustering
                st.session_state['scaled_data_for_pca'] = scaled_data

            except Exception as e:
                st.error(f"Ocorreu um erro ao executar o K-Means: {e}. Verifique a seleção das variáveis e o número de clusters.")
                return

        st.write("### 3. Visualização dos Clusters")
        df_with_clusters = df_cluster_clean.copy()
        df_with_clusters['Cluster'] = cluster_labels.astype(str)
        st.dataframe(df_with_clusters.head())

        st.write("#### Gráfico de Dispersão dos Clusters (PCA)")
        if principal_components is not None:
            pca_df = pd.DataFrame(data = principal_components, 
                                  columns = ['Componente Principal 1', 'Componente Principal 2'],
                                  index=df_cluster_clean.index)
            pca_df['Cluster'] = cluster_labels.astype(str)

            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                x='Componente Principal 1', 
                y='Componente Principal 2', 
                hue='Cluster', 
                palette='viridis', 
                data=pca_df, 
                ax=ax_scatter,
                s=100,
                alpha=0.7
            )
            ax_scatter.set_title('Visualização dos Clusters (PCA)')
            ax_scatter.set_xlabel(f'Componente Principal 1 ({pca_explained_variance_ratio[0]*100:.2f}%)')
            ax_scatter.set_ylabel(f'Componente Principal 2 ({pca_explained_variance_ratio[1]*100:.2f}%)')
            ax_scatter.legend(title='Cluster')
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
            st.info("Este gráfico projeta os dados em duas dimensões principais (PCA) para visualizar a separação dos clusters. Os rótulos nos eixos indicam a proporção da variância total explicada por cada componente principal.")
        else:
            st.warning("Não é possível gerar um gráfico de dispersão 2D usando PCA. As variáveis selecionadas para clustering têm menos de 2 dimensões. Selecione pelo menos duas variáveis numéricas.")
    


    if 'cluster_labels_latest' in st.session_state and 'cluster_features_latest' in st.session_state:
        st.write("### 4. Características dos Clusters")

        df_with_clusters_for_means = df_cluster_clean.copy()
        # Ensure 'Cluster' column is string type before converting to category
        df_with_clusters_for_means['Cluster'] = st.session_state['cluster_labels_latest'].astype(str)
        st.dataframe(df_with_clusters_for_means.groupby('Cluster')[st.session_state['cluster_features_latest']].mean().round(2))

        if st.button("Adicionar rótulos de Cluster ao DataFrame processado", key="add_cluster_to_df_button"):
            cluster_labels = st.session_state['cluster_labels_latest']
            selected_cols_for_clustering = st.session_state['cluster_features_latest']

            # Convert cluster_labels to string type BEFORE creating the series and filling NaNs
            cluster_series = pd.Series(cluster_labels.astype(str), index=df_cluster_clean.index, name='Cluster_KMeans')

            df_processed_temp = st.session_state.df_processed.copy()
            
            # Reindex, fill NaN with the string, then convert to category
            df_processed_temp['Cluster_KMeans'] = cluster_series.reindex(df_processed_temp.index).fillna('Não_Clusterizado').astype('category')
            
            st.session_state.df_processed = df_processed_temp

            st.success("Coluna 'Cluster_KMeans' adicionada ao DataFrame processado na sessão.")

def show_exploratory_analysis():
    key_prefix = "ea_"
    st.header("📊 Análise Exploratória de Dados")
    if st.session_state['df_processed'] is None or st.session_state['df_processed'].empty:
        st.warning("⚠️ Dados não carregados ou pré-processados. Por favor, complete as etapas anteriores.")
        return

    df_ea = st.session_state['df_processed'].copy()

    st.info("Nesta seção, você pode realizar várias análises exploratórias no seu DataFrame processado. Use os expanders abaixo para selecionar o tipo de análise.")

    with st.expander("📈 Visualização Geral do DataFrame"):
        st.subheader("Visão Geral do DataFrame Processado")
        st.dataframe(df_ea.head())
        st.write(f"Dimensões: {df_ea.shape[0]} linhas, {df_ea.shape[1]} colunas.")

        st.subheader("Informações sobre as Colunas")
        buffer = pd.io.common.StringIO()
        df_ea.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with st.expander("📊 Análise Descritiva dos Dados Numéricos"):
        num_cols = df_ea.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            st.info("Nenhuma variável numérica disponível.")
        else:
            selected_num_cols = st.multiselect(
                "Selecione as variáveis numéricas que deseja explorar:",
                options=num_cols,
                default=[],
                key=key_prefix + "desc_num_multiselect"
            )

            if not selected_num_cols:
                st.info("Selecione pelo menos uma variável.")
            else:
                desc_df = df_ea[selected_num_cols].describe().T
                desc_df["coef_var"] = desc_df["std"] / desc_df["mean"]
                desc_df["amplitude"] = desc_df["max"] - desc_df["min"]
                desc_df["curtose"] = df_ea[selected_num_cols].kurtosis()
                desc_df["assimetria"] = df_ea[selected_num_cols].skew()

                st.markdown("#### Estatísticas Descritivas com Indicadores Ampliados")
                st.dataframe(desc_df)

                st.markdown("#### Diagnóstico Interpretativo de Curtose e Assimetria")
                for var in selected_num_cols:
                    skew_val = desc_df.loc[var, "assimetria"]
                    kurt_val = desc_df.loc[var, "curtose"]

                    if abs(skew_val) < 0.5:
                        skew_txt = "distribuição aproximadamente simétrica"
                    elif skew_val >= 0.5:
                        skew_txt = "distribuição assimétrica à direita (cauda longa à direita)"
                    else:
                        skew_txt = "distribuição assimétrica à esquerda (cauda longa à esquerda)"

                    if kurt_val < -1:
                        kurt_txt = "distribuição platicúrtica (achatada)"
                    elif -1 <= kurt_val <= 1:
                        kurt_txt = "curtose próxima da normal (mesocúrtica)"
                    else:
                        kurt_txt = "distribuição leptocúrtica (pontuda)"

                    st.markdown(f"📌 **{var}**: {skew_txt} e {kurt_txt}.")

                st.markdown("#### Histogramas")
                for col in selected_num_cols:
                    st.plotly_chart(px.histogram(df_ea, x=col, nbins=30, title=f"Histograma - {col}"))

                st.markdown("#### Boxplots")
                for col in selected_num_cols:
                    st.plotly_chart(px.box(df_ea, y=col, points="all", title=f"Boxplot - {col}"))


                st.markdown("#### 📦 Exportar Diagnósticos em .zip")

                def gerar_pacote_diagnosticos():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        txt_path = os.path.join(tmpdir, "diagnosticos.txt")
                        with open(txt_path, "w", encoding="utf-8") as f_txt:
                            for col in selected_num_cols:
                                data = df_ea[col].dropna()

                                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                                sns.histplot(data, kde=True, ax=axes[0])
                                axes[0].set_title(f"Histograma - {col}")
                                sns.boxplot(x=data, ax=axes[1])
                                axes[1].set_title(f"Boxplot - {col}")
                                plt.tight_layout()

                                fig_path = os.path.join(tmpdir, f"{col}.png")
                                fig.savefig(fig_path)
                                plt.close(fig)

                                skew_val = skew(data)
                                kurt_val = kurtosis(data)

                                if abs(skew_val) < 0.5:
                                    skew_txt = "distribuição aproximadamente simétrica"
                                elif skew_val > 0.5:
                                    skew_txt = "distribuição assimétrica à direita (cauda longa à direita)"
                                else:
                                    skew_txt = "distribuição assimétrica à esquerda (cauda longa à esquerda)"

                                if kurt_val < -1:
                                    kurt_txt = "distribuição platicúrtica (achatada)"
                                elif -1 <= kurt_val <= 1:
                                    kurt_txt = "curtose próxima da normal (mesocúrtica)"
                                else:
                                    kurt_txt = "distribuição leptocúrtica (pontuda)"

                                diag_text = f"📌 {col}:\n- {skew_txt}\n- {kurt_txt}\n\n"
                                f_txt.write(diag_text)

                        zip_path = os.path.join(tmpdir, "diagnosticos.zip")
                        with zipfile.ZipFile(zip_path, "w") as zipf:
                            for filename in os.listdir(tmpdir):
                                zipf.write(os.path.join(tmpdir, filename), arcname=filename)

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Baixar pacote ZIP",
                                data=f,
                                file_name="diagnosticos.zip",
                                mime="application/zip"
                            )

                if st.button("Gerar pacote com gráficos e interpretações"):
                    gerar_pacote_diagnosticos()


    with st.expander("📊 Análise Descritiva dos Dados Categóricos"):
        cat_cols = df_ea.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if not cat_cols:
            st.info("Nenhuma variável categórica disponível.")
        else:
            selected_cat_cols = st.multiselect(
                "Selecione as variáveis categóricas que deseja explorar:",
                options=cat_cols,
                default=[],
                key=key_prefix + "desc_cat_multiselect"
            )

            if not selected_cat_cols:
                st.info("Selecione pelo menos uma variável.")
            else:
                for col in selected_cat_cols:
                    st.markdown(f"### 📌 {col}")

                    freq_abs = df_ea[col].value_counts(dropna=False)
                    freq_rel = df_ea[col].value_counts(normalize=True, dropna=False) * 100
                    freq_df = pd.DataFrame({
                        "Frequência Absoluta": freq_abs,
                        "Frequência Relativa (%)": freq_rel.round(2)
                    })

                    st.dataframe(freq_df)

                    num_unique = df_ea[col].nunique(dropna=False)
                    moda = df_ea[col].mode().iloc[0] if not df_ea[col].mode().empty else "N/A"

                    total = len(df_ea[col])
                    dominant_cat = freq_abs.idxmax()
                    dominant_pct = freq_rel[dominant_cat]

                    if num_unique == 1:
                        interpret = "📌 A variável é constante — possui apenas uma categoria."
                    elif dominant_pct > 90:
                        interpret = f"📌 A categoria '{dominant_cat}' domina com {dominant_pct:.1f}% dos casos."
                    elif num_unique > 20:
                        interpret = f"📌 A variável tem alta cardinalidade ({num_unique} categorias)."
                    else:
                        interpret = f"📌 Distribuição razoavelmente balanceada com {num_unique} categorias. Moda: '{moda}'."

                    st.info(interpret)
           
    with st.expander("📊 Análise de Contingência e Frequência"):
        show_contingency_analysis(df_ea)

    with st.expander("🔗 Correlações numéricas"):
        show_correlation_matrix_interface(df_ea)

    with st.expander("🧪 Testes T (Uma Amostra, Independentes, Pareadas)"):
        show_t_tests(df_ea)

    with st.expander("📈 Análise de Variância (ANOVA)"):
        show_anova_analysis(df_ea)

    with st.expander("🔍 Análise de Cluster (K-Means)"):
        show_clustering_analysis(df_ea)