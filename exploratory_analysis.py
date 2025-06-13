#30/05 11:21

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd # Para post-hoc Tukey
from statsmodels.stats.contingency_tables import Table
from pingouin import welch_anova, pairwise_gameshowell # Para Games-Howell e Welch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA # Importar PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 
import io # Importe io para capturar a saída do info()
from contextlib import redirect_stdout # Importe redirect_stdout
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
import plotly.express as px


# --- Funções Auxiliares para cálculo de tamanho de efeito ---
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


# --- Funções de Análise Exploratória (Refatoradas para Expander) ---

# 1. Análise de Contingência
def show_contingency_analysis(df):
    st.subheader("Análise de Contingência (Tabelas e Gráficos)")
    st.info("Utilize esta seção para explorar a relação entre duas variáveis categóricas ou a distribuição de uma única variável categórica.")

    # Criar uma cópia e preencher NaNs com placeholder para evitar erros de renderização em st.dataframe
    df_temp = df.copy()
    
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
        col1 = st.selectbox("Selecione a primeira variável categórica:", cat_cols, key="contingency_col1")
        col2 = st.selectbox("Selecione a segunda variável categórica:", cat_cols, key="contingency_col2")

        if st.button("Gerar Tabela de Contingência e Teste Qui-Quadrado", key="generate_contingency_table"):
            if col1 and col2:
                # Tabela de frequências observadas
                observed_table = pd.crosstab(df_temp[col1].fillna("Valor_Ausente"), df_temp[col2].fillna("Valor_Ausente"), dropna=False)
                
                st.write(f"### Análise de Contingência Completa para '{col1}' vs '{col2}'")
                
                try:
                    # Teste Qui-Quadrado e cálculo das esperadas
                    chi2, p, dof, expected = stats.chi2_contingency(observed_table)
                    
                    # Converte a matriz 'expected' para um DataFrame com os mesmos índices e colunas da observada
                    expected_df = pd.DataFrame(expected, index=observed_table.index, columns=observed_table.columns)
                    
                    # Usa statsmodels.stats.contingency_tables.Table para calcular os resíduos ajustados
                    table_sm = Table(observed_table)
                    standardized_residuals = table_sm.standardized_resids

                    # --- Combinando as tabelas ---
                    # Criar um dicionário de DataFrames para concatenar
                    data_to_combine = {
                        'Observado': observed_table,
                        'Esperado': expected_df,
                        'Res. Padronizado': standardized_residuals
                    }

                    # Usar pd.concat para combinar, criando um MultiIndex nas colunas
                    combined_table = pd.concat(data_to_combine, axis=1, keys=['Observado', 'Esperado', 'Res. Padronizado'])

                    # Reordenar os níveis das colunas para ter a categoria principal primeiro, depois o tipo de valor
                    # Ex: (Categoria_A, Observado), (Categoria_A, Esperado), (Categoria_A, Res. Padronizado)
                    combined_table = combined_table.reorder_levels([1, 0], axis=1).sort_index(axis=1)
                    
                    st.dataframe(combined_table.round(3)) # Arredonda para melhor visualização
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


def show_correlation_matrix_interface(df):
    st.subheader("Matriz de Correlação e Gráficos de Pares")
    st.info("Visualize a força e direção da relação entre variáveis numéricas.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        st.warning("Não há colunas numéricas no DataFrame para calcular a matriz de correlação.")
        return

    selected_corr_cols = st.multiselect(
        "Selecione as variáveis numéricas para a matriz de correlação:",
        num_cols,
        default=num_cols if len(num_cols) <= 10 else num_cols[:10],
        key="corr_matrix_cols"
    )

    if st.button("Gerar Matriz de Correlação", key="generate_corr_matrix"):
        if selected_corr_cols:
            st.write("### Matriz de Correlação (Pearson)")
            corr_matrix = df[selected_corr_cols].corr()
            st.dataframe(corr_matrix)

            st.write("### Mapa de Calor da Correlação")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title('Mapa de Calor da Matriz de Correlação')
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 📊 Matriz de Correlação (Spearman)")
            spearman_corr = df[selected_corr_cols].corr(method="spearman")
            st.dataframe(spearman_corr.style.background_gradient(cmap="Purples"), use_container_width=True)
            st.download_button("📥 Baixar matriz Spearman (.csv)", spearman_corr.to_csv().encode("utf-8"),
                               file_name="correlacao_spearman.csv", mime="text/csv")
        else:
            st.warning("Selecione pelo menos uma coluna numérica para gerar a matriz de correlação.")

    if st.checkbox("Gerar Gráfico de Pares (Pair Plot)", key="generate_pair_plot_checkbox"):
        if selected_corr_cols:
            st.info("O Pair Plot pode levar algum tempo para renderizar com muitas variáveis ou grandes conjuntos de dados.")
            pair_plot_subset = st.multiselect(
                "Selecione variáveis para o Pair Plot (máximo 5 recomendado):",
                selected_corr_cols,
                default=selected_corr_cols[:min(len(selected_corr_cols), 5)],
                key="pair_plot_subset_cols"
            )
            if st.button("Gerar Pair Plot", key="generate_pair_plot_button"):
                if pair_plot_subset:
                    if len(pair_plot_subset) > 7:
                        st.warning("Gerar Pair Plot com mais de 7 variáveis pode ser muito lento.")
                    else:
                        st.write("### Gráfico de Pares")
                        fig = sns.pairplot(df[pair_plot_subset].dropna())
                        st.pyplot(fig)
                        plt.close("all")
                else:
                    st.warning("Selecione as variáveis para o Pair Plot.")
        else:
            st.warning("Nenhuma variável numérica selecionada para o Pair Plot.")

    st.markdown("### 🧮 Correlação Parcial entre duas variáveis controlando uma terceira")
    var1 = st.selectbox("📌 Variável 1", num_cols, key="partial_var1")
    var2 = st.selectbox("📌 Variável 2", [v for v in num_cols if v != var1], key="partial_var2")
    control = st.selectbox("🎯 Controlar por", [v for v in num_cols if v not in [var1, var2]], key="partial_control")

    if var1 and var2 and control:
        try:
            df_pc = df[[var1, var2, control]].dropna()
            reg1 = LinearRegression().fit(df_pc[[control]], df_pc[var1])
            reg2 = LinearRegression().fit(df_pc[[control]], df_pc[var2])
            resid1 = df_pc[var1] - reg1.predict(df_pc[[control]])
            resid2 = df_pc[var2] - reg2.predict(df_pc[[control]])
            r_pearson, p_pearson = pearsonr(resid1, resid2)
            r_spearman, p_spearman = spearmanr(resid1, resid2)
            st.metric("Correlação parcial (Pearson)", f"{r_pearson:.4f}")
            st.metric("p-valor (Pearson)", f"{p_pearson:.4f}")
            st.metric("Correlação parcial (Spearman)", f"{r_spearman:.4f}")
            st.metric("p-valor (Spearman)", f"{p_spearman:.4f}")
        except Exception as e:
            st.error(f"Erro ao calcular correlação parcial: {e}")

    st.markdown("### 📊 Correlação por Subgrupo")
    if cat_cols:
        strat_col = st.selectbox("Variável categórica para estratificar:", cat_cols, key="corr_strat_col")
        group_corr_var = st.selectbox("Variável 1 para correlação por grupo", num_cols, key="corr_strat_x")
        group_corr_var2 = st.selectbox("Variável 2 para correlação por grupo", [v for v in num_cols if v != group_corr_var], key="corr_strat_y")

        if strat_col and group_corr_var and group_corr_var2:
            st.write(f"Correlação entre `{group_corr_var}` e `{group_corr_var2}` por `{strat_col}`:")
            grouped = df[[group_corr_var, group_corr_var2, strat_col]].dropna().groupby(strat_col)
            results = []
            for g, data in grouped:
                try:
                    r_p, _ = pearsonr(data[group_corr_var], data[group_corr_var2])
                    r_s, _ = spearmanr(data[group_corr_var], data[group_corr_var2])
                    results.append((g, round(r_p, 3), round(r_s, 3)))
                except:
                    results.append((g, np.nan, np.nan))
            st.dataframe(pd.DataFrame(results, columns=[strat_col, "Pearson", "Spearman"]))

    st.markdown("### 📈 Dispersão com Cores por Categoria (interativo)")
    x_disp = st.selectbox("Eixo X", num_cols, key="disp_x")
    y_disp = st.selectbox("Eixo Y", [v for v in num_cols if v != x_disp], key="disp_y")
    group_color = st.selectbox("Colorir por (opcional)", ["(nenhuma)"] + cat_cols, key="disp_color")

    if x_disp and y_disp:
        fig = px.scatter(
            df,
            x=x_disp,
            y=y_disp,
            color=df[group_color] if group_color != "(nenhuma)" else None,
            title=f"Dispersão: {x_disp} vs {y_disp}",
            opacity=0.7,
            trendline="ols",
            #trendline_color_override="black"
        )
        st.plotly_chart(fig, use_container_width=True)


# 3. Análise ANOVA
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
    dv_col = st.selectbox("Variável Dependente (Numérica):", num_cols, key="anova_dv_col")

    # 2. Seleção do Tipo de ANOVA (Unifatorial ou Fatorial)
    anova_type = st.radio(
        "Tipo de ANOVA:",
        ["ANOVA Unifatorial (um fator)", "ANOVA Fatorial (dois ou mais fatores)"],
        key="anova_selection_type"
    )

    iv_cols = []
    if anova_type == "ANOVA Unifatorial (um fator)":
        iv_col = st.selectbox("Variável Independente (Categórica - Fator):", cat_cols, key="anova_iv_col_unifactorial")
        if iv_col:
            iv_cols.append(iv_col)
    else: # ANOVA Fatorial
        iv_cols = st.multiselect("Variáveis Independentes (Categóricas - Fatores):", cat_cols, key="anova_iv_cols_factorial")
        if len(iv_cols) < 2:
            st.warning("Para ANOVA Fatorial, selecione pelo menos duas variáveis independentes.")
            return

    # Garante que temos as seleções mínimas
    if not dv_col or not iv_cols:
        st.info("Selecione a variável dependente e pelo menos uma variável independente para iniciar a análise.")
        return

    # Botão para Executar a ANOVA e armazenar os resultados no session_state
    if st.button("Executar Análise ANOVA", key="run_anova"):
        # Limpa resultados anteriores para evitar confusão se o usuário mudar as seleções
        if 'anova_results' in st.session_state:
            del st.session_state['anova_results']
        if 'anova_table' in st.session_state:
            del st.session_state['anova_table']
        if 'levene_p_anova' in st.session_state:
            del st.session_state['levene_p_anova']
        if 'anova_dv_current' in st.session_state:
            del st.session_state['anova_dv_current']
        if 'anova_ivs_current' in st.session_state:
            del st.session_state['anova_ivs_current']

        # Salva as seleções atuais no session_state
        st.session_state['anova_dv_current'] = dv_col
        st.session_state['anova_ivs_current'] = iv_cols

        st.write(f"### ANOVA para '{st.session_state['anova_dv_current']}' por {st.session_state['anova_ivs_current']}")

        # Prepare os dados, removendo NaNs apenas para as colunas selecionadas
        cols_for_anova = [st.session_state['anova_dv_current']] + st.session_state['anova_ivs_current']
        df_anova = df[cols_for_anova].dropna()

        if df_anova.empty:
            st.warning("Não há dados suficientes após remover valores faltantes para realizar a ANOVA com as colunas selecionadas.")
            return

        # Verifica o número de grupos para cada IV
        for iv in st.session_state['anova_ivs_current']:
            unique_groups = df_anova[iv].unique()
            if len(unique_groups) < 2:
                st.warning(f"A variável independente '{iv}' deve ter pelo menos dois grupos distintos.")
                return
            if anova_type == "ANOVA Unifatorial (um fator)" and len(unique_groups) < 3:
                st.warning(f"A variável independente '{iv}' tem apenas {len(unique_groups)} grupos. A ANOVA é mais apropriada para três ou mais grupos. Para dois grupos, considere um teste t.")

        # Teste de Homogeneidade das Variâncias (Levene's Test)
        st.write("#### Teste de Homogeneidade de Variâncias (Levene's Test)")
        
        # Levene para cada fator principal (não para interações)
        # Se for ANOVA fatorial, Levene testa as variâncias da VD entre os grupos combinados de todos os IVs.
        # stats.levene espera múltiplos arrays, um para cada grupo.
        
        # Cria a coluna de grupo combinada para Levene se houver múltiplos IVs
        if len(st.session_state['anova_ivs_current']) > 1:
            df_anova['_combined_group_'] = df_anova[st.session_state['anova_ivs_current']].astype(str).agg('_'.join, axis=1)
            levene_groups = [df_anova[st.session_state['anova_dv_current']][df_anova['_combined_group_'] == g].dropna() for g in df_anova['_combined_group_'].unique()]
        else:
            levene_groups = [df_anova[st.session_state['anova_dv_current']][df_anova[st.session_state['anova_ivs_current'][0]] == g].dropna() for g in df_anova[st.session_state['anova_ivs_current'][0]].unique()]

        levene_groups = [g for g in levene_groups if not g.empty] # Remove grupos vazios

        levene_p_val = 1.0 # Default para homogêneo
        if len(levene_groups) < 2:
            st.warning("Não há grupos suficientes com dados válidos para realizar o Teste de Levene.")
            st.info("Prosseguindo com a ANOVA sem avaliação detalhada de homogeneidade de variâncias.")
        else:
            try:
                stat_levene, levene_p_val = stats.levene(*levene_groups)
                st.write(f"Estatística de Levene: {stat_levene:.3f}")
                st.write(f"Valor p de Levene: {levene_p_val:.3f}")
                if levene_p_val < 0.05:
                    st.warning("As variâncias entre os grupos NÃO são homogêneas (p < 0.05). Considere usar o teste de Welch ANOVA para análises unifatoriais ou esteja ciente para as análises post-hoc.")
                else:
                    st.success("As variâncias entre os grupos são homogêneas (p >= 0.05).")
            except Exception as e:
                st.error(f"Erro ao calcular o Teste de Levene: {e}. Pode haver grupos com variância zero ou dados insuficientes. Assumindo homogeneidade para prosseguir.")
                levene_p_val = 1.0
        
        st.session_state['levene_p_anova'] = levene_p_val # Armazena para uso posterior

        # Construção da fórmula e execução da ANOVA
        st.write("#### Resultados da ANOVA")
        formula_terms = [f'C({col})' for col in st.session_state['anova_ivs_current']]
        
        if len(formula_terms) > 1:
            # Para ANOVA Fatorial, incluímos a interação.
            # Ex: Y ~ C(Fator1) * C(Fator2)
            formula = f"{st.session_state['anova_dv_current']} ~ {' * '.join(formula_terms)}"
        else:
            # Para ANOVA Unifatorial
            formula = f"{st.session_state['anova_dv_current']} ~ {formula_terms[0]}"
        
        try:
            model = ols(formula, data=df_anova).fit()
            # anova_lm(typ=2) é geralmente mais robusto para designs desbalanceados.
            # Inclui sum_sq, df, F, PR(>F) e para o Residual (Error) também
            anova_table = anova_lm(model, typ=2)
            st.dataframe(anova_table)

            # Cálculo e Exibição do Tamanho do Efeito (Eta-Quadrado Parcial)
            st.write("#### Tamanho do Efeito (Eta-Quadrado Parcial, ηp²)")
            eta_squared_data = []
            for term in anova_table.index:
                if term not in ['Residual', 'Intercept']: # Não calculamos eta-quadrado para Residual ou Intercept
                    eta_p2 = calculate_partial_eta_squared(anova_table, term)
                    eta_squared_data.append({'Termo': term, 'ηp²': eta_p2})
            
            if eta_squared_data:
                eta_squared_df = pd.DataFrame(eta_squared_data)
                st.dataframe(eta_squared_df.set_index('Termo').round(3))
                st.info("Interpretação do ηp² (Cohen): 0.01 (pequeno), 0.06 (médio), 0.14 (grande).")
            else:
                st.info("Nenhum termo de efeito para calcular ηp².")

            # Armazena os resultados no session_state para acesso pelos botões post-hoc
            st.session_state['anova_results'] = {
                'df_anova': df_anova,
                'dv_col': st.session_state['anova_dv_current'],
                'iv_cols': st.session_state['anova_ivs_current'],
                'anova_table': anova_table,
                'formula': formula
            }
            st.session_state['anova_table'] = anova_table # Salva a tabela ANOVA completa para post-hoc

            # Verificação de significância para ativar post-hoc
            # Se for fatorial, verifica se há algum termo significativo.
            p_val_overall_significant = False
            for term in st.session_state['anova_ivs_current']: # Checa os fatores principais
                 if term in anova_table.index and anova_table.loc[term, 'PR(>F)'] < 0.05:
                    p_val_overall_significant = True
                    break
            # Verifica também interações se existirem
            if len(st.session_state['anova_ivs_current']) > 1:
                interaction_term = ':'.join([f'C({c})' for c in st.session_state['anova_ivs_current']])
                if interaction_term in anova_table.index and anova_table.loc[interaction_term, 'PR(>F)'] < 0.05:
                    p_val_overall_significant = True

            if p_val_overall_significant:
                st.success("Há pelo menos um efeito estatisticamente significativo. Prossiga para a análise Post-Hoc.")
            else:
                st.info("Não há efeitos estatisticamente significativos em geral. Nenhuma análise post-hoc necessária.")


        except Exception as e:
            st.error(f"Erro ao executar ANOVA: {e}. Verifique se as variáveis selecionadas são apropriadas e se não há problemas nos dados (ex: poucas observações por grupo, variância zero).")
            # Limpa o session_state em caso de erro na ANOVA
            if 'anova_results' in st.session_state:
                del st.session_state['anova_results']
            if 'anova_table' in st.session_state:
                del st.session_state['anova_table']
            if 'levene_p_anova' in st.session_state:
                del st.session_state['levene_p_anova']
            
    # --- Seção para Análises Post-Hoc (APENAS se a ANOVA foi executada e há resultados) ---
    if 'anova_results' in st.session_state and 'anova_table' in st.session_state:
        st.write("---")
        st.write("#### Análise Post-Hoc (Comparações Múltiplas)")
        st.info("Esta seção permite realizar comparações post-hoc se a ANOVA geral indicar um efeito significativo.")

        df_anova_results = st.session_state['anova_results']['df_anova']
        dv_col_results = st.session_state['anova_results']['dv_col']
        iv_cols_results = st.session_state['anova_results']['iv_cols']
        anova_table_results = st.session_state['anova_table']
        levene_p_val_results = st.session_state['levene_p_anova']

        # Exibir opções de post-hoc para cada fator significativo
        # Fatores principais
        significant_main_factors = []
        for factor in iv_cols_results:
            term_name = f'C({factor})'
            if term_name in anova_table_results.index and anova_table_results.loc[term_name, 'PR(>F)'] < 0.05:
                significant_main_factors.append(factor)
        
        # Termos de interação (se houver)
        significant_interaction_terms = []
        if len(iv_cols_results) > 1:
            interaction_term_name = ':'.join([f'C({c})' for c in iv_cols_results])
            if interaction_term_name in anova_table_results.index and anova_table_results.loc[interaction_term_name, 'PR(>F)'] < 0.05:
                # Se a interação for significativa, muitas vezes não se interpreta os efeitos principais
                # diretamente, mas sim a interação. No entanto, para fins de exemplo, vamos listar a interação.
                significant_interaction_terms.append("Interação: " + " x ".join(iv_cols_results))


        if not significant_main_factors and not significant_interaction_terms:
            st.info("Não há fatores principais ou termos de interação significativos na ANOVA geral para realizar análises post-hoc.")
        else:
            st.markdown("**Fatores/Interações Significativas para Análise Post-Hoc:**")
            for factor in significant_main_factors:
                st.write(f"- {factor}")
            for term in significant_interaction_terms:
                st.write(f"- {term}")

            # Selecione qual fator/interação para o post-hoc
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
                        key=f"posthoc_method_select_{selected_posthoc_factor}" # Chave única
                    )

                    # Botão para executar Post-Hoc
                    # O botão do post-hoc agora aciona apenas o cálculo do post-hoc, não toda a ANOVA
                    if st.button(f"Executar Post-Hoc para {selected_posthoc_factor}", key=f"run_posthoc_button_{selected_posthoc_factor}"):
                        if posthoc_method == "Tukey HSD (se variâncias homogêneas)":
                            if levene_p_val_results < 0.05:
                                st.warning("Tukey HSD assume homogeneidade das variâncias. O teste de Levene indicou heterogeneidade. Considere Games-Howell.")
                            try:
                                tukey_result = pairwise_tukeyhsd(endog=df_anova_results[dv_col_results], 
                                                                 groups=df_anova_results[selected_posthoc_factor], 
                                                                 alpha=0.05)
                                st.text(tukey_result.summary().as_text())
                                fig = tukey_result.plot_simultaneous()
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Erro ao executar Tukey HSD: {e}")
                        elif posthoc_method == "Games-Howell (se variâncias heterogêneas)":
                            if levene_p_val_results >= 0.05:
                                st.warning("Games-Howell é para variâncias heterogêneas. O teste de Levene indicou homogeneidade. Considere Tukey HSD.")
                            try:
                                gameshowell_result = pairwise_gameshowell(data=df_anova_results, 
                                                                           dv=dv_col_results, 
                                                                           between=selected_posthoc_factor)
                                st.dataframe(gameshowell_result)
                            except Exception as e:
                                st.error(f"Erro ao executar Games-Howell: {e}")
            else:
                st.info("Nenhum fator principal significativo para análise post-hoc individual.")

            if significant_interaction_terms:
                st.write("---")
                st.write("#### Interpretação de Interação")
                st.info("Quando um termo de interação é significativo, a relação de um fator com a variável dependente muda dependendo dos níveis do outro fator. Isso geralmente exige a criação de gráficos de interação.")
                st.warning("A interpretação de interações é complexa e pode exigir visualizações personalizadas (e.g., gráficos de linha de interação).")
                # Futuras melhorias podem incluir gráficos de interação automáticos.

    # --- Gráficos de Visualização ---
    st.write("---")
    st.write("#### Gráficos de Visualização dos Grupos")
    if 'anova_results' in st.session_state:
        df_anova_plot = st.session_state['anova_results']['df_anova']
        dv_col_plot = st.session_state['anova_results']['dv_col']
        iv_cols_plot = st.session_state['anova_results']['iv_cols']

        # Gráfico de Barras de Médias com Erro Padrão para cada fator individual
        for iv_plot in iv_cols_plot:
            st.write(f"##### Gráfico de Médias de {dv_col_plot} por {iv_plot}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, errorbar='se', ax=ax, palette='viridis')
            ax.set_title(f'Média de {dv_col_plot} por {iv_plot} com Erro Padrão')
            ax.set_xlabel(iv_plot)
            ax.set_ylabel(f'Média de {dv_col_plot}')
            st.pyplot(fig)
            plt.close(fig)

            # Boxplot
            st.write(f"##### Boxplot de {dv_col_plot} por {iv_plot}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, ax=ax, palette='viridis')
            sns.stripplot(data=df_anova_plot, x=iv_plot, y=dv_col_plot, color='black', size=3, jitter=True, ax=ax)
            ax.set_title(f'Boxplot de {dv_col_plot} por {iv_plot}')
            ax.set_xlabel(iv_plot)
            ax.set_ylabel(dv_col_plot)
            st.pyplot(fig)
            plt.close(fig)
        
        # Gráfico de interação (se houver 2 fatores)
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

    # Teste T de Uma Amostra
    if test_type == "Teste T de Uma Amostra":
        st.markdown("#### Teste T de Uma Amostra")
        one_sample_col = st.selectbox("Selecione a variável numérica:", num_cols, key="one_sample_col")
        pop_mean = st.number_input("Média da população a ser testada (μ₀):", value=0.0, key="pop_mean")

        if st.button("Executar Teste T de Uma Amostra", key="run_one_sample_t_test"):
            if one_sample_col:
                sample_data = df[one_sample_col].dropna()
                if sample_data.empty:
                    st.warning("A coluna selecionada não possui dados válidos para o teste de uma amostra.")
                    return
                
                stat, p = stats.ttest_1samp(sample_data, pop_mean)
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
            else:
                st.warning("Selecione uma variável numérica.")

    # Teste T de Duas Amostras (Independentes)
    elif test_type == "Teste T de Duas Amostras (Independentes)":
        st.markdown("#### Teste T de Duas Amostras Independentes")
        dv_col_ind = st.selectbox("Variável Dependente (Numérica):", num_cols, key="dv_col_ind")
        group_col_ind = st.selectbox("Variável de Agrupamento (Categórica, 2 grupos):", cat_cols, key="group_col_ind")

        if group_col_ind:
            unique_groups = df[group_col_ind].dropna().unique().tolist()
            if len(unique_groups) != 2:
                st.warning("A variável de agrupamento deve ter exatamente dois valores únicos para o teste t independente.")
                if len(unique_groups) > 2:
                    st.info(f"Variável '{group_col_ind}' tem mais de 2 grupos: {unique_groups}. Considere usar ANOVA.")
                return

            group1_name = unique_groups[0]
            group2_name = unique_groups[1]
            st.write(f"Grupos detectados: '{group1_name}' e '{group2_name}'.")

            if st.button("Executar Teste T Independente", key="run_independent_t_test"):
                if dv_col_ind and group_col_ind:
                    # Remove NaNs apenas para as colunas e linhas relevantes
                    df_filtered = df[[dv_col_ind, group_col_ind]].dropna()
                    
                    group1_data = df_filtered[df_filtered[group_col_ind] == group1_name][dv_col_ind]
                    group2_data = df_filtered[df_filtered[group_col_ind] == group2_name][dv_col_ind]

                    if group1_data.empty or group2_data.empty:
                        st.warning("Um dos grupos está vazio após a remoção de valores faltantes. Verifique seus dados.")
                        return

                    st.write(f"### Resultados do Teste T Independente para '{dv_col_ind}'")
                    
                    st.write("#### Estatísticas Descritivas por Grupo")
                    st.write(f"**Grupo '{group1_name}':** Média={group1_data.mean():.3f}, DP={group1_data.std():.3f}, N={len(group1_data)}")
                    st.write(f"**Grupo '{group2_name}':** Média={group2_data.mean():.3f}, DP={group2_data.std():.3f}, N={len(group2_data)}")

                    # Teste de Levene para homogeneidade de variâncias
                    stat_levene, p_levene = stats.levene(group1_data, group2_data)
                    st.write("#### Teste de Homogeneidade de Variâncias (Levene's Test)")
                    st.write(f"Estatística de Levene: {stat_levene:.3f}")
                    st.write(f"Valor p de Levene: {p_levene:.3f}")

                    equal_var = True
                    if p_levene < 0.05:
                        st.warning("Variâncias NÃO são homogêneas (p < 0.05). Será utilizado o Teste t de Welch (não assume igualdade de variâncias).")
                        equal_var = False
                    else:
                        st.success("Variâncias são homogêneas (p >= 0.05).")

                    # Executa o teste t
                    stat_t, p_t = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                    d = cohens_d(group1_data, group2_data)

                    st.write("#### Resultados do Teste T")
                    st.write(f"Estatística T: {stat_t:.3f}")
                    st.write(f"Valor p: {p_t:.3f}")
                    st.write(f"Cohen's d (Tamanho do Efeito): {d:.3f}")

                    if p_t < 0.05:
                        st.success(f"Há uma diferença significativa na média de '{dv_col_ind}' entre '{group1_name}' e '{group2_name}'.")
                    else:
                        st.info(f"Não há diferença significativa na média de '{dv_col_ind}' entre '{group1_name}' e '{group2_name}'.")

                    # Boxplot
                    st.write("#### Boxplot por Grupo")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(data=df_filtered, x=group_col_ind, y=dv_col_ind, ax=ax, palette='Pastel1')
                    sns.stripplot(data=df_filtered, x=group_col_ind, y=dv_col_ind, color='black', size=4, jitter=True, ax=ax)
                    ax.set_title(f'Boxplot de {dv_col_ind} por {group_col_ind}')
                    ax.set_xlabel(group_col_ind)
                    ax.set_ylabel(dv_col_ind)
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    st.warning("Selecione a variável dependente e a variável de agrupamento.")
        else:
            st.info("Nenhuma coluna categórica disponível para agrupamento.")

    # Teste T de Amostras Pareadas
    elif test_type == "Teste T de Amostras Pareadas":
        st.markdown("#### Teste T de Amostras Pareadas")
        st.info("Este teste é para comparar duas medidas da *mesma* amostra em diferentes pontos no tempo ou sob diferentes condições (e.g., antes e depois).")
        
        # Colunas numéricas que podem ser candidatas a pareadas
        num_cols_for_paired = [col for col in num_cols if df[col].nunique() > 1] # Pelo menos 2 valores únicos

        col_pre = st.selectbox("Variável 'Pré' (e.g., antes da intervenção):", num_cols_for_paired, key="col_pre_paired")
        col_post = st.selectbox("Variável 'Pós' (e.g., depois da intervenção):", num_cols_for_paired, key="col_post_paired")

        if st.button("Executar Teste T Pareado", key="run_paired_t_test"):
            if col_pre and col_post and col_pre != col_post:
                # Garante que apenas as linhas onde ambas as colunas têm dados sejam usadas
                df_paired = df[[col_pre, col_post]].dropna()

                if df_paired.empty:
                    st.warning("Não há pares de dados válidos para o teste pareado após remover valores faltantes.")
                    return

                stat, p = stats.ttest_rel(df_paired[col_pre], df_paired[col_post])
                d = cohens_d_paired(df_paired[col_pre], df_paired[col_post])

                st.write(f"### Resultados do Teste T Pareado para '{col_pre}' vs '{col_post}'")
                st.write(f"Média de '{col_pre}': {df_paired[col_pre].mean():.3f}")
                st.write(f"Média de '{col_post}': {df_paired[col_post].mean():.3f}")
                st.write(f"Média da Diferença (Pós - Pré): {(df_paired[col_post] - df_paired[col_pre]).mean():.3f}")
                st.write(f"Estatística T: {stat:.3f}")
                st.write(f"Valor p: {p:.3f}")
                st.write(f"Graus de Liberdade: {len(df_paired) - 1}")
                st.write(f"Cohen's d (Tamanho do Efeito): {d:.3f}")

                if p < 0.05:
                    st.success(f"Há uma diferença estatisticamente significativa entre as médias de '{col_pre}' e '{col_post}'.")
                else:
                    st.info(f"Não há diferença significativa entre as médias de '{col_pre}' e '{col_post}'.")

                # Gráfico de Dispersão com Linha de Conexão (Paired Plot)
                st.markdown("#### Gráfico de Dispersão dos Pares")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Crie um DataFrame para o gráfico que facilite a visualização pareada
                plot_df = pd.DataFrame({
                    'Condição': [col_pre] * len(df_paired) + [col_post] * len(df_paired),
                    'Valor': pd.concat([df_paired[col_pre], df_paired[col_post]]),
                    'ID': list(range(len(df_paired))) * 2 # Para conectar os pontos
                })

                sns.lineplot(data=plot_df, x='Condição', y='Valor', units='ID', 
                             estimator=None, color='gray', alpha=0.6, ax=ax, marker='o')
                sns.pointplot(data=plot_df, x='Condição', y='Valor', 
                              estimator='mean', color='blue', ax=ax, markers='D', linestyles='--')

                ax.set_title(f'Comparação de {col_pre} vs {col_post} (Medidas Repetidas)')
                ax.set_xlabel('Condição')
                ax.set_ylabel('Valor')
                st.pyplot(fig)
                plt.close(fig)

            elif col_pre == col_post:
                st.warning("Por favor, selecione duas variáveis diferentes para o teste de medidas repetidas.")
            else:
                st.warning("Selecione ambas as variáveis para o teste pareado.")


# 5. Clustering
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
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(scaled_data)

            if num_clusters > 1:
                silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                st.success(f"Silhouette Score: {silhouette_avg:.3f}")
                st.info("O Silhouette Score varia de -1 (pior) a +1 (melhor). Valores próximos de 1 indicam clusters bem definidos e separados. Valores próximos de 0 indicam sobreposição. Valores negativos indicam atribuição incorreta.")

            # Persistência no session_state
            st.session_state['cluster_labels_latest'] = cluster_labels
            st.session_state['cluster_features_latest'] = selected_cols_for_clustering
            st.session_state['scaled_data_for_pca'] = scaled_data # Armazena dados escalados para PCA

        except Exception as e:
            st.error(f"Ocorreu um erro ao executar o K-Means: {e}. Verifique a seleção das variáveis e o número de clusters.")
            return

        # 3. Visualização dos Clusters
        st.write("### 3. Visualização dos Clusters")
        df_with_clusters = df_cluster_clean.copy()
        df_with_clusters['Cluster'] = cluster_labels.astype(str)
        st.dataframe(df_with_clusters.head())

        # --- Gráfico de Dispersão dos Clusters usando PCA ---
        st.write("#### Gráfico de Dispersão dos Clusters (PCA)")
        if scaled_data.shape[1] >= 2: # Precisa de pelo menos 2 dimensões para PCA 2D
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
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
                s=100, # Tamanho dos pontos
                alpha=0.7 # Transparência
            )
            ax_scatter.set_title('Visualização dos Clusters (PCA)')
            ax_scatter.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            ax_scatter.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            ax_scatter.legend(title='Cluster')
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
            st.info("Este gráfico projeta os dados em duas dimensões principais (PCA) para visualizar a separação dos clusters. Os rótulos nos eixos indicam a proporção da variância total explicada por cada componente principal.")
        else:
            st.warning("Não é possível gerar um gráfico de dispersão 2D usando PCA. As variáveis selecionadas para clustering têm menos de 2 dimensões. Selecione pelo menos duas variáveis numéricas.")
    
    # 4. Características dos Clusters e botão para adicionar rótulo
    if 'cluster_labels_latest' in st.session_state and 'cluster_features_latest' in st.session_state:
        st.write("### 4. Características dos Clusters")

        df_with_clusters_for_means = df_cluster_clean.copy()
        df_with_clusters_for_means['Cluster'] = st.session_state['cluster_labels_latest'].astype(str)
        st.dataframe(df_with_clusters_for_means.groupby('Cluster')[st.session_state['cluster_features_latest']].mean().round(2))

        if st.button("Adicionar rótulos de Cluster ao DataFrame processado", key="add_cluster_to_df_button"):
            cluster_labels = st.session_state['cluster_labels_latest']
            selected_cols_for_clustering = st.session_state['cluster_features_latest']

            cluster_series = pd.Series(cluster_labels, index=df_cluster_clean.index, name='Cluster_KMeans')

            df_processed_temp = st.session_state.df_processed.copy()
            df_processed_temp['Cluster_KMeans'] = cluster_series.reindex(df_processed_temp.index)
            df_processed_temp['Cluster_KMeans'] = df_processed_temp['Cluster_KMeans'].fillna('Não_Clusterizado').astype('category')
            st.session_state.df_processed = df_processed_temp

            st.success("Coluna 'Cluster_KMeans' adicionada ao DataFrame processado na sessão.")

def show_exploratory_analysis():
    st.header("📊 Análise Exploratória de Dados")

    if st.session_state['df_processed'] is None or st.session_state['df_processed'].empty:
        st.warning("⚠️ Dados não carregados ou pré-processados. Por favor, complete as etapas anteriores.")
        return

    df_ea = st.session_state['df_processed'].copy()  # Use uma cópia para as análises

    st.info("Nesta seção, você pode realizar várias análises exploratórias no seu DataFrame processado. Use os expanders abaixo para selecionar o tipo de análise.")

    # --- Cada tipo de análise em um expander ---

    # 1️⃣ Visualização Geral
    with st.expander("📈 Visualização Geral do DataFrame"):
        st.subheader("Visão Geral do DataFrame Processado")
        st.dataframe(df_ea.head())
        st.write(f"Dimensões: {df_ea.shape[0]} linhas, {df_ea.shape[1]} colunas.")

        st.subheader("Informações sobre as Colunas")
        buffer = pd.io.common.StringIO()
        df_ea.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # 2️⃣ 🔥 NOVO PAINEL - Análise Descritiva dos Dados Numéricos
    with st.expander("📊 Análise Descritiva dos Dados Numéricos"):
        st.subheader("Estatísticas Descritivas - Variáveis Numéricas")
        
        num_cols = df_ea.select_dtypes(include=np.number).columns.tolist()
        selected_num_cols = st.multiselect(
            "Selecione as variáveis numéricas para visualizar as estatísticas descritivas:",
            num_cols,
            default=num_cols
        )

        if selected_num_cols:
            st.dataframe(df_ea[selected_num_cols].describe())
        else:
            st.info("Selecione pelo menos uma variável numérica.")

    # 3️⃣ Análise de Contingência
    with st.expander("📊 Análise de Contingência e Frequência"):
        show_contingency_analysis(df_ea)

    # 4️⃣ Matriz de Correlação
    with st.expander("🔗 Correlações numéricas"):
        show_correlation_matrix_interface(df_ea)

    # 5️⃣ Testes T
    with st.expander("🧪 Testes T (Uma Amostra, Independentes, Pareadas)"):
        show_t_tests(df_ea)

    # 6️⃣ ANOVA
    with st.expander("📈 Análise de Variância (ANOVA)"):
        show_anova_analysis(df_ea)

    # 7️⃣ Cluster (K-Means)
    with st.expander("🔍 Análise de Cluster (K-Means)"):
        show_clustering_analysis(df_ea)