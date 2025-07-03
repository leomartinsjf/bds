import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import io
import xarray as xr

# ===============================
def reset_bayesian_analysis_state():
    keys_to_reset = [
        "bayes_reg_target", "bayes_reg_target_select", "bayes_reg_feature_cols", "bayes_reg_feature_select",
        "reg_chains", "reg_draws", "reg_tune", "reg_seed",
        "idata_bayesian", "bayesian_model", "bayesian_features",
        "bayesian_analysis_type_run",
        "bayes_ttest_num", "bayes_ttest_cat", "idata_bayesian_ttest",
        "ttest_chains", "ttest_draws", "ttest_tune", "ttest_seed",
        "bayes_anova_num", "bayes_anova_cat", "idata_bayesian_anova",
        "anova_chains", "anova_draws", "anova_tune", "anova_seed",
        "anova_group_pair_select"
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]

# ===============================
# INTERPRETAÇÕES BAYESIANAS GERAIS
# ===============================
def interpretar_bf(bf):
    # Valores de BF Baseados em Jeffreys (1961) ou Kass & Raftery (1995)
    if bf < 1/100:
        return "Evidência extrema a favor do modelo nulo."
    elif bf < 1/30:
        return "Evidência muito forte a favor do modelo nulo."
    elif bf < 1/10:
        return "Evidência forte a favor do modelo nulo."
    elif bf < 1/3:
        return "Evidência moderada a favor do modelo nulo."
    elif bf < 1:
        return "Evidência fraca a favor do modelo nulo."
    elif bf == 1:
        return "Nenhuma evidência."
    elif 1 < bf <= 3:
        return "Evidência fraca a favor do modelo alternativo."
    elif 3 < bf <= 10:
        return "Evidência moderada a favor do modelo alternativo."
    elif 10 < bf <= 30:
        return "Evidência forte a favor do modelo alternativo."
    elif 30 < bf <= 100:
        return "Evidência muito forte a favor do modelo alternativo."
    else: # bf > 100
        return "Evidência extrema a favor do modelo alternativo."

def interpretar_rope(rope_prob):
    # Probabilidade de que o intervalo de credibilidade esteja dentro da Região de Equivalência Prática (ROPE)
    if rope_prob > 0.95:
        return "A maior parte da distribuição posterior está dentro do ROPE. Indica que a diferença é **irrelevante** na prática."
    elif rope_prob < 0.05:
        return "A maior parte da distribuição posterior está fora do ROPE. Indica que a diferença é **relevante** na prática."
    else:
        return "A distribuição posterior cruza o ROPE. Há **incerteza** se a diferença é relevante ou irrelevante."

# ===============================
# FUNÇÕES DE ANÁLISE
# ===============================
def run_bayesian_regression(df, target, features, draws=2000, tune=1000, chains=4, seed=None):
    """
    Executa uma regressão linear bayesiana e retorna o InferenceData.
    """
    data = df[[target] + features].dropna()
    X = data[features].values
    y = data[target].values

    # Padronização das features para melhor desempenho do sampler
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    y_scaled = (y - y.mean()) / y.std() # Também padroniza o target para priors mais robustos

    with pm.Model() as model:
        # Priors para os coeficientes de regressão (beta)
        # Assumimos priors fracamente informativos
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X_scaled.shape[1])
        # Prior para o intercepto
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        # Prior para o desvio padrão do erro (deve ser positivo)
        sigma = pm.HalfCauchy("sigma", beta=1)

        # Média da distribuição da variável dependente
        mu = pm.Deterministic("mu", intercept + pm.math.dot(X_scaled, beta))

        # Variável observada (likelihood)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

        # Amostragem MCMC
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, return_inferencedata=True)

    # Calculando um Bayes Factor de inclusão (BFIncl) simples
    # Comparando a evidência de cada coeficiente ser positivo versus negativo
    # Isso não é um BF de modelo vs nulo, mas uma proporção de amostras positivas/negativas
    # Um BF real para regressão é mais complexo, geralmente usando bridge sampling ou WAIC/LOO para comparação de modelos.
    # Aqui, a métrica é mais sobre a direção do efeito.
    p_pos = (idata.posterior["beta"].mean(dim=["chain", "draw"]) > 0).values
    p_neg = (idata.posterior["beta"].mean(dim=["chain", "draw"]) < 0).values
    # Ajuste para evitar divisão por zero se todas as amostras forem de um lado
    bf_incl = (p_pos + 1e-6) / (p_neg + 1e-6)
    idata.attrs["BFIncl_beta"] = dict(zip(features, bf_incl))

    return idata

def run_bayesian_ttest(df, num_col, cat_col, draws=2000, tune=1000, chains=4, seed=None):
    """
    Executa um teste t bayesiano para dois grupos.
    """
    data = df[[num_col, cat_col]].dropna()
    grupos = data[cat_col].unique()
    if len(grupos) != 2:
        st.error("A variável categórica deve ter exatamente dois grupos para o Teste t Bayesiano.")
        return None

    v1 = data[data[cat_col] == grupos[0]][num_col].values
    v2 = data[data[cat_col] == grupos[1]][num_col].values

    with pm.Model() as model:
        # Priors para as médias dos grupos
        mu1 = pm.Normal("mu1", mu=np.mean(v1), sigma=np.std(v1)*2) # Priors levemente informativos
        mu2 = pm.Normal("mu2", mu=np.mean(v2), sigma=np.std(v2)*2)

        # Priors para os desvios padrão dos grupos
        sigma1 = pm.HalfNormal("sigma1", sigma=np.std(v1)*2)
        sigma2 = pm.HalfNormal("sigma2", sigma=np.std(v2)*2)

        # Likelihood
        y1 = pm.Normal("y1", mu=mu1, sigma=sigma1, observed=v1)
        y2 = pm.Normal("y2", mu=mu2, sigma=sigma2, observed=v2)

        # Parâmetro de interesse: diferença entre as médias
        diff_of_means = pm.Deterministic("diff", mu1 - mu2)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, return_inferencedata=True)

    # Calculando um Bayes Factor de inclusão (BFIncl) para a diferença
    # Novamente, uma simplificação: proporção de amostras positivas vs negativas para a diferença
    diff_samples = idata.posterior["diff"].values.flatten()
    p_diff_gt_0 = np.mean(diff_samples > 0)
    p_diff_lt_0 = np.mean(diff_samples < 0)
    bf_incl_diff = (p_diff_gt_0 + 1e-6) / (p_diff_lt_0 + 1e-6)
    idata.attrs["BFIncl_diff"] = bf_incl_diff

    return idata


def run_bayesian_anova(df, num_col, cat_col, draws=2000, tune=1000, chains=4, seed=None):
    """
    Executa uma ANOVA Bayesiana para um fator.
    Estima as médias dos grupos.
    """
    data = df[[num_col, cat_col]].dropna()
    y = data[num_col].values
    group_labels, x = pd.factorize(data[cat_col]) # Converte rótulos em inteiros
    k = len(np.unique(x)) # Número de grupos

    with pm.Model() as model:
        # Prior para a média global (populacional)
        mu_global = pm.Normal("mu_global", mu=np.mean(y), sigma=np.std(y)*2)

        # Prior para os desvios das médias dos grupos em relação à média global
        # Permite que cada grupo tenha uma média diferente
        # Usamos Non-centered parameterization para melhor amostragem se houver muitos grupos
        sigma_group = pm.HalfNormal("sigma_group", sigma=1) # Sigma para a variação entre grupos
        raw_deviations = pm.Normal("raw_deviations", mu=0, sigma=1, shape=k)
        mu_groups = pm.Deterministic("mu_groups", mu_global + raw_deviations * sigma_group)

        # Prior para o desvio padrão residual (erro)
        sigma_residual = pm.HalfNormal("sigma_residual", sigma=np.std(y)*2)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu_groups[group_labels], sigma=sigma_residual, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, return_inferencedata=True)

    # Armazena os nomes dos grupos no idata para plotagem e resumo
    idata.attrs["group_names"] = list(x)
    return idata

def run_bayesian_factorial_anova(df, num_col, cat1, cat2, draws=2000, tune=1000, chains=4, seed=None):
    """
    Executa uma ANOVA Fatorial Bayesiana para dois fatores.
    Estima efeitos principais e de interação.
    """
    data = df[[num_col, cat1, cat2]].dropna()
    y = data[num_col].values

    # Fatorizar colunas categóricas para obter códigos numéricos e labels
    labels_a, a_codes = pd.factorize(data[cat1])
    labels_b, b_codes = pd.factorize(data[cat2])

    na = len(a_codes) # Número de níveis do Fator 1
    nb = len(b_codes) # Número de níveis do Fator 2

    with pm.Model() as model:
        # Priors para a média global
        mu_global = pm.Normal("mu_global", mu=np.mean(y), sigma=np.std(y)*2)

        # Efeitos principais para Fator 1 (na-1 elementos para soma zero)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=na-1)
        alpha = pm.Deterministic("alpha", pm.math.concatenate([alpha_raw, [-alpha_raw.sum()]]))

        # Efeitos principais para Fator 2 (nb-1 elementos para soma zero)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=nb-1)
        beta = pm.Deterministic("beta", pm.math.concatenate([beta_raw, [-beta_raw.sum()]]))

        # Efeitos de interação ((na-1) x (nb-1) elementos para soma zero)
        interaction_raw = pm.Normal("interaction_raw", mu=0, sigma=1, shape=(na-1, nb-1))

        # Construir a matriz de interação completa (na x nb) aplicando as restrições de soma zero
        # 1. Calcular os elementos da última coluna com base nas linhas de interaction_raw
        last_col_elements = -pm.math.sum(interaction_raw, axis=1, keepdims=True) # Dimensão: (na-1, 1)

        # 2. Combinar interaction_raw com last_col_elements para formar as (na-1) linhas superiores
        top_na_minus_1_rows = pm.math.concatenate([interaction_raw, last_col_elements], axis=1) # Dimensão: (na-1, nb)

        # 3. Calcular os elementos da última linha com base nas colunas de interaction_raw
        last_row_elements = -pm.math.sum(interaction_raw, axis=0, keepdims=True) # Dimensão: (1, nb-1)

        # 4. Calcular o último elemento (canto inferior direito)
        very_last_element = pm.math.sum(interaction_raw) # Escalar

        # 5. Combinar last_row_elements com very_last_element para formar a linha inferior
        # Alterado para usar .reshape() diretamente no objeto tensor
        bottom_row = pm.math.concatenate([last_row_elements, very_last_element.reshape((1, 1))], axis=1) # Dimensão: (1, nb)

        # 6. Empilhar as linhas superiores e a linha inferior para formar a matriz de interação completa
        interaction = pm.Deterministic("interaction", pm.math.concatenate([top_na_minus_1_rows, bottom_row], axis=0))

        # Média esperada para cada observação
        mu_expected = mu_global + alpha[labels_a] + beta[labels_b] + interaction[labels_a, labels_b]

        # Prior para o desvio padrão residual
        sigma = pm.HalfNormal("sigma", sigma=np.std(y)*2)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu_expected, sigma=sigma, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, return_inferencedata=True)

    idata.attrs["cat1_names"] = list(a_codes)
    idata.attrs["cat2_names"] = list(b_codes)
    return idata

def compute_bayes_factor(idata_full, idata_reduced):
    """
    Compara dois modelos usando o Bayes Factor (via pseudo-BMA).
    Retorna um DataFrame com a comparação.
    """
    try:
        cmp = az.compare({"completo": idata_full, "reduzido": idata_reduced}, method="BB-pseudo-BMA")
        return cmp
    except Exception as e:
        st.error(f"Erro ao calcular Bayes Factor: {e}")
        return pd.DataFrame({"Erro": [str(e)]})

# ===============================
# INTEGRAÇÃO COM STREAMLIT
# ===============================
def show_bayesian_analysis_page(df):
    st.header("🔍 Painel de Análises Bayesiana")
    st.markdown("""
        Esta seção permite realizar análises estatísticas utilizando a abordagem Bayesiana.
        Ao contrário das análises frequentistas que focam em p-valores, a análise Bayesiana
        fornece distribuições de probabilidade para os parâmetros do modelo, permitindo
        inferências mais intuitivas e a quantificação da evidência a favor de uma hipótese.
    """)

    aba = st.selectbox("Escolha o tipo de análise:", [
        "Regressão Bayesiana", "Teste t Bayesiano", "ANOVA Bayesiana", "ANOVA Fatorial Bayesiana"
    ])

    # Configurações de amostragem comuns
    st.sidebar.subheader("⚙️ Configurações da Amostragem MCMC")
    chains = st.sidebar.slider("Número de cadeias (chains)", min_value=2, max_value=8, value=4)
    draws = st.sidebar.slider("Número de amostras (draws) por cadeia", min_value=500, max_value=5000, value=2000)
    tune = st.sidebar.slider("Warm-up/Tune (descartadas) por cadeia", min_value=500, max_value=5000, value=1000)
    seed = st.sidebar.number_input("Seed para reprodutibilidade", value=42)

    if aba == "Regressão Bayesiana":
        st.subheader("Regressão Linear Bayesiana")
        st.markdown("""
            A regressão bayesiana estima a relação entre uma variável dependente e uma ou mais variáveis independentes,
            fornecendo uma distribuição de probabilidade para os coeficientes.
        """)
        
        # Modified: Add "" as a blank option and set index=0
        numeric_cols_reg = df.select_dtypes(include=np.number).columns.tolist()
        target_options_reg = [""] + numeric_cols_reg
        target = st.selectbox("Variável dependente (target)", target_options_reg, index=0, key="bayes_reg_target")
        
        # Multiselect starts empty by default, no change needed for initial state
        features = st.multiselect("Variáveis independentes (preditoras)",
                                  [c for c in numeric_cols_reg if c != target],
                                  key="bayes_reg_feature_cols")
        
        if st.button("Rodar regressão bayesiana", key="run_reg_bayes"):
            if not target: # Check if target is empty string
                st.warning("Por favor, selecione uma variável dependente.")
                return
            if not features:
                st.warning("Por favor, selecione pelo menos uma variável independente.")
                return

            with st.spinner("Rodando regressão bayesiana..."):
                idata = run_bayesian_regression(df, target, features, draws=draws, tune=tune, chains=chains, seed=seed)
                st.session_state["idata_bayesian"] = idata
                st.session_state["bayesian_features"] = features # Guardar features para exibir BF

            st.success("Regressão Bayesiana concluída!")

        if "idata_bayesian" in st.session_state:
            idata = st.session_state["idata_bayesian"]
            features_used = st.session_state["bayesian_features"]

            st.write("---")
            st.subheader("📈 Resultados da Regressão Bayesiana")

            # 1. Resumo dos Parâmetros
            st.markdown("#### 1.1 Sumário da Distribuição Posterior")
            st.write("""
                O sumário apresenta estatísticas das distribuições posteriores dos parâmetros do modelo:
                - `mean`: Média da distribuição posterior (estimativa pontual).
                - `sd`: Desvio padrão da distribuição posterior (incerteza da estimativa).
                - `hdi_3%`, `hdi_97%`: Intervalo de Credibilidade de Alta Densidade (HDI) de 94%.
                  Representa o intervalo mais estreito que contém 94% da probabilidade posterior.
                - `r_hat`: Fator de escala de Gelman-Rubin. Valores próximos de 1 (idealmente < 1.01)
                  indicam que as cadeias de amostragem convergiram bem.
                - `ess_bulk`, `ess_tail`: Número Efetivo de Amostras (ESS). Indica o número de amostras
                  independentes efetivas. Valores baixos (< 400 por cadeia) sugerem problemas na amostragem.
            """)
            summary = az.summary(idata, var_names=["beta", "intercept", "sigma"]).round(3)
            # Renomear os índices para incluir o nome da feature para os betas
            beta_idx = [f"beta[{i}]" for i in range(len(features_used))]
            new_index = []
            beta_map = dict(zip(beta_idx, features_used))
            for idx in summary.index:
                if idx in beta_map:
                    new_index.append(f"beta ({beta_map[idx]})")
                else:
                    new_index.append(idx)
            summary.index = new_index
            st.dataframe(summary)

            # 2. Bayes Factor de Inclusão (simplificado)
            st.markdown("#### 1.2 Fator de Bayes (BFIncl) para Coeficientes")
            st.write("""
                O Fator de Bayes de Inclusão aqui é uma métrica simplificada que compara
                a proporção de evidência posterior de um coeficiente ser positivo versus negativo.
                Valores > 1 indicam mais evidência para o efeito positivo; < 1 para o negativo.
                Valores extremos indicam forte evidência para a direção do efeito.
            """)
            bf_incl_dict = idata.attrs.get("BFIncl_beta", {})
            for feature, bf in bf_incl_dict.items():
                st.write(f"- **{feature}**: BFIncl = {round(bf, 3)} ({interpretar_bf(bf)})")

            # 3. Visualizações da Distribuição Posterior
            st.markdown("#### 1.3 Visualização das Distribuições Posteriores") # Reordenado o título
            fig_post, ax_post = plt.subplots(figsize=(10, 6))
            az.plot_posterior(idata, var_names=["beta", "intercept"], ax=ax_post)
            plt.suptitle('Distribuições Posteriores dos Coeficientes de Regressão e Intercepto')
            st.pyplot(fig_post)
            plt.close(fig_post) # Fechar a figura para liberar memória

            # Correção para az.plot_trace: obter a figura atual após a chamada
            az.plot_trace(idata, var_names=["beta", "intercept"], compact=True)
            fig_trace = plt.gcf() # Obter a figura criada por az.plot_trace
            plt.suptitle('Gráficos de Rastreamento (Trace Plots) dos Parâmetros de Regressão')
            st.pyplot(fig_trace)
            plt.close(fig_trace)


    elif aba == "Teste t Bayesiano":
        st.subheader("Teste t Bayesiano")
        st.markdown("""
            O teste t Bayesiano compara as médias de dois grupos, fornecendo a distribuição da diferença entre as médias
            e a evidência a favor ou contra a hipótese de diferença.
        """)
        
        # Modified: Add "" as a blank option and set index=0
        numeric_cols_ttest = df.select_dtypes(include=np.number).columns.tolist()
        num_col_options_ttest = [""] + numeric_cols_ttest
        num_col = st.selectbox("Variável numérica", num_col_options_ttest, index=0, key="bayes_ttest_num")
        
        # Modified: Add "" as a blank option and set index=0
        object_cols_ttest = df.select_dtypes(include='object').columns.tolist()
        cat_col_options_ttest = [""] + object_cols_ttest
        cat_col = st.selectbox("Variável categórica binária (com 2 grupos)", cat_col_options_ttest, index=0, key="bayes_ttest_cat")

        if st.button("Rodar teste t bayesiano", key="run_ttest_bayes"):
            if not num_col:
                st.warning("Por favor, selecione a variável numérica.")
                return
            if not cat_col:
                st.warning("Por favor, selecione a variável categórica binária.")
                return

            # Verificar se a variável categórica tem exatamente dois grupos
            unique_groups = df[cat_col].dropna().unique()
            if len(unique_groups) != 2:
                st.error(f"A variável '{cat_col}' possui {len(unique_groups)} grupos. O Teste t Bayesiano requer exatamente 2 grupos.")
                if len(unique_groups) > 0:
                    st.info(f"Grupos detectados: {', '.join(map(str, unique_groups))}")
                return # Stop execution if condition not met

            with st.spinner("Rodando teste t bayesiano..."):
                idata = run_bayesian_ttest(df, num_col, cat_col, draws=draws, tune=tune, chains=chains, seed=seed)
                st.session_state["idata_bayesian_ttest"] = idata
            st.success("Teste t Bayesiano concluído!")

        if "idata_bayesian_ttest" in st.session_state and st.session_state["idata_bayesian_ttest"] is not None:
            idata = st.session_state["idata_bayesian_ttest"]

            st.write("---")
            st.subheader("📈 Resultados do Teste t Bayesiano")

            # 1. Resumo dos Parâmetros
            st.markdown("#### 2.1 Sumário da Distribuição Posterior")
            st.dataframe(az.summary(idata, var_names=["mu1", "mu2", "diff", "sigma1", "sigma2"]).round(3))

            # 2. Bayes Factor para a Diferença
            st.markdown("#### 2.2 Fator de Bayes (BFIncl) para a Diferença de Médias")
            bf_incl_diff = idata.attrs.get("BFIncl_diff", np.nan)
            
            # Ensure unique_groups is available for display
            current_cat_col = st.session_state["bayes_ttest_cat"]
            current_unique_groups = df[current_cat_col].dropna().unique() if current_cat_col else ["Grupo 1", "Grupo 2"]

            st.write(f"- **Diferença de Médias ({current_unique_groups[0]} vs {current_unique_groups[1]})**: BFIncl = {round(bf_incl_diff, 3)} ({interpretar_bf(bf_incl_diff)})")
            st.markdown("Um BFIncl > 1 indica evidência a favor de uma diferença positiva (grupo 1 > grupo 2), enquanto < 1 indica evidência para uma diferença negativa (grupo 1 < grupo 2).")

            # 3. Análise ROPE para a Diferença
            st.markdown("#### 2.3 Análise da Região de Equivalência Prática (ROPE)")
            st.write("""
                O ROPE (Region of Practical Equivalence) permite avaliar se uma diferença observada
                é *relevante* na prática, definindo um intervalo de valores que são considerados
                "praticamente equivalentes a zero".
            """)
            
            # Ensure num_col is available for std calculation
            current_num_col = st.session_state["bayes_ttest_num"]
            std_for_rope = df[current_num_col].std() if current_num_col else 1.0 # Fallback to 1.0 if not selected

            rope_lower = st.number_input("Limite Inferior do ROPE", value=-0.1 * std_for_rope)
            rope_upper = st.number_input("Limite Superior do ROPE", value=0.1 * std_for_rope)

            if rope_lower >= rope_upper:
                st.error("O limite inferior do ROPE deve ser menor que o superior.")
            else:
                try:
                    rope_interval = [rope_lower, rope_upper]
                    # Calcula a probabilidade do diff estar dentro do ROPE
                    diff_samples = idata.posterior["diff"].values.flatten()
                    prob_in_rope = np.mean((diff_samples >= rope_interval[0]) & (diff_samples <= rope_interval[1]))
                    st.write(f"Probabilidade da diferença estar dentro do ROPE [{round(rope_lower, 3)}, {round(rope_upper, 3)}]: **{round(prob_in_rope * 100, 2)}%**")
                    st.text(interpretar_rope(prob_in_rope))
                except Exception as e:
                    st.error(f"Erro ao calcular ROPE: {e}")

            # 4. Visualizações da Distribuição Posterior
            st.markdown("#### 2.4 Visualização das Distribuições Posteriores") # Reordenado o título
            fig_post, ax_post = plt.subplots(figsize=(10, 6))
            az.plot_posterior(idata, var_names=["mu1", "mu2", "diff"], ax=ax_post)
            plt.suptitle('Distribuições Posteriores das Médias dos Grupos e da Diferença')
            st.pyplot(fig_post)
            plt.close(fig_post)

            # Correção para az.plot_trace: obter a figura atual após a chamada
            az.plot_trace(idata, var_names=["mu1", "mu2", "diff"], compact=True)
            fig_trace = plt.gcf()
            plt.suptitle('Gráficos de Rastreamento (Trace Plots) do Teste t')
            st.pyplot(fig_trace)
            plt.close(fig_trace)


    elif aba == "ANOVA Bayesiana":
        st.subheader("ANOVA Bayesiana (Um Fator)")
        st.markdown("""
            A ANOVA Bayesiana com um fator permite comparar as médias de três ou mais grupos,
            fornecendo distribuições posteriores para as médias de cada grupo e o desvio padrão residual.
        """)
        
        # Modified: Add "" as a blank option and set index=0
        numeric_cols_anova = df.select_dtypes(include=np.number).columns.tolist()
        num_col_options_anova = [""] + numeric_cols_anova
        num_col = st.selectbox("Variável dependente (numérica)", num_col_options_anova, index=0, key="bayes_anova_num")
        
        # Modified: Add "" as a blank option and set index=0
        object_cols_anova = df.select_dtypes(include='object').columns.tolist()
        cat_col_options_anova = [""] + object_cols_anova
        cat_col = st.selectbox("Variável categórica (fator)", cat_col_options_anova, index=0, key="bayes_anova_cat")
        
        if st.button("Rodar ANOVA bayesiana", key="run_anova_bayes"):
            if not num_col:
                st.warning("Por favor, selecione a variável dependente numérica.")
                return
            if not cat_col:
                st.warning("Por favor, selecione a variável categórica (fator).")
                return

            if len(df[cat_col].dropna().unique()) < 2:
                st.warning("Selecione uma variável categórica com pelo menos dois grupos.")
                return

            with st.spinner("Rodando ANOVA Bayesiana..."):
                idata = run_bayesian_anova(df, num_col, cat_col, draws=draws, tune=tune, chains=chains, seed=seed)
                st.session_state["idata_bayesian_anova"] = idata
            st.success("ANOVA Bayesiana concluída!")

        if "idata_bayesian_anova" in st.session_state:
            idata = st.session_state["idata_bayesian_anova"]
            group_names = idata.attrs.get("group_names", [])

            st.write("---")
            st.subheader("📈 Resultados da ANOVA Bayesiana")

            # 1. Resumo dos Parâmetros
            st.markdown("#### 3.1 Sumário da Distribuição Posterior das Médias dos Grupos")
            summary = az.summary(idata, var_names=["mu_global", "mu_groups", "sigma_residual"]).round(3)
            # Renomear os índices para incluir o nome do grupo
            group_mu_idx = [f"mu_groups[{i}]" for i in range(len(group_names))]
            new_index = []
            group_mu_map = dict(zip(group_mu_idx, group_names))
            for idx in summary.index:
                if idx in group_mu_map:
                    new_index.append(f"mu_groups ({group_mu_map[idx]})")
                else:
                    new_index.append(idx)
            summary.index = new_index
            st.dataframe(summary)
            st.write("- `mu_global`: Média global estimada.")
            st.write("- `mu_groups`: Média estimada para cada grupo.")
            st.write("- `sigma_residual`: Desvio padrão residual (erro) do modelo.")

            # 3. Visualizações da Distribuição Posterior
            st.markdown("#### 3.2 Visualização das Distribuições Posteriores") # Reordenado o título
            # Correção para az.plot_forest: remover hdi_percent
            fig_forest, ax_forest = plt.subplots(figsize=(10, 6))
            az.plot_forest(idata, var_names=["mu_groups"], combined=True, ax=ax_forest)
            plt.title('Distribuições Posteriores das Médias dos Grupos (HDI 94%)')
            plt.yticks(ticks=np.arange(len(group_names)), labels=group_names) # Definir labels corretos
            st.pyplot(fig_forest)
            plt.close(fig_forest)

            # Correção para az.plot_trace: obter a figura atual após a chamada
            az.plot_trace(idata, var_names=["mu_groups", "sigma_residual"], compact=True)
            fig_trace = plt.gcf()
            plt.suptitle('Gráficos de Rastreamento (Trace Plots) da ANOVA')
            st.pyplot(fig_trace)
            plt.close(fig_trace)

            # 4. Comparação entre Pares de Grupos (Post-hoc Bayesiano)
            st.markdown("#### 3.3 Comparação Post-hoc Bayesiana (Diferenças entre Grupos)") # Reordenado o título
            st.write("Compare as distribuições de diferença entre pares de grupos.")

            all_groups = idata.attrs.get("group_names", [])
            if len(all_groups) > 1:
                group_pairs = list(combinations(all_groups, 2))
                selected_pair = st.selectbox("Selecione um par de grupos para comparar:", group_pairs, format_func=lambda x: f"{x[0]} vs {x[1]}")

                if selected_pair:
                    group1_name, group2_name = selected_pair
                    # Encontrar os índices dos grupos para acessar mu_groups
                    idx1 = idata.attrs["group_names"].index(group1_name)
                    idx2 = idata.attrs["group_names"].index(group2_name)

                    with idata.posterior:
                        # Criar a variável determinística para a diferença
                        diff_pair = idata.posterior["mu_groups"].sel(mu_groups_dim_0=idx1) - idata.posterior["mu_groups"].sel(mu_groups_dim_0=idx2)
                        idata.posterior["diff_pair"] = diff_pair

                    # Plotar a distribuição da diferença
                    fig_diff_pair, ax_diff_pair = plt.subplots(figsize=(8, 5))
                    az.plot_posterior(idata, var_names=["diff_pair"], ref_val=0, ax=ax_diff_pair)
                    ax_diff_pair.set_title(f'Distribuição Posterior da Diferença: {group1_name} - {group2_name}')
                    st.pyplot(fig_diff_pair)
                    plt.close(fig_diff_pair)

                    # Interpretação do BF e ROPE para a diferença de pares
                    diff_samples_pair = idata.posterior["diff_pair"].values.flatten()
                    p_diff_pair_gt_0 = np.mean(diff_samples_pair > 0)
                    p_diff_pair_lt_0 = np.mean(diff_samples_pair < 0)
                    bf_incl_pair = (p_diff_pair_gt_0 + 1e-6) / (p_diff_pair_lt_0 + 1e-6)
                    st.write(f"- **Fator de Bayes (BFIncl) para {group1_name} vs {group2_name}**: {round(bf_incl_pair, 3)} ({interpretar_bf(bf_incl_pair)})")

                    # ROPE para a diferença de pares
                    mean_diff_pair = np.mean(diff_samples_pair)
                    std_diff_pair = np.std(diff_samples_pair)
                    rope_lower_pair = st.number_input(f"Limite Inferior do ROPE para {group1_name}-{group2_name}",
                                                      value=-0.1 * std_diff_pair, key=f"rope_low_{group1_name}_{group2_name}")
                    rope_upper_pair = st.number_input(f"Limite Superior do ROPE para {group1_name}-{group2_name}",
                                                      value=0.1 * std_diff_pair, key=f"rope_up_{group1_name}_{group2_name}")
                    if rope_lower_pair < rope_upper_pair:
                        prob_in_rope_pair = np.mean((diff_samples_pair >= rope_lower_pair) & (diff_samples_pair <= rope_upper_pair))
                        st.write(f"Probabilidade da diferença {group1_name} - {group2_name} estar dentro do ROPE [{round(rope_lower_pair, 3)}, {round(rope_upper_pair, 3)}]: **{round(prob_in_rope_pair * 100, 2)}%**")
                        st.text(interpretar_rope(prob_in_rope_pair))
                    else:
                        st.error("O limite inferior do ROPE deve ser menor que o superior.")


    elif aba == "ANOVA Fatorial Bayesiana":
        st.subheader("ANOVA Fatorial Bayesiana")
        st.markdown("""
            A ANOVA Fatorial Bayesiana analisa o efeito de duas variáveis categóricas (fatores) e suas interações
            sobre uma variável dependente numérica.
        """)
        
        # Modified: Add "" as a blank option and set index=0
        numeric_cols_fact_anova = df.select_dtypes(include=np.number).columns.tolist()
        num_col_options_fact_anova = [""] + numeric_cols_fact_anova
        num_col = st.selectbox("Variável dependente (numérica)", num_col_options_fact_anova, index=0, key="bayes_anova_fact_num")
        
        # Modified: Add "" as a blank option and set index=0
        object_cols_fact_anova = df.select_dtypes(include='object').columns.tolist()
        cat1_options_fact_anova = [""] + object_cols_fact_anova
        cat1 = st.selectbox("Fator 1", cat1_options_fact_anova, index=0, key="bayes_anova_fact_cat1")
        
        # Ensure cat1 is selected before filtering for cat2 to avoid errors with `c != cat1`
        cat2_options_fact_anova = [""]
        if cat1: # Only populate options if cat1 is not blank
            cat2_options_fact_anova.extend([c for c in object_cols_fact_anova if c != cat1])
        cat2 = st.selectbox("Fator 2", cat2_options_fact_anova, index=0, key="bayes_anova_fact_cat2")

        if st.button("Rodar ANOVA fatorial", key="run_anova_fact_bayes"):
            if not num_col:
                st.warning("Por favor, selecione a variável dependente numérica.")
                return
            if not cat1:
                st.warning("Por favor, selecione o Fator 1.")
                return
            if not cat2:
                st.warning("Por favor, selecione o Fator 2.")
                return

            if len(df[cat1].dropna().unique()) < 2 or len(df[cat2].dropna().unique()) < 2:
                st.warning("Ambos os fatores devem ter pelo menos dois grupos.")
                return

            with st.spinner("Rodando ANOVA fatorial Bayesiana..."):
                idata = run_bayesian_factorial_anova(df, num_col, cat1, cat2, draws=draws, tune=tune, chains=chains, seed=seed)
                st.session_state["idata_bayesian_anova_fact"] = idata
            st.success("ANOVA Fatorial Bayesiana concluída!")

        if "idata_bayesian_anova_fact" in st.session_state:
            idata = st.session_state["idata_bayesian_anova_fact"]
            cat1_names = idata.attrs.get("cat1_names", [])
            cat2_names = idata.attrs.get("cat2_names", [])

            st.write("---")
            st.subheader("📈 Resultados da ANOVA Fatorial Bayesiana")

            # 1. Resumo dos Parâmetros
            st.markdown("#### 4.1 Sumário da Distribuição Posterior")
            st.write("""
                - `alpha`: Efeitos principais do Fator 1.
                - `beta`: Efeitos principais do Fator 2.
                - `interaction`: Efeitos de interação entre Fator 1 e Fator 2.
            """)
            summary = az.summary(idata, var_names=["alpha", "beta", "interaction", "mu_global", "sigma"]).round(3)
            
            # Renomear índices para melhor legibilidade
            new_index = []
            for idx in summary.index:
                if idx.startswith("alpha["):
                    i = int(idx.split('[')[1].split(']')[0])
                    new_index.append(f"alpha ({cat1_names[i]})")
                elif idx.startswith("beta["):
                    i = int(idx.split('[')[1].split(']')[0])
                    new_index.append(f"beta ({cat2_names[i]})")
                elif idx.startswith("interaction["):
                    parts = idx.split('[')[1].split(']')[0].split(',')
                    i = int(parts[0])
                    j = int(parts[1])
                    new_index.append(f"interaction ({cat1_names[i]}, {cat2_names[j]})")
                else:
                    new_index.append(idx)
            summary.index = new_index
            st.dataframe(summary)

            # 3. Visualizações da Distribuição Posterior
            st.markdown("#### 4.2 Visualização das Distribuições Posteriores") # Reordenado o título
            # Efeitos principais
            fig_forest_main, ax_forest_main = plt.subplots(1, 2, figsize=(12, 5))
            # Correção para az.plot_forest: remover hdi_percent
            az.plot_forest(idata, var_names=["alpha"], combined=True, ax=ax_forest_main[0])
            ax_forest_main[0].set_title(f'Efeitos Principais: {cat1} (HDI 94%)')
            ax_forest_main[0].set_yticks(ticks=np.arange(len(cat1_names)), labels=cat1_names)

            # Correção para az.plot_forest: remover hdi_percent
            az.plot_forest(idata, var_names=["beta"], combined=True, ax=ax_forest_main[1])
            ax_forest_main[1].set_title(f'Efeitos Principais: {cat2} (HDI 94%)')
            ax_forest_main[1].set_yticks(ticks=np.arange(len(cat2_names)), labels=cat2_names)
            plt.tight_layout()
            st.pyplot(fig_forest_main)
            plt.close(fig_forest_main)

            # Efeitos de interação
            st.write("#### Efeitos de Interação")
            st.write("Os efeitos de interação mostram como o efeito de um fator depende dos níveis do outro fator.")
            # Correção para az.plot_forest: remover hdi_percent
            fig_forest_interaction, ax_forest_interaction = plt.subplots(figsize=(10, 6))
            az.plot_forest(idata, var_names=["interaction"], combined=True, ax=ax_forest_interaction)
            plt.title('Efeitos de Interação (HDI 94%)')
            # Customizar labels do y-axis para interação
            interaction_labels = [f'{cat1_names[i]}, {cat2_names[j]}' for i in range(len(cat1_names)) for j in range(len(cat2_names))]
            plt.yticks(ticks=np.arange(len(interaction_labels)), labels=interaction_labels)
            st.pyplot(fig_forest_interaction)
            plt.close(fig_forest_interaction)


            # Correção para az.plot_trace: obter a figura atual após a chamada
            az.plot_trace(idata, var_names=["alpha", "beta", "interaction"], compact=True)
            fig_trace = plt.gcf()
            plt.suptitle('Gráficos de Rastreamento (Trace Plots) da ANOVA Fatorial')
            st.pyplot(fig_trace)
            plt.close(fig_trace)