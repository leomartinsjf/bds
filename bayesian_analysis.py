# bayesian_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importar bibliotecas Bayesianas
try:
    import pymc as pm
    import arviz as az
    st.session_state['pymc_installed'] = True
except ImportError:
    st.session_state['pymc_installed'] = False
    st.error("A funcionalidade de Análise Bayesiana está desativada porque as bibliotecas 'pymc' e 'arviz' não foram encontradas.")
    st.info("Por favor, instale-as para usar a análise Bayesiana: `pip install pymc arviz`")


def show_bayesian_analysis_page(df):
    st.title("🌌 Análise Bayesiana")

    if not st.session_state.get('pymc_installed', False):
        st.warning("A funcionalidade de Análise Bayesiana está desativada porque as bibliotecas 'pymc' e 'arviz' não foram encontradas.")
        st.stop() # Stop execution if dependencies are not met

    if df.empty:
        st.warning("Por favor, carregue os dados na página 'Carregar Dados' para realizar análises Bayesianas.")
        return

    # --- Mover a Navegação para Tipos de Análise Bayesiana para o corpo da página ---
    st.markdown("### Selecione o Tipo de Análise Bayesiana:")
    bayesian_analysis_type = st.radio( # ALTERADO: de st.sidebar.radio para st.radio
        " ", # Título vazio porque já temos um markdown acima
        ("Regressão Linear Bayesiana", "Teste t Bayesiano", "ANOVA Bayesiana Simples"),
        key="main_bayes_analysis_type_selector" # Adicionado uma chave única
    )
    st.markdown("---") # Adiciona um separador visual

    # As configurações de amostragem (MCMC) permanecem na barra lateral
    st.sidebar.header("⚙️ Configurações de Amostragem (MCMC)")

    # --- REGRESSÃO LINEAR BAYESIANA (EXISTENTE) ---
    if bayesian_analysis_type == "Regressão Linear Bayesiana":
        st.subheader("Regressão Linear Bayesiana")
        st.info("Este módulo implementa um modelo de regressão linear Bayesiana para prever uma variável numérica com base em features numéricas.")

        all_columns = df.columns.tolist()
        
        target_column = st.selectbox(
            "Selecione a Coluna Alvo (Variável Dependente):",
            options=all_columns,
            key="bayes_reg_target_col"
        )

        feature_columns = st.multiselect(
            "Selecione as Features (Variáveis Independentes):",
            options=[col for col in all_columns if col != target_column],
            key="bayes_reg_feature_cols"
        )

        if not target_column or not feature_columns:
            st.info("Selecione a coluna alvo e pelo menos uma feature para prosseguir com a modelagem Bayesiana.")
            return

        data_for_model = df[[target_column] + feature_columns].copy()
        numeric_features = data_for_model[feature_columns].select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_features:
            st.warning("Nenhuma feature numérica selecionada para o modelo. A regressão linear Bayesiana tipicamente requer features numéricas.")
            st.info("Por favor, selecione features que sejam números.")
            return
        
        X = data_for_model[numeric_features]
        y = data_for_model[target_column]

        original_rows = X.shape[0]
        data_for_model.dropna(subset=[target_column] + numeric_features, inplace=True)
        if data_for_model.shape[0] < original_rows:
            st.warning(f"Foram removidas {original_rows - data_for_model.shape[0]} linhas com valores ausentes (NaNs) nas colunas selecionadas.")
            X = data_for_model[numeric_features]
            y = data_for_model[target_column]
        
        if X.empty or y.empty:
            st.error("Após a remoção de NaNs, o DataFrame de features ou a série da coluna alvo está vazio. Não é possível prosseguir com o modelo Bayesiano.")
            return

        # Configurações de Amostragem (MCMC) - Continuam na Sidebar
        n_chains = st.sidebar.slider("Número de Cadeias (Chains):", 1, 4, 2, key="reg_chains")
        n_draws = st.sidebar.slider("Número de Amostras por Cadeia (Draws):", 500, 5000, 2000, key="reg_draws")
        n_tune = st.sidebar.slider("Número de Amostras para Afinação (Tune):", 100, 1000, 500, key="reg_tune")
        random_seed = st.sidebar.slider("Seed Aleatória (Random Seed):", 0, 100, 42, key="reg_seed")

        if st.button("Executar Análise Bayesiana (Regressão)", key="run_bayes_reg"):
            if 'bayesian_model' in st.session_state:
                del st.session_state['bayesian_model']

            with st.spinner("Construindo e Amostrando o Modelo Bayesiano..."):
                try:
                    model_coords = {
                        "coefficients_dim_0": numeric_features 
                    }
                    bayesian_model = pm.Model(coords=model_coords) 

                    with bayesian_model:
                        intercept = pm.Normal('intercept', mu=0, sigma=10)
                        coeffs = pm.Normal('coefficients', mu=0, sigma=10, dims="coefficients_dim_0") 
                        sigma = pm.HalfNormal('sigma', sigma=1)
                        linear_predictor = intercept + pm.math.dot(X.values, coeffs)
                        Y_obs = pm.Normal('Y_obs', mu=linear_predictor, sigma=sigma, observed=y.values)
                    
                    idata = pm.sample(
                        model=bayesian_model, 
                        draws=n_draws,
                        tune=n_tune,
                        chains=n_chains,
                        random_seed=random_seed,
                        idata_kwargs={"log_likelihood": True} 
                    )
                    
                    st.success("Amostragem MCMC concluída com sucesso!")
                    st.session_state['idata_bayesian'] = idata
                    st.session_state['bayesian_model'] = bayesian_model # Store model object if needed later
                    st.session_state['bayesian_features'] = numeric_features
                    st.session_state['bayesian_analysis_type_run'] = "Regressão Linear Bayesiana" # Store type of analysis run

                except Exception as e:
                    st.error(f"Erro ao executar a amostragem Bayesiana: {e}")
                    st.info("Verifique se as features selecionadas são apropriadas para o modelo ou se há problemas com os dados (ex: NaNs).")
                    st.session_state['idata_bayesian'] = None
                    st.session_state['bayesian_model'] = None
                    st.session_state['bayesian_features'] = None

        # --- Exibir Resultados da Regressão ---
        if st.session_state.get('idata_bayesian') is not None and st.session_state.get('bayesian_analysis_type_run') == "Regressão Linear Bayesiana":
            idata = st.session_state['idata_bayesian']
            bayesian_model = st.session_state['bayesian_model']
            bayesian_features = st.session_state['bayesian_features']

            st.markdown("### Resultados da Amostragem Bayesiana (Regressão)")

            st.markdown("#### Sumário Posterior")
            summary_df = az.summary(idata, var_names=["intercept", "coefficients", "sigma"]).round(2)
            st.dataframe(summary_df)
            st.write("`r_hat` deve ser próximo de 1.0 para indicar convergência.")

            st.markdown("#### Trace Plots")
            _ = az.plot_trace(idata, var_names=["intercept", "coefficients", "sigma"], compact=True, figsize=(10, 8))
            fig_trace = plt.gcf() 
            plt.tight_layout()
            st.pyplot(fig_trace)
            plt.close(fig_trace) 

            st.markdown("#### Densidades Posteriores das Variáveis")
            _ = az.plot_posterior(idata, var_names=["intercept", "coefficients", "sigma"], figsize=(10, 6))
            fig_posterior = plt.gcf()
            plt.tight_layout()
            st.pyplot(fig_posterior)
            plt.close(fig_posterior)

            st.markdown("#### Forest Plot (Coeficientes)")
            var_names_forest = ["intercept", "coefficients"] 
            coord_coeffs = {"coefficients_dim_0": bayesian_features} 
            
            _ = az.plot_forest(idata, var_names=var_names_forest, coords=coord_coeffs, figsize=(8, 6))
            fig_forest = plt.gcf()
            plt.tight_layout()
            st.pyplot(fig_forest)
            plt.close(fig_forest) 
            
            # --- Seção para WAIC e LOO (Comparação de Modelos) ---
            st.markdown("---")
            st.markdown("### 📊 Comparação de Modelos (WAIC / LOO)")
            st.info("O Bayes Factor completo é complexo de calcular e requer a evidência marginal. Alternativas mais comuns e computacionalmente acessíveis para comparação de modelos são o WAIC (Watanabe-Akaike Information Criterion) e LOO (Leave-One-Out cross-validation).")

            if st.checkbox("Calcular WAIC e LOO para o modelo atual", key="reg_calc_waic_loo"):
                with st.spinner("Calculando WAIC e LOO..."):
                    try:
                        waic_result = az.waic(idata, scale="log") 
                        st.write("#### WAIC (Watanabe-Akaike Information Criterion):")
                        st.write(waic_result) 
                        st.markdown(f"**WAIC (Log Scale - ELPD):** `{waic_result.elpd_waic:.2f}`")
                        st.markdown(f"**P_WAIC (Penalidade por complexidade):** `{waic_result.p_waic:.2f}`")
                        st.info("Valores de WAIC (ELPD) **mais baixos** indicam melhor ajuste ao modelo.")

                        st.markdown("---")

                        loo_result = az.loo(idata, scale="log") 
                        st.write("#### LOO (Leave-One-Out cross-validation):")
                        st.write(loo_result)
                        st.markdown(f"**LOO (Log Scale - ELPD):** `{loo_result.elpd_loo:.2f}`")
                        st.markdown(f"**P_LOO (Penalidade por complexidade):** `{loo_result.p_loo:.2f}`")
                        st.info("Valores de LOO (ELPD) **mais baixos (menos negativos)** indicam melhor ajuste ao modelo.")

                        st.success("Cálculo de WAIC e LOO concluído!")
                    except Exception as e:
                        st.error(f"Erro ao calcular WAIC/LOO: {e}")
                        st.info("Certifique-se de que o modelo foi amostrado com sucesso e que os dados são apropriados para WAIC/LOO. Verifique se o log_likelihood está sendo incluído no InferenceData.")

            st.markdown("##### Como Comparar Múltiplos Modelos:")
            st.info("Para comparar múltiplos modelos usando WAIC/LOO, você precisaria: "
                    "1. Ajustar cada modelo (ex: com diferentes conjuntos de features, distribuições de priors, etc.)."
                    "2. Armazenar seus respectivos objetos `InferenceData`."
                    "3. Então, usar a função `az.compare()` passando um dicionário de todos os `InferenceData` (ex: `az.compare({'modelo_A': idata_A, 'modelo_B': idata_B})`).")
            st.markdown("Para Bayes Factors diretos (e.g., usando Bridge Sampling), as implementações podem ser mais complexas e requerem ferramentas adicionais ou abordagens específicas para calcular a evidência marginal de cada modelo.")

            st.markdown("##### Sobre Bayes Factors (BFs) e Evidência Marginal:")
            st.info("Os **Bayes Factors** medem a evidência de um modelo sobre outro, baseando-se na **evidência marginal** (ou verossimilhança do modelo) de cada modelo. A evidência marginal é o denominador do teorema de Bayes e representa a probabilidade dos dados dado o modelo, integrando sobre todo o espaço de parâmetros do prior.")
            st.info("Calcular a evidência marginal é computacionalmente muito desafiador em modelos complexos. Técnicas como **Bridge Sampling** ou **Thermodynamic Integration** são usadas para aproximá-la, mas não são implementações de 'um clique' como WAIC/LOO.")
            st.info("Enquanto WAIC e LOO são **critérios de informação** para comparação de modelos (valores mais baixos são melhores), eles são aproximações da verossimilhança preditiva e são mais fáceis de obter do que os Bayes Factors diretos. Para a maioria dos casos de uso, eles fornecem uma forma robusta de comparar o desempenho de diferentes modelos.")
            st.info("Se a necessidade de Bayes Factors diretos for crítica, você precisará explorar implementações mais avançadas, como as disponíveis em pacotes como `PyMC-Bridge` (um projeto independente) ou `pymc.sampling.algorithms.importance` para casos específicos, que exigem uma configuração mais manual e cuidadosa do que podemos oferecer facilmente nesta interface simples do Streamlit.")

            # Opcional: Download de Resultados (InferenceData)
            st.markdown("---")
            st.markdown("### Exportar Resultados")
            @st.cache_data
            def convert_idata_to_netcdf(idata_obj): 
                import xarray as xr
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                    idata_obj.to_netcdf(tmp_file.name)
                with open(tmp_file.name, "rb") as f:
                    nc_bytes = f.read()
                os.remove(tmp_file.name) 
                return nc_bytes

            if st.button("Baixar InferenceData (NetCDF)", key="reg_download_idata_nc"):
                if idata:
                    nc_data = convert_idata_to_netcdf(idata) 
                    st.download_button(
                        label="Baixar idata.nc",
                        data=nc_data,
                        file_name="bayesian_idata_reg.nc",
                        mime="application/x-netcdf"
                    )
                else:
                    st.warning("Nenhum resultado de inferência para baixar.")

    # --- TESTE T BAYESIANO ---
    elif bayesian_analysis_type == "Teste t Bayesiano":
        st.subheader("Teste t Bayesiano")
        st.info("Compara as médias de dois grupos em uma variável numérica, fornecendo a distribuição posterior da diferença das médias.")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no seu dataset para o Teste t Bayesiano.")
            return
        if not categorical_cols:
            st.warning("Não há colunas categóricas para definir os grupos para o Teste t Bayesiano.")
            return

        col_numeric = st.selectbox("Selecione a Variável Numérica:", numeric_cols, key="bayes_ttest_num")
        col_group = st.selectbox("Selecione a Variável Categórica (para agrupar 2 categorias):", categorical_cols, key="bayes_ttest_cat")

        if col_numeric and col_group:
            temp_df = df[[col_numeric, col_group]].dropna()
            temp_df[col_group] = temp_df[col_group].astype('category')
            
            groups = temp_df[col_group].unique()
            if len(groups) != 2:
                st.warning(f"A variável de agrupamento '{col_group}' deve ter exatamente dois valores únicos para o Teste t Bayesiano. Encontrados: {list(groups)}")
                return
            
            # Mapear grupos para índices 0 e 1
            group_map = {g: i for i, g in enumerate(groups)}
            temp_df['group_idx'] = temp_df[col_group].map(group_map)
            
            y_data = temp_df[col_numeric].values
            # CORREÇÃO AQUI: Garante que group_idx_data seja um array de inteiros puros
            group_idx_data = temp_df['group_idx'].values.astype(int) 

            if y_data.size == 0:
                st.error("Não há dados para realizar o Teste t Bayesiano após a limpeza. Verifique suas seleções.")
                return

            # Configurações de Amostragem (MCMC) - Continuam na Sidebar
            n_chains = st.sidebar.slider("Número de Cadeias (Chains):", 1, 4, 2, key="ttest_chains")
            n_draws = st.sidebar.slider("Número de Amostras por Cadeia (Draws):", 500, 5000, 2000, key="ttest_draws")
            n_tune = st.sidebar.slider("Número de Amostras para Afinação (Tune):", 100, 1000, 500, key="ttest_tune")
            random_seed = st.sidebar.slider("Seed Aleatória (Random Seed):", 0, 100, 42, key="ttest_seed")

            if st.button("Executar Teste t Bayesiano", key="run_bayes_ttest"):
                with st.spinner("Construindo e Amostrando o Modelo Bayesiano..."):
                    try:
                        # Priors informativos baseados nos dados
                        y_mean = y_data.mean()
                        y_std = y_data.std()

                        with pm.Model() as bayesian_ttest_model:
                            # Priors para as médias de cada grupo
                            mu_group1 = pm.Normal('mu_group1', mu=y_mean, sigma=y_std*2)
                            mu_group2 = pm.Normal('mu_group2', mu=y_mean, sigma=y_std*2)
                            
                            # Prior para o desvio padrão (assumindo igual para ambos os grupos)
                            sigma = pm.HalfNormal('sigma', sigma=y_std)
                            
                            # Likelihood: usa pm.math.switch para atribuir a média correta a cada ponto de dado
                            likelihood = pm.Normal(
                                'likelihood', 
                                mu=pm.math.switch(group_idx_data == 0, mu_group1, mu_group2), 
                                sigma=sigma, 
                                observed=y_data
                            )
                            
                            # Variável Determinística: Diferença entre as médias
                            difference = pm.Deterministic('difference', mu_group1 - mu_group2)
                            
                        idata = pm.sample(
                            model=bayesian_ttest_model,
                            draws=n_draws,
                            tune=n_tune,
                            chains=n_chains,
                            random_seed=random_seed,
                            idata_kwargs={"log_likelihood": True}
                        )

                        st.success("Amostragem MCMC concluída com sucesso!")
                        st.session_state['idata_bayesian_ttest'] = idata
                        st.session_state['bayesian_analysis_type_run'] = "Teste t Bayesiano" # Store type of analysis run

                    except Exception as e:
                        st.error(f"Erro ao executar o Teste t Bayesiano: {e}")
                        st.info("Verifique os dados e as seleções. Pode haver problemas de dados insuficientes ou colunas inadequadas.")
                        st.session_state['idata_bayesian_ttest'] = None

            # --- Exibir Resultados do Teste t Bayesiano ---
            if st.session_state.get('idata_bayesian_ttest') is not None and st.session_state.get('bayesian_analysis_type_run') == "Teste t Bayesiano":
                idata_ttest = st.session_state['idata_bayesian_ttest']

                st.markdown("### Resultados do Teste t Bayesiano")

                st.markdown("#### Sumário Posterior")
                summary_df_ttest = az.summary(idata_ttest, var_names=["mu_group1", "mu_group2", "sigma", "difference"]).round(2)
                st.dataframe(summary_df_ttest)
                st.write("`r_hat` deve ser próximo de 1.0 para indicar convergência.")

                st.markdown("#### Densidades Posteriores e HDIs")
                _ = az.plot_posterior(idata_ttest, var_names=["mu_group1", "mu_group2", "difference"], figsize=(10, 8), hdi_prob=0.95)
                fig_ttest_posterior = plt.gcf()
                plt.tight_layout()
                st.pyplot(fig_ttest_posterior)
                plt.close(fig_ttest_posterior)

                # Probabilidade da diferença ser positiva/negativa
                diff_samples = idata_ttest.posterior["difference"].values.flatten()
                prob_diff_gt_0 = (diff_samples > 0).mean()
                prob_diff_lt_0 = (diff_samples < 0).mean()

                st.markdown("#### Interpretação da Diferença entre as Médias:")
                st.write(f"**Probabilidade (Média {groups[0]} > Média {groups[1]}):** `{prob_diff_gt_0:.3f}`")
                st.write(f"**Probabilidade (Média {groups[0]} < Média {groups[1]}):** `{prob_diff_lt_0:.3f}`")
                st.write(f"**95% HDI para a Diferença:** `[{summary_df_ttest.loc['difference', 'hdi_3%']:.2f}, {summary_df_ttest.loc['difference', 'hdi_97%']:.2f}]`")
                
                if prob_diff_gt_0 > 0.95: # Ou 0.975 dependendo da convenção
                    st.success(f"Há forte evidência Bayesiana de que a média de `{groups[0]}` é maior que a de `{groups[1]}`.")
                elif prob_diff_lt_0 > 0.95:
                    st.success(f"Há forte evidência Bayesiana de que a média de `{groups[1]}` é maior que a de `{groups[0]}`.")
                else:
                    st.info("Não há evidência Bayesiana forte o suficiente para afirmar que uma média é consistentemente maior que a outra. A diferença pode estar próxima de zero.")
                
                # Plot de caixa para visualização dos grupos
                st.markdown("#### Visualização dos Grupos")
                fig_boxplot, ax_boxplot = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=col_group, y=col_numeric, data=temp_df, ax=ax_boxplot)
                sns.stripplot(x=col_group, y=col_numeric, data=temp_df, color='black', size=5, jitter=0.2, ax=ax_boxplot)
                ax_boxplot.set_title(f'Distribuição de {col_numeric} por {col_group}')
                ax_boxplot.set_xlabel(col_group)
                ax_boxplot.set_ylabel(col_numeric)
                st.pyplot(fig_boxplot)
                plt.close(fig_boxplot)


    # --- ANOVA BAYESIANA SIMPLES ---
    elif bayesian_analysis_type == "ANOVA Bayesiana Simples":
        st.subheader("ANOVA Bayesiana Simples (One-Way)")
        st.info("Compara as médias de três ou mais grupos em uma variável numérica, fornecendo as distribuições posteriores das médias de cada grupo.")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no seu dataset para a ANOVA Bayesiana.")
            return
        if not categorical_cols:
            st.warning("Não há colunas categóricas para definir os grupos para a ANOVA Bayesiana.")
            return

        col_numeric = st.selectbox("Selecione a Variável Numérica (Dependente):", numeric_cols, key="bayes_anova_num")
        col_group = st.selectbox("Selecione a Variável Categórica (Fator com 2+ categorias):", categorical_cols, key="bayes_anova_cat")

        if col_numeric and col_group:
            temp_df = df[[col_numeric, col_group]].dropna()
            temp_df[col_group] = temp_df[col_group].astype('category')
            
            groups_unique = temp_df[col_group].unique()
            if len(groups_unique) < 2:
                st.warning(f"A variável fator '{col_group}' deve ter pelo menos dois valores únicos para a ANOVA Bayesiana. Encontrados: {list(groups_unique)}")
                return
            
            # Mapear grupos para índices numéricos para PyMC
            group_map = {g: i for i, g in enumerate(groups_unique)}
            temp_df['group_idx'] = temp_df[col_group].map(group_map)
            
            y_data = temp_df[col_numeric].values
            # CORREÇÃO AQUI: Garante que group_idx_data seja um array de inteiros puros
            group_idx_data = temp_df['group_idx'].values.astype(int) 

            if y_data.size == 0:
                st.error("Não há dados para realizar a ANOVA Bayesiana após a limpeza. Verifique suas seleções.")
                return

            # Configurações de Amostragem (MCMC) - Continuam na Sidebar
            n_chains = st.sidebar.slider("Número de Cadeias (Chains):", 1, 4, 2, key="anova_chains")
            n_draws = st.sidebar.slider("Número de Amostras por Cadeia (Draws):", 500, 5000, 2000, key="anova_draws")
            n_tune = st.sidebar.slider("Número de Amostras para Afinação (Tune):", 100, 1000, 500, key="anova_tune")
            random_seed = st.sidebar.slider("Seed Aleatória (Random Seed):", 0, 100, 42, key="anova_seed")

            if st.button("Executar ANOVA Bayesiana Simples", key="run_bayes_anova"):
                with st.spinner("Construindo e Amostrando o Modelo Bayesiano..."):
                    try:
                        y_mean = y_data.mean()
                        y_std = y_data.std()

                        with pm.Model(coords={"group_unique": groups_unique}) as bayesian_anova_model:
                            # Priors para as médias de cada grupo
                            # Usamos `dims` para criar um array de médias, uma para cada grupo
                            group_means = pm.Normal(
                                'group_means', 
                                mu=y_mean, 
                                sigma=y_std*2, 
                                dims="group_unique"
                            )
                            
                            # Prior para o desvio padrão (assumindo igual para todos os grupos)
                            sigma = pm.HalfNormal('sigma', sigma=y_std)
                            
                            # Likelihood: Cada ponto de dado é associado à média do seu grupo
                            likelihood = pm.Normal(
                                'likelihood', 
                                mu=group_means[group_idx_data], # Usamos o índice do grupo para selecionar a média correta
                                sigma=sigma, 
                                observed=y_data
                            )
                            
                        idata = pm.sample(
                            model=bayesian_anova_model,
                            draws=n_draws,
                            tune=n_tune,
                            chains=n_chains,
                            random_seed=random_seed,
                            idata_kwargs={"log_likelihood": True}
                        )

                        st.success("Amostragem MCMC concluída com sucesso!")
                        st.session_state['idata_bayesian_anova'] = idata
                        st.session_state['bayesian_anova_groups'] = groups_unique
                        st.session_state['bayesian_analysis_type_run'] = "ANOVA Bayesiana Simples" # Store type of analysis run

                    except Exception as e:
                        st.error(f"Erro ao executar a ANOVA Bayesiana Simples: {e}")
                        st.info("Verifique os dados e as seleções. Pode haver problemas de dados insuficientes ou colunas inadequadas.")
                        st.session_state['idata_bayesian_anova'] = None

            # --- Exibir Resultados da ANOVA Bayesiana Simples ---
            if st.session_state.get('idata_bayesian_anova') is not None and st.session_state.get('bayesian_analysis_type_run') == "ANOVA Bayesiana Simples":
                idata_anova = st.session_state['idata_bayesian_anova']
                anova_groups = st.session_state['bayesian_anova_groups']

                st.markdown("### Resultados da ANOVA Bayesiana Simples")

                st.markdown("#### Sumário Posterior das Médias dos Grupos e Desvio Padrão")
                summary_df_anova = az.summary(idata_anova, var_names=["group_means", "sigma"]).round(2)
                
                # Renomear os índices para mostrar os nomes dos grupos
                if "group_means_dim_0" in summary_df_anova.index.names:
                    summary_df_anova = summary_df_anova.reset_index()
                    # Melhor forma de renomear para grupos específicos
                    rename_map = {f'group_means[{i}]': f'Média do Grupo: {anova_groups[i]}' for i in range(len(anova_groups))}
                    summary_df_anova['index'] = summary_df_anova['index'].replace(rename_map)
                    summary_df_anova = summary_df_anova.set_index('index')

                st.dataframe(summary_df_anova)
                st.write("`r_hat` deve ser próximo de 1.0 para indicar convergência.")

                st.markdown("#### Densidades Posteriores e HDIs das Médias dos Grupos")
                # Plotar as densidades das médias dos grupos
                _ = az.plot_posterior(idata_anova, var_names=["group_means"], figsize=(10, 6), hdi_prob=0.95, 
                                     coords={"group_unique": anova_groups}) # Usar coords para nomes no plot
                fig_anova_posterior = plt.gcf()
                plt.tight_layout()
                st.pyplot(fig_anova_posterior)
                plt.close(fig_anova_posterior)
                
                # Opcional: Comparação Post-Hoc Bayesiana de Pares de Grupos
                st.markdown("#### Comparação Bayesiana de Pares de Grupos (Post-Hoc)")
                st.info("Se houver interesse em diferenças específicas entre pares de grupos, você pode examinar as distribuições posteriores das diferenças.")
                
                if len(anova_groups) > 1:
                    selected_group_pair = st.multiselect(
                        "Selecione dois grupos para comparar as médias:",
                        options=anova_groups,
                        max_selections=2,
                        key="anova_group_pair_select"
                    )

                    if len(selected_group_pair) == 2:
                        g1_name, g2_name = selected_group_pair
                        g1_idx = np.where(anova_groups == g1_name)[0][0]
                        g2_idx = np.where(anova_groups == g2_name)[0][0]

                        # Obter amostras das médias dos grupos
                        g1_samples = idata_anova.posterior["group_means"].sel(group_unique=g1_name).values.flatten()
                        g2_samples = idata_anova.posterior["group_means"].sel(group_unique=g2_name).values.flatten()

                        # Calcular a diferença entre as amostras
                        diff_samples = g1_samples - g2_samples

                        st.markdown(f"##### Diferença Posterior: {g1_name} - {g2_name}")
                        st.write(f"**Média da Diferença:** `{diff_samples.mean():.3f}`")
                        
                        # HDI da diferença
                        hdi_diff = az.hdi(diff_samples, hdi_prob=0.95)
                        st.write(f"**95% HDI da Diferença:** `[{hdi_diff[0]:.2f}, {hdi_diff[1]:.2f}]`")

                        # Probabilidade da diferença ser > 0 ou < 0
                        prob_g1_gt_g2 = (diff_samples > 0).mean()
                        st.write(f"**Probabilidade ({g1_name} > {g2_name}):** `{prob_g1_gt_g2:.3f}`")
                        st.write(f"**Probabilidade ({g1_name} < {g2_name}):** `{(diff_samples < 0).mean():.3f}`")

                        if prob_g1_gt_g2 > 0.95:
                            st.success(f"Há forte evidência Bayesiana de que a média de `{g1_name}` é maior que a de `{g2_name}`.")
                        elif (diff_samples < 0).mean() > 0.95:
                            st.success(f"Há forte evidência Bayesiana de que a média de `{g2_name}` é maior que a de `{g1_name}`.")
                        else:
                            st.info(f"Não há evidência Bayesiana forte o suficiente para afirmar uma diferença clara entre `{g1_name}` e `{g2_name}`.")

                        # Plot da distribuição da diferença
                        fig_diff, ax_diff = plt.subplots(figsize=(8, 5))
                        sns.histplot(diff_samples, kde=True, ax=ax_diff)
                        ax_diff.axvline(0, color='red', linestyle='--', label='Diferença = 0')
                        ax_diff.axvspan(hdi_diff[0], hdi_diff[1], color='gray', alpha=0.3, label='95% HDI')
                        ax_diff.set_title(f'Distribuição Posterior da Diferença: {g1_name} - {g2_name}')
                        ax_diff.set_xlabel('Diferença nas Médias')
                        ax_diff.legend()
                        st.pyplot(fig_diff)
                        plt.close(fig_diff)


                # Plot de caixa para visualização dos grupos
                st.markdown("#### Visualização dos Grupos")
                fig_boxplot_anova, ax_boxplot_anova = plt.subplots(figsize=(10, 7))
                sns.boxplot(x=col_group, y=col_numeric, data=temp_df, ax=ax_boxplot_anova)
                sns.stripplot(x=col_group, y=col_numeric, data=temp_df, color='black', size=5, jitter=0.2, ax=ax_boxplot_anova)
                ax_boxplot_anova.set_title(f'Distribuição de {col_numeric} por {col_group}')
                ax_boxplot_anova.set_xlabel(col_group)
                ax_boxplot_anova.set_ylabel(col_numeric)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_boxplot_anova)
                plt.close(fig_boxplot_anova)