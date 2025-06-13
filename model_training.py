#correto
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


# Apenas para fins de demonstração, crie um DataFrame dummy se não existir
if 'df_processed' not in st.session_state:
    np.random.seed(42) 
    data = {
        'Target_Linear': np.random.normal(50, 15, 200),
        'Feature_A': np.random.normal(10, 2, 200),
        'Feature_B': np.random.rand(200) * 5,
        'Category_C': np.random.choice(['X', 'Y', 'Z'], 200),
        'Target_Logistic': np.random.randint(0, 2, 200),
        'Feature_D': np.random.normal(25, 5, 200),
        'Feature_E': np.random.rand(200) * 10,
        'Group_ID': np.random.choice([f'G{i}' for i in range(1, 11)], 200),
        'Random_Slope_Var': np.random.rand(200) * 3,
        'Target_Multilevel': np.random.normal(100, 20, 200),
        'X_Predictor': np.random.normal(10, 2, 200), # Para Path Analysis
        'M_Mediator': np.random.normal(5, 1, 200),   # Para Path Analysis
        'Y_Outcome': np.random.normal(20, 4, 200),    # Para Path Analysis
        'Z_Covariate': np.random.normal(15, 3, 200) # Para Path Analysis
    }
    data['Target_Linear'] += data['Feature_A'] * 2 + data['Feature_B'] * 1.5 + np.random.normal(0, 5, 200)
    data['Target_Logistic'] = (data['Feature_D'] * 0.2 + data['Feature_E'] * 0.1 + np.random.normal(0, 0.5, 200) > 6).astype(int)
    
    group_effects = {f'G{i}': np.random.normal(0, 5) for i in range(1, 11)}
    data['Target_Multilevel'] += [group_effects[g] for g in data['Group_ID']]
    data['Target_Multilevel'] += data['Feature_A'] * 3 + data['Random_Slope_Var'] * 2 + np.random.normal(0, 10, 200)

    # Simular relações para Path Analysis: X -> M -> Y e X -> Y
    data['M_Mediator'] += data['X_Predictor'] * 0.6 + np.random.normal(0, 1, 200)
    data['Y_Outcome'] += data['X_Predictor'] * 0.3 + data['M_Mediator'] * 0.5 + np.random.normal(0, 2, 200)
    data['M_Mediator'] += data['Z_Covariate'] * 0.2 # Z covaria com M
    
    st.session_state['df_processed'] = pd.DataFrame(data)
    st.session_state['df_processed']['Group_ID'] = st.session_state['df_processed']['Group_ID'].astype('category')


# --- Função para Regressão Linear ---
def show_linear_regression_model():
    st.subheader("📈 Regressão Linear")
    st.info("Utilize a Regressão Linear para prever uma variável dependente contínua com base em uma ou mais variáveis independentes numéricas.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    # Adicionando o expander para toda a seção de configuração e resultados da regressão linear
    with st.expander("Configurar e Executar Regressão Linear", expanded=False): # 'expanded=True' para começar aberto
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            st.warning("Não há colunas numéricas no DataFrame para realizar a Regressão Linear.")
            return

        st.markdown("---")
        st.markdown("#### Configuração do Modelo de Regressão Linear")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 1. Selecione a Variável Dependente (Y)")
        dependent_var = st.selectbox(
            "Escolha a variável contínua a ser prevista:", 
            numeric_cols, 
            key="lr_dependent"
        )
        if dependent_var:
            st.success(f"Variável Dependente selecionada: **{dependent_var}**")
        else:
            st.warning("Por favor, selecione uma variável dependente.")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 2. Selecione as Variáveis Independentes (X)")
        options_independent_vars = [col for col in numeric_cols if col != dependent_var]
        if not options_independent_vars:
            st.warning("Não há variáveis numéricas disponíveis para usar como preditoras, exceto a variável dependente selecionada.")
        independent_vars = st.multiselect(
            "Escolha uma ou mais variáveis numéricas para prever a variável dependente:", 
            options_independent_vars, 
            key="lr_independent"
        )
        if independent_vars:
            st.success(f"Variáveis Independentes selecionadas: **{', '.join(independent_vars)}**")
        else:
            st.warning("Por favor, selecione pelo menos uma variável independente.")

        if dependent_var is None or not independent_vars:
            st.info("Por favor, complete as seleções obrigatórias para a Variável Dependente e Variáveis Independentes.")
            return

        if st.button("Executar Regressão Linear", key="run_lr_model"):
            with st.spinner("Treinando Modelo de Regressão Linear..."):
                try:
                    model_vars = [dependent_var] + independent_vars
                    df_model = df[model_vars].dropna()
                    
                    initial_rows = df.shape[0]
                    if initial_rows > df_model.shape[0]:
                        st.warning(f"Foram removidas {initial_rows - df_model.shape[0]} linhas devido a valores NaN nas variáveis selecionadas para o modelo.")

                    if df_model.empty:
                        st.error("Não há dados suficientes após o tratamento de NaN para construir o modelo.")
                        return

                    Y = df_model[dependent_var]
                    X = df_model[independent_vars]
                    X = sm.add_constant(X)

                    model = sm.OLS(Y, X)
                    results = model.fit()

                    st.markdown("---")
                    st.subheader("Resultados da Regressão Linear")

                    st.write("#### Sumário do Modelo (Statsmodels OLS)")
                    st.code(results.summary().as_text())

                    st.info("Os coeficientes ('coef') indicam a mudança média na variável dependente para cada aumento de uma unidade na variável independente correspondente, mantendo as outras constantes.")
                    st.info("O 'P>|t|' (p-valor) indica a significância estatística do coeficiente. Valores abaixo de 0.05 (geralmente) sugerem um efeito significativo.")
                    st.info("O R-quadrado ajustado ('Adj. R-squared') representa a proporção da variância na variável dependente que é explicada pelas variáveis independentes.")

                    st.write("#### Métricas de Avaliação")
                    predictions = results.predict(X)
                    r2 = r2_score(Y, predictions)
                    mse = mean_squared_error(Y, predictions)
                    rmse = np.sqrt(mse)

                    st.write(f"**R-quadrado (R²):** `{r2:.4f}`")
                    st.write(f"**Erro Quadrático Médio (MSE):** `{mse:.4f}`")
                    st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** `{rmse:.4f}`")

                    st.write("#### Análise de Resíduos")
                    fig_res, ax_res = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=predictions, y=results.resid, ax=ax_res)
                    ax_res.axhline(0, color='red', linestyle='--')
                    ax_res.set_xlabel("Valores Preditos")
                    ax_res.set_ylabel("Resíduos")
                    ax_res.set_title("Resíduos vs. Valores Preditos")
                    st.pyplot(fig_res)
                    plt.close(fig_res)
                    st.info("Este gráfico ajuda a verificar a suposição de homocedasticidade (variância constante dos resíduos). Os resíduos devem estar espalhados aleatoriamente em torno de zero.")
                    
                    fig_hist_res, ax_hist_res = plt.subplots(figsize=(10, 6))
                    sns.histplot(results.resid, kde=True, ax=ax_hist_res)
                    ax_hist_res.set_xlabel("Resíduos")
                    ax_hist_res.set_ylabel("Frequência")
                    ax_hist_res.set_title("Distribuição dos Resíduos")
                    st.pyplot(fig_hist_res)
                    plt.close(fig_hist_res)
                    st.info("A distribuição dos resíduos deve ser aproximadamente normal. Uma distribuição em forma de sino sugere que a suposição de normalidade foi atendida.")

                except Exception as e:
                    st.error(f"Erro ao executar a Regressão Linear: {e}")
                    st.info("Verifique se as variáveis selecionadas são numéricas e se há dados suficientes. Erros comuns incluem multicolinearidade ou problemas de dados.")


# --- Função para Regressão Logística ---
def show_logistic_regression_model():
    st.subheader("📊 Regressão Logística")
    st.info("Utilize a Regressão Logística para prever uma variável dependente categórica binária (0 ou 1, sim/não) com base em uma ou mais variáveis independentes.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    # Adicionando o expander para toda a seção de configuração e resultados da regressão logística
    with st.expander("Configurar e Executar Regressão Logística", expanded=False): # 'expanded=True' para começar aberto
        df = st.session_state.df_processed.copy()
        all_cols = df.columns.tolist()
        
        binary_cols = [col for col in all_cols if df[col].nunique() == 2 and df[col].isin([0, 1]).all()]
        
        if not binary_cols:
            st.warning("Não há colunas binárias (0 ou 1) no DataFrame para a variável dependente.")
            return

        st.markdown("---")
        st.markdown("#### Configuração do Modelo de Regressão Logística")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 1. Selecione a Variável Dependente (Y - Binária)")
        dependent_var = st.selectbox(
            "Escolha a variável binária (0 ou 1) a ser prevista:", 
            binary_cols, 
            key="logr_dependent"
        )
        if dependent_var:
            st.success(f"Variável Dependente selecionada: **{dependent_var}**")
            unique_vals = df[dependent_var].unique()
            st.info(f"Valores únicos da variável dependente: {unique_vals[0]} e {unique_vals[1]}. O modelo prediz a probabilidade do valor `1`.")
        else:
            st.warning("Por favor, selecione uma variável dependente binária.")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 2. Selecione as Variáveis Independentes (X)")
        options_independent_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect(
            "Escolha uma ou mais variáveis para prever a variável dependente:", 
            options_independent_vars, 
            key="logr_independent"
        )
        if independent_vars:
            st.success(f"Variáveis Independentes selecionadas: **{', '.join(independent_vars)}**")
        else:
            st.warning("Por favor, selecione pelo menos uma variável independente.")

        if dependent_var is None or not independent_vars:
            st.info("Por favor, complete as seleções obrigatórias para a Variável Dependente e Variáveis Independentes.")
            return

        if st.button("Executar Regressão Logística", key="run_logr_model"):
            with st.spinner("Treinando Modelo de Regressão Logística..."):
                try:
                    model_vars = [dependent_var] + independent_vars
                    df_model = df[model_vars].dropna()
                    
                    initial_rows = df.shape[0]
                    if initial_rows > df_model.shape[0]:
                        st.warning(f"Foram removidas {initial_rows - df_model.shape[0]} linhas devido a valores NaN nas variáveis selecionadas para o modelo.")

                    if df_model.empty:
                        st.error("Não há dados suficientes após o tratamento de NaN para construir o modelo.")
                        return

                    X = df_model[independent_vars]
                    for col in X.columns:
                        if pd.api.types.is_categorical_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                            X = pd.get_dummies(X, columns=[col], drop_first=True, dtype=int)

                    Y = df_model[dependent_var]
                    X = sm.add_constant(X, prepend=False)

                    model = sm.GLM(Y, X, family=sm.families.Binomial())
                    results = model.fit()

                    st.markdown("---")
                    st.subheader("Resultados da Regressão Logística")

                    st.write("#### Sumário do Modelo (Statsmodels GLM - Logística)")
                    st.code(results.summary().as_text())

                    st.info("Os coeficientes ('coef') indicam a mudança no log-odds da variável dependente para cada aumento de uma unidade na variável independente. Valores positivos aumentam a probabilidade de ocorrência do evento (valor 1).")
                    st.info("O 'P>|z|' (p-valor) indica a significância estatística do coeficiente. Valores abaixo de 0.05 (geralmente) sugerem um efeito significativo.")
                    
                    st.write("#### Métricas de Avaliação")
                    predictions_proba = results.predict(X)
                    
                    threshold = st.slider("Selecione o Ponto de Corte (Threshold) para Classificação:", 0.0, 1.0, 0.5, 0.01, key="logr_threshold")
                    predictions_class = (predictions_proba >= threshold).astype(int)

                    accuracy = accuracy_score(Y, predictions_class)
                    precision = precision_score(Y, predictions_class)
                    recall = recall_score(Y, predictions_class)
                    f1 = f1_score(Y, predictions_class)
                    auc_score = roc_auc_score(Y, predictions_proba)
                    cm = confusion_matrix(Y, predictions_class)

                    st.write(f"**Acurácia:** `{accuracy:.4f}`")
                    st.write(f"**Precisão:** `{precision:.4f}`")
                    st.write(f"**Recall:** `{recall:.4f}`")
                    st.write(f"**F1-Score:** `{f1:.4f}`")
                    st.write(f"**AUC (Area Under ROC Curve):** `{auc_score:.4f}`")

                    st.write("##### Matriz de Confusão:")
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                                xticklabels=['Predito 0', 'Predito 1'], yticklabels=['Real 0', 'Real 1'])
                    ax_cm.set_xlabel("Predito")
                    ax_cm.set_ylabel("Real")
                    ax_cm.set_title("Matriz de Confusão")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                    st.info("A matriz de confusão mostra o número de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.")

                    st.write("##### Curva ROC")
                    fpr, tpr, _ = roc_curve(Y, predictions_proba)
                    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                    ax_roc.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
                    ax_roc.plot([0, 1], [0, 1], 'r--', label='Linha de Referência')
                    ax_roc.set_xlabel("Taxa de Falso Positivo (FPR)")
                    ax_roc.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
                    ax_roc.set_title("Curva Característica de Operação do Receptor (ROC)")
                    ax_roc.legend()
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)
                    st.info("A Curva ROC avalia o desempenho do classificador em vários pontos de corte. Um AUC próximo de 1.0 indica um excelente poder de discriminação.")

                except Exception as e:
                    st.error(f"Erro ao executar a Regressão Logística: {e}")
                    st.info("Verifique se a variável dependente é binária (0 ou 1) e se há dados suficientes. Erros comuns incluem classes desbalanceadas ou problemas de convergência.")


# --- Função para Análise Multinível (Modelos Lineares Mistos) ---
def show_multilevel_model():
    st.subheader("🌳 Análise Multinível (Modelos Lineares Mistos)")
    st.info("Utilize a Análise Multinível para modelar dados com estrutura hierárquica ou aninhada (ex: estudantes em escolas, pacientes em hospitais).")
    st.info("Esta análise permite que os efeitos das variáveis variem entre os diferentes grupos.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    # Adicionando o expander para toda a seção de configuração e resultados da análise multinível
    with st.expander("Configurar e Executar Análise Multinível", expanded=False): # 'expanded=True' para começar aberto
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        all_cols = df.columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no DataFrame para a variável dependente.")
            return
        if not categorical_cols:
            st.warning("Não há colunas categóricas no DataFrame para a variável de agrupamento (nível superior).")
            return

        st.markdown("---")
        st.markdown("#### Configuração do Modelo Multinível")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 1. Selecione a Variável Dependente (Y)")
        dependent_var = st.selectbox(
            "Escolha a variável contínua a ser prevista:", 
            numeric_cols, 
            key="mlm_dependent"
        )
        if dependent_var:
            st.success(f"Variável Dependente selecionada: **{dependent_var}**")
        else:
            st.warning("Por favor, selecione uma variável dependente.")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 2. Selecione a Variável de Agrupamento")
        group_var = st.selectbox(
            "Escolha a variável categórica que define os grupos (nível superior):", 
            categorical_cols, 
            key="mlm_group"
        )
        if group_var:
            st.success(f"Variável de Agrupamento selecionada: **{group_var}**")
        else:
            st.warning("Por favor, selecione uma variável de agrupamento.")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 3. Selecione as Variáveis de Efeitos Fixos")
        options_fixed_effects = [col for col in all_cols if col != dependent_var and col != group_var]
        fixed_effects_vars = st.multiselect(
            "Escolha as variáveis preditoras que afetam todos os grupos da mesma forma:", 
            options_fixed_effects, 
            key="mlm_fixed_effects"
        )
        if fixed_effects_vars:
            st.success(f"Variáveis de Efeitos Fixos selecionadas: **{', '.join(fixed_effects_vars)}**")
        else:
            st.info("Nenhuma variável de efeito fixo selecionada. O modelo pode incluir apenas um intercepto fixo e efeitos aleatórios.")

        # NÃO MAIS EXPANDER ANINHADO AQUI
        st.markdown("##### 4. Selecione os Efeitos Aleatórios")
        selected_random_effects = []
        random_effects_options = ["Intercepto Aleatório"] + [
            col for col in fixed_effects_vars if col in numeric_cols 
        ]
        
        default_random_effects = []
        if "Intercepto Aleatório" in random_effects_options:
            default_random_effects.append("Intercepto Aleatório")

        selected_random_effects = st.multiselect(
            "Escolha quais componentes devem variar aleatoriamente entre os grupos (o intercepto é comum):",
            random_effects_options,
            default=default_random_effects,
            key="mlm_random_effects"
        )
        if selected_random_effects:
            st.success(f"Efeitos Aleatórios selecionados: **{', '.join(selected_random_effects)}**")
        else:
            st.warning("Por favor, selecione pelo menos um efeito aleatório (geralmente o Intercepto Aleatório).")

        if dependent_var is None or group_var is None or not selected_random_effects:
            st.info("Por favor, complete as seleções obrigatórias para a Variável Dependente, Variável de Agrupamento e Efeitos Aleatórios.")
            return

        if st.button("Executar Análise Multinível", key="run_mlm_model"):
            with st.spinner("Treinando Modelo Multinível..."):
                try:
                    model_vars = [dependent_var, group_var] + fixed_effects_vars 
                    
                    for re_var in selected_random_effects:
                        if re_var != "Intercepto Aleatório" and re_var not in model_vars:
                            model_vars.append(re_var)

                    df_model = df[model_vars].dropna()
                    
                    initial_rows = df.shape[0]
                    if initial_rows > df_model.shape[0]:
                        st.warning(f"Foram removidas {initial_rows - df_model.shape[0]} linhas devido a valores NaN nas variáveis selecionadas para o modelo multinível.")

                    if df_model.empty:
                        st.error("Não há dados suficientes após o tratamento de NaN para construir o modelo multinível.")
                        return

                    if not pd.api.types.is_categorical_dtype(df_model[group_var]):
                        df_model[group_var] = df_model[group_var].astype('category')
                        st.info(f"A variável de agrupamento '{group_var}' foi convertida para tipo categórica.")

                    fixed_part = " + ".join(fixed_effects_vars) if fixed_effects_vars else "1" 
                    
                    random_part_for_re_formula = []
                    if "Intercepto Aleatório" in selected_random_effects:
                        random_part_for_re_formula.append("1") 
                    
                    for re_var in selected_random_effects:
                        if re_var != "Intercepto Aleatório":
                            if re_var in numeric_cols:
                                random_part_for_re_formula.append(re_var)
                            else:
                                st.warning(f"Variável '{re_var}' selecionada para efeito aleatório não é numérica e será ignorada para slopes aleatórios.")
                    
                    re_formula_str = "~ " + " + ".join(random_part_for_re_formula)
                    
                    formula = f"{dependent_var} ~ {fixed_part}"

                    model = mixedlm(
                        formula=formula, 
                        data=df_model, 
                        re_formula=re_formula_str, 
                        groups=df_model[group_var]
                    )
                    
                    results = model.fit()

                    st.markdown("---")
                    st.subheader("Resultados do Modelo Multinível")

                    st.write("#### Sumário Completo do Modelo (Statsmodels MixedLM)")
                    st.code(results.summary().as_text())

                    st.info("Os coeficientes na seção 'Coefs' representam os efeitos fixos. Eles são interpretados como em uma regressão linear comum.")
                    st.info("A seção **'Random Effects'** no sumário completo do modelo (exibido acima) contém a variância e o desvio padrão dos efeitos aleatórios (intercepto e/ou slopes), indicando o quanto esses efeitos variam entre os grupos. **Por favor, consulte essa seção para os valores numéricos detalhados.**")
                    st.info("Um 'P>|z|' baixo (geralmente < 0.05) para um efeito fixo indica significância estatística.")
                    
                    st.subheader("Variância e Desvio Padrão dos Componentes do Modelo")
                    
                    if hasattr(results, 'vc_params') and results.vc_params is not None and not results.vc_params.empty:
                        st.write("##### Variância e Desvio Padrão dos Efeitos Aleatórios por Componente")
                        
                        vc_df = results.vc_params.to_frame(name='Variância Estimada')
                        vc_df['Desvio Padrão Estimado'] = np.sqrt(vc_df['Variância Estimada'])
                        st.dataframe(vc_df)
                        st.markdown(f"**Interpretação:** Estas são as variâncias e desvios padrão dos componentes de efeitos aleatórios que variam entre os grupos de `{group_var}`.")
                        
                    else:
                        st.info("Não foi possível extrair a variância de efeitos aleatórios explicitamente de `vc_params`. Por favor, consulte a seção 'Random Effects' no sumário completo do modelo acima.")

                    st.write("##### Variância e Desvio Padrão do Termo de Erro (Residual)")
                    st.write(f"Variância Residual (Scale): `{results.scale:.4f}`")
                    st.write(f"Desvio Padrão Residual (Scale): `{np.sqrt(results.scale):.4f}`")
                    st.markdown("**Interpretação:** Esta é a variância do erro não explicada pelo modelo.")


                    st.subheader("Visualização dos Efeitos Aleatórios Estimados por Grupo")
                    
                    random_effects_df = pd.DataFrame(results.random_effects).T 

                    if not random_effects_df.empty:
                        st.write("Tabela de Efeitos Aleatórios Estimados (Primeiras 5 Linhas):")
                        st.dataframe(random_effects_df.head())

                        for effect_name in random_effects_df.columns:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                            
                            sns.histplot(random_effects_df[effect_name], kde=True, ax=axes[0])
                            axes[0].set_title(f"Distribuição do Efeito Aleatório: {effect_name}")
                            axes[0].set_xlabel(f"Valor do Efeito Aleatório ({effect_name})")
                            axes[0].set_ylabel("Frequência")

                            sns.boxplot(x=random_effects_df[effect_name], ax=axes[1])
                            axes[1].set_title(f"Box Plot do Efeito Aleatório: {effect_name}")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.info(f"O gráfico de '{effect_name}' mostra a distribuição dos desvios para cada grupo em relação ao valor médio fixo deste efeito. Uma maior dispersão indica maior variabilidade entre os grupos.")
                    else:
                        st.info("Nenhum efeito aleatório estimado para visualizar ou o modelo não convergiu adequadamente.")

                except Exception as e:
                    st.error(f"Erro ao executar a análise multinível: {e}")
                    st.info("Verifique se as variáveis selecionadas são apropriadas para o tipo de modelo (ex: variáveis numéricas para dependente e slopes aleatórios, categóricas para agrupamento). Também verifique se não há valores NaN nas variáveis do modelo.")


# --- FUNÇÃO PARA ANÁLISE DE CAMINHO (PATH ANALYSIS) com Statsmodels e cálculo manual ---
def show_path_analysis_model():
    st.subheader("🕸️ Análise de Caminhos (Path Analysis)")
    st.info("A Análise de Caminhos permite testar um modelo teórico de relações causais entre variáveis observadas, utilizando uma série de regressões. Serão exibidos os resultados numéricos dos efeitos diretos, indiretos e totais.")
    st.info("Para uma interpretação correta dos coeficientes de caminho, as variáveis são padronizadas antes da modelagem.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    # Adicionando o expander para toda a seção de configuração e resultados da análise de caminhos
    with st.expander("Configurar e Executar Análise de Caminhos", expanded=False): # 'expanded=True' para começar aberto
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no DataFrame para realizar a Análise de Caminhos.")
            return

        st.markdown("---")
        st.markdown("#### Definir Relações de Caminho")
        st.info("Para cada variável dependente (endógena) no seu modelo, defina as variáveis independentes (exógenas ou outras endógenas) que a predizem.")
        st.info("Use o formato SEM: `dependente ~ independente1 + independente2`. Cada linha define uma equação de regressão.")

        model_syntax_input = st.text_area(
            "Defina as relações do modelo:",
            value="""# Exemplo de modelo de caminhos:
# M_Mediator ~ X_Predictor
# Y_Outcome ~ X_Predictor + M_Mediator
# Observação: 'M_Mediator' é uma variável mediadora aqui.
# A ordem das equações não importa para o cálculo, mas sim para a interpretação.
""",
            height=200,
            key="pa_model_syntax_input"
        )

        path_equations = []
        all_involved_vars = set() # Para coletar todas as variáveis que serão padronizadas
        pattern = re.compile(r"^\s*#.*$|^\s*(\w+)\s*~\s*(.+)$")
        
        for line in model_syntax_input.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if match:
                dependent_var = match.group(1).strip()
                independent_vars_str = match.group(2).strip()
                independent_vars = [var.strip() for var in independent_vars_str.split('+')]
                
                valid_dependent = dependent_var in numeric_cols
                valid_independent = all(var in numeric_cols for var in independent_vars)
                
                if valid_dependent and valid_independent:
                    path_equations.append({'dependent': dependent_var, 'independent': independent_vars, 'syntax': line})
                    all_involved_vars.add(dependent_var)
                    all_involved_vars.update(independent_vars)
                    st.success(f"Relação adicionada: `{line}`")
                else:
                    missing_vars = []
                    if not valid_dependent:
                        missing_vars.append(dependent_var)
                    for var in independent_vars:
                        if var not in numeric_cols:
                            missing_vars.append(var)
                    st.warning(f"Variáveis não encontradas no DataFrame para a relação `{line}`: {', '.join(missing_vars)}. Esta relação será ignorada.")
            else:
                st.warning(f"Sintaxe inválida para a linha: `{line}`. Use o formato `dependente ~ independente1 + independente2`.")

        if st.button("Executar Análise de Caminhos", key="run_path_analysis_model"):
            if not path_equations:
                st.error("Por favor, defina pelo menos uma relação de caminho válida no formato SEM.")
                return

            
            # Padronizar todas as variáveis envolvidas ANTES de qualquer regressão
            df_standardized = df.copy() 
            
            # Apenas padroniza as colunas que estão na lista all_involved_vars
            if all_involved_vars:
                scaler = StandardScaler()
                cols_to_standardize = [col for col in list(all_involved_vars) if col in numeric_cols]
                
                if cols_to_standardize:
                    # Cria um DataFrame temporário para escalonamento que contém apenas as colunas relevantes
                    # e as linhas SEM NaNs nessas colunas específicas.
                    temp_df_for_scaling = df_standardized[cols_to_standardize].dropna()
                    
                    if not temp_df_for_scaling.empty:
                        # Realiza o escalonamento no DataFrame temporário
                        scaled_values = scaler.fit_transform(temp_df_for_scaling)
                        
                        # Cria um novo DataFrame a partir dos valores escalonados,
                        # PRESERVANDO O ÍNDICE E OS NOMES DAS COLUNAS do temp_df_for_scaling.
                        scaled_df_part = pd.DataFrame(scaled_values, columns=cols_to_standardize, index=temp_df_for_scaling.index)
                        
                        # Atualiza o DataFrame ORIGINAL df_standardized com os valores escalonados,
                        # APENAS PARA AS LINHAS QUE FORAM ESCALONADAS (usando .loc para alinhamento de índice).
                        df_standardized.loc[scaled_df_part.index, cols_to_standardize] = scaled_df_part
                        
                        st.info(f"Variáveis padronizadas (Z-score) para a análise de caminhos: {', '.join(cols_to_standardize)}")
                    else:
                        st.warning("Não há dados completos (sem NaNs) para padronizar as variáveis envolvidas. A análise de caminhos pode não ter coeficientes padronizados confiáveis.")
                else:
                    st.warning("Nenhuma variável numérica válida para padronização. A análise de caminhos pode não ter coeficientes padronizados confiáveis.")
            else:
                st.warning("Nenhuma variável foi definida nas relações de caminho para padronização.")


            st.markdown("---")
            st.subheader("Resultados Detalhados das Regressões (com Coeficientes Padronizados)")

            # Dicionário para armazenar os coeficientes de caminho (betas) para o cálculo dos efeitos
            # Formato: {dependente: {independente: coeficiente_beta}}
            path_coefficients = {} 

            for j, equation in enumerate(path_equations):
                dependent = equation['dependent']
                independent = equation['independent']

                if not independent:
                    st.warning(f"A Relação {j+1} ('{dependent}' predita por nada) não será processada. Selecione variáveis independentes.")
                    continue

                st.markdown(f"#### Modelo para: **{dependent}**")
                
                # Usar o DataFrame PADRONIZADO
                X = df_standardized[independent].copy()
                y = df_standardized[dependent].copy()

                # Tratamento de valores NaN (removendo linhas). Importante que X e y correspondam.
                initial_rows = X.shape[0]
                data = pd.concat([X, y], axis=1).dropna()
                X = data[independent]
                y = data[dependent]

                
                if initial_rows > X.shape[0]:
                    st.warning(f"Foram removidas {initial_rows - X.shape[0]} linhas na relação de '{dependent}' devido a valores NaN.")

                if X.empty or y.empty:
                    st.error(f"Não há dados suficientes após o tratamento de NaN para construir o modelo para '{dependent}'. Por favor, verifique se há dados completos para as variáveis selecionadas.") # Mensagem mais informativa
                    continue

                # Adicionar uma constante ao X para o Statsmodels
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X) # Usar y e X completos para ajuste, ou treinar/testar se quiser métricas de previsão
                    results = model.fit()

                    st.write("##### Sumário da Regressão (Statsmodels):")
                    st.code(results.summary().as_text())

                    st.write("##### Coeficientes de Caminho Padronizados (Betas) e P-valores:")
                    
                    # Os coeficientes do statsmodels com dados padronizados JÁ SÃO os betas
                    coefs_df = results.summary2().tables[1][['Coef.', 'P>|t|']]
                    coefs_df.rename(columns={'Coef.': 'Beta (Coef. Padronizado)', 'P>|t|': 'P-valor'}, inplace=True)
                    coefs_df.index.name = 'Variável Preditiva'
                    
                    # Armazenar betas para cálculo de efeitos indiretos/totais
                    path_coefficients[dependent] = {
                        idx: row['Beta (Coef. Padronizado)'] 
                        for idx, row in coefs_df.iterrows() if idx != 'const'
                    }
                    
                    st.dataframe(coefs_df)

                    r2_eq = r2_score(y, results.predict(X))
                    st.metric(label=f"R-squared para '{dependent}'", value=f"{r2_eq:.4f}")

                except Exception as e:
                    st.error(f"Ocorreu um erro ao treinar o modelo de regressão para '{dependent}': {e}")
                    st.warning("Verifique se as colunas selecionadas não contêm valores infinitos ou colunas com variância zero.")
                st.markdown("---") # Separador para cada modelo
            
            
            
            st.subheader("Resultados da Análise de Caminhos: Efeitos Diretos, Indiretos e Totais")
            st.info("Nesta seção, calculamos os efeitos baseados nos coeficientes padronizados (Betas) obtidos nas regressões acima.")

            # Calculando efeitos diretos, indiretos e totais
            effects = {}

            # Identificar variáveis endógenas (dependentes em alguma equação)
            endogenous_vars = set(eq['dependent'] for eq in path_equations)
            # Identificar variáveis exógenas (nunca dependentes)
            # Devemos considerar as variáveis que estão em all_involved_vars mas NUNCA são dependentes
            # em nenhuma das equações definidas.
            defined_dependent_vars = set(eq['dependent'] for eq in path_equations)
            exogenous_vars = all_involved_vars - defined_dependent_vars

            st.write("#### Efeitos Diretos (Betas)")
            direct_effects_df = []
            for dependent, coefs in path_coefficients.items():
                for predictor, beta in coefs.items():
                    direct_effects_df.append({
                        'Origem': predictor,
                        'Destino': dependent,
                        'Efeito Direto (Beta)': f"{beta:.4f}"
                    })
            if direct_effects_df:
                st.dataframe(pd.DataFrame(direct_effects_df))
            else:
                st.info("Nenhum efeito direto calculado.")


            st.write("#### Efeitos Indiretos e Totais (Caminhos de Mediação Simples)")
            st.warning("Esta seção calcula efeitos indiretos e totais para cenários de mediação simples (X -> M -> Y). Para modelos de caminhos mais complexos, uma interpretação manual ou ferramentas mais avançadas seriam necessárias.")

            indirect_effects_list = []

            # Itere sobre as variáveis exógenas (potenciais X)
            for exog_var in exogenous_vars:
                # Encontre variáveis que são preditas por exog_var (potenciais mediadoras M)
                for mediator_var in endogenous_vars:
                    # Condição 1: Existe um caminho direto de X para M? (exog_var prediz mediator_var)
                    if exog_var in path_coefficients.get(mediator_var, {}): 
                        beta_xm = path_coefficients[mediator_var][exog_var] # Coeficiente X -> M
                        
                        # Encontre variáveis que são preditas por mediator_var (potenciais Y)
                        for final_dep_var in endogenous_vars:
                            # Condição 2: Existe um caminho direto de M para Y? (mediator_var prediz final_dep_var)
                            if mediator_var in path_coefficients.get(final_dep_var, {}): 
                                # Condição 3: Y não é o próprio X (evita ciclos ou redundâncias)
                                if final_dep_var != exog_var:
                                    beta_my = path_coefficients[final_dep_var][mediator_var] # Coeficiente M -> Y
                                    
                                    indirect_effect = beta_xm * beta_my
                                    
                                    # Coletar o efeito direto de X para Y, se existir na equação de Y
                                    # Se X prediz Y diretamente, ele estará em path_coefficients[final_dep_var][exog_var]
                                    direct_effect_xy = path_coefficients.get(final_dep_var, {}).get(exog_var, 0)
                                    total_effect = direct_effect_xy + indirect_effect

                                    indirect_effects_list.append({
                                        'Origem': exog_var,
                                        'Mediador': mediator_var,
                                        'Destino': final_dep_var,
                                        'Efeito Indireto (Beta)': f"{indirect_effect:.4f}",
                                        'Efeito Direto (Beta)': f"{direct_effect_xy:.4f}", # Re-exibe para contexto
                                        'Efeito Total (Beta)': f"{total_effect:.4f}"
                                    })
            
            if indirect_effects_list:
                st.dataframe(pd.DataFrame(indirect_effects_list))
            else:
                st.info("Nenhum efeito indireto de mediação simples encontrado com as relações definidas.")
                st.info("Verifique se há cadeias X -> M e M -> Y onde X é exógena (não é predita por nada no modelo), M é endógena (é predita por X e preditora de Y), e X é preditora de M.")

                
            st.markdown("---")
            st.info("O diagrama de caminhos visual não pode ser gerado no momento. Você pode usar os resultados dos coeficientes para desenhar o diagrama manualmente ou tentar instalar o Graphviz mais tarde.")