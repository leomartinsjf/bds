#versão com path indireto
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


def show_linear_regression_model():
    st.subheader("📈 Regressão Linear")
    st.info("Utilize a Regressão Linear para prever uma variável dependente contínua com base em uma ou mais variáveis independentes numéricas.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    with st.expander("Configurar e Executar Regressão Linear", expanded=False):
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no DataFrame para realizar a Regressão Linear.")
            return

        st.markdown("---")
        st.markdown("#### Configuração do Modelo de Regressão Linear")

        st.markdown("##### 1. Selecione a Variável Dependente (Y)")
        dependent_var = st.selectbox(
            "Escolha a variável contínua a ser prevista:",
            options=numeric_cols,
            index=None,
            placeholder="Selecione uma variável dependente..."
        )
        if dependent_var is None:
            st.warning("Por favor, selecione uma variável dependente.")
            return

        st.markdown("##### 2. Selecione as Variáveis Independentes (X)")
        options_independent_vars = [col for col in numeric_cols if col != dependent_var]
        if not options_independent_vars:
            st.warning("Não há variáveis numéricas disponíveis para usar como preditoras, exceto a variável dependente selecionada.")
            return

        independent_vars = st.multiselect(
            "Escolha uma ou mais variáveis numéricas para prever a variável dependente:",
            options=options_independent_vars,
            default=[]
        )
        if not independent_vars:
            st.warning("Por favor, selecione pelo menos uma variável independente.")
            return

        if st.button("Executar Regressão Linear", key="run_lr_model"):
            with st.spinner("Treinando Modelo de Regressão Linear..."):
                try:
                    model_vars = [dependent_var] + independent_vars
                    df_model = df[model_vars].dropna()

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

                    # Exibe coeficientes padronizados
                    st.write("#### Coeficientes Padronizados")
                    scaler = StandardScaler()
                    df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=model_vars)
                    Y_std = df_scaled[dependent_var]
                    X_std = df_scaled[independent_vars]
                    X_std = sm.add_constant(X_std)
                    model_std = sm.OLS(Y_std, X_std)
                    results_std = model_std.fit()
                    st.dataframe(results_std.summary2().tables[1].round(4))

                    st.info("Esses são os coeficientes da regressão após padronização (Z-score) das variáveis. Facilitam a comparação do peso relativo das variáveis.")

                    st.write("#### Métricas de Avaliação")
                    predictions = results.predict(X)
                    r2 = r2_score(Y, predictions)
                    mse = mean_squared_error(Y, predictions)
                    rmse = np.sqrt(mse)

                    st.write(f"**R-quadrado (R²):** `{r2:.4f}`")
                    st.write(f"**Erro Quadrático Médio (MSE):** `{mse:.4f}`")
                    st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** `{rmse:.4f}`")

                    st.write("#### Análise de Resíduos")

                    # Diagnóstico: Shapiro-Wilk para normalidade dos resíduos
                    from scipy.stats import shapiro
                    shapiro_stat, shapiro_p = shapiro(results.resid)
                    st.write("**Shapiro-Wilk (normalidade dos resíduos):**")
                    st.write(f"Estatística = `{shapiro_stat:.4f}`, p-valor = `{shapiro_p:.4f}`")
                    if shapiro_p < 0.05:
                        st.warning("Os resíduos provavelmente não seguem distribuição normal (p < 0.05).")
                    else:
                        st.success("Os resíduos são compatíveis com normalidade (p >= 0.05).")

                    # Diagnóstico: VIF
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    vif_data = pd.DataFrame()
                    vif_data['Variável'] = X.columns
                    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    st.write("#### Fatores de Inflação da Variância (VIF)")
                    st.dataframe(vif_data.round(2))
                    st.info("VIF > 5 sugere multicolinearidade moderada; VIF > 10 é preocupante.")

                    # Diagnóstico: Cook's Distance
                    influence = results.get_influence()
                    cooks_d = influence.cooks_distance[0]
                    st.write("#### Cook's Distance")
                    fig_cook, ax_cook = plt.subplots(figsize=(10, 4))
                    ax_cook.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
                    ax_cook.axhline(4/len(X), color='red', linestyle='--', label='Limite (4/n)')
                    ax_cook.set_title("Cook's Distance")
                    ax_cook.set_xlabel("Observação")
                    ax_cook.set_ylabel("Influência")
                    ax_cook.legend()
                    st.pyplot(fig_cook)
                    plt.close(fig_cook)
                    st.info("Observações com Cook's Distance acima de 4/n podem ter influência excessiva no modelo.")
                    fig_res, ax_res = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=predictions, y=results.resid, ax=ax_res)
                    ax_res.axhline(0, color='red', linestyle='--')
                    ax_res.set_xlabel("Valores Preditos")
                    ax_res.set_ylabel("Resíduos")
                    ax_res.set_title("Resíduos vs. Valores Preditos")
                    st.pyplot(fig_res)
                    plt.close(fig_res)

                    fig_hist_res, ax_hist_res = plt.subplots(figsize=(10, 6))
                    sns.histplot(results.resid, kde=True, ax=ax_hist_res)
                    ax_hist_res.set_xlabel("Resíduos")
                    ax_hist_res.set_ylabel("Frequência")
                    ax_hist_res.set_title("Distribuição dos Resíduos")
                    st.pyplot(fig_hist_res)
                    plt.close(fig_hist_res)

                except Exception as e:
                    st.error(f"Erro ao executar a Regressão Linear: {e}")


def show_logistic_regression_model():
    st.subheader("📊 Regressão Logística")
    st.info("Utilize a Regressão Logística para prever uma variável dependente categórica binária (0 ou 1, sim/não) com base em uma ou mais variáveis independentes.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    with st.expander("Configurar e Executar Regressão Logística", expanded=False):
        df = st.session_state.df_processed.copy()
        all_cols = df.columns.tolist()
        binary_cols = [col for col in all_cols if df[col].nunique() == 2 and df[col].dropna().isin([0, 1]).all()]

        if not binary_cols:
            st.warning("Não há colunas binárias (0 ou 1) no DataFrame para a variável dependente.")
            return

        st.markdown("---")
        st.markdown("#### Configuração do Modelo de Regressão Logística")

        st.markdown("##### 1. Selecione a Variável Dependente (Y - Binária)")
        dependent_var = st.selectbox(
            "Escolha a variável binária (0 ou 1) a ser prevista:",
            options=[""] + binary_cols,
            index=0
        )
        if dependent_var == "":
            st.warning("Por favor, selecione uma variável dependente binária.")
            return

        st.markdown("##### 2. Selecione as Variáveis Independentes (X)")
        options_independent_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect(
            "Escolha uma ou mais variáveis para prever a variável dependente:",
            options=options_independent_vars,
            default=[]
        )
        if not independent_vars:
            st.warning("Por favor, selecione pelo menos uma variável independente.")
            return

        if st.button("Executar Regressão Logística", key="run_logr_model"):
            with st.spinner("Treinando Modelo de Regressão Logística..."):
                try:
                    model_vars = [dependent_var] + independent_vars
                    df_model = df[model_vars].dropna()

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

                    # Tabela de coeficientes, odds ratio, IC95% e p-valor
                    st.write("#### Tabela de Coeficientes e Odds Ratios")
                    coef = results.params
                    conf = results.conf_int()
                    conf.columns = ['IC 2.5%', 'IC 97.5%']
                    or_df = pd.DataFrame({
                        'Coeficiente': coef,
                        'Odds Ratio': np.exp(coef),
                        'IC 2.5%': np.exp(conf['IC 2.5%']),
                        'IC 97.5%': np.exp(conf['IC 97.5%']),
                        'P-valor': results.pvalues
                    })
                    st.dataframe(or_df.round(4))
                    st.info("Odds Ratios representam o fator multiplicativo da chance de ocorrência do evento para cada unidade de aumento na variável.")

                    # Coeficientes padronizados (com variáveis normalizadas)
                    st.write("#### Coeficientes Padronizados (Z-score)")
                    scaler = StandardScaler()
                    df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=model_vars)
                    X_std = df_scaled[independent_vars]
                    Y_std = df_scaled[dependent_var]
                    X_std = sm.add_constant(X_std, prepend=False)
                    model_std = sm.GLM(Y_std, X_std, family=sm.families.Binomial())
                    results_std = model_std.fit()
                    st.dataframe(results_std.summary2().tables[1].round(4))
                    st.info("Esses coeficientes permitem comparar a influência relativa das variáveis na escala padronizada.")

                    # Diagnóstico de multicolinearidade
                    st.write("#### Fatores de Inflação da Variância (VIF)")
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    vif_data = pd.DataFrame()
                    vif_data['Variável'] = X.columns
                    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    st.dataframe(vif_data.round(2))
                    st.info("VIF > 5 sugere multicolinearidade moderada; VIF > 10 é preocupante.")

                    st.write("#### Métricas de Avaliação")
                    predictions_proba = results.predict(X)
                    threshold = st.slider("Selecione o Ponto de Corte (Threshold) para Classificação:", 0.0, 1.0, 0.5, 0.01)
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

                except Exception as e:
                    st.error(f"Erro ao executar a Regressão Logística: {e}")


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


# --- FUNÇÃO PARA ANÁLISE DE CAMINHO (PATH ANALYSIS) com Statsmodels e cálculo manual ---
def show_path_analysis_model():
    st.subheader("🕸️ Análise de Caminhos (Path Analysis)")
    st.info("A Análise de Caminhos permite testar um modelo teórico de relações causais entre variáveis observadas, utilizando uma série de regressões.")

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Nenhum dado processado disponível. Por favor, carregue e pré-processe os dados primeiro.")
        return

    with st.expander("Configurar e Executar Análise de Caminhos", expanded=False):
        df = st.session_state.df_processed.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("Não há colunas numéricas no DataFrame para realizar a Análise de Caminhos.")
            return

        st.markdown("---")
        st.markdown("#### Definir Relações de Caminho")
        st.info("Para cada variável dependente (endógena), defina as variáveis independentes que a predizem no formato `Y ~ X1 + X2`.")

        model_syntax_input = st.text_area(
            "Defina as relações do modelo:",
            value="""# Exemplo de modelo de caminhos:
# M_Mediator ~ X_Predictor
# Y_Outcome ~ X_Predictor + M_Mediator
# Observação: 'M_Mediator' é uma variável mediadora aqui.
# A ordem das equações não importa para o cálculo, mas sim para a interpretação.
""",
            height=200
        )

        st.write("Modelo recebido:")
        st.code(model_syntax_input)

        path_equations = []
        all_involved_vars = set()
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
                    path_equations.append({'dependent': dependent_var, 'independent': independent_vars})
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
                st.error("Nenhuma equação válida definida.")
                return

            # Construir grafo dos caminhos
            G = nx.DiGraph()
            for eq in path_equations:
                for pred in eq['independent']:
                    G.add_edge(pred, eq['dependent'])

            st.subheader("Visualização Gráfica do Modelo de Caminhos")
            fig, ax = plt.subplots(figsize=(10, 7)) # Aumentei o tamanho da figura para melhor visualização
            st.markdown("##### Grafo com Betas e P-valores<br>Exógenas à esquerda, Endógenas à direita", unsafe_allow_html=True)
            
            exogenas = list(set(pred for eq in path_equations for pred in eq['independent']) - set(eq['dependent'] for eq in path_equations))
            endogenas = list(set(eq['dependent'] for eq in path_equations))
            
            # Ajuste de layout para uma visualização mais clara
            pos = {}
            # Posiciona exógenas à esquerda
            for i, node in enumerate(exogenas):
                pos[node] = (0, i * 0.3)
            # Posiciona mediadores (se houver, no meio)
            mediadores = list(set(endogenas) & set(pred for eq in path_equations for pred in eq['independent']))
            for i, node in enumerate(mediadores):
                pos[node] = (0.5, i * 0.3 + 0.15) # Leve ajuste para evitar sobreposição
            # Posiciona endógenas à direita
            final_endogenas = list(set(endogenas) - set(mediadores))
            for i, node in enumerate(final_endogenas):
                pos[node] = (1, i * 0.3)

            # Se alguma variável está em 'all_involved_vars' mas não foi posicionada,
            # tenta dar uma posição padrão para evitar erros no draw
            for node in all_involved_vars:
                if node not in pos:
                    # Posiciona nós não conectados ou isolados no centro
                    pos[node] = (0.5, 0.5)

            edge_labels = {}
            for eq in path_equations:
                dep = eq['dependent']
                temp = df[[eq['dependent']] + eq['independent']].dropna()
                X = temp[eq['independent']]
                y = temp[eq['dependent']]
                
                # Tratar caso de multicolinearidade ou poucas observações
                if len(X) <= len(X.columns):
                    st.warning(f"Não há observações suficientes para a equação '{dep} ~ {' + '.join(eq['independent'])}' ou as variáveis são colineares. Betas não serão calculados para esta equação.")
                    continue

                X = sm.add_constant(X)
                try:
                    model = sm.OLS(y, X).fit()
                    for ind in eq['independent']:
                        beta = model.params.get(ind, np.nan) # Usar .get para evitar KeyError se o termo for removido
                        pval = model.pvalues.get(ind, np.nan)
                        
                        if not np.isnan(beta) and not np.isnan(pval):
                            label = f"β={beta:.2f}\n(p={pval:.3f})"
                            edge_labels[(ind, dep)] = label
                        else:
                            edge_labels[(ind, dep)] = "β=N/A\n(p=N/A)" # Indicar que não foi possível calcular
                except Exception as e:
                    st.warning(f"Erro ao ajustar modelo para '{dep} ~ {' + '.join(eq['independent'])}': {e}. Esta relação pode estar mal especificada ou ter problemas de dados.")
                    for ind in eq['independent']:
                        edge_labels[(ind, dep)] = "β=Erro\n(p=Erro)" # Indicar erro de cálculo

            edge_colors = [
                'green' if '(p=' in edge_labels[edge] and not 'N/A' in edge_labels[edge] and float(edge_labels[edge].split('(p=')[1].rstrip(')')) < 0.05
                else 'red' if '(p=' in edge_labels[edge] and not 'N/A' in edge_labels[edge] and float(edge_labels[edge].split('(p=')[1].rstrip(')')) >= 0.05
                else 'gray' for edge in G.edges()
            ]
            node_colors = []
            for node in G.nodes():
                if node in final_endogenas:
                    node_colors.append('gold')  # Endógenas finais
                elif node in mediadores:
                    node_colors.append('lightgreen') # Mediadores
                else:
                    node_colors.append('lightblue')  # Exógenas

            nx.draw(
                G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
                edgecolors='black', node_size=2500, font_size=10, arrows=True, ax=ax
            )
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', ax=ax)
            
            # --- Cálculo dos coeficientes diretos e efeitos ---
            path_coefficients = {}
            for eq in path_equations:
                # O dataframe temporário já foi criado e tratado para NaNs antes
                temp = df[[eq['dependent']] + eq['independent']].dropna()
                X = temp[eq['independent']]
                y = temp[eq['dependent']]
                
                if len(X) > len(X.columns): # Verifica se há observações suficientes para o modelo
                    X = sm.add_constant(X)
                    try:
                        model = sm.OLS(y, X).fit()
                        path_coefficients[eq['dependent']] = model.params.to_dict()
                    except Exception as e:
                        st.warning(f"Não foi possível calcular coeficientes para '{eq['dependent']}' devido a: {e}")
                        path_coefficients[eq['dependent']] = {} # Garante que a chave existe, mas vazia
                else:
                    path_coefficients[eq['dependent']] = {} # Nenhum coeficiente se não houver dados suficientes

            st.subheader("Efeitos Diretos")
            direct_effects = []
            for dep, preds in path_coefficients.items():
                for ind, beta in preds.items():
                    if ind != 'const':
                        direct_effects.append({"Origem": ind, "Destino": dep, "Efeito Direto (Beta)": round(beta, 4)})
            df_direct = pd.DataFrame(direct_effects)
            if not df_direct.empty:
                st.dataframe(df_direct)
                st.download_button(
                    "📥 Baixar Efeitos Diretos",
                    df_direct.to_csv(index=False).encode("utf-8"),
                    file_name="efeitos_diretos.csv",
                    mime="text/csv"
                )
            else:
                st.info("Nenhum efeito direto calculado. Verifique se as equações foram definidas corretamente ou se há dados suficientes.")


            # --- Efeitos Indiretos e Totais (via mediação simples) ---
            st.subheader("Efeitos Indiretos e Totais (via mediação simples)")
            indirect_effects = []
            
            # Lógica para encontrar caminhos indiretos X -> M -> Y
            for eq_m in path_equations: # Equação que define o mediador M
                mediator = eq_m['dependent']
                for predictor_x in eq_m['independent']: # X prediz M
                    # Buscar equações onde M prediz Y
                    for eq_y in path_equations:
                        dependent_y = eq_y['dependent']
                        if mediator in eq_y['independent'] and predictor_x != dependent_y: # M prediz Y, e X não é Y
                            
                            beta_xm = path_coefficients.get(mediator, {}).get(predictor_x)
                            beta_my = path_coefficients.get(dependent_y, {}).get(mediator)
                            
                            if beta_xm is not None and beta_my is not None:
                                indirect = beta_xm * beta_my
                                # Efeito direto de X para Y
                                direct = path_coefficients.get(dependent_y, {}).get(predictor_x, 0)
                                total = direct + indirect
                                
                                # Adicionar ao dicionário de efeitos indiretos
                                indirect_effects.append({
                                    "Origem": predictor_x,
                                    "Mediador": mediator,
                                    "Destino": dependent_y,
                                    "Efeito Indireto (Beta)": round(indirect, 4),
                                    "Efeito Direto (Beta)": round(direct, 4),
                                    "Efeito Total (Beta)": round(total, 4)
                                })
            
            if indirect_effects:
                # Desenhar linhas pontilhadas com beta indireto no mesmo gráfico
                for effect in indirect_effects:
                    origem = effect['Origem']
                    destino = effect['Destino']
                    beta_indireto = effect['Efeito Indireto (Beta)']

                    # Apenas desenha se origem e destino estiverem no grafo (tiverem posições)
                    if origem in pos and destino in pos:
                        x0, y0 = pos[origem]
                        x1, y1 = pos[destino]
                        
                        # Calcula um pequeno offset perpendicular para a linha pontilhada
                        # para que ela não se sobreponha exatamente à linha direta
                        dx = x1 - x0
                        dy = y1 - y0
                        length = np.sqrt(dx**2 + dy**2)
                        
                        if length > 0:
                            unit_dx = dx / length
                            unit_dy = dy / length
                        else:
                            unit_dx, unit_dy = 0, 0 # Evita divisão por zero
                            
                        # Vetor perpendicular (normalizado)
                        perp_dx = -unit_dy
                        perp_dy = unit_dx
                        
                        offset_magnitude = 0.015 # Ajuste a magnitude do offset conforme necessário
                        
                        offset_x = perp_dx * offset_magnitude
                        offset_y = perp_dy * offset_magnitude
                        
                        ax.plot([x0 + offset_x, x1 + offset_x], [y0 + offset_y, y1 + offset_y],
                                linestyle='--', color='purple', alpha=0.7, linewidth=1.5,
                                label=f"Indireto: {origem} -> {destino}")
                        
                        mid_x, mid_y = (x0 + x1) / 2 + offset_x, (y0 + y1) / 2 + offset_y
                        
                        # Ajusta a posição do texto para não sobrepor a linha
                        ax.text(mid_x, mid_y + 0.02, # Ajuste o 0.03 para mover o texto para cima/baixo
                                f"β_ind={beta_indireto:.2f}",
                                fontsize=9, color='purple', ha='center', va='bottom',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

                df_indirect = pd.DataFrame(indirect_effects)
                st.dataframe(df_indirect)
                st.download_button("📥 Baixar Efeitos Indiretos e Totais", df_indirect.to_csv(index=False).encode("utf-8"), file_name="efeitos_indiretos_totais.csv", mime="text/csv")
            else:
                st.info("Nenhum efeito indireto simples detectado com base nas equações fornecidas.")

            # --- RENDERIZAR O GRÁFICO FINAL AQUI ---
            # Salvar e exibir o gráfico DEPOIS de adicionar todos os elementos (diretos e indiretos)
            fig.savefig("grafo_modelo_caminhos.png")
            st.pyplot(fig)
            plt.close(fig) # Libera a memória da figura do Matplotlib

            with open("grafo_modelo_caminhos.png", "rb") as f: # Adicionei .png aqui
                st.download_button(
                    label="📸 Baixar Imagem do Grafo",
                    data=f,
                    file_name="grafo_modelo_caminhos.png",
                    mime="image/png"
                )

            # Exportar CSV com estrutura do modelo
            path_df = pd.DataFrame([
                {"Dependente": eq["dependent"], "Independente": pred}
                for eq in path_equations for pred in eq["independent"]
            ])
            st.download_button(
                label="📥 Baixar Estrutura do Modelo (.csv)",
                data=path_df.to_csv(index=False).encode("utf-8"),
                file_name="modelo_caminhos.csv",
                mime="text/csv"
            )