#ok
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error,
    r2_score, mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import plotly.graph_objs as go

def compute_shap_values(model, X_sampled, model_type="tree", predict_fn=None, class_index=None):
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_output = explainer(X_sampled)
        if isinstance(shap_output, list):
            if class_index is not None:
                return shap_output[class_index]
            else:
                raise ValueError("Modelo multiclasse requer class_index para SHAP.")
        return shap_output

    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_sampled)
        shap_values_raw = explainer.shap_values(X_sampled)
        if isinstance(shap_values_raw, list):
            if class_index is not None:
                return shap.Explanation(
                    values=shap_values_raw[class_index],
                    base_values=explainer.expected_value[class_index],
                    data=X_sampled.values,
                    feature_names=X_sampled.columns.tolist()
                )
            else:
                raise ValueError("Modelo multiclasse requer class_index para SHAP.")
        return shap.Explanation(
            values=shap_values_raw,
            base_values=explainer.expected_value,
            data=X_sampled.values,
            feature_names=X_sampled.columns.tolist()
        )

    elif model_type == "kernel":
        explainer = shap.KernelExplainer(predict_fn, shap.utils.sample(X_sampled, 50))
        shap_values_raw = explainer.shap_values(X_sampled)
        if isinstance(shap_values_raw, list):
            if class_index is not None:
                return shap.Explanation(
                    values=shap_values_raw[class_index],
                    base_values=explainer.expected_value[class_index],
                    data=X_sampled.values,
                    feature_names=X_sampled.columns.tolist()
                )
            else:
                raise ValueError("Modelo multiclasse requer class_index para SHAP.")
        return shap.Explanation(
            values=shap_values_raw,
            base_values=explainer.expected_value,
            data=X_sampled.values,
            feature_names=X_sampled.columns.tolist()
        )
    else:
        raise ValueError("Tipo de modelo não suportado para SHAP.")

# Função de gráfico scatter SHAP

def shap_scatter_plot(shap_values, feature_idx, feature_names):
    x = shap_values.values[:, feature_idx]
    y = shap_values.base_values + shap_values.values.sum(axis=1)

    color = shap_values.values[:, feature_idx]

    fig = go.Figure(data=go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color=color, colorscale='RdBu', showscale=True),
        hovertemplate=f'{feature_names[feature_idx]} SHAP: %{{x:.3f}}<br>Model Output (f(x)): %{{y:.3f}}<extra></extra>'
    ))
    fig.update_layout(
        title=f'SHAP Scatter Plot - Feature: {feature_names[feature_idx]}',
        xaxis_title=f'SHAP value for {feature_names[feature_idx]}',
        yaxis_title='Model Output (f(x))'
    )
    st.plotly_chart(fig)
    return fig



def shap_scatter_plot(shap_values, feature_idx, feature_names):
    x = shap_values.values[:, feature_idx]
    # Y-axis should represent the actual model output (f(x)) = base_value + sum of SHAP values
    y = shap_values.base_values + shap_values.values.sum(axis=1)

    color = shap_values.values[:, feature_idx]

    fig = go.Figure(data=go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color=color, colorscale='RdBu', showscale=True),
        hovertemplate=f'{feature_names[feature_idx]} SHAP: %{{x:.3f}}<br>Model Output (f(x)): %{{y:.3f}}<extra></extra>'
    ))
    fig.update_layout(
        title=f'SHAP Scatter Plot - Feature: {feature_names[feature_idx]}',
        xaxis_title=f'SHAP value for {feature_names[feature_idx]}',
        yaxis_title='Model Output (f(x))'
    )
    st.plotly_chart(fig)
    return fig

# Renomeando a função para ser mais genérica e adaptada à nova estrutura
def show_machine_learning_page():
    st.subheader("📈 Modelos Supervisionados e Não-Supervisionados com Explicabilidade e Hiperparametrização")

    df = st.session_state.get("df_processed", None)
    if df is None or df.empty:
        st.warning("Carregue e processe os dados antes de realizar a modelagem.")
        return

    # Initialize session state for storing model metrics if not already present
    if 'reg_model_metrics' not in st.session_state:
        st.session_state['reg_model_metrics'] = []
    if 'clf_model_metrics' not in st.session_state:
        st.session_state['clf_model_metrics'] = []

    # Novo seletor principal para os tipos de modelo
    model_category_selection = st.radio(
        "Selecione a Categoria de Modelo:",
        ["Modelos Preditivos (Regressão)", "Modelos de Classificação", "Modelos Não-Supervisionados"],
        help="Escolha entre modelos de regressão, classificação ou não-supervisionados."
    )

    if model_category_selection == "Modelos Preditivos (Regressão)":
        with st.expander("⚙️ Configuração e Treinamento de Modelos Preditivos (Regressão)", expanded=True):
            st.markdown("### 📊 Configuração de Regressão")
            st.markdown("Configure os parâmetros do modelo de regressão. Use hiperparametrização (GridSearchCV) se desejar encontrar os melhores hiperparâmetros.")

            target = st.selectbox("🎯 Variável alvo (target) para Regressão:", df.columns, key="reg_target")
            features = st.multiselect("📌 Variáveis preditoras (features) para Regressão:", [col for col in df.columns if col != target], key="reg_features")

            if not target or not features:
                st.info("Selecione a variável alvo e pelo menos uma preditora para a regressão.")
                return

            test_size = st.slider("🔀 Proporção de dados para teste", 0.1, 0.5, 0.2, 0.05, key="reg_test_size")
            random_state = st.number_input("Seed (random_state)", min_value=0, value=42, step=1, key="reg_random_state")
            enable_gridsearch = st.checkbox("🔍 Ativar GridSearchCV para hiperparametrização", key="reg_gridsearch")
            
            # --- New SHAP toggle ---
            enable_shap = st.checkbox("⚙️ Ativar Explicabilidade com SHAP (pode ser computacionalmente intensivo)", value=False, key="reg_enable_shap")
            # --- End New SHAP toggle ---

            # Criando cópia do dataframe para evitar SettingWithCopyWarning e manipulação segura
            df_model = df.copy()

            # TRATAMENTO DE NaN NA VARIÁVEL ALVO (REGRESSÃO)
            if df_model[target].isnull().any():
                st.warning(f"A variável alvo '{target}' contém valores ausentes. Preenchendo com a média.")
                df_model[target] = df_model[target].fillna(df_model[target].mean())

            X = df_model[features]
            y = df_model[target]
            task_type = "regressao"

            # Preprocessing Pipeline with Imputation
            numeric_features = X.select_dtypes(include=np.number).columns
            categorical_features = X.select_dtypes(include='object').columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )

            model_option = st.selectbox("🧠 Escolha o modelo de Regressão:", [
                "Regressão Linear", "Random Forest Regressor", "SVM Regressor", "KNN Regressor", "XGBoost Regressor", "LightGBM Regressor"
            ], key="reg_model_select")

            base_model = None
            param_grid = {}

            if model_option == "Regressão Linear":
                base_model = LinearRegression()
            elif model_option == "Random Forest Regressor":
                base_model = RandomForestRegressor(random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [None, 10, 20]}
            elif model_option == "SVM Regressor":
                base_model = SVR()
                if enable_gridsearch:
                    param_grid = {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']}
            elif model_option == "KNN Regressor":
                base_model = KNeighborsRegressor()
                if enable_gridsearch:
                    param_grid = {'regressor__n_neighbors': [3, 5, 7]}
            elif model_option == "XGBoost Regressor":
                base_model = XGBRegressor(random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]}
            elif model_option == "LightGBM Regressor":
                base_model = LGBMRegressor(random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]}

            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', base_model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            trained_model = None
            best_params_found = "N/A"

            if st.button(f"Treinar Modelo de {model_option}", key="train_reg_model_button"):
                with st.spinner(f"Treinando {model_option} e otimizando hiperparâmetros..." if enable_gridsearch else f"Treinando {model_option}..."):
                    if enable_gridsearch and param_grid:
                        st.info("Executando GridSearchCV...")
                        grid = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
                        grid.fit(X_train, y_train)
                        trained_model = grid.best_estimator_
                        best_params_found = grid.best_params_
                        st.success(f"Melhor modelo encontrado com GridSearchCV: {trained_model}")
                        st.write("Melhores Hiperparâmetros:", best_params_found)
                    else:
                        trained_model = model_pipeline
                        trained_model.fit(X_train, y_train)
                        st.success(f"Modelo {model_option} treinado com sucesso!")

                    st.session_state['trained_reg_model'] = trained_model
                    st.session_state['reg_y_test'] = y_test
                    st.session_state['reg_X_test'] = X_test
                    st.session_state['reg_X_train'] = X_train
                    st.session_state['reg_y_train'] = y_train
                    st.session_state['reg_task_type'] = task_type
                    st.session_state['reg_model_option'] = model_option
                    st.session_state['reg_enable_gridsearch'] = enable_gridsearch
                    st.session_state['reg_best_params_found'] = best_params_found
                    st.session_state['reg_original_target_values'] = None # Not applicable for regression
                    st.session_state['reg_label_encoder_mapping'] = None # Not applicable for regression
                    # st.session_state['reg_enable_shap'] = enable_shap # This is already set by the widget key

                    # Calculate and store metrics for the comparison table
                    y_pred = trained_model.predict(X_test)
                    current_r2 = r2_score(y_test, y_pred)
                    current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    current_mae = mean_absolute_error(y_test, y_pred)
                    
                    cv_scores = cross_val_score(trained_model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=random_state), scoring='r2', n_jobs=-1)
                    current_mean_cv_score = np.mean(cv_scores)

                    model_metrics_entry = {
                        'Modelo': model_option,
                        'R2_Score': f"{current_r2:.4f}",
                        'RMSE': f"{current_rmse:.4f}",
                        'MAE': f"{current_mae:.4f}",
                        'Media_CV_R2': f"{current_mean_cv_score:.4f}",
                        'GridSearchCV_Ativado': enable_gridsearch,
                        'Melhores_Hiperparametros': str(best_params_found),
                        'SHAP_Ativado': enable_shap
                    }
                    st.session_state['reg_model_metrics'].append(model_metrics_entry)


            if 'trained_reg_model' in st.session_state and st.session_state['trained_reg_model'] is not None and st.session_state['reg_task_type'] == "regressao":
                trained_model = st.session_state['trained_reg_model']
                y_test = st.session_state['reg_y_test']
                X_test = st.session_state['reg_X_test']
                X_train = st.session_state['reg_X_train']
                y_train = st.session_state['reg_y_train']
                model_option = st.session_state['reg_model_option']
                target = st.session_state['reg_target']
                features = st.session_state['reg_features']
                enable_gridsearch = st.session_state['reg_enable_gridsearch']
                best_params_found = st.session_state['reg_best_params_found']
                current_enable_shap = st.session_state['reg_enable_shap'] # Retrieve SHAP toggle state

                y_pred = trained_model.predict(X_test)

                st.markdown("### 📊 Avaliação - Regressão")
                st.write("**R²:**", r2_score(y_test, y_pred))
                st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test, y_pred)))
                st.write("**MAE:**", mean_absolute_error(y_test, y_pred))

                fig_resid, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction')
                ax.set_xlabel("Valores Reais")
                ax.set_ylabel("Valores Preditos")
                ax.set_title('Real vs. Predicted Values')
                ax.legend()
                st.pyplot(fig_resid)

                st.markdown("### 🔁 Validação Cruzada")
                cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
                # Use X e y originais (tratados de NaN) para validação cruzada
                # Re-obtenha X e y do df_model (que já teve o target tratado)
                current_X_for_cv = df_model[features]
                current_y_for_cv = df_model[target]
                scores = cross_val_score(trained_model, current_X_for_cv, current_y_for_cv, cv=cv, scoring='r2', n_jobs=-1)
                st.write("Scores de Validação Cruzada:", scores)
                st.write("Média dos Scores de Validação Cruzada:", np.mean(scores))
                st.write("Desvio Padrão dos Scores de Validação Cruzada:", np.std(scores))

                # --- Conditionally generate SHAP plots ---
                if current_enable_shap:
                    st.markdown("### 🧠 Explicabilidade com SHAP")
                    try:
                        X_test_preprocessed = trained_model.named_steps['preprocessor'].transform(X_test)
                        feature_names_out_raw = trained_model.named_steps['preprocessor'].get_feature_names_out()
                        cleaned_feature_names = []
                        for name in feature_names_out_raw:
                            name = name.replace('num__', '')
                            if 'cat__' in name:
                                parts = name.split('__')
                                if len(parts) > 1:
                                    feature_and_value = parts[1]
                                    last_underscore_idx = feature_and_value.rfind('_')
                                    if last_underscore_idx != -1:
                                        original_feature_name = feature_and_value[:last_underscore_idx]
                                        category_value = feature_and_value[last_underscore_idx+1:]
                                        name = f"{original_feature_name}: {category_value}"
                                    else:
                                        name = feature_and_value.replace('cat__', '')
                                else:
                                    name = name.replace('cat__', '')
                            cleaned_feature_names.append(name)

                        X_test_df_shap = pd.DataFrame(X_test_preprocessed, columns=cleaned_feature_names)
                        model_for_shap = trained_model.named_steps['regressor'] 

                        if isinstance(model_for_shap, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
                            explainer = shap.TreeExplainer(model_for_shap) 
                            shap_values = explainer(X_test_df_shap)
                        elif isinstance(model_for_shap, LinearRegression):
                            background_data_for_linear = trained_model.named_steps['preprocessor'].transform(X_train)
                            if background_data_for_linear.shape[0] > 1000:
                                background_data_for_linear = shap.utils.sample(background_data_for_linear, 1000, random_state=random_state)
                            explainer = shap.LinearExplainer(model_for_shap, background_data_for_linear)
                            shap_values_raw = explainer.shap_values(X_test_df_shap)
                            shap_values = shap.Explanation(
                                values=shap_values_raw,
                                base_values=explainer.expected_value,
                                data=X_test_df_shap.values,
                                feature_names=X_test_df_shap.columns.tolist()
                            )
                        else:
                            st.warning("O modelo selecionado não é um modelo de árvore nem linear. Usando `KernelExplainer` do SHAP, que pode ser muito lento para grandes conjuntos de dados. **Recomenda-se reduzir o tamanho do conjunto de dados de teste** para explicabilidade SHAP para este modelo.")
                            X_test_df_shap_sampled = X_test_df_shap
                            if X_test_df_shap.shape[0] > 100: 
                                 X_test_df_shap_sampled = shap.utils.sample(X_test_df_shap, 100, random_state=random_state) 
                                 st.info(f"Amostrando {X_test_df_shap_sampled.shape[0]} observações para `KernelExplainer` para melhorar o desempenho.")

                            background_data = trained_model.named_steps['preprocessor'].transform(X_train)
                            if background_data.shape[0] > 50: 
                                background_data = shap.utils.sample(background_data, 50, random_state=random_state) 

                            explainer = shap.KernelExplainer(model_for_shap.predict, background_data)
                            shap_values_raw = explainer.shap_values(X_test_df_shap_sampled)
                            shap_values = shap.Explanation(
                                values=shap_values_raw,
                                base_values=explainer.expected_value,
                                data=X_test_df_shap_sampled.values,
                                feature_names=X_test_df_shap_sampled.columns.tolist()
                            )

                        if not isinstance(shap_values, shap.Explanation):
                            st.error("Erro interno: `shap_values` não é um objeto `shap.Explanation` após a inicialização. Isso pode causar problemas de plotagem.")
                        else:
                            st.markdown("#### 🔍 Summary Plot (Importância Global das Features)")
                            fig_summary = plt.figure(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_test_df_shap, show=False)
                            plt.tight_layout()
                            st.pyplot(fig_summary)

                            shap_var = st.selectbox("Escolha uma variável para Scatter Plot SHAP:", list(X_test_df_shap.columns), key="reg_shap_var")
                            feature_idx = list(X_test_df_shap.columns).index(shap_var)
                            fig_scatter_shap = shap_scatter_plot(shap_values, feature_idx, list(X_test_df_shap.columns))

                            st.markdown("#### 🌊 Waterfall Plot (Explicabilidade para uma Observação Individual)")
                            obs_idx = st.slider("Selecione o índice da observação para Waterfall Plot", 0, len(X_test_df_shap) - 1, 0, key="reg_obs_idx")
                            fig_waterfall = plt.figure(figsize=(10, 6))
                            shap.plots.waterfall(shap_values[obs_idx], show=False)
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)

                            st.session_state['reg_fig_summary'] = fig_summary
                            st.session_state['reg_fig_waterfall'] = fig_waterfall
                            st.session_state['reg_fig_scatter_shap'] = fig_scatter_shap

                    except Exception as e:
                        st.error(f"Não foi possível gerar os gráficos SHAP para o modelo de Regressão ({model_option}). Detalhes do erro: {e}")
                        st.info("Isso pode ocorrer devido a incompatibilidades de versão do SHAP, modelos não totalmente suportados, ou problemas de desempenho com grandes volumes de dados. Por favor, tente um modelo diferente ou verifique a versão da biblioteca SHAP.")
                else:
                    st.info("A geração de gráficos SHAP está desativada. Ative o checkbox 'Ativar Explicabilidade com SHAP' para visualizá-los.")
                    # Clear SHAP figures from session state if not enabled in the current run (optional but good practice)
                    if 'reg_fig_summary' in st.session_state: del st.session_state['reg_fig_summary']
                    if 'reg_fig_waterfall' in st.session_state: del st.session_state['reg_fig_waterfall']
                    if 'reg_fig_scatter_shap' in st.session_state: del st.session_state['reg_fig_scatter_shap']
                # --- End Conditional SHAP plots ---

                st.markdown("### 📥 Exportar Resultados e Relatório")
                metrics = {
                    'Modelo': [model_option],
                    'Task': [task_type],
                    'Score_Medio_CV': [np.mean(scores)],
                    'Target': [target],
                    'Features': [", ".join(features)],
                    'GridSearchCV_Ativado': [enable_gridsearch],
                    'Melhores_Hiperparametros': [str(best_params_found)],
                    'R2_Score': [r2_score(y_test, y_pred)],
                    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred))],
                    'MAE': [mean_absolute_error(y_test, y_pred)]
                }
                result_df = pd.DataFrame(metrics)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar CSV de Resultados", data=csv, file_name="resultados_regressao.csv", mime="text/csv", key="download_reg_csv")

                if st.button("📄 Gerar PDF do Relatório de Regressão", key="generate_reg_pdf_button"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=16, style='B')
                    pdf.cell(200, 10, txt=f"Relatório de Análise do Modelo: {model_option} (Regressão)", ln=True, align='C')
                    pdf.ln(10)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Configuração do Modelo:", ln=True)
                    pdf.set_font("Arial", size=12)
                    for k, v in metrics.items():
                        pdf.cell(200, 7, txt=f"{k.replace('_', ' ')}: {v[0]}", ln=True)
                    pdf.ln(5)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Resultados da Avaliação:", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 7, txt=f"R²: {r2_score(y_test, y_pred):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"MAE: {mean_absolute_error(y_test, y_pred):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"Média CV Scores: {np.mean(scores):.4f}", ln=True)
                    pdf.ln(5)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Gráficos de Avaliação:", ln=True)
                    if fig_resid:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            fig_resid.savefig(tmp_file.name)
                            pdf.image(tmp_file.name, x=10, w=180)
                        tmp_file.close()
                        import os
                        os.unlink(tmp_file.name)
                        pdf.ln(5)

                    # --- Conditionally add SHAP plots to PDF ---
                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Gráficos de Explicabilidade (SHAP):", ln=True)
                    if current_enable_shap: # Only attempt to add if SHAP was enabled
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 7, txt="Os gráficos SHAP (Summary Plot, Scatter Plot e Waterfall Plot) fornecem insights sobre a importância das features e a contribuição individual de cada feature para as previsões do modelo. Eles estão disponíveis na interface da aplicação para exploração interativa. Gráficos SHAP podem não estar disponíveis em PDF se houve um erro na geração na interface.")

                        if 'reg_fig_summary' in st.session_state and st.session_state['reg_fig_summary']:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_shap_summary_file:
                                    st.session_state['reg_fig_summary'].savefig(tmp_shap_summary_file.name)
                                    pdf.image(tmp_shap_summary_file.name, x=10, w=180)
                                tmp_shap_summary_file.close()
                                import os
                                os.unlink(tmp_shap_summary_file.name)
                                pdf.ln(5)
                            except Exception as e:
                                st.warning(f"Não foi possível incorporar o Summary Plot SHAP no PDF: {e}")
                    else:
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 7, txt="A geração de gráficos SHAP estava desativada para este modelo e, portanto, não foram incluídos neste relatório PDF.")
                    # --- End Conditionally add SHAP plots to PDF ---

                    pdf.output("relatorio_regressao.pdf")
                    with open("relatorio_regressao.pdf", "rb") as f:
                        pdf_bytes = f.read()
                        st.download_button("📄 Baixar Relatório PDF de Regressão", pdf_bytes, file_name="relatorio_regressao.pdf", mime="application/pdf", key="download_reg_pdf")

                # --- Comparison Table for Regression Models ---
                st.markdown("---")
                st.markdown("### 📊 Comparação de Modelos de Regressão")
                if st.session_state['reg_model_metrics']:
                    metrics_df_reg = pd.DataFrame(st.session_state['reg_model_metrics'])
                    st.dataframe(metrics_df_reg.set_index('Modelo'))
                else:
                    st.info("Nenhum modelo de regressão foi treinado ainda para comparação.")


    elif model_category_selection == "Modelos de Classificação":
        with st.expander("⚙️ Configuração e Treinamento de Modelos de Classificação", expanded=True):
            st.markdown("### 📊 Configuração de Classificação")
            st.markdown("Configure os parâmetros do modelo de classificação. Use hiperparametrização (GridSearchCV) se desejar encontrar os melhores hiperparâmetros.")

            target = st.selectbox("🎯 Variável alvo (target) para Classificação:", df.columns, key="clf_target")
            features = st.multiselect("📌 Variáveis preditoras (features) para Classificação:", [col for col in df.columns if col != target], key="clf_features")

            if not target or not features:
                st.info("Selecione a variável alvo e pelo menos uma preditora para a classificação.")
                return

            test_size = st.slider("🔀 Proporção de dados para teste", 0.1, 0.5, 0.2, 0.05, key="clf_test_size")
            random_state = st.number_input("Seed (random_state)", min_value=0, value=42, step=1, key="clf_random_state")
            enable_gridsearch = st.checkbox("🔍 Ativar GridSearchCV para hiperparametrização", key="clf_gridsearch")
            
            # --- New SHAP toggle ---
            enable_shap = st.checkbox("⚙️ Ativar Explicabilidade com SHAP (pode ser computacionalmente intensivo)", value=False, key="clf_enable_shap")
            # --- End New SHAP toggle ---

            # Criando cópia do dataframe para evitar SettingWithCopyWarning e manipulação segura
            df_model = df.copy()

            # TRATAMENTO DE NaN NA VARIÁVEL ALVO (CLASSIFICAÇÃO)
            if df_model[target].isnull().any():
                st.warning(f"A variável alvo '{target}' contém valores ausentes. Removendo linhas com NaNs na variável alvo.")
                df_model.dropna(subset=[target], inplace=True)
                if df_model.empty:
                    st.error("Após remover linhas com NaNs na variável alvo, o DataFrame ficou vazio. Não é possível prosseguir com a modelagem.")
                    return

            X = df_model[features]
            y = df_model[target]

            original_target_values = y.unique().tolist()
            label_encoder_mapping = None

            task_type = "classificacao"
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoder_mapping = dict(zip(le.transform(original_target_values), original_target_values))


            # Preprocessing Pipeline with Imputation
            numeric_features = X.select_dtypes(include=np.number).columns
            categorical_features = X.select_dtypes(include='object').columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )

            model_option = st.selectbox("🧠 Escolha o modelo de Classificação:", [
                "Regressão Logística", "Random Forest Classifier", "SVM Classifier", "KNN Classifier", "XGBoost Classifier", "LightGBM Classifier"
            ], key="clf_model_select")

            base_model = None
            param_grid = {}

            if model_option == "Regressão Logística":
                base_model = LogisticRegression(max_iter=1000, random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'classifier__C': [0.01, 0.1, 1, 10]}
            elif model_option == "Random Forest Classifier":
                base_model = RandomForestClassifier(random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]}
            elif model_option == "SVM Classifier":
                base_model = SVC(probability=True, random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
            elif model_option == "KNN Classifier":
                base_model = KNeighborsClassifier()
                if enable_gridsearch:
                    param_grid = {'classifier__n_neighbors': [3, 5, 7]}
            elif model_option == "XGBoost Classifier":
                base_model = XGBClassifier(eval_metric='logloss', random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1, 0.2]}
            elif model_option == "LightGBM Classifier":
                base_model = LGBMClassifier(random_state=random_state)
                if enable_gridsearch:
                    param_grid = {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1, 0.2]}

            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', base_model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            trained_model = None
            best_params_found = "N/A"

            if st.button(f"Treinar Modelo de {model_option}", key="train_clf_model_button"):
                with st.spinner(f"Treinando {model_option} e otimizando hiperparâmetros..." if enable_gridsearch else f"Treinando {model_option}..."):
                    if enable_gridsearch and param_grid:
                        st.info("Executando GridSearchCV...")
                        grid = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                        grid.fit(X_train, y_train)
                        trained_model = grid.best_estimator_
                        best_params_found = grid.best_params_
                        st.success(f"Melhor modelo encontrado com GridSearchCV: {trained_model}")
                        st.write("Melhores Hiperparâmetros:", best_params_found)
                    else:
                        trained_model = model_pipeline
                        trained_model.fit(X_train, y_train)
                        st.success(f"Modelo {model_option} treinado com sucesso!")

                    st.session_state['trained_clf_model'] = trained_model
                    st.session_state['clf_y_test'] = y_test
                    st.session_state['clf_X_test'] = X_test
                    st.session_state['clf_X_train'] = X_train
                    st.session_state['clf_y_train'] = y_train
                    st.session_state['clf_task_type'] = task_type
                    st.session_state['clf_model_option'] = model_option
                    st.session_state['clf_enable_gridsearch'] = enable_gridsearch
                    st.session_state['clf_best_params_found'] = best_params_found
                    st.session_state['clf_original_target_values'] = original_target_values
                    st.session_state['clf_label_encoder_mapping'] = label_encoder_mapping
                    # st.session_state['clf_enable_shap'] = enable_shap # This is already set by the widget key

                    # Calculate and store metrics for the comparison table
                    y_pred = trained_model.predict(X_test)
                    current_accuracy = accuracy_score(y_test, y_pred)
                    current_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    current_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    current_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    # Use X e y originais (tratados de NaN) para validação cruzada
                    # Re-obtenha X e y do df_model (que já teve o target tratado)
                    current_X_for_cv = df_model[features]
                    current_y_for_cv = y # y já está codificado e sem NaNs aqui
                    cv_scores = cross_val_score(trained_model, current_X_for_cv, current_y_for_cv, cv=KFold(n_splits=5, shuffle=True, random_state=random_state), scoring='accuracy', n_jobs=-1)
                    current_mean_cv_score = np.mean(cv_scores)

                    model_metrics_entry = {
                        'Modelo': model_option,
                        'Acuracia': f"{current_accuracy:.4f}",
                        'Precision': f"{current_precision:.4f}",
                        'Recall': f"{current_recall:.4f}",
                        'F1_Score': f"{current_f1:.4f}",
                        'Media_CV_Acuracia': f"{current_mean_cv_score:.4f}",
                        'GridSearchCV_Ativado': enable_gridsearch,
                        'Melhores_Hiperparametros': str(best_params_found),
                        'SHAP_Ativado': enable_shap
                    }
                    st.session_state['clf_model_metrics'].append(model_metrics_entry)


            if 'trained_clf_model' in st.session_state and st.session_state['trained_clf_model'] is not None and st.session_state['clf_task_type'] == "classificacao":
                trained_model = st.session_state['trained_clf_model']
                y_test = st.session_state['clf_y_test']
                X_test = st.session_state['clf_X_test']
                X_train = st.session_state['clf_X_train']
                y_train = st.session_state['clf_y_train']
                model_option = st.session_state['clf_model_option']
                target = st.session_state['clf_target']
                features = st.session_state['clf_features']
                enable_gridsearch = st.session_state['clf_enable_gridsearch']
                best_params_found = st.session_state['clf_best_params_found']
                original_target_values = st.session_state['clf_original_target_values']
                label_encoder_mapping = st.session_state['clf_label_encoder_mapping']
                current_enable_shap = st.session_state['clf_enable_shap'] # Retrieve SHAP toggle state

                y_pred = trained_model.predict(X_test)

                st.markdown("### 📊 Avaliação - Classificação")
                st.write("**Acurácia:**", accuracy_score(y_test, y_pred))
                st.write("**Precision:**", precision_score(y_test, y_pred, average='weighted', zero_division=0))
                st.write("**Recall:**", recall_score(y_test, y_pred, average='weighted', zero_division=0))
                st.write("**F1 Score:**", f1_score(y_test, y_pred, average='weighted', zero_division=0))

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax = plt.subplots(figsize=(6, 4))
                # Use mapped labels if available for better readability
                display_labels = [label_encoder_mapping[i] for i in sorted(label_encoder_mapping.keys())] if label_encoder_mapping else None
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax, cbar=False,
                            xticklabels=display_labels, yticklabels=display_labels)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig_cm)

                st.markdown("### 🔁 Validação Cruzada")
                cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
                # Re-obtenha X e y do df_model (que já teve o target tratado)
                current_X_for_cv = df_model[features]
                current_y_for_cv = y # y já está codificado e sem NaNs aqui
                scores = cross_val_score(trained_model, current_X_for_cv, current_y_for_cv, cv=cv, scoring='accuracy', n_jobs=-1)
                st.write("Scores de Validação Cruzada:", scores)
                st.write("Média dos Scores de Validação Cruzada:", np.mean(scores))
                st.write("Desvio Padrão dos Scores de Validação Cruzada:", np.std(scores))

                # --- Conditionally generate SHAP plots ---
                if current_enable_shap:
                    st.markdown("### 🧠 Explicabilidade com SHAP")
                    try:
                        X_test_preprocessed = trained_model.named_steps['preprocessor'].transform(X_test)
                        feature_names_out_raw = trained_model.named_steps['preprocessor'].get_feature_names_out()
                        cleaned_feature_names = []
                        for name in feature_names_out_raw:
                            name = name.replace('num__', '')
                            if 'cat__' in name:
                                parts = name.split('__')
                                if len(parts) > 1:
                                    feature_and_value = parts[1]
                                    last_underscore_idx = feature_and_value.rfind('_')
                                    if last_underscore_idx != -1:
                                        original_feature_name = feature_and_value[:last_underscore_idx]
                                        category_value = feature_and_value[last_underscore_idx+1:]
                                        name = f"{original_feature_name}: {category_value}"
                                    else:
                                        name = feature_and_value.replace('cat__', '')
                                else:
                                    name = name.replace('cat__', '')
                            cleaned_feature_names.append(name)

                        X_test_df_shap = pd.DataFrame(X_test_preprocessed, columns=cleaned_feature_names)
                        model_for_shap = trained_model.named_steps['classifier']

                        # Determine the class to explain for SHAP
                        num_classes_shap = len(original_target_values)
                        class_to_explain_idx = 0 # Default for multi-class or single-class

                        if num_classes_shap == 2:
                            class_to_explain_idx = 1 # For binary classification, explain the positive class (often encoded as 1)
                            st.info(f"Explicando os valores SHAP para a classe: '{label_encoder_mapping.get(class_to_explain_idx, class_to_explain_idx)}'.")
                        elif num_classes_shap > 2:
                            # Allow user to select the class to explain for multi-class
                            class_options = {label_encoder_mapping[i]: i for i in sorted(label_encoder_mapping.keys())}
                            selected_class_name = st.selectbox(
                                "Selecione a Classe para Explicação SHAP (Multi-classe):",
                                options=list(class_options.keys()),
                                index=0, # Default to the first class
                                key="clf_shap_class_select"
                            )
                            class_to_explain_idx = class_options[selected_class_name]
                            st.info(f"Explicando os valores SHAP para a classe: '{selected_class_name}' (codificada como {class_to_explain_idx}).")
                        else:
                            st.warning("Não foi possível determinar as classes para explicação SHAP ou há apenas uma classe. A explicação SHAP pode não ser aplicável.")
                            class_to_explain_idx = 0 # Fallback default
                            
                        shap_values = None # Initialize to None

                        if isinstance(model_for_shap, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
                            explainer = shap.TreeExplainer(model_for_shap) 
                            shap_output = explainer(X_test_df_shap)
                            if isinstance(shap_output, list): # Multi-class output (list of Explanation objects)
                                if len(shap_output) > class_to_explain_idx:
                                    shap_values = shap_output[class_to_explain_idx]
                                else:
                                    st.error(f"Não foi possível explicar a classe {class_to_explain_idx} para o modelo de árvore. Verifique a seleção da classe.")
                                    return 
                            else: # Binary classification (single Explanation object)
                                shap_values = shap_output
                        elif isinstance(model_for_shap, LogisticRegression):
                            background_data_for_linear = trained_model.named_steps['preprocessor'].transform(X_train)
                            if background_data_for_linear.shape[0] > 1000:
                                background_data_for_linear = shap.utils.sample(background_data_for_linear, 1000, random_state=random_state)
                            
                            explainer = shap.LinearExplainer(model_for_shap, background_data_for_linear)
                            shap_values_raw = explainer.shap_values(X_test_df_shap)
                            if isinstance(shap_values_raw, list): # Multi-class output (list of arrays)
                                if len(shap_values_raw) > class_to_explain_idx:
                                    shap_values = shap.Explanation(
                                        values=shap_values_raw[class_to_explain_idx],
                                        base_values=explainer.expected_value[class_to_explain_idx],
                                        data=X_test_df_shap.values,
                                        feature_names=X_test_df_shap.columns.tolist()
                                    )
                                else:
                                    st.error(f"Não foi possível explicar a classe {class_to_explain_idx} para o modelo linear. Verifique a seleção da classe.")
                                    return
                            else: # Binary classification (single array)
                                shap_values = shap.Explanation(
                                    values=shap_values_raw,
                                    base_values=explainer.expected_value,
                                    data=X_test_df_shap.values,
                                    feature_names=X_test_df_shap.columns.tolist()
                                )
                        else: # KernelExplainer
                            st.warning("O modelo selecionado não é um modelo de árvore nem linear. Usando `KernelExplainer` do SHAP, que pode ser muito lento para grandes conjuntos de dados. **Recomenda-se reduzir o tamanho do conjunto de dados de teste** para explicabilidade SHAP para este modelo.")
                            X_test_df_shap_sampled = X_test_df_shap
                            if X_test_df_shap.shape[0] > 100: 
                                 X_test_df_shap_sampled = shap.utils.sample(X_test_df_shap, 100, random_state=random_state) 
                                 st.info(f"Amostrando {X_test_df_shap_sampled.shape[0]} observações para `KernelExplainer` para melhorar o desempenho.")

                            background_data = trained_model.named_steps['preprocessor'].transform(X_train)
                            if background_data.shape[0] > 50: 
                                background_data = shap.utils.sample(background_data, 50, random_state=random_state) 
                            
                            explainer = shap.KernelExplainer(model_for_shap.predict_proba, background_data)
                            shap_values_raw = explainer.shap_values(X_test_df_shap_sampled)
                            
                            if isinstance(shap_values_raw, list): # Multi-class output (list of arrays)
                                if len(shap_values_raw) > class_to_explain_idx:
                                    shap_values = shap.Explanation(
                                        values=shap_values_raw[class_to_explain_idx],
                                        base_values=explainer.expected_value[class_to_explain_idx],
                                        data=X_test_df_shap_sampled.values,
                                        feature_names=X_test_df_shap_sampled.columns.tolist()
                                    )
                                else:
                                    st.error(f"Não foi possível explicar a classe {class_to_explain_idx} para o KernelExplainer. Verifique a seleção da classe.")
                                    return
                            else: # Binary classification (single array)
                                shap_values = shap.Explanation(
                                    values=shap_values_raw,
                                    base_values=explainer.expected_value,
                                    data=X_test_df_shap_sampled.values,
                                    feature_names=X_test_df_shap_sampled.columns.tolist()
                                )
                        
                        if not isinstance(shap_values, shap.Explanation):
                            st.error("Erro interno: `shap_values` não é um objeto `shap.Explanation` válido após a inicialização. Isso pode causar problemas de plotagem.")
                            return # Exit SHAP section to prevent further errors
                        else:
                            st.markdown("#### 🔍 Summary Plot (Importância Global das Features)")
                            fig_summary = plt.figure(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_test_df_shap, show=False)
                            plt.tight_layout()
                            st.pyplot(fig_summary)

                            shap_var = st.selectbox("Escolha uma variável para Scatter Plot SHAP:", list(X_test_df_shap.columns), key="clf_shap_var")
                            feature_idx = list(X_test_df_shap.columns).index(shap_var)
                            # shap_values is already a single Explanation object for the chosen class here
                            fig_scatter_shap = shap_scatter_plot(shap_values, feature_idx, list(X_test_df_shap.columns))

                            st.markdown("#### 🌊 Waterfall Plot (Explicabilidade para uma Observação Individual)")
                            obs_idx = st.slider("Selecione o índice da observação para Waterfall Plot", 0, len(X_test_df_shap) - 1, 0, key="clf_obs_idx")
                            fig_waterfall = plt.figure(figsize=(10, 6))
                            
                            # Create a single Explanation object for the chosen observation
                            # Ensure base_values is scalar for waterfall plot
                            single_explanation_for_waterfall = shap.Explanation(
                                values=shap_values.values[obs_idx],
                                base_values=shap_values.base_values if np.isscalar(shap_values.base_values) else shap_values.base_values[obs_idx],
                                data=X_test_df_shap.iloc[obs_idx].values,
                                feature_names=X_test_df_shap.columns.tolist()
                            )
                            shap.plots.waterfall(single_explanation_for_waterfall, show=False)
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)

                            st.session_state['clf_fig_summary'] = fig_summary
                            st.session_state['clf_fig_waterfall'] = fig_waterfall
                            st.session_state['clf_fig_scatter_shap'] = fig_scatter_shap

                    except Exception as e:
                        st.error(f"Não foi possível gerar os gráficos SHAP para o modelo de Classificação ({model_option}). Detalhes do erro: {e}")
                        st.info("Isso pode ocorrer devido a incompatibilidades de versão do SHAP, modelos não totalmente suportados, ou problemas de desempenho com grandes volumes de dados. Por favor, tente um modelo diferente ou verifique a versão da biblioteca SHAP.")
                else:
                    st.info("A geração de gráficos SHAP está desativada. Ative o checkbox 'Ativar Explicabilidade com SHAP' para visualizá-los.")
                    # Clear SHAP figures from session state if not enabled in the current run (optional but good practice)
                    if 'clf_fig_summary' in st.session_state: del st.session_state['clf_fig_summary']
                    if 'clf_fig_waterfall' in st.session_state: del st.session_state['clf_fig_waterfall']
                    if 'clf_fig_scatter_shap' in st.session_state: del st.session_state['clf_fig_scatter_shap']
                # --- End Conditional SHAP plots ---

                st.markdown("### 📥 Exportar Resultados e Relatório")
                metrics = {
                    'Modelo': [model_option],
                    'Task': [task_type],
                    'Score_Medio_CV': [np.mean(scores)],
                    'Target': [target],
                    'Features': [", ".join(features)],
                    'GridSearchCV_Ativado': [enable_gridsearch],
                    'Melhores_Hiperparametros': [str(best_params_found)],
                    'Acuracia': [accuracy_score(y_test, y_pred)],
                    'Precision': [precision_score(y_test, y_pred, average='weighted', zero_division=0)],
                    'Recall': [recall_score(y_test, y_pred, average='weighted', zero_division=0)],
                    'F1_Score': [f1_score(y_test, y_pred, average='weighted', zero_division=0)]
                }
                result_df = pd.DataFrame(metrics)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar CSV de Resultados", data=csv, file_name="resultados_classificacao.csv", mime="text/csv", key="download_clf_csv")

                if st.button("📄 Gerar PDF do Relatório de Classificação", key="generate_clf_pdf_button"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=16, style='B')
                    pdf.cell(200, 10, txt=f"Relatório de Análise do Modelo: {model_option} (Classificação)", ln=True, align='C')
                    pdf.ln(10)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Configuração do Modelo:", ln=True)
                    pdf.set_font("Arial", size=12)
                    for k, v in metrics.items():
                        pdf.cell(200, 7, txt=f"{k.replace('_', ' ')}: {v[0]}", ln=True)
                    pdf.ln(5)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Resultados da Avaliação:", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 7, txt=f"Acurácia: {accuracy_score(y_test, y_pred):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", ln=True)
                    pdf.cell(200, 7, txt=f"Média CV Scores: {np.mean(scores):.4f}", ln=True)
                    pdf.ln(5)

                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Gráficos de Avaliação:", ln=True)
                    if fig_cm:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            fig_cm.savefig(tmp_file.name)
                            pdf.image(tmp_file.name, x=10, w=180)
                        tmp_file.close()
                        import os
                        os.unlink(tmp_file.name)
                        pdf.ln(5)

                    # --- Conditionally add SHAP plots to PDF ---
                    pdf.set_font("Arial", size=12, style='B')
                    pdf.cell(200, 10, txt="Gráficos de Explicabilidade (SHAP):", ln=True)
                    if current_enable_shap: # Only attempt to add if SHAP was enabled
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 7, txt="Os gráficos SHAP (Summary Plot, Scatter Plot e Waterfall Plot) fornecem insights sobre a importância das features e a contribuição individual de cada feature para as previsões do modelo. Eles estão disponíveis na interface da aplicação para exploração interativa. Gráficos SHAP podem não estar disponíveis em PDF se houve um erro na geração na interface.")

                        if 'clf_fig_summary' in st.session_state and st.session_state['clf_fig_summary']:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_shap_summary_file:
                                    st.session_state['clf_fig_summary'].savefig(tmp_shap_summary_file.name)
                                    pdf.image(tmp_shap_summary_file.name, x=10, w=180)
                                tmp_shap_summary_file.close()
                                import os
                                os.unlink(tmp_shap_summary_file.name)
                                pdf.ln(5)
                            except Exception as e:
                                st.warning(f"Não foi possível incorporar o Summary Plot SHAP no PDF: {e}")
                    else:
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 7, txt="A geração de gráficos SHAP estava desativada para este modelo e, portanto, não foram incluídos neste relatório PDF.")
                    # --- End Conditionally add SHAP plots to PDF ---

                    pdf.output("relatorio_classificacao.pdf")
                    with open("relatorio_classificacao.pdf", "rb") as f:
                        pdf_bytes = f.read()
                        st.download_button("📄 Baixar Relatório PDF de Classificação", pdf_bytes, file_name="relatorio_classificacao.pdf", mime="application/pdf", key="download_clf_pdf")
                
                # --- Comparison Table for Classification Models ---
                st.markdown("---")
                st.markdown("### 📊 Comparação de Modelos de Classificação")
                if st.session_state['clf_model_metrics']:
                    metrics_df_clf = pd.DataFrame(st.session_state['clf_model_metrics'])
                    st.dataframe(metrics_df_clf.set_index('Modelo'))
                else:
                    st.info("Nenhum modelo de classificação foi treinado ainda para comparação.")


    elif model_category_selection == "Modelos Não-Supervisionados":
        with st.expander("⚙️ Configuração e Treinamento de Modelos Não-Supervisionados", expanded=True):
            st.markdown("### 🛠️ Em Desenvolvimento")
            st.info("Esta seção para Modelos Não-Supervisionados (ex: Clustering, Redução de Dimensionalidade) está em desenvolvimento. Por favor, aguarde futuras atualizações!")