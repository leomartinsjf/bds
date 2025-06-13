import pandas as pd
import numpy as np
import streamlit as st

# --- Função Safe Predict ---
def safe_predict(modelo, df, label="Predição"):
    try:
        preds = modelo.predict(df)
        return preds
    except Exception as e:
        st.warning(f"⚠️ Erro ao calcular predições do modelo completo: {e}")
        return np.full(len(df), np.nan)

# --- ICC ROBUSTO PRO ---
def calcular_metricas_icc_robusto(modelo, df, grupos_l4, log_visual=True):
    metricas = []
    log_text = ""

    for grupo_nome, vars_grupo in grupos_l4.items():
        for var in vars_grupo:
            try:
                var_random = modelo.cov_re.iloc[0, 0]
                var_residual = modelo.scale

                if (var_random + var_residual) > 0:
                    icc = var_random / (var_random + var_residual)
                    # --- Se for negativo (numérico, mas inconsistente), colocar NaN
                    if icc < 0 or icc > 1:
                        icc = np.nan
                else:
                    icc = np.nan

                metricas.append({
                    'Grupo': grupo_nome,
                    'Variável': var,
                    'metrica': icc
                })

            except Exception as e_var:
                log_text += f"[ICC] Variável '{var}' não convergiu — colocando NaN.\n"
                metricas.append({
                    'Grupo': grupo_nome,
                    'Variável': var,
                    'metrica': np.nan
                })

    df_metricas = pd.DataFrame(metricas)

    if log_visual and log_text != "":
        st.warning(f"⚠️ ICC ROBUSTO — Variáveis que não convergiram:\n{log_text}")

    return df_metricas


def calcular_metricas_pseudoR2_robusto(modelo, df, grupos_l4, log_visual=True):
    metricas = []

    # --- Safe predict ---
    try:
        preds = modelo.predict(df)
    except Exception as e_pred:
        st.warning(f"⚠️ Erro ao calcular predições do modelo completo: {e_pred}")
        preds = np.full(len(df), np.nan)

    y = df[st.session_state.y_var].values
    ss_total = np.nansum((y - np.nanmean(y)) ** 2)

    log_text = ""

    for grupo_nome, vars_grupo in grupos_l4.items():
        for var in vars_grupo:
            try:
                ss_residual = np.nansum((y - preds) ** 2)

                if ss_total > 0 and ss_residual >= 0:
                    pseudo_r2 = 1 - (ss_residual / ss_total)
                    # --- Se for negativo extremo (indicando falha), colocar NaN
                    if pseudo_r2 < -1 or pseudo_r2 > 1:
                        pseudo_r2 = np.nan
                else:
                    pseudo_r2 = np.nan

                metricas.append({
                    'Grupo': grupo_nome,
                    'Variável': var,
                    'metrica': pseudo_r2
                })

            except Exception as e_var:
                log_text += f"[Pseudo-R²] Variável '{var}' não convergiu — colocando NaN.\n"
                metricas.append({
                    'Grupo': grupo_nome,
                    'Variável': var,
                    'metrica': np.nan
                })

    df_metricas = pd.DataFrame(metricas)

    if log_visual and log_text != "":
        st.warning(f"⚠️ Pseudo-R² ROBUSTO — Variáveis que não convergiram:\n{log_text}")

    return df_metricas

def calcular_icc_global(modelo, log_visual=True):
    """
    Calcula o ICC global do modelo MixedLM.
    Retorna: float ICC ou np.nan
    """

    try:
        var_random = modelo.cov_re.iloc[0, 0]
        var_residual = modelo.scale

        if (var_random + var_residual) > 0:
            icc_global = var_random / (var_random + var_residual)

            # Proteção contra valores absurdos
            if icc_global < 0 or icc_global > 1:
                icc_global = np.nan
        else:
            icc_global = np.nan

    except Exception as e_icc:
        icc_global = np.nan
        if log_visual:
            st.warning(f"⚠️ Não foi possível calcular o ICC global: {e_icc}")

    return icc_global

