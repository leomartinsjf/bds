import plotly.graph_objects as go
import numpy as np

def gerar_radar_l4_por_grupo(df_metricas_grouped, titulo="Radar L4 (por Grupo)"):
    """
    Gera um Radar L4 com as 4 funções (grupos) L4:
    - df_metricas_grouped deve ter colunas: 'Grupo', 'metrica'
    """

    # --- Verificação básica ---
    required_cols = ['Grupo', 'metrica']
    for col in required_cols:
        if col not in df_metricas_grouped.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada em df_metricas_grouped.")

    # --- Garante que os 4 grupos estejam presentes (mesmo que com NaN) ---
    grupos_l4 = [
        "Trocas materiais",
        "Subjetividades",
        "Estrutura/Instituições",
        "Relações interpessoais"
    ]

    # --- Prepara a lista final de valores ---
    valores = []
    for grupo in grupos_l4:
        if grupo in df_metricas_grouped['Grupo'].values:
            val = df_metricas_grouped[df_metricas_grouped['Grupo'] == grupo]['metrica'].values[0]
            valores.append(val)
        else:
            valores.append(np.nan)  # Se não tem, coloca NaN

    # --- Fecha o loop do radar ---
    valores += [valores[0]]
    grupos_l4 += [grupos_l4[0]]

    # --- Cria o Radar Plot ---
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=grupos_l4,
        fill='toself',
        name='Métrica'
    ))

    # --- Layout do gráfico ---
    fig.update_layout(
        title=titulo,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Ajustável se quiser
            )
        ),
        showlegend=False
    )

    return fig
