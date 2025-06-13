# modules/variable_grouping.py — Versão atualizada e robusta

def get_variable_groups(df):
    """
    Retorna os grupos L4, contendo apenas as variáveis que existem no df atual.
    """

    # --- Define o L4 teórico original ---
    l4_original = {
        "Trocas materiais": [
            "var_tm_1", "var_tm_2", "var_tm_3"
            # Adicione aqui suas variáveis reais
        ],
        "Subjetividades": [
            "var_subj_1", "var_subj_2", "var_subj_3"
        ],
        "Estrutura/Instituições": [
            "var_estr_1", "var_estr_2", "var_estr_3"
        ],
        "Relações interpessoais": [
            "var_rel_1", "var_rel_2", "var_rel_3"
        ]
    }

    # --- Monta o grupos_l4 final, compatível com o df ---
    grupos_l4 = {}
    for grupo, variaveis in l4_original.items():
        vars_presentes = [var for var in variaveis if var in df.columns]
        grupos_l4[grupo] = vars_presentes

    return grupos_l4
