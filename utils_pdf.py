# modules/utils_pdf.py — versão FINALIZADA

from fpdf import FPDF
import datetime
import io
import os

def gerar_pdf_radar_l4(df_metricas, fig_radar, modelo, metrica):
    # Cria buffer
    pdf_buffer = io.BytesIO()

    # Cria o PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Caminho para as fontes (você deve ter os 3 arquivos .ttf na pasta modules/fonts/)
    font_path_regular = os.path.join('modules', 'fonts', 'DejaVuSans.ttf')
    font_path_bold = os.path.join('modules', 'fonts', 'DejaVuSans-Bold.ttf')
    font_path_italic = os.path.join('modules', 'fonts', 'DejaVuSans-Oblique.ttf')

    # Adiciona fontes
    pdf.add_font('DejaVu', '', font_path_regular, uni=True)
    pdf.add_font('DejaVu', 'B', font_path_bold, uni=True)
    pdf.add_font('DejaVu', 'I', font_path_italic, uni=True)

    # Página 1 — Capa
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, "Relatório do Radar L4 - Projeto OPPES", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font('DejaVu', '', 12)
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 10, f"Data de geração: {now}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Este relatório apresenta o gráfico de Radar L4, as métricas por grupo e o resumo do modelo ajustado.")
    pdf.ln(20)
    pdf.set_font('DejaVu', 'I', 10)
    pdf.cell(0, 10, "Prof. Marcos / OPPES / UFBA / UFS", ln=True, align="C")

    # Página 2 — Gráfico Radar
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, f"Gráfico Radar L4 - Métrica: {metrica}", ln=True)

    # Salva o radar em imagem temporária
    img_bytes = fig_radar.to_image(format="png")
    img_filename = "temp_radar.png"
    with open(img_filename, "wb") as f:
        f.write(img_bytes)

    # Insere imagem
    pdf.image(img_filename, x=25, y=40, w=160)

    # Página 3 — Tabela de métricas
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Tabela de Métricas", ln=True)
    pdf.ln(5)

    pdf.set_font('DejaVu', '', 10)
    for idx, row in df_metricas.iterrows():
        linha = f"{row['grupo']} - {row['variavel']}: {row['metrica']:.4f}"
        pdf.cell(0, 8, linha, ln=True)

    # Página 4 — Resumo do modelo
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Resumo do Modelo", ln=True)
    pdf.ln(5)

    pdf.set_font('DejaVu', '', 8)
    summary_text = modelo.summary().as_text().split('\n')
    for line in summary_text:
        # Força encoding seguro para latin-1
        line_encoded = line.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 5, line_encoded)

    # Salva no buffer → dest='S' + encode → 100% compatível com Streamlit
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_buffer = io.BytesIO(pdf_bytes)

    return pdf_buffer
