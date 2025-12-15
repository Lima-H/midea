import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="Detecção de Rebarbas")

st.title("🔍 Detecção de Rebarbas - Placas Midea")
st.markdown("Faça upload da imagem da placa para analisar rugosidade e textura.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

def redimensionar_imagem(img, max_height=800):
    h, w = img.shape[:2]
    if h > max_height:
        fator = max_height / h
        img = cv2.resize(img, (0, 0), fx=fator, fy=fator)
    return img

def processar_rebarbas(img):
    # Configuração fixa do script detectar_rebarbas.py
    limiar_sensibilidade = 1

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    ksize = (3, 3)
    gray_float = gray.astype(np.float32)
    
    blur = cv2.blur(gray_float, ksize)
    blur_sq = cv2.blur(gray_float ** 2, ksize)
    variance = blur_sq - (blur ** 2)
    sigma = np.sqrt(np.maximum(variance, 0))
    
    sigma_norm = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX)
    sigma_uint8 = np.uint8(sigma_norm)
    
    _, mask_textura = cv2.threshold(sigma_uint8, limiar_sensibilidade, 255, cv2.THRESH_TOZERO)
    
    # Contar pontos de rebarba (regiões conectadas acima do limiar)
    _, mask_binaria = cv2.threshold(sigma_uint8, limiar_sensibilidade, 255, cv2.THRESH_BINARY)
    contornos_rebarbas, _ = cv2.findContours(mask_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qtd_rebarbas = len(contornos_rebarbas)

    heatmap = cv2.applyColorMap(mask_textura, cv2.COLORMAP_JET)
    # Converter heatmap de BGR (OpenCV) para RGB (Streamlit)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    sobreposicao = cv2.addWeighted(img, 0.6, heatmap_rgb, 0.4, 0)
    
    return sobreposicao, heatmap_rgb, qtd_rebarbas

if uploaded_file is not None:
    # Ler imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Redimensionar (usando 800px de altura como padrão para visualização)
    img = redimensionar_imagem(img, max_height=800)
    
    # Converter BGR para RGB para exibição no Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Imagem Original")
    st.image(img_rgb, use_column_width=True)

    st.header("🔬 Detecção de Rebarbas")
    
    # Processamento sem sliders, usando configurações fixas
    img_rebarbas, heatmap, qtd_rebarbas = processar_rebarbas(img_rgb.copy())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sobreposição com Mapa de Calor")
        st.image(img_rebarbas, use_column_width=True)
    
    with col2:
        st.subheader("Apenas Mapa de Calor")
        st.image(heatmap, use_column_width=True)
    
    # Estatísticas
    st.markdown("---")
    st.markdown(f"### 🟠 Pontos de Rugosidade Detectados: {qtd_rebarbas}")
    
    st.caption("💡 O mapa de calor mostra áreas com alta variação de textura, indicando possíveis rebarbas ou irregularidades.")

else:
    st.info("Por favor, faça o upload de uma imagem para começar.")