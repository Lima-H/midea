import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import feature, exposure
from scipy import stats

# Suporte para imagens HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

st.set_page_config(layout="wide", page_title="Análise de Placas Midea")

st.title("Análise de Qualidade - Placas Midea")
st.markdown("Faça upload da imagem da placa para analisar furos e rebarbas simultaneamente.")

# Especificações da placa
DIAMETRO_NOMINAL_MM = 10.0  # Diâmetro interno do furo
TOLERANCIA_MM = 1.0
DISTANCIA_CAMERA = 10  # cm - iPhone SE 2020

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png", "heic", "HEIC"])

def redimensionar_imagem(img, max_height=None):
    """NÃO redimensiona - mantém resolução original do S23 para calibração correta"""
    # Para 40cm com S23, manter imagem original
    return img

def refinar_borda_interna(gray, cx_inicial, cy_inicial, r_inicial, num_raios=72):
    """
    Refina a detecção da borda interna usando análise de gradiente radial.
    Dispara raios do centro para fora e encontra a transição escuro→claro (borda interna).
    
    CALIBRADO para imagens 10cm: HoughCircles tende a detectar ~11px menor que a borda real.
    
    Args:
        gray: Imagem em escala de cinza
        cx_inicial, cy_inicial: Centro inicial estimado
        r_inicial: Raio inicial estimado
        num_raios: Número de raios a disparar (mais = mais preciso)
    
    Returns:
        (cx_refinado, cy_refinado, raio_refinado) ou None se falhar
    """
    altura, largura = gray.shape
    
    # Converter para float32 para evitar overflow
    gray_float = gray.astype(np.float32)
    
    # Suavizar para reduzir ruído no gradiente
    gray_smooth = cv2.GaussianBlur(gray_float, (5, 5), 1.5)
    
    # Calcular gradiente (Sobel)
    grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Pontos da borda detectados
    pontos_borda = []
    
    # Calcular intensidade média do centro (região escura do furo)
    intensidades_centro = []
    for r in range(5, int(r_inicial * 0.3)):
        for ang in range(0, 360, 45):
            angulo = np.radians(ang)
            px = int(cx_inicial + r * np.cos(angulo))
            py = int(cy_inicial + r * np.sin(angulo))
            if 0 <= px < largura and 0 <= py < altura:
                intensidades_centro.append(gray_smooth[py, px])
    
    intensidade_centro = np.median(intensidades_centro) if intensidades_centro else 70
    
    # Disparar raios do centro para fora
    for i in range(num_raios):
        angulo = 2 * np.pi * i / num_raios
        cos_a = np.cos(angulo)
        sin_a = np.sin(angulo)
        
        # Começar de 55% do raio e ir até 1.3x
        r_min = int(r_inicial * 0.55)
        r_max = int(r_inicial * 1.3)
        
        melhor_r = None
        melhor_score = 0
        
        # Procurar a transição mais forte de escuro para claro
        for r in range(r_min, r_max):
            px = int(cx_inicial + r * cos_a)
            py = int(cy_inicial + r * sin_a)
            
            # Verificar limites
            if not (0 <= px < largura and 0 <= py < altura):
                continue
            
            grad = grad_magnitude[py, px]
            
            # Olhar alguns pixels para dentro e para fora
            px_in = int(cx_inicial + (r - 5) * cos_a)
            py_in = int(cy_inicial + (r - 5) * sin_a)
            px_out = int(cx_inicial + (r + 5) * cos_a)
            py_out = int(cy_inicial + (r + 5) * sin_a)
            
            if not (0 <= px_in < largura and 0 <= py_in < altura and 
                    0 <= px_out < largura and 0 <= py_out < altura):
                continue
            
            val_in = gray_smooth[py_in, px_in]
            val_out = gray_smooth[py_out, px_out]
            
            # Critérios para borda interna:
            # 1. Gradiente significativo
            # 2. Interior mais escuro que exterior (transição escuro→claro)
            # 3. Interior próximo da intensidade do centro do furo
            
            diferenca = float(val_out) - float(val_in)
            proximidade_centro = max(0, 50 - abs(val_in - intensidade_centro))
            
            # Score combina gradiente + diferença de intensidade + proximidade ao padrão do centro
            if diferenca > 12:  # Transição clara escuro→claro
                score = grad * 0.3 + diferenca * 0.5 + proximidade_centro * 0.2
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_r = r
        
        # Se encontrou borda válida, adicionar ponto
        if melhor_r is not None and melhor_score > 18:
            px_borda = cx_inicial + melhor_r * cos_a
            py_borda = cy_inicial + melhor_r * sin_a
            pontos_borda.append((px_borda, py_borda, melhor_r))
    
    # Precisamos de pelo menos 55% dos raios para ter confiança
    if len(pontos_borda) < num_raios * 0.55:
        return None
    
    # Filtrar outliers usando IQR nos raios detectados
    raios_detectados = [p[2] for p in pontos_borda]
    q1, q3 = np.percentile(raios_detectados, [25, 75])
    iqr = q3 - q1
    raio_min_valido = q1 - 1.2 * iqr
    raio_max_valido = q3 + 1.2 * iqr
    
    pontos_filtrados = [(p[0], p[1]) for p in pontos_borda 
                        if raio_min_valido <= p[2] <= raio_max_valido]
    
    if len(pontos_filtrados) < 5:
        return None
    
    # Converter para array numpy para fitEllipse
    pontos_array = np.array(pontos_filtrados, dtype=np.float32).reshape(-1, 1, 2)
    
    if len(pontos_array) >= 5:
        try:
            # Ajustar elipse aos pontos da borda
            ellipse = cv2.fitEllipse(pontos_array)
            (cx_fit, cy_fit), (w, h), angle = ellipse
            
            # Raio médio
            raio_fit = (w + h) / 4
            
            # Validar resultado
            dist_centro = np.sqrt((cx_fit - cx_inicial)**2 + (cy_fit - cy_inicial)**2)
            diff_raio = (raio_fit - r_inicial) / r_inicial
            
            # Aceitar se centro próximo e raio entre -20% e +15% do original
            # HoughCircles frequentemente superestima o raio, então permitir maior redução
            if dist_centro < r_inicial * 0.18 and -0.20 <= diff_raio <= 0.15:
                return (cx_fit, cy_fit, raio_fit)
        except:
            pass
    
    return None

def processar_furos(img):
    """
    Processa imagem para detectar furos usando HoughCircles + refinamento por fitEllipse.
    O HoughCircles detecta os círculos iniciais, depois refina o centro usando
    threshold local na ROI + fitEllipse para centralização precisa na borda interna.
    Retorna: img_resultado, estatisticas (dict), circles
    """
    # Conversão para grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # ========== Método: HoughCircles + Refinamento com fitEllipse ==========
    # Pré-processamento para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    blurred = cv2.medianBlur(blurred, 7)
    
    # Calcular raio esperado baseado no tamanho da imagem
    altura = img.shape[0]
    escala = altura / 4032.0
    min_radius = max(50, int(140 * escala))
    max_radius = max(100, int(300 * escala))
    min_dist = max(150, int(250 * escala))
    
    # HoughCircles para detecção inicial
    circles_hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Refinar centros usando ROI local + threshold + fitEllipse
    furos_detectados = []
    if circles_hough is not None:
        circles_hough = np.uint16(np.around(circles_hough))
        for c in circles_hough[0]:
            cx_hough, cy_hough, r_hough = int(c[0]), int(c[1]), int(c[2])
            
            # Extrair ROI em torno do círculo detectado
            margin = int(100 * escala) if escala > 0.5 else 50
            x1 = max(0, cx_hough - r_hough - margin)
            y1 = max(0, cy_hough - r_hough - margin)
            x2 = min(gray.shape[1], cx_hough + r_hough + margin)
            y2 = min(gray.shape[0], cy_hough + r_hough + margin)
            
            roi_gray = gray[y1:y2, x1:x2]
            
            # Usar Canny para detectar bordas com mais precisão
            roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 1)
            edges = cv2.Canny(roi_blur, 30, 100)
            
            # Dilatar bordas para conectar
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Também usar threshold para região escura como backup
            _, roi_thresh = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            # Combinar: usar região escura que tenha borda definida
            roi_combined = cv2.bitwise_and(roi_thresh, edges_dilated)
            
            # Se combinação muito pequena, usar só threshold
            if cv2.countNonZero(roi_combined) < 10000:
                roi_combined = roi_thresh
            
            # Limpar ruído
            roi_combined = cv2.morphologyEx(roi_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            roi_combined = cv2.morphologyEx(roi_combined, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Encontrar contornos na ROI
            contours, _ = cv2.findContours(roi_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Centro esperado na ROI local
            cx_local = cx_hough - x1
            cy_local = cy_hough - y1
            
            # Encontrar o contorno mais próximo do centro HoughCircles
            best_cnt = None
            best_dist = float('inf')
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                min_area = 40000 * (escala ** 2) if escala > 0.5 else 5000
                if area < min_area:
                    continue
                
                # Centro do contorno
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cnt_cx = int(M['m10'] / M['m00'])
                    cnt_cy = int(M['m01'] / M['m00'])
                    
                    # Distância ao centro esperado
                    dist = np.sqrt((cnt_cx - cx_local)**2 + (cnt_cy - cy_local)**2)
                    
                    # Deve estar próximo do centro HoughCircles
                    if dist < best_dist and dist < r_hough * 0.5:
                        best_dist = dist
                        best_cnt = cnt
            
            if best_cnt is not None:
                # Ajustar coordenadas de volta para imagem original
                best_cnt_global = best_cnt + np.array([x1, y1])
                
                # fitEllipse para melhor ajuste à borda interna
                if len(best_cnt) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(best_cnt_global)
                        (cx_fit, cy_fit), (w, h), angle = ellipse
                        raio_fit = (w + h) / 4  # média dos semi-eixos
                        
                        # ========== REFINAMENTO ADICIONAL: Gradiente Radial ==========
                        # Tentar refinar ainda mais usando análise de gradiente
                        resultado_refinado = refinar_borda_interna(
                            gray, 
                            int(cx_fit), int(cy_fit), 
                            int(raio_fit),
                            num_raios=72
                        )
                        
                        if resultado_refinado is not None:
                            cx_final, cy_final, raio_final = resultado_refinado
                            furos_detectados.append({
                                'centro': (int(cx_final), int(cy_final)),
                                'raio': raio_final
                            })
                        else:
                            # Usar resultado do fitEllipse
                            furos_detectados.append({
                                'centro': (int(cx_fit), int(cy_fit)),
                                'raio': raio_fit
                            })
                    except:
                        # Fallback para minEnclosingCircle
                        (cx_real, cy_real), raio_real = cv2.minEnclosingCircle(best_cnt_global)
                        furos_detectados.append({
                            'centro': (int(cx_real), int(cy_real)),
                            'raio': raio_real
                        })
                else:
                    # Fallback para minEnclosingCircle
                    (cx_real, cy_real), raio_real = cv2.minEnclosingCircle(best_cnt_global)
                    furos_detectados.append({
                        'centro': (int(cx_real), int(cy_real)),
                        'raio': raio_real
                    })
            else:
                # Tentar refinamento direto com gradiente radial usando HoughCircles como ponto inicial
                resultado_refinado = refinar_borda_interna(
                    gray, 
                    cx_hough, cy_hough, 
                    r_hough,
                    num_raios=72
                )
                
                if resultado_refinado is not None:
                    cx_final, cy_final, raio_final = resultado_refinado
                    furos_detectados.append({
                        'centro': (int(cx_final), int(cy_final)),
                        'raio': raio_final
                    })
                else:
                    # Usar HoughCircles original como fallback
                    furos_detectados.append({
                    'centro': (cx_hough, cy_hough),
                    'raio': r_hough
                })
    
    # Calcular calibração usando mediana dos raios
    # TODOS os furos são 10mm por definição
    if len(furos_detectados) > 0:
        raios = [f['raio'] for f in furos_detectados]
        raio_mediano = np.median(raios)
        diametro_mediano_px = raio_mediano * 2
        pixels_por_mm = diametro_mediano_px / DIAMETRO_NOMINAL_MM
        
        # Filtrar outliers (±30% da mediana)
        raio_min = raio_mediano * 0.7
        raio_max = raio_mediano * 1.3
        furos_validos = [f for f in furos_detectados if raio_min <= f['raio'] <= raio_max]
    else:
        furos_validos = []
        pixels_por_mm = 45.0  # valor padrão para iPhone SE a 10cm (4032x3024)
    
    # Calcular diâmetros em mm e criar resultado
    img_resultado = img.copy()
    diametros_mm = []
    furos_ok = 0
    furos_fora = 0
    
    for furo in furos_validos:
        diametro_mm = (furo['raio'] * 2) / pixels_por_mm
        diametros_mm.append(diametro_mm)
        
        cx, cy = furo['centro']
        raio_px = int(furo['raio'])
        
        # Verificar tolerância
        diff = abs(diametro_mm - DIAMETRO_NOMINAL_MM)
        if diff <= TOLERANCIA_MM:
            cor = (0, 255, 0)  # Verde
            furos_ok += 1
        else:
            cor = (255, 0, 0)  # Vermelho
            furos_fora += 1
        
        # Desenhar círculo (espessura proporcional ao tamanho da imagem)
        espessura = max(2, int(raio_px / 30))
        cv2.circle(img_resultado, (cx, cy), raio_px, cor, espessura)
        cv2.circle(img_resultado, (cx, cy), max(3, int(raio_px / 20)), (255, 0, 0), -1)  # Centro
        
        # Mostrar diâmetro em mm ao lado do furo
        texto = f"{diametro_mm:.1f}"
        # Tamanho do texto proporcional ao raio
        font_scale = max(0.5, raio_px / 60)
        font_thickness = max(1, int(raio_px / 40))
        pos_texto = (cx - int(raio_px/3), cy - raio_px - int(raio_px/5))
        cv2.putText(img_resultado, texto, pos_texto, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
    
    # Preparar estatísticas
    estatisticas = {
        'total': len(furos_validos),
        'ok': furos_ok,
        'fora': furos_fora,
        'diametros_mm': diametros_mm,
        'pixels_por_mm': pixels_por_mm
    }
    
    if len(diametros_mm) > 0:
        estatisticas['media_mm'] = np.mean(diametros_mm)
        estatisticas['mediana_mm'] = np.median(diametros_mm)
        estatisticas['std_mm'] = np.std(diametros_mm)
        estatisticas['min_mm'] = np.min(diametros_mm)
        estatisticas['max_mm'] = np.max(diametros_mm)
        estatisticas['dentro_tolerancia_pct'] = (furos_ok / len(furos_validos)) * 100
    
    # Círculos para mascarar em rebarbas
    circles = [(f['centro'][0], f['centro'][1], int(f['raio'])) for f in furos_validos]
    
    return img_resultado, estatisticas, circles

def processar_rebarbas(img, circles=None, colormap_tipo='TURBO'):
    """
    Analisa rebarbas APENAS no centro dos furos, ignorando as bordas.
    Região de análise: do centro até 50% do raio (metade interna do furo)
    """
    # Configuração de sensibilidade
    limiar_sensibilidade = 90  # Mais restritivo para evitar falsos positivos

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    ksize = (5, 5)  # Kernel maior para suavizar pequenas variações
    gray_float = gray.astype(np.float32)
    
    blur = cv2.blur(gray_float, ksize)
    blur_sq = cv2.blur(gray_float ** 2, ksize)
    variance = blur_sq - (blur ** 2)
    sigma = np.sqrt(np.maximum(variance, 0))
    
    sigma_norm = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX)
    sigma_uint8 = np.uint8(sigma_norm)
    
    _, mask_textura = cv2.threshold(sigma_uint8, limiar_sensibilidade, 255, cv2.THRESH_TOZERO)
    
    # Máscara binária de rugosidade
    _, mask_binaria = cv2.threshold(sigma_uint8, limiar_sensibilidade, 255, cv2.THRESH_BINARY)

    # --- Adicional: combinar com detecção de bordas (Canny) para riscos finos ---
    edges = cv2.Canny(np.uint8(gray_float), 40, 120, apertureSize=3, L2gradient=True)
    # Somar bordas ao mapa de rugosidade
    mask_binaria = cv2.bitwise_or(mask_binaria, edges)

    # Selecionar colormap
    colormaps = {
        'JET': cv2.COLORMAP_JET,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'TURBO': cv2.COLORMAP_TURBO,
        'HOT': cv2.COLORMAP_HOT,
        'BONE': cv2.COLORMAP_BONE,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
    }
    
    colormap_id = colormaps.get(colormap_tipo, cv2.COLORMAP_TURBO)
    heatmap = cv2.applyColorMap(mask_textura, colormap_id)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    sobreposicao = cv2.addWeighted(img, 0.6, heatmap_rgb, 0.4, 0)

    # Converter heatmap para HSV para análise de cor
    heatmap_hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

    # Analisar rebarbas APENAS na região central de cada furo
    qtd_rebarbas_in_circles = 0
    circles_with_rebarba = []
    total_pixels_rebarba = 0
    
    if circles is not None:
        for (x, y, r) in circles:
            raio_centro = int(r * 0.77)
            mask_circular = np.zeros(heatmap_hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask_circular, (int(x), int(y)), raio_centro, 255, -1)

            # Vasculhar área circular no heatmap HSV
            h, s, v = cv2.split(heatmap_hsv)
            h_masked = h[mask_circular == 255]
            s_masked = s[mask_circular == 255]
            v_masked = v[mask_circular == 255]

            # Faixas típicas TURBO: amarelo (20-40), verde (40-90), vermelho (0-15 e 160-180)
            amarelo = ((h_masked >= 20) & (h_masked <= 40))
            verde = ((h_masked >= 41) & (h_masked <= 90))
            vermelho1 = ((h_masked >= 0) & (h_masked <= 15))
            vermelho2 = ((h_masked >= 160) & (h_masked <= 180))
            cor_detectada = amarelo | verde | vermelho1 | vermelho2

            # Considerar só pixels com saturação e valor altos (descartar tons escuros)
            cor_detectada = cor_detectada & (s_masked > 80) & (v_masked > 80)

            pct_cor = np.sum(cor_detectada) / len(h_masked) * 100 if len(h_masked) > 0 else 0

            # Threshold: se mais de 2% da área circular tem essas cores, marcar como rebarba
            tem_rebarba = pct_cor > 0.001

            if tem_rebarba:
                qtd_rebarbas_in_circles += 1
                circles_with_rebarba.append((int(x), int(y), int(r), True, pct_cor, pct_cor))
                cv2.circle(sobreposicao, (int(x), int(y)), int(r), (255, 0, 0), 3)
                cv2.circle(sobreposicao, (int(x), int(y)), raio_centro, (255, 100, 100), 1)
            else:
                circles_with_rebarba.append((int(x), int(y), int(r), False, pct_cor, pct_cor))
                cv2.circle(sobreposicao, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(sobreposicao, (int(x), int(y)), raio_centro, (100, 255, 100), 1)
    
    # Contar total de regiões com rugosidade na imagem toda (informativo)
    contornos_rebarbas, _ = cv2.findContours(mask_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qtd_rebarbas_total = len(contornos_rebarbas)
    
    return sobreposicao, heatmap_rgb, qtd_rebarbas_total, qtd_rebarbas_in_circles

if uploaded_file is not None:
    # Ler imagem (suporte a HEIC e formatos normais)
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.heic'):
        # Usar PIL com pillow-heif para HEIC
        pil_img = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # Formato normal (jpg, png, etc)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    
    # Verificar se imagem foi carregada
    if img is None:
        st.error("Erro ao carregar imagem. Tente outro formato.")
    else:
        # Redimensionar (mantém até 1600px para 40cm - mais detalhes)
        img = redimensionar_imagem(img, max_height=1600)
        
        # Converter BGR para RGB para exibição no Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.subheader("Imagem Original")
        st.image(img_rgb, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.header("1. Análise de Furos")
            # Processamento usando Canny edge detection
            img_furos, estatisticas, circles = processar_furos(img_rgb.copy())
            st.image(img_furos, use_container_width=True)
            
            # Estatísticas detalhadas
            st.markdown(f"### 🔵 Furos Detectados: {estatisticas['total']}")
            
            if estatisticas['total'] > 0:
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Diâmetro Médio", f"{estatisticas.get('media_mm', 0):.2f} mm")
                    st.metric("Diâmetro Mediana", f"{estatisticas.get('mediana_mm', 0):.2f} mm")
                    st.metric("Desvio Padrão", f"{estatisticas.get('std_mm', 0):.2f} mm")
                
                with col_stat2:
                    st.metric("Mín / Máx", f"{estatisticas.get('min_mm', 0):.2f} / {estatisticas.get('max_mm', 0):.2f} mm")
                    st.metric(f"Dentro Tolerância (±{TOLERANCIA_MM}mm)", f"{estatisticas.get('dentro_tolerancia_pct', 0):.1f}%")
                    st.metric("Calibração", f"{estatisticas.get('pixels_por_mm', 0):.2f} px/mm")
            
            if estatisticas['fora'] > 0:
                st.markdown(f"### 🔴 Fora do padrão: {estatisticas['fora']}")
            else:
                st.markdown(f"### 🟢 Todos dentro do padrão!")

        with col2:
            st.header("2. Detecção de Rebarbas")
            
            # Processamento com TURBO apenas e verificar dentro dos círculos
            img_rebarbas, heatmap, qtd_rebarbas, qtd_rebarbas_in_circles = processar_rebarbas(img_rgb.copy(), circles, colormap_tipo='TURBO')
            st.image(img_rebarbas, use_container_width=True)
            
            # Descrição maior usando Markdown
            st.markdown(f"### 🔴 Pontos de Rugosidade dentro dos furos: {qtd_rebarbas_in_circles}")
        
        # Seção de visualização detalhada das rebarbas
        st.markdown("---")
        st.header("📊 Análise Detalhada de Rugosidades")
        
        # Duas colunas: tabela primeiro, depois mapa de calor
        col_tabela, col_heat = st.columns([1, 2])
        
        with col_tabela:
            st.subheader("📏 Diâmetros dos Furos")
            if estatisticas['total'] > 0 and 'diametros_mm' in estatisticas:
                diametros = estatisticas['diametros_mm']
                
                # Criar tabela com número do furo e diâmetro
                df_diametros = pd.DataFrame({
                    'Furo': [f"#{i+1}" for i in range(len(diametros))],
                    'Diâmetro (mm)': [f"{d:.2f}" for d in diametros],
                    'Status': ['✅' if abs(d - DIAMETRO_NOMINAL_MM) <= TOLERANCIA_MM else '❌' for d in diametros]
                })
                
                # Altura dinâmica: ~35px por linha + header
                altura_tabela = min(35 * len(diametros) + 40, 600)
                st.dataframe(df_diametros, use_container_width=True, height=altura_tabela)
                
                # Resumo
                st.markdown(f"""
                **Resumo:**
                - Total: **{len(diametros)}** furos
                - ✅ OK: **{estatisticas['ok']}**
                - ❌ Fora: **{estatisticas['fora']}**
                """)
            else:
                st.info("Nenhum furo detectado")
        
        with col_heat:
            st.subheader("Mapa de Calor - Rugosidades")
            st.image(heatmap, use_container_width=True)
        
        # Expander para ver imagem em tamanho real
        with st.expander("🔍 Clique para ver imagem em TAMANHO REAL (zoom)"):
            st.image(img_rebarbas, caption="Imagem com detecção de rugosidades - Tamanho Real")
            st.image(heatmap, caption="Mapa de calor - Tamanho Real")

else:
    st.info("Por favor, faça o upload de uma imagem para começar.")
