"""
Script para testar o algoritmo melhorado nas imagens de 10cm
"""
import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

DIAMETRO_NOMINAL_MM = 10.0

def carregar_imagem(caminho):
    if caminho.lower().endswith('.heic'):
        pil_img = Image.open(caminho)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(caminho)
    return img

def refinar_borda_interna(gray, cx_inicial, cy_inicial, r_inicial, num_raios=72):
    """Versão melhorada do refinamento"""
    altura, largura = gray.shape
    
    # Converter para float32 para evitar overflow
    gray_float = gray.astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray_float, (5, 5), 1.5)
    
    grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calcular intensidade média do centro
    intensidades_centro = []
    for r in range(5, int(r_inicial * 0.3)):
        for ang in range(0, 360, 45):
            angulo = np.radians(ang)
            px = int(cx_inicial + r * np.cos(angulo))
            py = int(cy_inicial + r * np.sin(angulo))
            if 0 <= px < largura and 0 <= py < altura:
                intensidades_centro.append(gray_smooth[py, px])
    
    intensidade_centro = np.median(intensidades_centro) if intensidades_centro else 70
    
    pontos_borda = []
    
    for i in range(num_raios):
        angulo = 2 * np.pi * i / num_raios
        cos_a = np.cos(angulo)
        sin_a = np.sin(angulo)
        
        r_min = int(r_inicial * 0.55)
        r_max = int(r_inicial * 1.3)
        
        melhor_r = None
        melhor_score = 0
        
        for r in range(r_min, r_max):
            px = int(cx_inicial + r * cos_a)
            py = int(cy_inicial + r * sin_a)
            
            if not (0 <= px < largura and 0 <= py < altura):
                continue
            
            grad = grad_magnitude[py, px]
            
            px_in = int(cx_inicial + (r - 5) * cos_a)
            py_in = int(cy_inicial + (r - 5) * sin_a)
            px_out = int(cx_inicial + (r + 5) * cos_a)
            py_out = int(cy_inicial + (r + 5) * sin_a)
            
            if not (0 <= px_in < largura and 0 <= py_in < altura and 
                    0 <= px_out < largura and 0 <= py_out < altura):
                continue
            
            val_in = gray_smooth[py_in, px_in]
            val_out = gray_smooth[py_out, px_out]
            
            diferenca = float(val_out) - float(val_in)
            proximidade_centro = max(0, 50 - abs(val_in - intensidade_centro))
            
            if diferenca > 12:
                score = grad * 0.3 + diferenca * 0.5 + proximidade_centro * 0.2
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_r = r
        
        if melhor_r is not None and melhor_score > 18:
            px_borda = cx_inicial + melhor_r * cos_a
            py_borda = cy_inicial + melhor_r * sin_a
            pontos_borda.append((px_borda, py_borda, melhor_r))
    
    if len(pontos_borda) < num_raios * 0.55:
        return None
    
    raios_detectados = [p[2] for p in pontos_borda]
    q1, q3 = np.percentile(raios_detectados, [25, 75])
    iqr = q3 - q1
    raio_min_valido = q1 - 1.2 * iqr
    raio_max_valido = q3 + 1.2 * iqr
    
    pontos_filtrados = [(p[0], p[1]) for p in pontos_borda 
                        if raio_min_valido <= p[2] <= raio_max_valido]
    
    if len(pontos_filtrados) < 5:
        return None
    
    pontos_array = np.array(pontos_filtrados, dtype=np.float32).reshape(-1, 1, 2)
    
    if len(pontos_array) >= 5:
        try:
            ellipse = cv2.fitEllipse(pontos_array)
            (cx_fit, cy_fit), (w, h), angle = ellipse
            raio_fit = (w + h) / 4
            
            dist_centro = np.sqrt((cx_fit - cx_inicial)**2 + (cy_fit - cy_inicial)**2)
            diff_raio = (raio_fit - r_inicial) / r_inicial
            
            if dist_centro < r_inicial * 0.18 and -0.20 <= diff_raio <= 0.15:
                return (cx_fit, cy_fit, raio_fit)
        except:
            pass
    
    return None

def processar_imagem_teste(img_path):
    """Processa uma imagem e compara HoughCircles vs Refinado"""
    img = carregar_imagem(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    blurred = cv2.medianBlur(blurred, 7)
    
    altura = img.shape[0]
    escala = altura / 4032.0
    min_radius = max(50, int(140 * escala))
    max_radius = max(100, int(300 * escala))
    min_dist = max(150, int(250 * escala))
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    resultados = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            
            # Tentar refinamento
            resultado_refinado = refinar_borda_interna(gray, cx, cy, r, num_raios=72)
            
            if resultado_refinado:
                cx_ref, cy_ref, r_ref = resultado_refinado
                resultados.append({
                    'hough_raio': r,
                    'refinado_raio': r_ref,
                    'diferenca': r_ref - r,
                    'refinamento_sucesso': True
                })
            else:
                resultados.append({
                    'hough_raio': r,
                    'refinado_raio': r,
                    'diferenca': 0,
                    'refinamento_sucesso': False
                })
    
    return resultados

# Testar nas imagens
pasta = "imagens_10cm"
arquivos = sorted([f for f in os.listdir(pasta) if f.upper().endswith('.HEIC')])

print("="*70)
print("TESTE DO ALGORITMO MELHORADO")
print("="*70)

todos_hough = []
todos_refinados = []
sucessos = 0
total_furos = 0

for arq in arquivos:  # Testar com TODAS imagens
    caminho = os.path.join(pasta, arq)
    print(f"\n{arq}:")
    
    resultados = processar_imagem_teste(caminho)
    if resultados:
        for i, res in enumerate(resultados):
            total_furos += 1
            todos_hough.append(res['hough_raio'])
            todos_refinados.append(res['refinado_raio'])
            
            if res['refinamento_sucesso']:
                sucessos += 1
                status = "✅"
            else:
                status = "⚠️"
            
            print(f"  Furo {i+1}: Hough={res['hough_raio']}px → Refinado={res['refinado_raio']:.1f}px (diff={res['diferenca']:+.1f}px) {status}")

print("\n" + "="*70)
print("RESUMO")
print("="*70)

print(f"\nTotal de furos: {total_furos}")
print(f"Refinamento bem-sucedido: {sucessos} ({100*sucessos/total_furos:.1f}%)")

if todos_hough:
    raio_med_hough = np.median(todos_hough)
    raio_med_ref = np.median(todos_refinados)
    
    diam_hough = raio_med_hough * 2
    diam_ref = raio_med_ref * 2
    
    # Calibração pelo refinado (assumindo 10mm de diâmetro real)
    px_por_mm = diam_ref / DIAMETRO_NOMINAL_MM
    
    print(f"\n📊 RAIOS MEDIANOS:")
    print(f"   HoughCircles: {raio_med_hough:.1f} px (diâmetro: {diam_hough:.1f} px)")
    print(f"   Refinado: {raio_med_ref:.1f} px (diâmetro: {diam_ref:.1f} px)")
    print(f"   Diferença: {raio_med_ref - raio_med_hough:+.1f} px")
    
    print(f"\n📐 CALIBRAÇÃO (usando diâmetro refinado):")
    print(f"   {px_por_mm:.2f} pixels/mm")
    
    # Calcular diâmetros em mm para cada furo
    print(f"\n📏 DIÂMETROS CALCULADOS (mm):")
    diametros_mm = [(r * 2) / px_por_mm for r in todos_refinados]
    print(f"   Média: {np.mean(diametros_mm):.2f} mm")
    print(f"   Mediana: {np.median(diametros_mm):.2f} mm")
    print(f"   Desvio Padrão: {np.std(diametros_mm):.2f} mm")
    print(f"   Min/Max: {np.min(diametros_mm):.2f} / {np.max(diametros_mm):.2f} mm")
