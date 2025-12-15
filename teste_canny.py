import cv2
import numpy as np
import os

# Carregar uma imagem de teste
# Altere o caminho para sua imagem
# pasta_imagens = "imagens_iphone"
# pasta_imagens = "Imagens"
pasta_imagens = "novas fotos"

# Listar imagens disponíveis
print("Imagens disponíveis:")
for f in os.listdir(pasta_imagens):
    print(f"  - {f}")

# Escolha uma imagem
nome_imagem = "40cm.png"  # Altere conforme necessário
caminho = os.path.join(pasta_imagens, nome_imagem)

print(f"\nCarregando: {caminho}")
img = cv2.imread(caminho)

if img is None:
    print("ERRO: Não foi possível carregar a imagem!")
    exit()

print(f"Imagem carregada: {img.shape}")

# Redimensionar se muito grande
max_height = 800
h, w = img.shape[:2]
if h > max_height:
    fator = max_height / h
    img = cv2.resize(img, (0, 0), fx=fator, fy=fator)
    print(f"Redimensionada para: {img.shape}")

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ========== PRÉ-PROCESSAMENTO ==========
print(f"\nEstatísticas da imagem cinza:")
print(f"  Min: {gray.min()}, Max: {gray.max()}, Média: {gray.mean():.1f}")

# Se a imagem for muito escura, inverter pode ajudar
if gray.mean() < 50:
    print("  -> Imagem muito escura, tentando inverter...")
    gray_inv = 255 - gray
else:
    gray_inv = gray

# CLAHE para melhorar contraste (mais agressivo para imagens escuras)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray_inv)

# Normalizar para usar toda a faixa 0-255
gray_norm = cv2.normalize(gray_eq, None, 0, 255, cv2.NORM_MINMAX)

print(f"\nEstatísticas após normalização:")
print(f"  Min: {gray_norm.min()}, Max: {gray_norm.max()}, Média: {gray_norm.mean():.1f}")

# Blur para reduzir ruído
blurred = cv2.GaussianBlur(gray_norm, (5, 5), 0)

# ========== DETECTOR DE CANNY ==========
# Testar diferentes limiares

# Método 1: Limiares automáticos baseados na mediana
median_val = np.median(blurred)
auto_lower = int(max(10, 0.33 * median_val))
auto_upper = int(min(255, 1.0 * median_val))
print(f"\nMediana da imagem (após normalização): {median_val}")
print(f"Limiares automáticos: lower={auto_lower}, upper={auto_upper}")

# Aplicar Canny com diferentes configurações
canny_auto = cv2.Canny(blurred, auto_lower, auto_upper)
canny_low = cv2.Canny(blurred, 20, 60)      # Limiares baixos (mais sensível)
canny_mid = cv2.Canny(blurred, 50, 150)     # Limiares médios
canny_high = cv2.Canny(blurred, 100, 200)   # Limiares altos (menos sensível)

# ========== PÓS-PROCESSAMENTO ==========
def processar_canny(edges, nome):
    """Processa bordas Canny e encontra contornos circulares"""
    print(f"\n--- {nome} ---")
    
    # Dilatar para fechar gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Fechar contornos
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Preencher buracos
    filled = closed.copy()
    h, w = filled.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    final = closed | filled_inv
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos circulares
    img_resultado = img.copy()
    furos = []
    
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        if perimetro == 0:
            continue
        
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))
        
        # Filtro menos restritivo
        if area > 50 and circularidade > 0.3:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5 and radius < 200:
                furos.append((int(x), int(y), int(radius), area, circularidade))
                cv2.circle(img_resultado, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(img_resultado, f"{int(radius*2)}px", 
                           (int(x)-20, int(y)-int(radius)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    print(f"Contornos totais: {len(contornos)}")
    print(f"Furos detectados: {len(furos)}")
    for i, (x, y, r, a, c) in enumerate(furos[:10]):  # Mostrar até 10
        print(f"  Furo {i+1}: pos=({x},{y}), raio={r}px, diâmetro={r*2}px, área={a:.0f}, circ={c:.2f}")
    
    return img_resultado, edges, final, len(furos)

# Processar cada configuração
img_auto, edges_auto, final_auto, n_auto = processar_canny(canny_auto, "CANNY AUTO")
img_low, edges_low, final_low, n_low = processar_canny(canny_low, "CANNY LOW (20-60)")
img_mid, edges_mid, final_mid, n_mid = processar_canny(canny_mid, "CANNY MID (50-150)")
img_high, edges_high, final_high, n_high = processar_canny(canny_high, "CANNY HIGH (100-200)")

# ========== MÉTODO ALTERNATIVO: HOUGH CIRCLES ==========
print("\n--- HOUGH CIRCLES ---")
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=50,
    param2=25,
    minRadius=10,
    maxRadius=150
)

img_hough = img.copy()
n_hough = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    n_hough = len(circles[0])
    print(f"Círculos detectados: {n_hough}")
    for i, (x, y, r) in enumerate(circles[0][:10]):
        print(f"  Círculo {i+1}: pos=({x},{y}), raio={r}px, diâmetro={r*2}px")
        cv2.circle(img_hough, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img_hough, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(img_hough, f"{r*2}px", (x-20, y-r-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
else:
    print("Nenhum círculo detectado!")

# ========== MÉTODO ALTERNATIVO: BINARIZAÇÃO ==========
print("\n--- BINARIZAÇÃO OTSU ---")
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

contornos_bin, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_bin = img.copy()
n_bin = 0
for cnt in contornos_bin:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    if perimetro == 0:
        continue
    circularidade = 4 * np.pi * (area / (perimetro * perimetro))
    if area > 50 and circularidade > 0.3:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5 and radius < 200:
            n_bin += 1
            cv2.circle(img_bin, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(img_bin, f"{int(radius*2)}px", 
                       (int(x)-20, int(y)-int(radius)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

print(f"Furos detectados: {n_bin}")

# ========== SALVAR RESULTADOS ==========
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f"{output_dir}/01_original.png", img)
cv2.imwrite(f"{output_dir}/02_gray.png", gray)
cv2.imwrite(f"{output_dir}/03_gray_eq.png", gray_eq)
cv2.imwrite(f"{output_dir}/03b_gray_norm.png", gray_norm)
cv2.imwrite(f"{output_dir}/04_blurred.png", blurred)

cv2.imwrite(f"{output_dir}/05_canny_auto_edges.png", edges_auto)
cv2.imwrite(f"{output_dir}/06_canny_auto_final.png", final_auto)
cv2.imwrite(f"{output_dir}/07_canny_auto_result.png", img_auto)

cv2.imwrite(f"{output_dir}/08_canny_low_edges.png", edges_low)
cv2.imwrite(f"{output_dir}/09_canny_low_result.png", img_low)

cv2.imwrite(f"{output_dir}/10_canny_mid_edges.png", edges_mid)
cv2.imwrite(f"{output_dir}/11_canny_mid_result.png", img_mid)

cv2.imwrite(f"{output_dir}/12_canny_high_edges.png", edges_high)
cv2.imwrite(f"{output_dir}/13_canny_high_result.png", img_high)

cv2.imwrite(f"{output_dir}/14_hough_result.png", img_hough)

cv2.imwrite(f"{output_dir}/15_thresh.png", thresh)
cv2.imwrite(f"{output_dir}/16_thresh_clean.png", thresh_clean)
cv2.imwrite(f"{output_dir}/17_bin_result.png", img_bin)

print(f"\n========== RESUMO ==========")
print(f"Canny Auto:  {n_auto} furos")
print(f"Canny Low:   {n_low} furos")
print(f"Canny Mid:   {n_mid} furos")
print(f"Canny High:  {n_high} furos")
print(f"Hough:       {n_hough} furos")
print(f"Binarização: {n_bin} furos")
print(f"\nResultados salvos em: {output_dir}/")
print("Abra as imagens para comparar os métodos!")
