"""
Detector de Furos - Placa Midea

ESPECIFICAÇÕES DA PLACA:
- 26 furos por fileira
- 3 fileiras
- Total: 78 furos
- Diâmetro interno: 11mm
- Diâmetro externo: 12mm
- Distância entre furos: 13mm
- Altura da placa: 70cm (700mm)
- Largura da placa: 8,4cm (84mm)
- Profundidade: 1,5cm (15mm)

DISTÂNCIAS DE FOTO:
- 40cm
- 60cm
"""

import cv2
import numpy as np
import os

# ========== ESPECIFICAÇÕES DA PLACA MIDEA ==========
ESPECIFICACOES = {
    "furos_por_fileira": 26,
    "num_fileiras": 3,
    "total_furos": 78,
    "diametro_interno_mm": 11.0,
    "diametro_externo_mm": 12.0,
    "distancia_entre_furos_mm": 13.0,
    "altura_placa_mm": 700.0,
    "largura_placa_mm": 84.0,
    "profundidade_mm": 15.0,
}

# Tolerância para validação (±mm)
TOLERANCIA_MM = 1.0

# ========== CONFIGURAÇÃO ==========
pasta_imagens = "novas fotos"
nome_imagem = "60cm.png"
distancia_camera = "60cm"  # "40cm" ou "60cm"

# Calibração estimada (pixels por mm) - AJUSTAR CONFORME NECESSÁRIO
# Esses valores dependem da resolução da câmera e distância
CALIBRACAO = {
    "40cm": 13.25,   # pixels por mm a 40cm
    "60cm": 14.14    # pixels por mm a 60cm (calibrado!)
}

# ========== CARREGAR IMAGEM ==========
print("=" * 60)
print("DETECTOR DE FUROS - PLACA MIDEA")
print("=" * 60)

print(f"\nEspecificações esperadas:")
print(f"  - Furos: {ESPECIFICACOES['total_furos']} ({ESPECIFICACOES['furos_por_fileira']} x {ESPECIFICACOES['num_fileiras']})")
print(f"  - Diâmetro interno: {ESPECIFICACOES['diametro_interno_mm']}mm")
print(f"  - Diâmetro externo: {ESPECIFICACOES['diametro_externo_mm']}mm")
print(f"  - Tolerância: ±{TOLERANCIA_MM}mm")

print(f"\nImagens disponíveis em '{pasta_imagens}':")
for f in os.listdir(pasta_imagens):
    print(f"  - {f}")

caminho = os.path.join(pasta_imagens, nome_imagem)
print(f"\nCarregando: {caminho}")

img_original = cv2.imread(caminho)
if img_original is None:
    print("ERRO: Não foi possível carregar a imagem!")
    exit()

print(f"Dimensões originais: {img_original.shape}")

# Manter resolução alta para melhor precisão
img = img_original.copy()
h_orig, w_orig = img.shape[:2]

# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"\nEstatísticas da imagem:")
print(f"  Min: {gray.min()}, Max: {gray.max()}, Média: {gray.mean():.1f}")

# ========== PRÉ-PROCESSAMENTO ==========
# Se imagem muito escura, melhorar contraste
if gray.mean() < 50:
    print("  -> Imagem escura detectada, aplicando correção de brilho...")
    # Aumentar brilho
    gray_bright = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
else:
    gray_bright = gray

# Equalização adaptativa
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray_bright)

# Normalizar para usar toda a faixa
gray_norm = cv2.normalize(gray_eq, None, 0, 255, cv2.NORM_MINMAX)

# Blur para reduzir ruído
blurred = cv2.GaussianBlur(gray_norm, (5, 5), 0)

print(f"  Após processamento - Média: {blurred.mean():.1f}")

# ========== CÁLCULO DE TAMANHO ESPERADO EM PIXELS ==========
pixels_por_mm = CALIBRACAO[distancia_camera]
diametro_esperado_px = ESPECIFICACOES['diametro_interno_mm'] * pixels_por_mm
raio_esperado_px = diametro_esperado_px / 2
tolerancia_px = TOLERANCIA_MM * pixels_por_mm

print(f"\nCalibração para {distancia_camera}:")
print(f"  Pixels/mm: {pixels_por_mm}")
print(f"  Diâmetro esperado: {diametro_esperado_px:.1f}px ({ESPECIFICACOES['diametro_interno_mm']}mm)")
print(f"  Raio esperado: {raio_esperado_px:.1f}px")
print(f"  Tolerância: ±{tolerancia_px:.1f}px")

# Limites para filtrar círculos
raio_min = max(5, int(raio_esperado_px * 0.5))
raio_max = int(raio_esperado_px * 2.0)
print(f"  Buscar raios entre: {raio_min}px e {raio_max}px")

# ========== MÉTODO 1: HOUGH CIRCLES ==========
print("\n--- MÉTODO 1: HOUGH CIRCLES ---")

# Calcular distância mínima entre furos - REDUZIDA para pegar mais furos
# Os furos estão a 13mm de distância, na imagem seria ~185px
# Mas precisamos permitir furos mais próximos
distancia_min_px = 50  # Distância mínima fixa menor

# Testar vários parâmetros
print(f"  Dist. mínima entre círculos: {distancia_min_px}px")
for param2 in [10, 15, 20, 25]:
    circles_test = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=distancia_min_px,
        param1=50,
        param2=param2,
        minRadius=10,
        maxRadius=100
    )
    if circles_test is not None:
        raios = circles_test[0][:, 2]
        print(f"  param2={param2}: {len(circles_test[0])} círculos, raio médio: {np.mean(raios):.1f}px")

# Usar param2=15 
circles_hough = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=distancia_min_px,
    param1=50,
    param2=15,
    minRadius=10,
    maxRadius=100
)

img_hough = img.copy()
furos_hough = []

# Definir faixa de raio aceitável (11mm ± 3mm para pré-filtro)
raio_min_filtro = (ESPECIFICACOES['diametro_interno_mm'] - 3) * pixels_por_mm / 2  # 8mm / 2
raio_max_filtro = (ESPECIFICACOES['diametro_interno_mm'] + 3) * pixels_por_mm / 2  # 14mm / 2

if circles_hough is not None:
    circles_hough = np.uint16(np.around(circles_hough))
    print(f"  Círculos brutos detectados: {len(circles_hough[0])}")
    print(f"  Filtrando raios entre {raio_min_filtro:.1f}px e {raio_max_filtro:.1f}px...")
    
    for i, (x, y, r) in enumerate(circles_hough[0]):
        # Pré-filtro por tamanho (apenas furos próximos de 11mm)
        if r < raio_min_filtro or r > raio_max_filtro:
            continue
            
        diametro_px = r * 2
        diametro_mm = diametro_px / pixels_por_mm
        
        # Verificar se está dentro da tolerância estrita (±1mm)
        diferenca = abs(diametro_mm - ESPECIFICACOES['diametro_interno_mm'])
        if diferenca <= TOLERANCIA_MM:
            status = "OK"
            cor = (0, 255, 0)
        else:
            status = "NOK"
            cor = (0, 0, 255)
        
        furos_hough.append((x, y, r, diametro_mm, status))
        
        # Desenhar círculo
        cv2.circle(img_hough, (x, y), r, cor, 2)
        cv2.circle(img_hough, (x, y), 2, cor, 3)  # Centro
        
        # Número do furo e diâmetro
        num_furo = len(furos_hough)
        label = f"#{num_furo} {diametro_mm:.1f}mm"
        cv2.putText(img_hough, label, (x - 40, y - r - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)
        
    print(f"  Círculos após filtro: {len(furos_hough)}")
else:
    print("Nenhum círculo detectado!")

# ========== MÉTODO 2: CANNY + CONTORNOS ==========
print("\n--- MÉTODO 2: CANNY + CONTORNOS ---")

# Aplicar Canny
canny = cv2.Canny(blurred, 30, 100)

# Dilatar bordas
kernel = np.ones((3, 3), np.uint8)
canny_dilated = cv2.dilate(canny, kernel, iterations=1)

# Encontrar contornos
contornos, _ = cv2.findContours(canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_canny = img.copy()
furos_canny = []

for cnt in contornos:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    if perimetro == 0:
        continue
    
    # Circularidade
    circularidade = 4 * np.pi * (area / (perimetro * perimetro))
    
    # Filtrar por circularidade e área
    area_esperada = np.pi * (raio_esperado_px ** 2)
    area_min = area_esperada * 0.3
    area_max = area_esperada * 3.0
    
    if circularidade > 0.5 and area_min < area < area_max:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), radius
        
        diametro_px = radius * 2
        diametro_mm = diametro_px / pixels_por_mm
        
        diferenca = abs(diametro_mm - ESPECIFICACOES['diametro_interno_mm'])
        if diferenca <= TOLERANCIA_MM:
            status = "OK"
            cor = (0, 255, 0)
        else:
            status = "NOK"
            cor = (0, 0, 255)
        
        furos_canny.append((x, y, radius, diametro_mm, status, circularidade))
        cv2.circle(img_canny, (x, y), int(radius), cor, 2)

print(f"Furos detectados: {len(furos_canny)}")

# ========== MÉTODO 3: BINARIZAÇÃO + CONTORNOS ==========
print("\n--- MÉTODO 3: BINARIZAÇÃO + CONTORNOS ---")

# Binarização adaptativa
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)

# Limpar ruído
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Encontrar contornos
contornos_bin, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_bin = img.copy()
furos_bin = []

for cnt in contornos_bin:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    if perimetro == 0:
        continue
    
    circularidade = 4 * np.pi * (area / (perimetro * perimetro))
    
    area_esperada = np.pi * (raio_esperado_px ** 2)
    area_min = area_esperada * 0.3
    area_max = area_esperada * 3.0
    
    if circularidade > 0.4 and area_min < area < area_max:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), radius
        
        diametro_px = radius * 2
        diametro_mm = diametro_px / pixels_por_mm
        
        diferenca = abs(diametro_mm - ESPECIFICACOES['diametro_interno_mm'])
        if diferenca <= TOLERANCIA_MM:
            status = "OK"
            cor = (0, 255, 0)
        else:
            status = "NOK"
            cor = (0, 0, 255)
        
        furos_bin.append((x, y, radius, diametro_mm, status, circularidade))
        cv2.circle(img_bin, (x, y), int(radius), cor, 2)

print(f"Furos detectados: {len(furos_bin)}")

# ========== MÉTODO 4: BLOB DETECTOR ==========
print("\n--- MÉTODO 4: BLOB DETECTOR ---")

# Configurar detector de blobs
params = cv2.SimpleBlobDetector_Params()

# Filtrar por área
params.filterByArea = True
params.minArea = np.pi * (raio_min ** 2)
params.maxArea = np.pi * (raio_max ** 2)

# Filtrar por circularidade
params.filterByCircularity = True
params.minCircularity = 0.5

# Filtrar por convexidade
params.filterByConvexity = True
params.minConvexity = 0.5

# Filtrar por inércia
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Filtrar por cor (furos escuros)
params.filterByColor = True
params.blobColor = 0

detector = cv2.SimpleBlobDetector_create(params)

# Detectar blobs
keypoints = detector.detect(blurred)

img_blob = img.copy()
furos_blob = []

for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    radius = kp.size / 2
    
    diametro_px = radius * 2
    diametro_mm = diametro_px / pixels_por_mm
    
    diferenca = abs(diametro_mm - ESPECIFICACOES['diametro_interno_mm'])
    if diferenca <= TOLERANCIA_MM:
        status = "OK"
        cor = (0, 255, 0)
    else:
        status = "NOK"
        cor = (0, 0, 255)
    
    furos_blob.append((x, y, radius, diametro_mm, status))
    cv2.circle(img_blob, (x, y), int(radius), cor, 2)

print(f"Furos detectados: {len(furos_blob)}")

# ========== SALVAR RESULTADOS ==========
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f"{output_dir}/01_original.png", img)
cv2.imwrite(f"{output_dir}/02_gray.png", gray)
cv2.imwrite(f"{output_dir}/03_gray_eq.png", gray_eq)
cv2.imwrite(f"{output_dir}/04_blurred.png", blurred)
cv2.imwrite(f"{output_dir}/05_canny.png", canny)
cv2.imwrite(f"{output_dir}/06_canny_dilated.png", canny_dilated)
cv2.imwrite(f"{output_dir}/07_thresh.png", thresh)
cv2.imwrite(f"{output_dir}/08_result_hough.png", img_hough)
cv2.imwrite(f"{output_dir}/09_result_canny.png", img_canny)
cv2.imwrite(f"{output_dir}/10_result_bin.png", img_bin)
cv2.imwrite(f"{output_dir}/11_result_blob.png", img_blob)

# ========== RESUMO ==========
print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print(f"\nEsperado: {ESPECIFICACOES['total_furos']} furos de {ESPECIFICACOES['diametro_interno_mm']}mm")
print(f"\nDetectados:")
print(f"  Hough Circles:  {len(furos_hough)} furos")
print(f"  Canny+Contorno: {len(furos_canny)} furos")
print(f"  Binarização:    {len(furos_bin)} furos")
print(f"  Blob Detector:  {len(furos_blob)} furos")

# Estatísticas detalhadas do melhor método
melhor_metodo = max([
    ("Hough", furos_hough),
    ("Canny", furos_canny),
    ("Binarização", furos_bin),
    ("Blob", furos_blob)
], key=lambda x: len(x[1]))

print(f"\nMelhor método: {melhor_metodo[0]} ({len(melhor_metodo[1])} furos)")

if melhor_metodo[1]:
    diametros = [f[3] for f in melhor_metodo[1]]
    ok_count = sum(1 for f in melhor_metodo[1] if f[4] == "OK")
    nok_count = len(melhor_metodo[1]) - ok_count
    
    print(f"\nEstatísticas de diâmetro ({melhor_metodo[0]}):")
    print(f"  Média: {np.mean(diametros):.2f}mm")
    print(f"  Mediana: {np.median(diametros):.2f}mm")
    print(f"  Min: {np.min(diametros):.2f}mm")
    print(f"  Max: {np.max(diametros):.2f}mm")
    print(f"  Desvio padrão: {np.std(diametros):.2f}mm")
    print(f"\n  OK (dentro de {ESPECIFICACOES['diametro_interno_mm']}±{TOLERANCIA_MM}mm): {ok_count}")
    print(f"  NOK (fora da tolerância): {nok_count}")

print(f"\nResultados salvos em: {output_dir}/")
print("\n" + "=" * 60)

# ========== MOSTRAR IMAGEM COM OS FUROS ==========
print("\n📸 Mostrando imagem com furos detectados...")
print("   Pressione qualquer tecla para fechar a janela")

# Redimensionar para visualização se muito grande
h, w = img_hough.shape[:2]
max_display = 1200
if max(h, w) > max_display:
    scale = max_display / max(h, w)
    img_display = cv2.resize(img_hough, None, fx=scale, fy=scale)
else:
    img_display = img_hough

# Adicionar legenda na imagem
legenda_y = 30
cv2.putText(img_display, f"FUROS DETECTADOS: {len(furos_hough)}", (10, legenda_y), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(img_display, f"Verde = OK (11mm +/- 1mm) | Vermelho = Fora da tolerancia", 
           (10, legenda_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Mostrar janela
cv2.imshow("Furos Detectados - Hough Circles", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ========== CALIBRAÇÃO AUTOMÁTICA ==========
print("\n💡 DICA DE CALIBRAÇÃO:")
if melhor_metodo[1]:
    diametros_px = [f[2] * 2 for f in melhor_metodo[1]]  # raio * 2
    media_px = np.mean(diametros_px)
    calibracao_sugerida = media_px / ESPECIFICACOES['diametro_interno_mm']
    print(f"   Diâmetro médio detectado: {media_px:.1f}px")
    print(f"   Se os furos são de {ESPECIFICACOES['diametro_interno_mm']}mm,")
    print(f"   a calibração deveria ser: {calibracao_sugerida:.2f} px/mm")
    print(f"   (Atual: {pixels_por_mm} px/mm)")
