"""
Teste de detecção de furos usando Canny do scikit-image
FOCO: Estimar o tamanho dos círculos a partir das bordas detectadas
Os círculos têm o mesmo tamanho das bordas!
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, color, exposure, morphology, measure
from skimage.draw import circle_perimeter
from scipy import ndimage
import cv2
import os

# ========== CONFIGURAÇÃO ==========
pasta_imagens = "novas fotos"
nome_imagem = "60cm.png"

# ========== CARREGAR IMAGEM ==========
caminho = os.path.join(pasta_imagens, nome_imagem)
print(f"Carregando: {caminho}")

image_rgb = io.imread(caminho)
print(f"Dimensões: {image_rgb.shape}")

# Converter para escala de cinza
if len(image_rgb.shape) == 3:
    if image_rgb.shape[2] == 4:
        image_rgb = color.rgba2rgb(image_rgb)
    image = color.rgb2gray(image_rgb)
else:
    image = image_rgb

print(f"Estatísticas: Min={image.min():.3f}, Max={image.max():.3f}, Média={image.mean():.3f}")

# ========== PRÉ-PROCESSAMENTO ==========
print("\n--- PRÉ-PROCESSAMENTO ---")

# Equalização adaptativa para melhorar contraste
image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
print(f"Após equalização: Min={image_eq.min():.3f}, Max={image_eq.max():.3f}, Média={image_eq.mean():.3f}")

# ========== APLICAR CANNY COM PARÂMETROS OTIMIZADOS ==========
print("\n--- APLICANDO CANNY ---")

# Parâmetros otimizados: σ=4, L=0.05, H=0.15
sigma = 4
low_threshold = 0.05
high_threshold = 0.15

edges = feature.canny(image_eq, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
edge_count = np.sum(edges)

print(f"  Parâmetros: σ={sigma}, low={low_threshold}, high={high_threshold}")
print(f"  Pixels de borda: {edge_count}")

# ========== ANALISAR CONTORNOS DAS BORDAS ==========
print("\n--- ANALISANDO CONTORNOS DAS BORDAS ---")

# Converter bordas para uint8
edges_uint8 = (edges * 255).astype(np.uint8)

# Dilatar bordas para fechar gaps pequenos
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges_uint8, kernel, iterations=1)

# Fechar contornos
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# Encontrar contornos
contornos, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"  Contornos encontrados: {len(contornos)}")

furos_detectados = []
for cnt in contornos:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    if perimetro == 0:
        continue
    
    circularidade = 4 * np.pi * area / (perimetro ** 2)
    
    # Filtrar: área mínima e circularidade
    # Ajuste esses valores conforme necessário
    if area > 200 and circularidade > 0.3:
        (x, y), raio = cv2.minEnclosingCircle(cnt)
        
        # Verificar se o raio faz sentido (não muito pequeno, não muito grande)
        if 10 < raio < 200:
            furos_detectados.append({
                'centro': (int(x), int(y)),
                'raio': raio,
                'diametro': raio * 2,
                'area': area,
                'circularidade': circularidade,
                'contorno': cnt
            })

print(f"  Furos circulares detectados: {len(furos_detectados)}")

# ========== REMOVER DUPLICATAS - MANTER APENAS CONTORNO INTERNO ==========
print("\n--- REMOVENDO DUPLICATAS (MANTENDO CONTORNO INTERNO) ---")

# Agrupar furos por proximidade de centro (dentro de 10px)
furos_unicos = []
usados = set()

for i, furo in enumerate(furos_detectados):
    if i in usados:
        continue
    
    cx, cy = furo['centro']
    grupo = [furo]
    usados.add(i)
    
    # Encontrar outros furos próximos ao mesmo centro
    for j, outro in enumerate(furos_detectados):
        if j in usados:
            continue
        ox, oy = outro['centro']
        distancia = np.sqrt((cx - ox)**2 + (cy - oy)**2)
        if distancia < 10:  # Mesmo furo se centro está a menos de 10px
            grupo.append(outro)
            usados.add(j)
    
    # Do grupo, pegar o de MENOR raio (contorno interno)
    furo_interno = min(grupo, key=lambda f: f['raio'])
    furos_unicos.append(furo_interno)

print(f"  Furos únicos (contorno interno): {len(furos_unicos)}")

# Substituir lista
furos_detectados = furos_unicos

# ========== SEPARAR FUROS VÁLIDOS DE FALSOS POSITIVOS ==========
print("\n--- ANALISANDO FALSOS POSITIVOS ---")

# Calcular estatísticas para identificar outliers
raios_todos = [f['raio'] for f in furos_detectados]
raio_mediana = np.median(raios_todos)
raio_std = np.std(raios_todos)

print(f"  Raio mediana: {raio_mediana:.1f} px")
print(f"  Desvio padrão: {raio_std:.1f} px")

# Definir faixa aceitável: mediana ± 2*desvio (ou faixa fixa baseada na mediana)
raio_min_valido = raio_mediana * 0.7  # 70% da mediana
raio_max_valido = raio_mediana * 1.3  # 130% da mediana
print(f"  Faixa válida: {raio_min_valido:.1f} - {raio_max_valido:.1f} px")

furos_validos = []
falsos_positivos = []

for furo in furos_detectados:
    raio = furo['raio']
    circ = furo['circularidade']
    
    # Critérios para ser válido:
    # 1. Raio dentro da faixa esperada
    # 2. Circularidade alta (> 0.7)
    if raio_min_valido <= raio <= raio_max_valido and circ > 0.7:
        furos_validos.append(furo)
    else:
        # Guardar motivo do falso positivo
        motivos = []
        if raio < raio_min_valido:
            motivos.append(f"raio pequeno ({raio:.1f})")
        elif raio > raio_max_valido:
            motivos.append(f"raio grande ({raio:.1f})")
        if circ <= 0.7:
            motivos.append(f"baixa circularidade ({circ:.2f})")
        furo['motivo'] = ", ".join(motivos)
        falsos_positivos.append(furo)

print(f"\n  Furos VÁLIDOS: {len(furos_validos)}")
print(f"  FALSOS POSITIVOS: {len(falsos_positivos)}")

# Mostrar detalhes dos falsos positivos
if falsos_positivos:
    print(f"\n  Detalhes dos falsos positivos:")
    for i, fp in enumerate(falsos_positivos):
        cx, cy = fp['centro']
        print(f"    {i+1}. Centro ({cx}, {cy}), Raio={fp['raio']:.1f}px, Circ={fp['circularidade']:.2f} - {fp['motivo']}")

# ========== ESTATÍSTICAS DOS RAIOS ==========
print("\n" + "=" * 60)
print("ESTATÍSTICAS DOS FUROS VÁLIDOS")
print("=" * 60)

if furos_validos:
    raios = [f['raio'] for f in furos_validos]
    diametros = [f['diametro'] for f in furos_validos]
    
    print(f"\nFuros válidos: {len(furos_validos)}")
    print(f"  Raio médio: {np.mean(raios):.1f} px")
    print(f"  Raio mediana: {np.median(raios):.1f} px")
    print(f"  Raio min: {np.min(raios):.1f} px")
    print(f"  Raio max: {np.max(raios):.1f} px")
    print(f"  Desvio padrão: {np.std(raios):.1f} px")
    print(f"\n  Diâmetro médio: {np.mean(diametros):.1f} px")
    print(f"  Diâmetro mediana: {np.median(diametros):.1f} px")

# ========== DESENHAR RESULTADOS ==========
print("\n--- GERANDO VISUALIZAÇÃO ---")

# Criar imagem para desenhar (OpenCV BGR)
img_resultado = cv2.cvtColor((image_eq * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Desenhar FALSOS POSITIVOS em VERMELHO
for i, furo in enumerate(falsos_positivos):
    cx, cy = furo['centro']
    raio = int(furo['raio'])
    
    # Círculo vermelho
    cv2.circle(img_resultado, (cx, cy), raio, (0, 0, 255), 2)
    cv2.circle(img_resultado, (cx, cy), 3, (0, 0, 255), -1)
    # Label
    cv2.putText(img_resultado, f"FP", (cx - 10, cy - raio - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Desenhar FUROS VÁLIDOS em VERDE
for i, furo in enumerate(furos_validos):
    cx, cy = furo['centro']
    raio = int(furo['raio'])
    diametro = furo['diametro']
    
    # Círculo verde
    cv2.circle(img_resultado, (cx, cy), raio, (0, 255, 0), 2)
    cv2.circle(img_resultado, (cx, cy), 3, (0, 255, 0), -1)
    # Número e diâmetro
    label = f"#{i+1} D={diametro:.0f}px"
    cv2.putText(img_resultado, label, (cx - 50, cy - raio - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# ========== VISUALIZAÇÃO COMPLETA ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Imagem original
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Imagem Original', fontsize=14)
axes[0, 0].axis('off')

# Imagem equalizada
axes[0, 1].imshow(image_eq, cmap='gray')
axes[0, 1].set_title('Equalização Adaptativa', fontsize=14)
axes[0, 1].axis('off')

# Bordas Canny
axes[0, 2].imshow(edges, cmap='gray')
axes[0, 2].set_title(f'Canny σ={sigma}, L={low_threshold}, H={high_threshold}', fontsize=14)
axes[0, 2].axis('off')

# Bordas dilatadas
axes[1, 0].imshow(edges_dilated, cmap='gray')
axes[1, 0].set_title('Bordas Dilatadas', fontsize=14)
axes[1, 0].axis('off')

# Resultado com círculos
axes[1, 1].imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f'Válidos: {len(furos_validos)} (verde) | FP: {len(falsos_positivos)} (vermelho)', fontsize=14)
axes[1, 1].axis('off')

# Gráfico de dispersão dos diâmetros ordenados (11mm no centro)
if furos_validos:
    # Calcular calibração
    diametro_mediana_px = np.median([f['diametro'] for f in furos_validos])
    px_por_mm = diametro_mediana_px / 11.0
    
    # Converter para mm
    diametros_mm = [f['diametro'] / px_por_mm for f in furos_validos]
    
    # Ordenar os diâmetros
    diametros_ordenados = sorted(diametros_mm)
    
    # Estatísticas
    media = np.mean(diametros_mm)
    desvio = np.std(diametros_mm)
    n = len(diametros_mm)
    
    from scipy import stats
    
    # Posição X (índice ordenado)
    x_pos = np.arange(n)
    
    # Cores: verde se dentro da tolerância (±1mm), vermelho se fora
    cores = ['green' if 10.0 <= d <= 12.0 else 'red' for d in diametros_ordenados]
    
    # Plotar pontos de dispersão
    axes[1, 2].scatter(x_pos, diametros_ordenados, c=cores, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Linha conectando os pontos (mostra a distribuição)
    axes[1, 2].plot(x_pos, diametros_ordenados, 'b-', linewidth=1, alpha=0.5)
    
    # Linha central (11mm)
    axes[1, 2].axhline(11.0, color='blue', linestyle='-', linewidth=2, label='Nominal: 11mm')
    
    # Linhas de tolerância ±1mm
    axes[1, 2].axhline(12.0, color='red', linestyle='--', linewidth=1.5, label='LSE: 12mm')
    axes[1, 2].axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='LIE: 10mm')
    
    # Faixa de tolerância
    axes[1, 2].axhspan(10, 12, alpha=0.15, color='green')
    
    # Linha da média
    axes[1, 2].axhline(media, color='orange', linestyle=':', linewidth=2, label=f'Média: {media:.2f}mm')
    
    # Linhas de ±1σ
    axes[1, 2].axhline(media + desvio, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    axes[1, 2].axhline(media - desvio, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'±1σ ({desvio:.2f}mm)')
    
    axes[1, 2].set_xlabel('Furos (ordenados do menor ao maior)', fontsize=12)
    axes[1, 2].set_ylabel('Diâmetro (mm)', fontsize=12)
    axes[1, 2].set_title(f'Dispersão dos Diâmetros\nn={n} | μ={media:.2f}mm | σ={desvio:.2f}mm', fontsize=13)
    axes[1, 2].legend(fontsize=8, loc='upper left')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(9, 13)

plt.suptitle('Detecção de Furos - Estimativa de Tamanho pelas Bordas', fontsize=16)
plt.tight_layout()
plt.savefig('test_outputs/canny_contornos_resultado.png', dpi=150)
plt.show()

# Salvar imagem com detecção
cv2.imwrite('test_outputs/furos_detectados_opencv.png', img_resultado)

# ========== LISTA DE FUROS VÁLIDOS ==========
if furos_validos:
    print(f"\n--- LISTA DOS {len(furos_validos)} FUROS VÁLIDOS ---")
    print("  #   |  Centro (x,y)  |  Raio (px)  |  Diâmetro (px) | Circularidade")
    print("-" * 75)
    for i, furo in enumerate(furos_validos):
        cx, cy = furo['centro']
        print(f"  {i+1:3d} |  ({cx:4d}, {cy:4d}) |   {furo['raio']:6.1f}   |    {furo['diametro']:6.1f}     |    {furo['circularidade']:.2f}")

# ========== CALIBRAÇÃO E NORMALIZAÇÃO PARA 11mm ==========
print("\n" + "=" * 60)
print("CALIBRAÇÃO E NORMALIZAÇÃO PARA 11mm")
print("=" * 60)

if furos_validos:
    diametro_medio_px = np.mean([f['diametro'] for f in furos_validos])
    diametro_mediana_px = np.median([f['diametro'] for f in furos_validos])
    
    # Usar mediana para calibração (mais robusta a outliers)
    pixels_por_mm = diametro_mediana_px / 11.0
    
    print(f"\nCalibração baseada na mediana:")
    print(f"  Diâmetro mediana: {diametro_mediana_px:.1f} px = 11mm")
    print(f"  → pixels_por_mm = {pixels_por_mm:.2f}")
    
    # Normalizar todos os furos para mm
    print(f"\n--- FUROS NORMALIZADOS PARA mm ---")
    print("  #   |  Centro (x,y)  |  Diâmetro (mm) | Diferença de 11mm")
    print("-" * 65)
    
    diametros_mm = []
    for i, furo in enumerate(furos_validos):
        cx, cy = furo['centro']
        diametro_mm = furo['diametro'] / pixels_por_mm
        diferenca = diametro_mm - 11.0
        diametros_mm.append(diametro_mm)
        
        # Marcar se está fora da tolerância (±1mm)
        if abs(diferenca) <= 1.0:
            status = "OK"
        else:
            status = "FORA"
        
        print(f"  {i+1:3d} |  ({cx:4d}, {cy:4d}) |     {diametro_mm:5.2f}     |   {diferenca:+5.2f}mm  {status}")
    
    # Estatísticas em mm
    print(f"\n" + "=" * 60)
    print("ESTATÍSTICAS NORMALIZADAS")
    print("=" * 60)
    print(f"\n  Total de furos válidos: {len(furos_validos)}")
    print(f"  Diâmetro médio: {np.mean(diametros_mm):.2f} mm")
    print(f"  Diâmetro mediana: {np.median(diametros_mm):.2f} mm")
    print(f"  Diâmetro min: {np.min(diametros_mm):.2f} mm")
    print(f"  Diâmetro max: {np.max(diametros_mm):.2f} mm")
    print(f"  Desvio padrão: {np.std(diametros_mm):.2f} mm")
    
    # Contar dentro/fora da tolerância
    dentro_tolerancia = sum(1 for d in diametros_mm if abs(d - 11.0) <= 1.0)
    fora_tolerancia = len(diametros_mm) - dentro_tolerancia
    
    print(f"\n  Dentro de 11mm ±1mm: {dentro_tolerancia} ({100*dentro_tolerancia/len(diametros_mm):.1f}%)")
    print(f"  Fora da tolerância: {fora_tolerancia} ({100*fora_tolerancia/len(diametros_mm):.1f}%)")
    
    # ========== ANÁLISE DE CAPACIDADE DO PROCESSO (Cp, Cpk) ==========
    print(f"\n" + "=" * 60)
    print("ANÁLISE DE CAPACIDADE DO PROCESSO")
    print("=" * 60)
    
    media = np.mean(diametros_mm)
    sigma = np.std(diametros_mm)
    LSE = 12.0  # Limite Superior de Especificação (11 + 1)
    LIE = 10.0  # Limite Inferior de Especificação (11 - 1)
    
    # Cp = (LSE - LIE) / (6 * sigma)
    Cp = (LSE - LIE) / (6 * sigma)
    
    # Cpk = min((LSE - media) / (3 * sigma), (media - LIE) / (3 * sigma))
    Cpk_superior = (LSE - media) / (3 * sigma)
    Cpk_inferior = (media - LIE) / (3 * sigma)
    Cpk = min(Cpk_superior, Cpk_inferior)
    
    print(f"\n  Especificação: 11mm ±1mm (LIE={LIE}, LSE={LSE})")
    print(f"  Média do processo: {media:.3f} mm")
    print(f"  Desvio padrão (σ): {sigma:.3f} mm")
    print(f"\n  Cp  = {Cp:.2f}  (capacidade potencial)")
    print(f"  Cpk = {Cpk:.2f}  (capacidade real)")
    
    # Interpretação
    print(f"\n  Interpretação:")
    if Cpk >= 1.33:
        print(f"    ✓ Processo CAPAZ (Cpk ≥ 1.33)")
    elif Cpk >= 1.0:
        print(f"    ⚠ Processo MARGINALMENTE CAPAZ (1.0 ≤ Cpk < 1.33)")
    else:
        print(f"    ✗ Processo NÃO CAPAZ (Cpk < 1.0)")
    
    # % fora dos limites (usando distribuição normal)
    from scipy import stats
    prob_fora_superior = 1 - stats.norm.cdf(LSE, media, sigma)
    prob_fora_inferior = stats.norm.cdf(LIE, media, sigma)
    prob_fora_total = (prob_fora_superior + prob_fora_inferior) * 100
    
    print(f"\n  % esperado fora de especificação: {prob_fora_total:.2f}%")
    print(f"  PPM (partes por milhão) fora: {prob_fora_total * 10000:.0f}")

print("\n" + "=" * 60)
print("Resultados salvos em test_outputs/")
