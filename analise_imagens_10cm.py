"""
Script para analisar automaticamente as imagens da pasta imagens_10cm
e gerar diagnóstico sobre a detecção de bordas internas.
"""
import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# Parâmetros
DIAMETRO_NOMINAL_MM = 10.0
pasta = "imagens_10cm"

def carregar_imagem(caminho):
    """Carrega imagem HEIC ou formato normal"""
    if caminho.lower().endswith('.heic'):
        pil_img = Image.open(caminho)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(caminho)
    return img

def analisar_furo_individual(gray, cx, cy, r, idx_furo):
    """Analisa um furo individual e retorna diagnóstico detalhado"""
    altura, largura = gray.shape
    
    # Análise de perfil radial - média em múltiplas direções
    num_raios = 72
    perfis = []
    
    for i in range(num_raios):
        angulo = 2 * np.pi * i / num_raios
        cos_a = np.cos(angulo)
        sin_a = np.sin(angulo)
        
        perfil = []
        for dist in range(0, int(r * 1.8)):
            px = int(cx + dist * cos_a)
            py = int(cy + dist * sin_a)
            if 0 <= px < largura and 0 <= py < altura:
                perfil.append(gray[py, px])
            else:
                perfil.append(0)
        perfis.append(perfil)
    
    # Média dos perfis
    max_len = min(len(p) for p in perfis)
    perfil_medio = np.mean([p[:max_len] for p in perfis], axis=0)
    
    # Calcular gradiente do perfil médio
    gradiente = np.gradient(perfil_medio)
    
    # Encontrar transições significativas (picos de gradiente)
    threshold_grad = np.std(gradiente) * 1.5
    
    # Procurar borda interna (transição escuro->claro) na região esperada
    r_min = int(r * 0.5)
    r_max = int(r * 1.3)
    
    # Encontrar máximo gradiente positivo (escuro para claro) na região
    gradiente_regiao = gradiente[r_min:r_max]
    if len(gradiente_regiao) > 0:
        idx_max_grad = np.argmax(gradiente_regiao) + r_min
        valor_grad_max = gradiente[idx_max_grad]
    else:
        idx_max_grad = r
        valor_grad_max = 0
    
    # Valores de intensidade
    valor_centro = np.mean(perfil_medio[:int(r*0.3)])  # Centro do furo
    valor_borda_interna = perfil_medio[idx_max_grad] if idx_max_grad < len(perfil_medio) else 0
    valor_metal = np.mean(perfil_medio[int(r*1.2):int(r*1.5)]) if int(r*1.5) < len(perfil_medio) else 255
    
    return {
        'furo': idx_furo,
        'centro': (cx, cy),
        'raio_hough': r,
        'raio_borda_detectada': idx_max_grad,
        'diferenca_raio': idx_max_grad - r,
        'gradiente_maximo': valor_grad_max,
        'intensidade_centro': valor_centro,
        'intensidade_borda': valor_borda_interna,
        'intensidade_metal': valor_metal,
        'contraste': valor_metal - valor_centro,
        'perfil_medio': perfil_medio[:min(200, len(perfil_medio))]
    }

def processar_imagem_diagnostico(img_path):
    """Processa uma imagem e retorna diagnóstico completo"""
    img = carregar_imagem(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pré-processamento
    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    blurred = cv2.medianBlur(blurred, 7)
    
    # Calcular escala
    altura = img.shape[0]
    escala = altura / 4032.0
    min_radius = max(50, int(140 * escala))
    max_radius = max(100, int(300 * escala))
    min_dist = max(150, int(250 * escala))
    
    # HoughCircles
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
    
    resultados = {
        'arquivo': os.path.basename(img_path),
        'dimensoes': img.shape[:2],
        'escala': escala,
        'num_furos': 0,
        'furos': []
    }
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        resultados['num_furos'] = len(circles[0])
        
        for idx, c in enumerate(circles[0]):
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            diag = analisar_furo_individual(gray, cx, cy, r, idx)
            resultados['furos'].append(diag)
    
    return resultados

# Processar todas as imagens
print("="*70)
print("ANÁLISE AUTOMÁTICA DAS IMAGENS DE 10CM")
print("="*70)

arquivos = sorted([f for f in os.listdir(pasta) if f.upper().endswith('.HEIC')])
todos_resultados = []

for arq in arquivos:
    caminho = os.path.join(pasta, arq)
    print(f"\nProcessando: {arq}")
    resultado = processar_imagem_diagnostico(caminho)
    if resultado:
        todos_resultados.append(resultado)
        print(f"  Dimensões: {resultado['dimensoes']}")
        print(f"  Escala: {resultado['escala']:.3f}")
        print(f"  Furos detectados: {resultado['num_furos']}")

# Análise consolidada
print("\n" + "="*70)
print("ANÁLISE CONSOLIDADA")
print("="*70)

# Coletar estatísticas de todos os furos
todos_raios = []
todas_diferencas = []
todos_gradientes = []
todos_contrastes = []

for res in todos_resultados:
    for furo in res['furos']:
        todos_raios.append(furo['raio_hough'])
        todas_diferencas.append(furo['diferenca_raio'])
        todos_gradientes.append(furo['gradiente_maximo'])
        todos_contrastes.append(furo['contraste'])

if todos_raios:
    print(f"\n📊 ESTATÍSTICAS DOS RAIOS (HoughCircles):")
    print(f"   Média: {np.mean(todos_raios):.1f} px")
    print(f"   Mediana: {np.median(todos_raios):.1f} px")
    print(f"   Desvio Padrão: {np.std(todos_raios):.1f} px")
    print(f"   Min/Max: {np.min(todos_raios):.1f} / {np.max(todos_raios):.1f} px")
    
    print(f"\n📏 DIFERENÇA ENTRE RAIO HOUGH E BORDA REAL:")
    print(f"   Média: {np.mean(todas_diferencas):.1f} px")
    print(f"   Mediana: {np.median(todas_diferencas):.1f} px")
    print(f"   Isso indica se HoughCircles está pegando borda externa ou interna")
    
    print(f"\n🔍 GRADIENTE NA BORDA:")
    print(f"   Média: {np.mean(todos_gradientes):.1f}")
    print(f"   Mínimo: {np.min(todos_gradientes):.1f}")
    print(f"   Bom gradiente > 10 indica transição clara")
    
    print(f"\n🎨 CONTRASTE (Metal - Centro do furo):")
    print(f"   Média: {np.mean(todos_contrastes):.1f}")
    print(f"   Min/Max: {np.min(todos_contrastes):.1f} / {np.max(todos_contrastes):.1f}")

# Calcular calibração ideal
if todos_raios:
    raio_mediano = np.median(todos_raios)
    pixels_por_mm = (raio_mediano * 2) / DIAMETRO_NOMINAL_MM
    print(f"\n📐 CALIBRAÇÃO ESTIMADA:")
    print(f"   Raio mediano: {raio_mediano:.1f} px")
    print(f"   Diâmetro mediano: {raio_mediano*2:.1f} px")
    print(f"   Pixels/mm: {pixels_por_mm:.2f}")
    print(f"   1mm = {pixels_por_mm:.2f} pixels")

# Analisar um furo em detalhe para ver o perfil
print("\n" + "="*70)
print("PERFIL DETALHADO DE UM FURO TÍPICO")
print("="*70)

if todos_resultados and todos_resultados[0]['furos']:
    furo_exemplo = todos_resultados[0]['furos'][0]
    perfil = furo_exemplo['perfil_medio']
    
    print(f"\nFuro #{furo_exemplo['furo']} da imagem {todos_resultados[0]['arquivo']}:")
    print(f"  Raio HoughCircles: {furo_exemplo['raio_hough']} px")
    print(f"  Raio da borda real detectada: {furo_exemplo['raio_borda_detectada']} px")
    print(f"  Diferença: {furo_exemplo['diferenca_raio']} px")
    print(f"  Intensidade centro: {furo_exemplo['intensidade_centro']:.0f}")
    print(f"  Intensidade metal: {furo_exemplo['intensidade_metal']:.0f}")
    
    print(f"\n  Perfil de intensidade (distância do centro):")
    for i in range(0, min(len(perfil), 250), 20):
        bar = "█" * int(perfil[i] / 10)
        marcador = " ← RAIO HOUGH" if abs(i - furo_exemplo['raio_hough']) < 10 else ""
        marcador = " ← BORDA DETECTADA" if abs(i - furo_exemplo['raio_borda_detectada']) < 10 else marcador
        print(f"    {i:3d}px: {perfil[i]:6.1f} {bar}{marcador}")

print("\n" + "="*70)
print("RECOMENDAÇÕES PARA MELHORAR O CÓDIGO")
print("="*70)

if todas_diferencas:
    diff_media = np.mean(todas_diferencas)
    if diff_media > 5:
        print("⚠️  HoughCircles está detectando MAIOR que a borda real")
        print("   → Ajustar refinamento para REDUZIR o raio")
    elif diff_media < -5:
        print("⚠️  HoughCircles está detectando MENOR que a borda real")
        print("   → Ajustar refinamento para AUMENTAR o raio")
    else:
        print("✅ HoughCircles está próximo da borda real")
        print("   → Refinamento deve fazer ajuste fino")
