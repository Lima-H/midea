

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Adicionar diretório principal ao sys.path para importar app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import processar_furos, processar_rebarbas

print('Critérios atuais: limiar_sensibilidade=15, percentual mínimo=5%')

# Suporte para HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Pasta de entrada e saída
PASTA_IMAGENS = 'imagens_10cm'  # Altere para a pasta desejada
PASTA_SAIDA = 'test_outputs_rebarba'
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Extensões suportadas
EXTENSOES = ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']

# Listar imagens
imagens = [f for f in os.listdir(PASTA_IMAGENS) if os.path.splitext(f)[1].lower() in EXTENSOES]

for nome_img in imagens:
    caminho_img = os.path.join(PASTA_IMAGENS, nome_img)
    print(f'Processando: {nome_img}')
    
    # Ler imagem

    # Abrir imagem (HEIC com Pillow, outros com OpenCV)
    if nome_img.lower().endswith('.heic'):
        try:
            pil_img = Image.open(caminho_img)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f'Erro ao ler {nome_img} (HEIC): {e}')
            continue
    else:
        img = cv2.imread(caminho_img)
        if img is None:
            print(f'Erro ao ler {nome_img}')
            continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar furos
    img_furos, estatisticas, circles = processar_furos(img_rgb.copy())
    print(f'  Furos detectados: {estatisticas["total"]}')

    # Detectar rebarbas/riscos
    img_rebarbas, heatmap, qtd_rebarbas, qtd_rebarbas_in_circles = processar_rebarbas(img_rgb.copy(), circles, colormap_tipo='TURBO')
    print(f'  Rugosidades dentro dos furos: {qtd_rebarbas_in_circles}')

    # Salvar resultados
    nome_base = os.path.splitext(nome_img)[0]
    cv2.imwrite(os.path.join(PASTA_SAIDA, f'{nome_base}_furos.png'), cv2.cvtColor(img_furos, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(PASTA_SAIDA, f'{nome_base}_rebarba.png'), cv2.cvtColor(img_rebarbas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(PASTA_SAIDA, f'{nome_base}_heatmap.png'), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

print('Teste finalizado! Veja os resultados na pasta test_outputs_rebarba/')
