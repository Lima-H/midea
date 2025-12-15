import cv2
import numpy as np

def identificar_rebarbas_visuais(imagem_path):
    # 1. Carregar imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return

    # Redimensionar (Mantendo a lógica que você já usa)
    h, w = img.shape[:2]
    if h > 800:
        fator = 800 / h
        img = cv2.resize(img, (0, 0), fx=fator, fy=fator)

    # Converter para cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CALCULAR A VARIÂNCIA LOCAL (Sua lógica original)
    ksize = (3, 3) 
    gray_float = gray.astype(np.float32)

    blur = cv2.blur(gray_float, ksize)
    blur_sq = cv2.blur(gray_float ** 2, ksize)
    variance = blur_sq - (blur ** 2)
    sigma = np.sqrt(np.maximum(variance, 0))
    
    # Normalizar para 0-255
    sigma_norm = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX)
    sigma_uint8 = np.uint8(sigma_norm)

    # --- AQUI COMEÇA A ALTERAÇÃO PARA IDENTIFICAR ---
    
    # 3. Segmentar o "Vermelho Forte"
    # No seu mapa de calor, o vermelho aparece onde a textura é muito alta.
    # Vamos definir que "Vermelho" é qualquer valor de textura acima de 150 (ajustável).
    LIMIAR_REBARBA = 150 
    _, mascara_defeitos = cv2.threshold(sigma_uint8, LIMIAR_REBARBA, 255, cv2.THRESH_BINARY)

    # Limpeza de ruído (remove pontinhos isolados que não são rebarba real)
    kernel = np.ones((3,3), np.uint8)
    mascara_defeitos = cv2.morphologyEx(mascara_defeitos, cv2.MORPH_OPEN, kernel)
    
    # Dilatação (conecta partes próximas da mesma rebarba para formar um bloco só)
    mascara_defeitos = cv2.dilate(mascara_defeitos, kernel, iterations=2)

    # 4. Encontrar onde estão esses defeitos
    contornos, _ = cv2.findContours(mascara_defeitos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_final = img.copy()
    total_rebarbas = 0

    print(f"Analisando contornos de textura...")

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        
        # Filtro de Tamanho: Ignora sujeira muito pequena (ajuste conforme necessário)
        if area > 20: 
            # Pega o retângulo que envolve a rebarba
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            
            # Desenha um retângulo VERMELHO na imagem original
            cv2.rectangle(img_final, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 2)
            
            # Escreve "Defeito"
            cv2.putText(img_final, "REBARBA", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            total_rebarbas += 1

    # 5. Visualização (Mapa de Calor + Resultado Final)
    # Gera o mapa colorido apenas para comparação visual
    heatmap = cv2.applyColorMap(sigma_uint8, cv2.COLORMAP_JET)
    
    print(f"Total de rebarbas identificadas: {total_rebarbas}")

    cv2.imshow("1. Mapa de Calor (O que o PC ve)", heatmap)
    cv2.imshow("2. Identificacao Final", img_final)
    
    print("Pressione qualquer tecla para sair.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- EXECUÇÃO ---
# Substitua pelo caminho da sua imagem
caminho_arquivo = '/Users/andreonmagagna/Downloads/midea/Imagens/WhatsApp Image 2025-11-04 at 14.58.31 (2).jpeg'
identificar_rebarbas_visuais(caminho_arquivo)