import cv2
import numpy as np

def inspecionar_rebarbas_finais(imagem_path):
    # 1. Carregar a imagem
    img_original = cv2.imread(imagem_path)
    if img_original is None: print("Erro ao abrir imagem."); return
    
    # Redimensionar para facilitar visualização
    h_orig, w_orig = img_original.shape[:2]
    if h_orig > 800:
        fator = 800 / h_orig
        img = cv2.resize(img_original, (0, 0), fx=fator, fy=fator)
    else:
        img = img_original.copy()
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resultado = img.copy()

    # ====================================================================
    # PASSO A: Encontrar ONDE estão os furos (Método de Contraste)
    # ====================================================================
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Binariza para achar os buracos escuros
    _, thresh_furos = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contornos_furos, _ = cv2.findContours(thresh_furos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    furos_candidatos = []
    for cnt in contornos_furos:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if perimetro == 0: continue
        circularidade = 4 * np.pi * (area / (perimetro**2))
        
        # Filtra para pegar apenas o que parece um furo (ajuste se necessário)
        if area > 50 and circularidade > 0.5:
            furos_candidatos.append(cnt)
            
    print(f"Total de furos detectados para análise: {len(furos_candidatos)}")

    # ====================================================================
    # PASSO B: Encontrar o "VERMELHO FORTE" (Método de Variância/Calor)
    # ====================================================================
    ksize = (3, 3)
    gray_float = gray.astype(np.float32)
    blur = cv2.blur(gray_float, ksize)
    blur_sq = cv2.blur(gray_float ** 2, ksize)
    variance = blur_sq - (blur ** 2)
    sigma = np.sqrt(np.maximum(variance, 0))
    sigma_uint8 = np.uint8(cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX))

    # --- O PULO DO GATO ---
    # Aqui definimos o que é "Vermelho Forte".
    # O mapa vai de 0 (azul) a 255 (vermelho intenso).
    # Um valor alto aqui (ex: 170) isola apenas as piores rebarbas.
    LIMIAR_REBARBA_FORTE = 170
    _, mask_rebarba_forte = cv2.threshold(sigma_uint8, LIMIAR_REBARBA_FORTE, 255, cv2.THRESH_BINARY)

    # Opcional: Limpar pequenos ruídos na máscara de rebarba
    kernel_limpeza = np.ones((3,3), np.uint8)
    mask_rebarba_forte = cv2.morphologyEx(mask_rebarba_forte, cv2.MORPH_OPEN, kernel_limpeza)

    # ====================================================================
    # PASSO C: Cruzar os dados (Quem tem rebarba?)
    # ====================================================================
    furos_com_rebarba = 0
    
    for cnt in furos_candidatos:
        # 1. Achar o centro e o raio aproximado do furo
        (cx, cy), raio = cv2.minEnclosingCircle(cnt)
        center = (int(cx), int(cy))
        radius = int(raio)

        # 2. Criar uma "zona de inspeção" ao redor da borda do furo.
        # A rebarba fica na borda, então olhamos um pouco além do raio do furo.
        margem_inspecao = int(radius * 1.3) # Olha 30% além do tamanho do furo
        
        # Criar uma máscara temporária só para este furo
        mask_furo_atual = np.zeros_like(gray)
        cv2.circle(mask_furo_atual, center, margem_inspecao, 255, -1)
        
        # 3. Verificar se dentro desta zona existe "vermelho forte"
        # Usamos 'bitwise_and' para ver a intersecção entre a zona do furo e a máscara de rebarba
        interseccao = cv2.bitwise_and(mask_rebarba_forte, mask_furo_atual)
        
        # Conta quantos pixels de rebarba forte existem nessa área
        pixels_rebarba = cv2.countNonZero(interseccao)

        # Se houver um número mínimo de pixels de rebarba, marcamos o furo.
        if pixels_rebarba > 10: # Tolerância mínima de pixels
            # TEM REBARBA! Desenha círculo VERMELHO grosso
            cv2.drawContours(img_resultado, [cnt], -1, (0, 0, 255), 3)
            cv2.putText(img_resultado, "REBARBA", (center[0]-20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            furos_com_rebarba += 1
        else:
            # OK! Desenha círculo VERDE fino
            cv2.drawContours(img_resultado, [cnt], -1, (0, 255, 0), 1)

    # ====================================================================
    # Resultados
    # ====================================================================
    print(f"Furos com Rebarba Forte detectados: {furos_com_rebarba}")

    # Mostra o mapa de calor das rebarbas fortes para você entender o que o PC viu
    heatmap_debug = cv2.applyColorMap(mask_rebarba_forte, cv2.COLORMAP_JET)
    
    cv2.imshow("1. O que o PC considera 'Vermelho Forte'", heatmap_debug)
    cv2.imshow("2. Resultado Final da Inspecao", img_resultado)
    
    print("Pressione qualquer tecla na imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- EXECUÇÃO ---
# Use a imagem original em tons de cinza para melhor resultado
# Se usar a imagem já colorida do mapa de calor, não vai funcionar direito.
inspecionar_rebarbas_finais('/Users/andreonmagagna/Downloads/midea/Imagens/WhatsApp Image 2025-11-04 at 14.58.31 (1).jpeg') # Substitua pelo nome da sua imagem P&B