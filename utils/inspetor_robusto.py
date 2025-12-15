import cv2
import numpy as np

def inspecionar_furos_roscados(imagem_path):
    # 1. Carregar imagem
    img_original = cv2.imread(imagem_path)
    if img_original is None: print("Erro ao abrir imagem."); return
    
    # Redimensionar para manter consistência nos parâmetros (altura 800px)
    h_orig, w_orig = img_original.shape[:2]
    fator = 800 / h_orig
    img = cv2.resize(img_original, (0, 0), fx=fator, fy=fator)
    
    # Converter para cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. SUAVIZAÇÃO (Crucial para Hough)
    # MedianBlur é melhor que Gaussian para remover o "ruído" das roscas
    gray_blur = cv2.medianBlur(gray, 5)

    # 3. DETECÇÃO DE CÍRCULOS (Hough Transform)
    # Este método não liga se o buraco é preto ou branco, ele procura curvas.
    rows = gray_blur.shape[0]
    
    # --- PARÂMETROS SENSÍVEIS (Ajuste aqui se não detectar) ---
    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1,           # Resolução (1 = mesma da imagem, 2 = metade)
        minDist=40,     # Distância mínima entre centros de furos (em pixels)
        param1=100,     # Sensibilidade de borda (Canny high threshold)
        param2=30,      # Limiar de contagem (quanto menor, mais falsos círculos detecta)
        minRadius=15,   # Raio mínimo do furo
        maxRadius=45    # Raio máximo do furo
    )

    img_resultado = img.copy()
    furos_detectados = 0
    furos_com_rebarba = 0

    # Se encontrou círculos...
    if circles is not None:
        circles = np.uint16(np.around(circles))
        furos_detectados = len(circles[0, :])
        print(f"Furos geométricos encontrados: {furos_detectados}")

        # 4. CALCULAR MAPA DE CALOR DE REBARBA (Mesma lógica de antes)
        gray_float = gray.astype(np.float32)
        ksize = (3, 3)
        blur_map = cv2.blur(gray_float, ksize)
        blur_sq_map = cv2.blur(gray_float ** 2, ksize)
        variance = blur_sq_map - (blur_map ** 2)
        sigma = np.sqrt(np.maximum(variance, 0))
        sigma_uint8 = np.uint8(cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX))
        
        # Limiar de sensibilidade da rebarba
        _, mask_rebarba = cv2.threshold(sigma_uint8, 160, 255, cv2.THRESH_BINARY)

        for i in circles[0, :]:
            center = (i[0], i[1]) # Centro (x, y)
            radius = i[2]         # Raio r

            # --- ANÁLISE DE DEFEITO ---
            # Vamos olhar apenas um anel fino na borda do círculo detectado
            mask_analise = np.zeros_like(gray)
            
            # Desenha um anel branco na borda do furo (espessura 4)
            cv2.circle(mask_analise, center, radius, 255, 4)
            
            # Verifica se esse anel bate com a máscara de rebarba (rugosidade)
            interseccao = cv2.bitwise_and(mask_rebarba, mask_analise)
            pixels_ruins = cv2.countNonZero(interseccao)

            # Se tiver mais de X pixels de rugosidade na borda, é defeito
            if pixels_ruins > 8:
                # REBARBA (Círculo Vermelho Grosso)
                cv2.circle(img_resultado, center, radius, (0, 0, 255), 3)
                cv2.circle(img_resultado, center, 2, (0, 0, 255), 3) # Ponto no centro
                furos_com_rebarba += 1
            else:
                # OK (Círculo Verde Fino)
                cv2.circle(img_resultado, center, radius, (0, 255, 0), 2)
    else:
        print("Nenhum círculo encontrado. Tente diminuir o 'param2'.")

    # 5. RESULTADOS
    # Texto informativo na tela
    texto = f"Total: {furos_detectados} | Defeitos: {furos_com_rebarba}"
    cv2.putText(img_resultado, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_resultado, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imshow("Inspecao (Metodo Hough)", img_resultado)
    cv2.imshow("Mapa de Rugosidade", mask_rebarba)
    
    print("Pressione qualquer tecla para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- EXECUÇÃO ---
inspecionar_furos_roscados('image_2.png')