import cv2
import numpy as np

def analisar_placa(imagem_path):
    # 1. Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: Imagem não encontrada. Verifique o nome do arquivo.")
        return

    # Redimensionar para facilitar a visualização se a imagem for muito grande
    altura, largura = img.shape[:2]
    fator_escala = 1.0
    if altura > 1000: # Se for muito alta, diminui um pouco
        fator_escala = 1000 / altura
        img = cv2.resize(img, (0, 0), fx=fator_escala, fy=fator_escala)

    # 2. Pré-processamento
    # Converte para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplica um desfoque para reduzir ruído e suavizar bordas
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Detecção de Bordas e Binarização
    # Usa o método de Otsu para encontrar o melhor limiar automaticamente
    # THRESH_BINARY_INV: Inverte, pois queremos os furos (escuros) como branco
    val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 4. Encontrar Contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    furos_validos = []
    areas = []

    # Configurações de filtro (ajuste conforme necessário)
    area_minima = 50   # Ignora ruídos muito pequenos
    area_maxima = 2000 # Ignora contornos gigantes (como a borda da placa)
    circularidade_minima = 0.6 # 1.0 é um círculo perfeito. 0.6 aceita ovais (perspectiva)

    print(f"Analisando {len(contornos)} contornos detectados...")

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        if perimetro == 0: continue
        
        # Cálculo de circularidade: 4*pi*area / perimetro^2
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))

        if area_minima < area < area_maxima and circularidade > circularidade_minima:
            furos_validos.append(cnt)
            areas.append(area)

    if not areas:
        print("Nenhum furo válido encontrado. Tente ajustar os filtros de área.")
        return

    # 5. Análise Estatística
    # Calculamos a mediana (menos sensível a outliers que a média)
    mediana_area = np.median(areas)
    tolerancia = 0.30 # 30% de tolerância (alto devido à perspectiva da foto)

    print(f"\nResultados:")
    print(f"Furos detectados: {len(areas)}")
    print(f"Área Mediana (pixels): {mediana_area:.2f}")

    # 6. Desenhar Resultados
    img_resultado = img.copy()
    
    for i, cnt in enumerate(furos_validos):
        area = areas[i]
        raio = int(np.sqrt(area / np.pi))
        
        # Calcula o centro do furo para escrever o texto
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Verifica se o furo foge da tolerância
        diferenca = abs(area - mediana_area) / mediana_area
        
        if diferenca <= tolerancia:
            cor = (0, 255, 0) # Verde (OK)
            status = "OK"
        else:
            cor = (0, 0, 255) # Vermelho (Diferente)
            status = "DIFF"

        # Desenha o contorno do furo
        cv2.drawContours(img_resultado, [cnt], -1, cor, 2)
        # Escreve a área ou status ao lado
        cv2.putText(img_resultado, status, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

    # 7. Mostrar Imagem
    cv2.imshow("Analise de Furos", img_resultado)
    print("\nPressione qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- EXECUÇÃO ---
# Substitua pelo caminho da sua imagem
analisar_placa('/Users/andreonmagagna/Downloads/midea/image_2.png')