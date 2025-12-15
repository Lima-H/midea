"""
Diagnóstico detalhado do refinamento para um furo específico
"""
import cv2
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def carregar_imagem(caminho):
    pil_img = Image.open(caminho)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

# Carregar imagem
img = carregar_imagem("imagens_10cm/IMG_6719.HEIC")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar círculos
blurred = cv2.GaussianBlur(gray, (15, 15), 3)
blurred = cv2.medianBlur(blurred, 7)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=250,
    param1=80,
    param2=50,
    minRadius=140,
    maxRadius=300
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    print("DIAGNÓSTICO DETALHADO DO REFINAMENTO")
    print("="*60)
    
    for idx, c in enumerate(circles[0][:3]):  # Analisar 3 primeiros furos
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        print(f"\n{'='*60}")
        print(f"FURO #{idx+1}: Centro ({cx}, {cy}), Raio HoughCircles = {r}px")
        print("="*60)
        
        # Simular refinamento com debug
        altura, largura = gray.shape
        gray_float = gray.astype(np.float32)
        gray_smooth = cv2.GaussianBlur(gray_float, (5, 5), 1.5)
        
        grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Intensidade do centro
        intensidades_centro = []
        for rad in range(5, int(r * 0.3)):
            for ang in range(0, 360, 45):
                angulo = np.radians(ang)
                px = int(cx + rad * np.cos(angulo))
                py = int(cy + rad * np.sin(angulo))
                if 0 <= px < largura and 0 <= py < altura:
                    intensidades_centro.append(gray_smooth[py, px])
        
        intensidade_centro = np.median(intensidades_centro)
        print(f"Intensidade média do centro: {intensidade_centro:.1f}")
        
        # Testar alguns raios específicos
        num_raios = 72
        pontos_borda = []
        
        for i in range(num_raios):
            angulo = 2 * np.pi * i / num_raios
            cos_a = np.cos(angulo)
            sin_a = np.sin(angulo)
            
            r_min = int(r * 0.55)
            r_max = int(r * 1.3)
            
            melhor_r = None
            melhor_score = 0
            
            for rad in range(r_min, r_max):
                px = int(cx + rad * cos_a)
                py = int(cy + rad * sin_a)
                
                if not (0 <= px < largura and 0 <= py < altura):
                    continue
                
                grad = grad_magnitude[py, px]
                
                px_in = int(cx + (rad - 5) * cos_a)
                py_in = int(cy + (rad - 5) * sin_a)
                px_out = int(cx + (rad + 5) * cos_a)
                py_out = int(cy + (rad + 5) * sin_a)
                
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
                        melhor_r = rad
            
            if melhor_r is not None and melhor_score > 18:
                pontos_borda.append((cx + melhor_r * cos_a, cy + melhor_r * sin_a, melhor_r))
        
        print(f"Pontos de borda detectados: {len(pontos_borda)}/{num_raios} ({100*len(pontos_borda)/num_raios:.1f}%)")
        
        if len(pontos_borda) > 0:
            raios = [p[2] for p in pontos_borda]
            print(f"Raios detectados: min={min(raios):.0f}, max={max(raios):.0f}, média={np.mean(raios):.1f}, mediana={np.median(raios):.1f}")
            
            # Filtrar outliers
            q1, q3 = np.percentile(raios, [25, 75])
            iqr = q3 - q1
            raio_min_valido = q1 - 1.2 * iqr
            raio_max_valido = q3 + 1.2 * iqr
            
            pontos_filtrados = [(p[0], p[1]) for p in pontos_borda 
                                if raio_min_valido <= p[2] <= raio_max_valido]
            
            print(f"Após filtrar outliers: {len(pontos_filtrados)} pontos")
            
            if len(pontos_filtrados) >= 5:
                pontos_array = np.array(pontos_filtrados, dtype=np.float32).reshape(-1, 1, 2)
                try:
                    ellipse = cv2.fitEllipse(pontos_array)
                    (cx_fit, cy_fit), (w, h), angle = ellipse
                    raio_fit = (w + h) / 4
                    
                    dist_centro = np.sqrt((cx_fit - cx)**2 + (cy_fit - cy)**2)
                    diff_raio = (raio_fit - r) / r
                    
                    print(f"\nResultado fitEllipse:")
                    print(f"  Centro: ({cx_fit:.1f}, {cy_fit:.1f}) - dist do original: {dist_centro:.1f}px")
                    print(f"  Raio: {raio_fit:.1f}px (diff: {diff_raio*100:+.1f}%)")
                    print(f"  Critérios: dist < {r*0.18:.0f}px, diff entre -8% e +18%")
                    
                    if dist_centro < r * 0.18 and -0.08 <= diff_raio <= 0.18:
                        print("  ✅ PASSOU nos critérios!")
                    else:
                        print("  ❌ FALHOU nos critérios!")
                        if dist_centro >= r * 0.18:
                            print(f"     → Centro muito longe ({dist_centro:.1f} >= {r*0.18:.1f})")
                        if diff_raio < -0.08:
                            print(f"     → Raio diminuiu muito ({diff_raio*100:.1f}% < -8%)")
                        if diff_raio > 0.18:
                            print(f"     → Raio aumentou muito ({diff_raio*100:.1f}% > 18%)")
                except Exception as e:
                    print(f"Erro no fitEllipse: {e}")
        else:
            print("❌ Poucos pontos de borda detectados")
            
            # Diagnóstico: verificar intensidades em diferentes raios
            print("\nDiagnóstico de intensidades ao longo de um raio:")
            angulo = 0  # horizontal
            cos_a = np.cos(angulo)
            sin_a = np.sin(angulo)
            
            print("  Raio | Int_in | Int_out | Diff | Grad | Score")
            for rad in range(int(r * 0.5), int(r * 1.3), 10):
                px = int(cx + rad * cos_a)
                py = int(cy + rad * sin_a)
                
                if 0 <= px < largura and 0 <= py < altura:
                    grad = grad_magnitude[py, px]
                    
                    px_in = int(cx + (rad - 5) * cos_a)
                    py_in = int(cy + (rad - 5) * sin_a)
                    px_out = int(cx + (rad + 5) * cos_a)
                    py_out = int(cy + (rad + 5) * sin_a)
                    
                    if (0 <= px_in < largura and 0 <= py_in < altura and 
                        0 <= px_out < largura and 0 <= py_out < altura):
                        
                        val_in = gray_smooth[py_in, px_in]
                        val_out = gray_smooth[py_out, px_out]
                        diferenca = float(val_out) - float(val_in)
                        
                        score = 0
                        if diferenca > 12:
                            proximidade = max(0, 50 - abs(val_in - intensidade_centro))
                            score = grad * 0.3 + diferenca * 0.5 + proximidade * 0.2
                        
                        marcador = " ← RAIO HOUGH" if abs(rad - r) < 5 else ""
                        print(f"  {rad:4d} | {val_in:6.1f} | {val_out:7.1f} | {diferenca:+5.1f} | {grad:5.1f} | {score:5.1f}{marcador}")
