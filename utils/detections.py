import cv2
import numpy as np

def _ensure_odd(x: int) -> int:
    x = max(1, int(x))
    return x if x % 2 == 1 else x + 1

def _to_gray(img_rgb):
    if len(img_rgb.shape) == 3:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_rgb

def processar_furos_blackhat(img_rgb, tamanho_kernel=None, area_minima=None, area_maxima=None, tolerancia=0.4, cleaning_kernel_size=5, debug=False):
    """
    Detecta furos usando MORPH_BLACKHAT com alguns ajustes para evitar perda de furos pequenos.
    Retorna: img_resultado_rgb, qtd_detectados, qtd_fora, intermediates_dict
    """
    gray = _to_gray(img_rgb)
    h, w = gray.shape[:2]

    # Defaults relative to image size
    if tamanho_kernel is None:
        # kernel ~ 1/30 da menor dimensão
        k = max(3, min(h, w) // 30)
        tamanho_kernel = (k, k)
    if area_minima is None:
        area_minima = max(10, int(h * w * 0.0002))  # ~0.02% da área da imagem
    if area_maxima is None:
        area_maxima = h * w  # large

    # 1. Black Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( _ensure_odd(tamanho_kernel[0]), _ensure_odd(tamanho_kernel[1]) ))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. Binarização e Limpeza
    # equalize or blur to help OTSU in some images
    blur = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ensure_odd(cleaning_kernel_size), _ensure_odd(cleaning_kernel_size)))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # 3. Contornos
    contornos, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_result = img_bgr.copy()
    furos_validos = []
    areas = []
    # geometric filtering
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area_minima <= area <= area_maxima:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4 * np.pi * (area / (perimetro ** 2)) if perimetro > 0 else 0
            # use either circularity or ellipse aspect
            if area >= 20:  # approximate minimal area
                if len(cnt) >= 5:
                    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                    aspect_ratio = min(MA, ma) / max(MA, ma) if max(MA, ma) != 0 else 0
                else:
                    # fallback to min enclosing circle for small contours
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    aspect_ratio = 1.0  # circle-like by default for small ones

                # Accept shapes that are not extremely elongated
                if aspect_ratio > 0.2 or circularidade > 0.35:
                    furos_validos.append(cnt)
                    areas.append(area)

    furos_fora = 0
    mediana_area = np.median(areas) if areas else 0
    for i, cnt in enumerate(furos_validos):
        area = areas[i]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        diferenca = abs(area - mediana_area) / mediana_area if mediana_area != 0 else 0
        if diferenca <= tolerancia:
            color = (0, 255, 0)
            status = ""
        else:
            color = (0, 0, 255)  # red
            status = "DIFF"
            furos_fora += 1

        # draw as ellipse/circle for readability
        if len(cnt) >= 5:
            elipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img_result, elipse, color, 2)
            if status:
                cv2.putText(img_result, status, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            cv2.drawContours(img_result, [cnt], -1, color, 1)

    # Convert back to RGB for display
    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

    intermediates = {
        'gray': gray,
        'blackhat': blackhat,
        'thresh': thresh,
        'thresh_clean': thresh_clean,
        'contours_mask': cv2.drawContours(np.zeros_like(gray), contornos, -1, 255, 1)
    }

    if debug:
        return img_result_rgb, len(areas), furos_fora, intermediates
    return img_result_rgb, len(areas), furos_fora


def processar_furos_threshold(img_rgb, area_minima=None, area_maxima=None, blur_ks=(5, 5), circularity_threshold=0.5, debug=False):
    gray = _to_gray(img_rgb)
    h, w = gray.shape[:2]
    if area_minima is None:
        area_minima = max(10, int(h * w * 0.00005))
    if area_maxima is None:
        area_maxima = h * w

    blurred = cv2.GaussianBlur(gray, blur_ks, 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Cleaning small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_result = img_bgr.copy()
    furos_validos = []
    areas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area_minima <= area <= area_maxima:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4 * np.pi * (area / (perimetro ** 2))
            if circularidade >= circularity_threshold:
                furos_validos.append(cnt)
                areas.append(area)

    for cnt in furos_validos:
        cv2.drawContours(img_result, [cnt], -1, (0, 255, 0), 1)

    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    intermediates = {
        'gray': gray,
        'thresh': thresh,
        'thresh_clean': thresh_clean,
        'contours_mask': cv2.drawContours(np.zeros_like(gray), contornos, -1, 255, 1)
    }

    if debug:
        return img_result_rgb, len(areas), intermediates
    return img_result_rgb, len(areas)


def detectar_furos_combinado(img_rgb, debug=False, **kwargs):
    """
    Tenta detectar furos com abordagens combinadas: Blackhat e Threshold. Usa a união das máscaras.
    Retorna: img_resultado_rgb, qtd_detectados, qtd_fora, intermediates (optional)
    """
    # Run both detectors and combine masks
    # Get blackhat intermediate mask
    bh_img, bh_count, bh_fora, bh_inter = processar_furos_blackhat(img_rgb, debug=True, **kwargs)
    th_img, th_count, th_inter = processar_furos_threshold(img_rgb, debug=True, **kwargs)

    bh_mask = bh_inter['thresh_clean']
    th_mask = th_inter['thresh_clean']
    combined_mask = cv2.bitwise_or(bh_mask, th_mask)
    # Clean combined
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_result = img_bgr.copy()

    areas = []
    vals = []
    # Basic area-based filter
    h, w = combined_mask.shape[:2]
    area_minima = max(5, int(h * w * 0.00002))
    area_maxima = h * w
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area_minima <= area <= area_maxima:
            # optional geometric tests
            per = cv2.arcLength(cnt, True)
            circularidade = 4 * np.pi * (area / (per * per)) if per > 0 else 0
            if circularidade > 0.25 or area > 20:
                areas.append(area)
                vals.append(cnt)
                cv2.drawContours(img_result, [cnt], -1, (0, 255, 0), 1)

    qtd_detectados = len(areas)
    qtd_fora = 0
    mediana_area = np.median(areas) if areas else 0
    for i, cnt in enumerate(vals):
        area = areas[i]
        M = cv2.moments(cnt)
        if M.get("m00", 0) != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        diferenca = abs(area - mediana_area) / mediana_area if mediana_area != 0 else 0
        if diferenca > kwargs.get('tolerancia', 0.4):
            qtd_fora += 1
            cv2.putText(img_result, "DIFF", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    intermediates = {
        'blackhat': bh_inter,
        'threshold': th_inter,
        'combined_mask': combined_mask
    }

    if debug:
        return img_result_rgb, qtd_detectados, qtd_fora, intermediates
    return img_result_rgb, qtd_detectados, qtd_fora
