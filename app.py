import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Suporte para imagens HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- Constantes ---
HOLE_NOMINAL_DIAMETER_MM = 10.0
HOLE_TOLERANCE_MM = 1.0
REFERENCE_IMAGE_HEIGHT = 4032


# =============================================================================
# Pipeline de detecção de furos
# =============================================================================

def refine_inner_edge(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    num_rays: int = 72,
    score_threshold: float = 18.0,
    min_ray_ratio: float = 0.55,
    iqr_factor: float = 1.2,
    max_radius_diff: float = 0.25,
) -> tuple[float, float, float] | None:
    """Refina a borda interna do furo usando análise de gradiente radial."""
    height, width = gray.shape
    gray_smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)

    grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Intensidade média do centro do furo
    center_intensities = []
    for r in range(5, int(radius * 0.3)):
        for ang in range(0, 360, 45):
            px = int(cx + r * np.cos(np.radians(ang)))
            py = int(cy + r * np.sin(np.radians(ang)))
            if 0 <= px < width and 0 <= py < height:
                center_intensities.append(gray_smooth[py, px])

    center_intensity = np.median(center_intensities) if center_intensities else 70

    edge_points = []
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        r_min, r_max = int(radius * 0.55), int(radius * 1.3)
        best_r, best_score = None, 0

        for r in range(r_min, r_max):
            px, py = int(cx + r * cos_a), int(cy + r * sin_a)
            if not (0 <= px < width and 0 <= py < height):
                continue
            px_in = int(cx + (r - 5) * cos_a)
            py_in = int(cy + (r - 5) * sin_a)
            px_out = int(cx + (r + 5) * cos_a)
            py_out = int(cy + (r + 5) * sin_a)
            if not (0 <= px_in < width and 0 <= py_in < height
                    and 0 <= px_out < width and 0 <= py_out < height):
                continue

            diff = float(gray_smooth[py_out, px_out]) - float(gray_smooth[py_in, px_in])
            if diff > 12:
                grad = grad_mag[py, px]
                proximity = max(0, 50 - abs(gray_smooth[py_in, px_in] - center_intensity))
                score = grad * 0.3 + diff * 0.5 + proximity * 0.2
                if score > best_score:
                    best_score = score
                    best_r = r

        if best_r is not None and best_score > score_threshold:
            edge_points.append((cx + best_r * cos_a, cy + best_r * sin_a, best_r))

    if len(edge_points) < num_rays * min_ray_ratio:
        return None

    radii = [p[2] for p in edge_points]
    q1, q3 = np.percentile(radii, [25, 75])
    iqr = q3 - q1
    valid_points = [
        (p[0], p[1]) for p in edge_points
        if q1 - iqr_factor * iqr <= p[2] <= q3 + iqr_factor * iqr
    ]

    if len(valid_points) < 5:
        return None

    pts = np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        (cx_fit, cy_fit), (w, h), _ = cv2.fitEllipse(pts)
        radius_fit = (w + h) / 4
        dist = np.sqrt((cx_fit - cx)**2 + (cy_fit - cy)**2)
        radius_diff = (radius_fit - radius) / radius
        if dist < radius * 0.18 and -max_radius_diff <= radius_diff <= 0.15:
            return (cx_fit, cy_fit, radius_fit)
    except cv2.error:
        pass
    return None


def detect_holes(
    img: np.ndarray,
    blur_ksize: int = 15,
    blur_sigma: float = 3.0,
    median_ksize: int = 7,
    canny_low: int = 70,
    canny_high: int = 140,
    hough_dp: float = 1.2,
    hough_param1: int = 80,
    hough_param2: int = 50,
) -> list[dict]:
    """Detecta furos usando HoughCircles + refinamento com fitEllipse e gradiente."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    blurred = cv2.medianBlur(blurred, median_ksize)

    height = img.shape[0]
    scale = height / REFERENCE_IMAGE_HEIGHT
    min_radius = max(50, int(140 * scale))
    max_radius = max(100, int(300 * scale))
    min_dist = max(150, int(250 * scale))

    circles_hough = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=hough_dp,
        minDist=min_dist, param1=hough_param1, param2=hough_param2,
        minRadius=min_radius, maxRadius=max_radius,
    )

    detected = []
    if circles_hough is None:
        return detected

    circles_hough = np.uint16(np.around(circles_hough))
    kernel = np.ones((3, 3), np.uint8)

    for c in circles_hough[0]:
        cx_h, cy_h, r_h = int(c[0]), int(c[1]), int(c[2])

        # ROI ao redor do círculo
        margin = int(100 * scale) if scale > 0.5 else 50
        x1 = max(0, cx_h - r_h - margin)
        y1 = max(0, cy_h - r_h - margin)
        x2 = min(gray.shape[1], cx_h + r_h + margin)
        y2 = min(gray.shape[0], cy_h + r_h + margin)
        roi_gray = gray[y1:y2, x1:x2]

        # Canny + Threshold combinados
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 1)
        edges = cv2.Canny(roi_blur, canny_low, canny_high)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        _, roi_thresh = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY_INV)
        roi_combined = cv2.bitwise_and(roi_thresh, edges_dilated)

        if cv2.countNonZero(roi_combined) < 10000:
            roi_combined = roi_thresh

        roi_combined = cv2.morphologyEx(roi_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        roi_combined = cv2.morphologyEx(roi_combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Melhor contorno próximo ao centro
        contours, _ = cv2.findContours(roi_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx_local, cy_local = cx_h - x1, cy_h - y1
        best_cnt, best_dist = None, float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = 40000 * (scale ** 2) if scale > 0.5 else 5000
            if area < min_area:
                continue
            m = cv2.moments(cnt)
            if m['m00'] > 0:
                cnt_cx = int(m['m10'] / m['m00'])
                cnt_cy = int(m['m01'] / m['m00'])
                dist = np.sqrt((cnt_cx - cx_local)**2 + (cnt_cy - cy_local)**2)
                if dist < best_dist and dist < r_h * 0.5:
                    best_dist = dist
                    best_cnt = cnt

        hole = _refine_hole(gray, best_cnt, cx_h, cy_h, r_h, x1, y1)
        detected.append(hole)

    return detected


def _refine_hole(
    gray: np.ndarray,
    contour: np.ndarray | None,
    cx_hough: int,
    cy_hough: int,
    r_hough: int,
    roi_x: int,
    roi_y: int,
) -> dict:
    """Refina a posição e raio de um furo usando fitEllipse + gradiente radial."""
    if contour is not None:
        contour_global = contour + np.array([roi_x, roi_y])
        if len(contour) >= 5:
            try:
                (cx_fit, cy_fit), (w, h), _ = cv2.fitEllipse(contour_global)
                radius_fit = (w + h) / 4
                result = refine_inner_edge(gray, int(cx_fit), int(cy_fit), int(radius_fit))
                if result:
                    return {'center': (int(result[0]), int(result[1])), 'radius': result[2]}
                return {'center': (int(cx_fit), int(cy_fit)), 'radius': radius_fit}
            except cv2.error:
                (cx_r, cy_r), radius_r = cv2.minEnclosingCircle(contour_global)
                return {'center': (int(cx_r), int(cy_r)), 'radius': radius_r}
        else:
            (cx_r, cy_r), radius_r = cv2.minEnclosingCircle(contour_global)
            return {'center': (int(cx_r), int(cy_r)), 'radius': radius_r}

    # Sem contorno: tentar refinamento direto
    result = refine_inner_edge(gray, cx_hough, cy_hough, r_hough)
    if result:
        return {'center': (int(result[0]), int(result[1])), 'radius': result[2]}
    return {'center': (cx_hough, cy_hough), 'radius': r_hough}


def calibrate_pixels_per_mm(
    holes: list[dict],
    nominal_diameter: float = HOLE_NOMINAL_DIAMETER_MM,
) -> tuple[list[dict], float]:
    """Calibra px/mm pela mediana dos raios e filtra outliers."""
    if not holes:
        return [], 45.0

    radii = [h['radius'] for h in holes]
    median_radius = np.median(radii)
    pixels_per_mm = (median_radius * 2) / nominal_diameter

    radius_min = median_radius * 0.7
    radius_max = median_radius * 1.3
    valid_holes = [h for h in holes if radius_min <= h['radius'] <= radius_max]

    return valid_holes, pixels_per_mm


def compute_hole_stats(
    holes: list[dict],
    pixels_per_mm: float,
    nominal_diameter: float = HOLE_NOMINAL_DIAMETER_MM,
    tolerance: float = HOLE_TOLERANCE_MM,
) -> dict:
    """Calcula estatísticas de diâmetro dos furos."""
    diameters_mm = [(h['radius'] * 2) / pixels_per_mm for h in holes]
    ok_count = sum(1 for d in diameters_mm if abs(d - nominal_diameter) <= tolerance)
    fail_count = len(diameters_mm) - ok_count

    stats = {
        'total': len(holes),
        'ok': ok_count,
        'fail': fail_count,
        'diameters_mm': diameters_mm,
        'pixels_per_mm': pixels_per_mm,
    }

    if diameters_mm:
        stats.update({
            'mean_mm': np.mean(diameters_mm),
            'median_mm': np.median(diameters_mm),
            'std_mm': np.std(diameters_mm),
            'min_mm': np.min(diameters_mm),
            'max_mm': np.max(diameters_mm),
            'within_tolerance_pct': (ok_count / len(holes)) * 100,
        })

    return stats


def draw_holes(
    img: np.ndarray,
    holes: list[dict],
    pixels_per_mm: float,
    nominal_diameter: float = HOLE_NOMINAL_DIAMETER_MM,
    tolerance: float = HOLE_TOLERANCE_MM,
) -> np.ndarray:
    """Desenha círculos e diâmetros na imagem."""
    result = img.copy()

    for hole in holes:
        diameter_mm = (hole['radius'] * 2) / pixels_per_mm
        cx, cy = hole['center']
        radius_px = int(hole['radius'])

        diff = abs(diameter_mm - nominal_diameter)
        color = (0, 255, 0) if diff <= tolerance else (255, 0, 0)

        thickness = max(2, int(radius_px / 30))
        cv2.circle(result, (cx, cy), radius_px, color, thickness)
        cv2.circle(result, (cx, cy), max(3, int(radius_px / 20)), (255, 0, 0), -1)

        text = f"{diameter_mm:.1f}"
        font_scale = max(0.5, radius_px / 60)
        font_thickness = max(1, int(radius_px / 40))
        text_pos = (cx - int(radius_px / 3), cy - radius_px - int(radius_px / 5))
        cv2.putText(result, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)

    return result


# =============================================================================
# Pipeline de detecção de rebarbas (linhas)
# =============================================================================

def merge_lines(
    lines: list[np.ndarray],
    angle_thresh: float = 15.0,
    dist_thresh: float = 60.0,
) -> list[np.ndarray]:
    """Agrupa segmentos de linha próximos e com ângulo similar."""
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        segments.append((x1, y1, x2, y2, angle))

    used = [False] * len(segments)
    merged = []

    for i in range(len(segments)):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            avg_angle = np.mean([s[4] for s in group])

            for j in range(len(segments)):
                if used[j]:
                    continue
                da = abs(avg_angle - segments[j][4])
                if da > 90:
                    da = 180 - da
                if da > angle_thresh:
                    continue

                min_d = float('inf')
                for seg in group:
                    for px, py in [(seg[0], seg[1]), (seg[2], seg[3])]:
                        for qx, qy in [(segments[j][0], segments[j][1]),
                                        (segments[j][2], segments[j][3])]:
                            d = np.sqrt((px - qx)**2 + (py - qy)**2)
                            min_d = min(min_d, d)

                if min_d > dist_thresh:
                    continue
                group.append(segments[j])
                used[j] = True
                changed = True

        # Encontrar os dois pontos mais distantes do grupo
        all_pts = []
        for seg in group:
            all_pts.append((seg[0], seg[1]))
            all_pts.append((seg[2], seg[3]))

        max_d = 0
        p1, p2 = all_pts[0], all_pts[-1]
        for a in range(len(all_pts)):
            for b in range(a + 1, len(all_pts)):
                d = (all_pts[a][0] - all_pts[b][0])**2 + (all_pts[a][1] - all_pts[b][1])**2
                if d > max_d:
                    max_d = d
                    p1, p2 = all_pts[a], all_pts[b]

        merged.append(np.array([p1[0], p1[1], p2[0], p2[1]]))

    return merged


def point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Calcula a distância de um ponto a um segmento de reta."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def filter_edge_lines(
    lines: list[np.ndarray],
    cx: float,
    cy: float,
    r: float,
    edge_thresh: float = 0.75,
    min_length_ratio: float = 0.35,
    max_center_dist: float = 0.80,
) -> list[np.ndarray]:
    """Remove linhas que são artefatos da borda do furo."""
    diameter = 2 * r
    filtered = []

    for line in lines:
        x1, y1, x2, y2 = line

        # Critério 1: se todos os pontos estão na periferia, é borda
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist_mid = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
        dist_p1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        dist_p2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        if (dist_p1 > r * edge_thresh
                and dist_p2 > r * edge_thresh
                and dist_mid > r * edge_thresh):
            continue

        # Critério 2: comprimento mínimo
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < diameter * min_length_ratio:
            continue

        # Critério 3: a reta passa perto o suficiente do centro
        dist_center = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
        if dist_center > r * max_center_dist:
            continue

        filtered.append(line)

    return filtered


def detect_wires(
    gray: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    margin_factor: float = 1.4,
    gauss_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 100,
    hough_threshold: int = 20,
    min_line_length: int = 5,
    max_line_gap: int = 30,
    inner_mask_ratio: float = 0.85,
    merge_angle_thresh: float = 15.0,
    merge_dist_thresh: float = 60.0,
    edge_filter_thresh: float = 0.75,
    min_length_ratio: float = 0.35,
    max_center_dist: float = 0.80,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Detecta fios/rebarbas dentro de um furo usando detecção de linhas."""
    height, width = gray.shape
    margin = int(r * margin_factor)
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(width, cx + margin)
    y2 = min(height, cy + margin)

    roi = gray[y1:y2, x1:x2].copy()
    cx_rel, cy_rel = cx - x1, cy - y1

    # Máscara circular
    mask = np.zeros_like(roi, dtype=np.uint8)
    cv2.circle(mask, (cx_rel, cy_rel), r, 255, -1)
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

    # Gauss + Canny
    roi_gauss = cv2.GaussianBlur(roi_masked, (gauss_ksize, gauss_ksize), 0)
    edges = cv2.Canny(roi_gauss, canny_low, canny_high)

    # Máscara interna (exclui borda externa do furo)
    edge_mask = np.zeros_like(edges)
    cv2.circle(edge_mask, (cx_rel, cy_rel), int(r * inner_mask_ratio), 255, -1)
    edges = cv2.bitwise_and(edges, edge_mask)

    # HoughLinesP + merge + filtro
    lines_raw = cv2.HoughLinesP(
        edges, 2, np.pi / 180, hough_threshold,
        np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap,
    )

    wires = []
    if lines_raw is not None:
        lines_merged = merge_lines(lines_raw, merge_angle_thresh, merge_dist_thresh)
        lines_merged = filter_edge_lines(
            lines_merged, cx_rel, cy_rel, r,
            edge_filter_thresh, min_length_ratio, max_center_dist,
        )
        for line in lines_merged:
            lx1, ly1, lx2, ly2 = line
            # Converter coordenadas de volta para imagem original
            wires.append((
                (int(lx1 + x1), int(ly1 + y1)),
                (int(lx2 + x1), int(ly2 + y1)),
            ))

    return wires


def draw_wires(
    img: np.ndarray,
    all_wires: list[tuple[tuple[int, int], tuple[int, int]]],
    holes: list[dict],
    holes_with_wires: list[bool],
) -> np.ndarray:
    """Desenha linhas roxas das rebarbas e círculos verde/vermelho nos furos."""
    result = img.copy()

    for i, hole in enumerate(holes):
        cx, cy = hole['center']
        r = int(hole['radius'])
        has_wire = holes_with_wires[i]

        if has_wire:
            cv2.circle(result, (cx, cy), r, (255, 0, 0), 3)
        else:
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)

    for p1, p2 in all_wires:
        cv2.line(result, p1, p2, (255, 0, 255), 3)

    return result


# =============================================================================
# Interface Streamlit
# =============================================================================

st.set_page_config(layout="wide", page_title="Análise de Placas Midea")
st.title("Análise de Qualidade - Placas Midea")
st.markdown("Faça upload da imagem da placa para analisar furos e rebarbas.")

# --- Sidebar: parâmetros do pipeline ---
with st.sidebar:
    st.header("Parâmetros do Pipeline")

    st.subheader("Detecção de Furos")
    hole_blur_ksize = st.slider("Gaussian Blur kernel", 3, 31, 15, step=2,
                                key="hole_blur")
    hole_median_ksize = st.slider("Median Blur kernel", 3, 15, 7, step=2,
                                  key="hole_median")
    hole_canny_low = st.slider("Canny low (furos)", 10, 150, 70, key="hole_canny_l")
    hole_canny_high = st.slider("Canny high (furos)", 50, 300, 140, key="hole_canny_h")
    hole_hough_dp = st.slider("Hough dp", 1.0, 3.0, 1.2, step=0.1, key="hole_dp")
    hole_hough_param1 = st.slider("Hough param1", 20, 200, 80, key="hole_p1")
    hole_hough_param2 = st.slider("Hough param2", 10, 100, 50, key="hole_p2")

    st.markdown("---")
    st.subheader("Detecção de Rebarbas")
    wire_gauss_ksize = st.slider("Gaussian Blur kernel (ROI)", 3, 15, 5, step=2,
                                 key="wire_gauss")
    wire_canny_low = st.slider("Canny low (rebarbas)", 10, 150, 50, key="wire_canny_l")
    wire_canny_high = st.slider("Canny high (rebarbas)", 30, 300, 100, key="wire_canny_h")
    wire_inner_mask = st.slider("Máscara interna (% raio)", 0.50, 0.95, 0.85, step=0.05,
                                key="wire_mask")
    wire_hough_thresh = st.slider("Hough threshold", 5, 60, 20, key="wire_hough")
    wire_min_line_len = st.slider("Min line length", 1, 30, 5, key="wire_minlen")
    wire_max_line_gap = st.slider("Max line gap", 5, 80, 30, key="wire_maxgap")
    wire_merge_angle = st.slider("Merge angle threshold", 5.0, 45.0, 15.0, step=1.0,
                                 key="wire_angle")
    wire_merge_dist = st.slider("Merge distance threshold", 10.0, 150.0, 60.0, step=5.0,
                                key="wire_dist")
    wire_edge_thresh = st.slider("Edge filter threshold", 0.50, 0.95, 0.75, step=0.05,
                                 key="wire_edge")
    wire_min_length_ratio = st.slider("Min length ratio (% diâmetro)", 0.10, 0.60, 0.35,
                                      step=0.05, key="wire_lenratio")

# --- Upload ---
uploaded_file = st.file_uploader(
    "Escolha uma imagem...", type=["jpg", "jpeg", "png", "heic", "HEIC"],
)

if uploaded_file is not None:
    # Carregar imagem
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.heic'):
        pil_img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

    if img_bgr is None:
        st.error("Erro ao carregar imagem. Tente outro formato.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        st.subheader("Imagem Original")
        st.image(img_rgb, use_container_width=True)

        # === Pipeline de furos ===
        raw_holes = detect_holes(
            img_rgb,
            blur_ksize=hole_blur_ksize,
            blur_sigma=3.0,
            median_ksize=hole_median_ksize,
            canny_low=hole_canny_low,
            canny_high=hole_canny_high,
            hough_dp=hole_hough_dp,
            hough_param1=hole_hough_param1,
            hough_param2=hole_hough_param2,
        )
        valid_holes, pixels_per_mm = calibrate_pixels_per_mm(raw_holes)
        hole_stats = compute_hole_stats(valid_holes, pixels_per_mm)
        img_holes = draw_holes(img_rgb, valid_holes, pixels_per_mm)

        # === Pipeline de rebarbas ===
        all_wires = []
        holes_with_wires = []
        wire_counts = []

        for hole in valid_holes:
            cx, cy = hole['center']
            r = int(hole['radius'])
            wires = detect_wires(
                gray, cx, cy, r,
                gauss_ksize=wire_gauss_ksize,
                canny_low=wire_canny_low,
                canny_high=wire_canny_high,
                inner_mask_ratio=wire_inner_mask,
                hough_threshold=wire_hough_thresh,
                min_line_length=wire_min_line_len,
                max_line_gap=wire_max_line_gap,
                merge_angle_thresh=wire_merge_angle,
                merge_dist_thresh=wire_merge_dist,
                edge_filter_thresh=wire_edge_thresh,
                min_length_ratio=wire_min_length_ratio,
            )
            all_wires.extend(wires)
            has_wire = len(wires) > 0
            holes_with_wires.append(has_wire)
            wire_counts.append(len(wires))

        img_wires = draw_wires(img_rgb, all_wires, valid_holes, holes_with_wires)
        total_with_wires = sum(holes_with_wires)

        # === Layout de resultados ===
        col1, col2 = st.columns(2)

        with col1:
            st.header("1. Análise de Furos")
            st.image(img_holes, use_container_width=True)

            st.markdown(f"### Furos Detectados: {hole_stats['total']}")

            if hole_stats['total'] > 0:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Diâmetro Médio", f"{hole_stats.get('mean_mm', 0):.2f} mm")
                    st.metric("Diâmetro Mediana", f"{hole_stats.get('median_mm', 0):.2f} mm")
                    st.metric("Desvio Padrão", f"{hole_stats.get('std_mm', 0):.2f} mm")
                with col_s2:
                    st.metric("Mín / Máx",
                              f"{hole_stats.get('min_mm', 0):.2f} / {hole_stats.get('max_mm', 0):.2f} mm")
                    st.metric(f"Dentro Tolerância (±{HOLE_TOLERANCE_MM}mm)",
                              f"{hole_stats.get('within_tolerance_pct', 0):.1f}%")
                    st.metric("Calibração", f"{hole_stats.get('pixels_per_mm', 0):.2f} px/mm")

                if hole_stats['fail'] > 0:
                    st.markdown(f"### Fora do padrão: {hole_stats['fail']}")
                else:
                    st.markdown("### Todos dentro do padrão!")

        with col2:
            st.header("2. Detecção de Rebarbas")
            st.image(img_wires, use_container_width=True)

            if total_with_wires > 0:
                st.markdown(f"### Furos com rebarba: {total_with_wires}/{hole_stats['total']}")
            else:
                st.markdown(f"### Nenhuma rebarba detectada!")

        # === Tabela detalhada ===
        st.markdown("---")
        st.header("Análise Detalhada")

        if hole_stats['total'] > 0:
            diameters = hole_stats['diameters_mm']
            df = pd.DataFrame({
                'Furo': [f"#{i+1}" for i in range(len(diameters))],
                'Diâmetro (mm)': [f"{d:.2f}" for d in diameters],
                'Tolerância': [
                    'OK' if abs(d - HOLE_NOMINAL_DIAMETER_MM) <= HOLE_TOLERANCE_MM else 'NOK'
                    for d in diameters
                ],
                'Rebarbas': [
                    f"{wire_counts[i]} fio(s)" if holes_with_wires[i] else 'OK'
                    for i in range(len(diameters))
                ],
            })

            table_height = min(35 * len(diameters) + 40, 600)
            st.dataframe(df, use_container_width=True, height=table_height)

            st.markdown(f"""
            **Resumo:**
            - Total: **{len(diameters)}** furos
            - Diâmetro OK: **{hole_stats['ok']}** | NOK: **{hole_stats['fail']}**
            - Sem rebarba: **{sum(1 for w in holes_with_wires if not w)}** | Com rebarba: **{total_with_wires}**
            """)
        else:
            st.info("Nenhum furo detectado")

        # Expander para zoom
        with st.expander("Clique para ver imagem em TAMANHO REAL (zoom)"):
            st.image(img_wires, caption="Detecção de rebarbas - Tamanho Real")
            st.image(img_holes, caption="Detecção de furos - Tamanho Real")

else:
    st.info("Por favor, faça o upload de uma imagem para começar.")
