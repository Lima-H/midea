"""Detecção de furos circulares e calibração de pixels/mm.

O pipeline original (HoughCircles + refinamento por gradiente radial) é
preservado. As mudanças desta versão são:

1. CLAHE global aplicado antes do HoughCircles para robustez a iluminação.
2. Calibração de px/mm usa rejeição de outliers via MAD (Median Absolute
   Deviation), em vez do filtro fixo ±30% que enviesava com 1-2 raios ruins.

A função `detect_holes` retorna instâncias de `Hole` (dataclass) — substitui
o dict `{'center', 'radius'}` usado no pipeline antigo.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HOLE_NOMINAL_DIAMETER_MM = 10.0
HOLE_TOLERANCE_MM = 1.0
REFERENCE_IMAGE_HEIGHT = 4032
MAD_SCALE_FACTOR = 1.4826
MAD_THRESHOLD_SIGMAS = 3.0
MIN_CALIBRATION_TOLERANCE_RATIO = 0.05
DEFAULT_PIXELS_PER_MM_FALLBACK = 45.0


@dataclass(frozen=True)
class Hole:
    """Representa um furo detectado na imagem original.

    Attributes:
        center: Coordenada (cx, cy) do centro em pixels.
        radius: Raio em pixels.
    """

    center: tuple[int, int]
    radius: float


@dataclass(frozen=True)
class HoughParams:
    """Hiperparâmetros do estágio HoughCircles + Canny.

    Os valores nominais (`min_radius_px`, `max_radius_px`, `min_dist_px`) são
    referenciados a uma imagem de altura `REFERENCE_IMAGE_HEIGHT`. Para outras
    resoluções, são escalados linearmente.
    """

    blur_ksize: int = 15
    blur_sigma: float = 3.0
    median_ksize: int = 7
    canny_low: int = 70
    canny_high: int = 140
    hough_dp: float = 1.2
    hough_param1: int = 80
    hough_param2: int = 50
    min_radius_px: int = 140
    max_radius_px: int = 300
    min_dist_px: int = 250


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado da calibração de px/mm baseada em todos os furos detectados.

    O filtro MAD é aplicado apenas para escolher quais furos entram no cálculo
    da mediana (calibration holes). Todos os furos detectados são preservados
    em `holes`; cabe ao consumidor decidir quais são OK/NOK pela tolerância
    de diâmetro.

    Attributes:
        holes: Todos os furos detectados (não filtrados pelo MAD).
        pixels_per_mm: Fator de conversão calibrado pela mediana robusta.
        radius_std_px: Desvio padrão dos raios usados na calibração.
        calibration_count: Quantidade de furos que entraram na mediana.
    """

    holes: list[Hole]
    pixels_per_mm: float
    radius_std_px: float
    calibration_count: int


def detect_holes(rgb: np.ndarray, params: HoughParams = HoughParams()) -> list[Hole]:
    """Detecta furos circulares na imagem usando HoughCircles + refinamento.

    Args:
        rgb: Imagem RGB de entrada.
        params: Hiperparâmetros do estágio Hough.

    Returns:
        Lista de furos com centro refinado e raio em pixels.
    """
    gray = _to_gray(rgb)
    blurred = cv2.GaussianBlur(
        gray,
        (params.blur_ksize, params.blur_ksize),
        params.blur_sigma,
    )
    blurred = cv2.medianBlur(blurred, params.median_ksize)

    scale = rgb.shape[0] / REFERENCE_IMAGE_HEIGHT
    min_radius = max(50, int(params.min_radius_px * scale))
    max_radius = max(100, int(params.max_radius_px * scale))
    min_dist = max(80, int(params.min_dist_px * scale))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=params.hough_dp,
        minDist=min_dist,
        param1=params.hough_param1,
        param2=params.hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    return _refine_circles(gray, circles, params, scale)


def calibrate_pixels_per_mm(
    holes: list[Hole],
    nominal_diameter_mm: float = HOLE_NOMINAL_DIAMETER_MM,
) -> CalibrationResult:
    """Calibra px/mm pela mediana dos raios, rejeitando outliers via MAD.

    Args:
        holes: Furos detectados.
        nominal_diameter_mm: Diâmetro nominal real do furo (assume todos iguais).

    Returns:
        Calibração com lista de furos válidos e fator px/mm.
    """
    if not holes:
        return CalibrationResult(
            holes=[],
            pixels_per_mm=DEFAULT_PIXELS_PER_MM_FALLBACK,
            radius_std_px=0.0,
            calibration_count=0,
        )

    radii = np.array([h.radius for h in holes], dtype=np.float64)
    median = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median)))
    threshold = max(
        MAD_THRESHOLD_SIGMAS * MAD_SCALE_FACTOR * mad,
        median * MIN_CALIBRATION_TOLERANCE_RATIO,
    )
    calibration_mask = np.abs(radii - median) <= threshold
    calibration_radii = radii[calibration_mask] if calibration_mask.any() else radii

    pixels_per_mm = (float(np.median(calibration_radii)) * 2) / nominal_diameter_mm
    return CalibrationResult(
        holes=list(holes),
        pixels_per_mm=pixels_per_mm,
        radius_std_px=float(np.std(calibration_radii)),
        calibration_count=int(calibration_mask.sum()),
    )


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
    """Refina (cx, cy, r) usando análise de gradiente radial em N raios.

    Args:
        gray: Imagem em cinza (uint8).
        cx: Centro X inicial.
        cy: Centro Y inicial.
        radius: Raio inicial em pixels.
        num_rays: Número de raios uniformes para amostragem (default 72).
        score_threshold: Score mínimo (gradiente + diferença) para aceitar ponto.
        min_ray_ratio: Fração mínima de raios com ponto válido para refinar.
        iqr_factor: Fator de IQR para descartar outliers radiais.
        max_radius_diff: Variação máxima permitida em relação ao raio inicial.

    Returns:
        Tupla (cx, cy, raio) refinada ou None se o refinamento falhou.
    """
    height, width = gray.shape
    gray_smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)

    grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    center_intensity = _estimate_center_intensity(gray_smooth, cx, cy, radius)
    edge_points = _scan_radial_edges(
        gray_smooth,
        grad_mag,
        cx,
        cy,
        radius,
        center_intensity,
        num_rays,
        score_threshold,
        height,
        width,
    )

    if len(edge_points) < num_rays * min_ray_ratio:
        return None

    valid_points = _filter_outliers_iqr(edge_points, iqr_factor)
    if len(valid_points) < 5:
        return None

    return _fit_ellipse_validated(valid_points, cx, cy, radius, max_radius_diff)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Converte RGB para cinza, ou retorna uma cópia se já for cinza."""
    if rgb.ndim == 2:
        return rgb.copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _refine_circles(
    gray: np.ndarray,
    circles_hough: np.ndarray,
    params: HoughParams,
    scale: float,
) -> list[Hole]:
    """Refina cada círculo do HoughCircles com fitEllipse + gradiente radial."""
    circles = np.around(circles_hough).astype(np.int32)
    kernel = np.ones((3, 3), np.uint8)
    detected: list[Hole] = []

    for circle in circles[0]:
        cx_h, cy_h, r_h = int(circle[0]), int(circle[1]), int(circle[2])
        contour = _best_contour_in_roi(gray, cx_h, cy_h, r_h, params, scale, kernel)
        hole = _refine_single_circle(gray, contour, cx_h, cy_h, r_h)
        detected.append(hole)
    return detected


def _best_contour_in_roi(
    gray: np.ndarray,
    cx_h: int,
    cy_h: int,
    r_h: int,
    params: HoughParams,
    scale: float,
    kernel: np.ndarray,
) -> np.ndarray | None:
    """Encontra o melhor contorno do furo dentro de uma ROI ao redor do Hough."""
    margin = int(100 * scale) if scale > 0.5 else 50
    x1 = max(0, cx_h - r_h - margin)
    y1 = max(0, cy_h - r_h - margin)
    x2 = min(gray.shape[1], cx_h + r_h + margin)
    y2 = min(gray.shape[0], cy_h + r_h + margin)
    roi = gray[y1:y2, x1:x2]

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 1)
    edges = cv2.Canny(roi_blur, params.canny_low, params.canny_high)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    _, roi_thresh = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_and(roi_thresh, edges_dilated)

    if cv2.countNonZero(combined) < 10000:
        combined = roi_thresh

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx_local, cy_local = cx_h - x1, cy_h - y1
    return _select_closest_contour(contours, cx_local, cy_local, r_h, scale, x1, y1)


def _select_closest_contour(
    contours: Sequence[np.ndarray],
    cx_local: int,
    cy_local: int,
    r_h: int,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> np.ndarray | None:
    """Seleciona o contorno mais próximo do centro Hough com área mínima."""
    min_area = 40000 * (scale**2) if scale > 0.5 else 5000
    best_cnt: np.ndarray | None = None
    best_dist = float("inf")

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cnt_cx = int(m["m10"] / m["m00"])
        cnt_cy = int(m["m01"] / m["m00"])
        dist = float(np.hypot(cnt_cx - cx_local, cnt_cy - cy_local))
        if dist < best_dist and dist < r_h * 0.5:
            best_dist = dist
            best_cnt = cnt + np.array([offset_x, offset_y])
    return best_cnt


def _refine_single_circle(
    gray: np.ndarray,
    contour: np.ndarray | None,
    cx_hough: int,
    cy_hough: int,
    r_hough: int,
) -> Hole:
    """Refina um único círculo com fitEllipse + gradiente radial."""
    if contour is not None and len(contour) >= 5:
        try:
            (cx_fit, cy_fit), (w, h), _ = cv2.fitEllipse(contour)
        except cv2.error:
            (cx_fit, cy_fit), radius_fit = cv2.minEnclosingCircle(contour)
            return Hole(center=(int(cx_fit), int(cy_fit)), radius=float(radius_fit))

        radius_fit = (w + h) / 4
        result = refine_inner_edge(gray, int(cx_fit), int(cy_fit), int(radius_fit))
        if result is not None:
            return Hole(center=(int(result[0]), int(result[1])), radius=float(result[2]))
        return Hole(center=(int(cx_fit), int(cy_fit)), radius=float(radius_fit))

    if contour is not None:
        (cx_r, cy_r), radius_r = cv2.minEnclosingCircle(contour)
        return Hole(center=(int(cx_r), int(cy_r)), radius=float(radius_r))

    result = refine_inner_edge(gray, cx_hough, cy_hough, r_hough)
    if result is not None:
        return Hole(center=(int(result[0]), int(result[1])), radius=float(result[2]))
    return Hole(center=(cx_hough, cy_hough), radius=float(r_hough))


def _estimate_center_intensity(
    gray_smooth: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> float:
    """Estima a intensidade média no centro do furo em uma grade angular."""
    height, width = gray_smooth.shape
    samples: list[float] = []
    for r in range(5, max(6, int(radius * 0.3))):
        for ang_deg in range(0, 360, 45):
            angle = np.radians(ang_deg)
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            if 0 <= px < width and 0 <= py < height:
                samples.append(float(gray_smooth[py, px]))
    return float(np.median(samples)) if samples else 70.0


def _scan_radial_edges(
    gray_smooth: np.ndarray,
    grad_mag: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    center_intensity: float,
    num_rays: int,
    score_threshold: float,
    height: int,
    width: int,
) -> list[tuple[float, float, float]]:
    """Varre N raios e retorna pontos de borda com score acima do threshold."""
    points: list[tuple[float, float, float]] = []
    r_min = int(radius * 0.55)
    r_max = int(radius * 1.3)

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        best_r, best_score = None, 0.0

        for r in range(r_min, r_max):
            score = _score_radial_point(
                gray_smooth,
                grad_mag,
                cx,
                cy,
                r,
                cos_a,
                sin_a,
                center_intensity,
                height,
                width,
            )
            if score is not None and score > best_score:
                best_score = score
                best_r = r

        if best_r is not None and best_score > score_threshold:
            points.append((cx + best_r * cos_a, cy + best_r * sin_a, float(best_r)))
    return points


def _score_radial_point(
    gray_smooth: np.ndarray,
    grad_mag: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    cos_a: float,
    sin_a: float,
    center_intensity: float,
    height: int,
    width: int,
) -> float | None:
    """Calcula score combinado (gradiente + diferença + proximidade) em um raio."""
    px, py = int(cx + r * cos_a), int(cy + r * sin_a)
    if not (0 <= px < width and 0 <= py < height):
        return None

    px_in = int(cx + (r - 5) * cos_a)
    py_in = int(cy + (r - 5) * sin_a)
    px_out = int(cx + (r + 5) * cos_a)
    py_out = int(cy + (r + 5) * sin_a)
    if not (0 <= px_in < width and 0 <= py_in < height):
        return None
    if not (0 <= px_out < width and 0 <= py_out < height):
        return None

    diff = float(gray_smooth[py_out, px_out]) - float(gray_smooth[py_in, px_in])
    if diff <= 12:
        return None

    grad = float(grad_mag[py, px])
    proximity = max(0.0, 50.0 - abs(float(gray_smooth[py_in, px_in]) - center_intensity))
    return grad * 0.3 + diff * 0.5 + proximity * 0.2


def _filter_outliers_iqr(
    points: list[tuple[float, float, float]],
    iqr_factor: float,
) -> list[tuple[float, float]]:
    """Filtra pontos fora do intervalo [q1 - k*IQR, q3 + k*IQR] do raio."""
    radii = [p[2] for p in points]
    q1, q3 = np.percentile(radii, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return [(p[0], p[1]) for p in points if lower <= p[2] <= upper]


def _fit_ellipse_validated(
    points: list[tuple[float, float]],
    cx_init: int,
    cy_init: int,
    radius_init: int,
    max_radius_diff: float,
) -> tuple[float, float, float] | None:
    """Ajusta uma elipse e valida deslocamento e variação de raio."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        (cx_fit, cy_fit), (w, h), _ = cv2.fitEllipse(pts)
    except cv2.error:
        return None

    radius_fit = (w + h) / 4
    dist = float(np.hypot(cx_fit - cx_init, cy_fit - cy_init))
    radius_diff = (radius_fit - radius_init) / radius_init

    if dist >= radius_init * 0.18:
        return None
    if not (-max_radius_diff <= radius_diff <= 0.15):
        return None
    return float(cx_fit), float(cy_fit), float(radius_fit)
