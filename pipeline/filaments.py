"""Detector de filamentos lineares (fios e rebarbas-linha) dentro de furos.

Pipeline (por furo):
    1. Recortar ROI com margem proporcional ao raio.
    2. Aplicar CLAHE local.
    3. Detectar polaridade (claro-em-escuro vs escuro-em-claro) e normalizar.
    4. Realçar filamentos com dois detectores em paralelo:
        - Frangi vesselness (skimage) — multi-escala, ressalta estruturas tubulares.
        - Black-hat morfológico orientado em N ângulos.
    5. Fundir os mapas com pesos fixos e aplicar threshold por percentil.
    6. Esqueletizar (Guo-Hall) e extrair componentes conexos.
    7. Aplicar filtros geométricos (length, straightness, vesselness mean,
       anti-borda) e produzir endpoints do filamento.

A ideia é cobrir tanto fios finos completamente internos (caso 'fio dentro do
furo') quanto linhas longas atravessando (caso 'rebarba-linha'), em qualquer
polaridade. Frangi/black-hat operam em ROIs ~600×600 px — viável em CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np
from skimage.filters import frangi

from pipeline.holes import Hole
from pipeline.preprocess import (
    Polarity,
    RoiContext,
    apply_clahe,
    detect_polarity,
    extract_roi,
    normalize_polarity,
)

logger = logging.getLogger(__name__)
_thinning_fallback_warned = False

ORIENTATION_STEP_DEG = 15
BLACKHAT_KERNEL_LENGTH = 15
BLACKHAT_KERNEL_WIDTH = 1
FUSION_VESSELNESS_WEIGHT = 0.6
FUSION_BLACKHAT_WEIGHT = 0.4
SATURATION_THRESHOLD = 250
HULL_OPTIMIZATION_THRESHOLD = 200
TANGENTIAL_ANGLE_VARIATION_DEG = 5.0


@dataclass(frozen=True)
class FilamentParams:
    """Hiperparâmetros do detector de filamentos.

    Defaults validados qualitativamente. A UI Streamlit expõe os mais sensíveis
    como sliders (clahe, frangi_sigma_max, score_percentile, min_length_ratio).
    """

    clahe_clip_limit: float = 2.0
    high_pass_sigma: float = 8.0
    frangi_sigma_max: int = 3
    frangi_gamma: Optional[float] = None
    score_percentile: float = 99.5
    min_length_ratio: float = 0.20
    min_vesselness: float = 0.10
    short_length_ratio: float = 0.10
    short_max_straightness: float = 1.3
    short_max_center_dist: float = 0.5
    inner_mask_ratio: float = 0.95
    border_ring_inner: float = 0.92
    border_ring_outer: float = 1.08
    anti_edge_pixel_ratio: float = 0.80
    anti_edge_radius_ratio: float = 0.92
    force_polarity: Optional[Polarity] = None


@dataclass(frozen=True)
class FilamentSegment:
    """Filamento detectado, em coordenadas da imagem original.

    Attributes:
        p1: Endpoint 1 (x, y) em pixels.
        p2: Endpoint 2 (x, y) em pixels.
        length_px: Comprimento aproximado em pixels (do skeleton).
        mean_vesselness: Resposta média de vesselness no componente (0 a 1).
        polarity: Polaridade detectada para o furo de origem.
    """

    p1: tuple[int, int]
    p2: tuple[int, int]
    length_px: float
    mean_vesselness: float
    polarity: Polarity


def detect_filaments(
    gray: np.ndarray,
    hole: Hole,
    params: FilamentParams = FilamentParams(),
) -> list[FilamentSegment]:
    """Detecta filamentos lineares dentro de um furo.

    Args:
        gray: Imagem em cinza (uint8) original.
        hole: Furo já calibrado com centro e raio em pixels.
        params: Hiperparâmetros do pipeline.

    Returns:
        Lista de filamentos detectados com endpoints e métricas.
    """
    cx, cy = hole.center
    radius = int(hole.radius)
    if radius <= 0:
        return []

    ctx = extract_roi(gray, cx, cy, radius)
    roi_eq = apply_clahe(ctx.roi, clip_limit=params.clahe_clip_limit)
    ctx_eq = replace(ctx, roi=roi_eq)
    polarity = params.force_polarity or detect_polarity(ctx_eq)
    roi_norm = normalize_polarity(roi_eq, polarity)
    roi_proc = _high_pass(roi_norm, params.high_pass_sigma)

    valid_mask = _build_valid_mask(roi_proc, ctx, params)
    if not valid_mask.any():
        return []

    vesselness = _compute_vesselness(roi_proc, params, valid_mask)
    blackhat = _oriented_blackhat(roi_proc, valid_mask)
    score = _fuse_scores(vesselness, blackhat)

    filament_mask = _threshold_score(score, valid_mask, params.score_percentile)
    if not filament_mask.any():
        return []

    skeleton = _thinning(filament_mask)
    components = _split_components(skeleton)
    return _build_segments(components, vesselness, ctx, polarity, params)


def _build_valid_mask(
    roi_norm: np.ndarray,
    ctx: RoiContext,
    params: FilamentParams,
) -> np.ndarray:
    """Máscara booleana dos pixels válidos para detecção (sem borda nem saturados)."""
    inner = np.zeros(roi_norm.shape, dtype=np.uint8)
    cv2.circle(
        inner,
        (ctx.cx_local, ctx.cy_local),
        int(ctx.radius * params.inner_mask_ratio),
        255,
        -1,
    )
    border_outer = np.zeros(roi_norm.shape, dtype=np.uint8)
    border_inner = np.zeros(roi_norm.shape, dtype=np.uint8)
    cv2.circle(
        border_outer,
        (ctx.cx_local, ctx.cy_local),
        int(ctx.radius * params.border_ring_outer),
        255,
        -1,
    )
    cv2.circle(
        border_inner,
        (ctx.cx_local, ctx.cy_local),
        int(ctx.radius * params.border_ring_inner),
        255,
        -1,
    )
    border_ring = (border_outer > 0) & (border_inner == 0)

    saturated = roi_norm > SATURATION_THRESHOLD
    return (inner > 0) & ~border_ring & ~saturated


def _high_pass(roi: np.ndarray, sigma: float) -> np.ndarray:
    """Aplica filtro passa-alta subtraindo um blur Gaussiano grande.

    Remove o componente de baixa frequência (textura grossa, sombras de
    iluminação, baseline do material visível através do furo). Estruturas
    finas como fios são preservadas, pois mudam pouco sob o blur.

    Args:
        roi: Imagem em cinza (uint8) já com polaridade normalizada.
        sigma: Desvio padrão (px) do blur Gaussiano. Valores típicos: 6–12.
            Use 0 para desativar o filtro.

    Returns:
        ROI pós-filtro normalizada para [0, 255], dtype uint8.
    """
    if sigma <= 0:
        return roi
    blurred = cv2.GaussianBlur(roi, (0, 0), sigma)
    diff = cv2.subtract(roi, blurred)
    diff_max = float(diff.max())
    if diff_max <= 0:
        return diff
    return ((diff.astype(np.float32) / diff_max) * 255).astype(np.uint8)


def _compute_vesselness(
    roi_norm: np.ndarray,
    params: FilamentParams,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Calcula o mapa de vesselness Frangi (estruturas tubulares claras)."""
    sigmas = range(1, max(2, params.frangi_sigma_max + 1))
    roi_float = roi_norm.astype(np.float32) / 255.0
    response = frangi(
        roi_float,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=params.frangi_gamma,
        black_ridges=False,
    )
    response = np.where(valid_mask, response, 0.0)
    return response.astype(np.float32)


def _oriented_blackhat(roi_norm: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Black-hat morfológico com kernel linear varrendo orientações [0°, 180°)."""
    inverted = cv2.bitwise_not(roi_norm)  # Frangi e blackhat trabalham simétricos
    response = np.zeros_like(roi_norm, dtype=np.float32)

    for theta_deg in range(0, 180, ORIENTATION_STEP_DEG):
        kernel = _make_oriented_kernel(theta_deg)
        bh = cv2.morphologyEx(inverted, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, bh.astype(np.float32))

    response = np.where(valid_mask, response, 0.0)
    return response


def _make_oriented_kernel(theta_deg: float) -> np.ndarray:
    """Constrói kernel linear de comprimento fixo rotacionado em theta graus."""
    base_size = BLACKHAT_KERNEL_LENGTH
    kernel = np.zeros((base_size, base_size), dtype=np.uint8)
    cv2.line(
        kernel,
        (0, base_size // 2),
        (base_size - 1, base_size // 2),
        1,
        BLACKHAT_KERNEL_WIDTH,
    )
    matrix = cv2.getRotationMatrix2D((base_size / 2, base_size / 2), theta_deg, 1.0)
    rotated = cv2.warpAffine(
        kernel,
        matrix,
        (base_size, base_size),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated


def _fuse_scores(vesselness: np.ndarray, blackhat: np.ndarray) -> np.ndarray:
    """Combina vesselness e black-hat normalizados em um score único [0, 1]."""
    v_norm = _minmax_normalize(vesselness)
    b_norm = _minmax_normalize(blackhat)
    return FUSION_VESSELNESS_WEIGHT * v_norm + FUSION_BLACKHAT_WEIGHT * b_norm


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normaliza um array para [0, 1] com clipagem em max(arr)."""
    arr_max = float(arr.max())
    if arr_max <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / arr_max).astype(np.float32)


def _threshold_score(
    score: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Binariza o score com threshold no percentil dado dos pixels válidos."""
    valid_scores = score[valid_mask]
    if valid_scores.size == 0:
        return np.zeros_like(score, dtype=np.uint8)

    threshold = float(np.percentile(valid_scores, percentile))
    if threshold <= 0:
        return np.zeros_like(score, dtype=np.uint8)

    mask = (score >= threshold) & valid_mask
    return mask.astype(np.uint8) * 255


def _thinning(mask: np.ndarray) -> np.ndarray:
    """Esqueletização Guo-Hall (cv2.ximgproc.thinning)."""
    global _thinning_fallback_warned
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None:
        if not _thinning_fallback_warned:
            logger.warning(
                "cv2.ximgproc indisponível (instale opencv-contrib-python-headless); "
                "usando erosão iterativa como fallback.",
            )
            _thinning_fallback_warned = True
        return _fallback_thinning(mask)
    return ximgproc.thinning(mask, thinningType=ximgproc.THINNING_GUOHALL)


def _fallback_thinning(mask: np.ndarray) -> np.ndarray:
    """Esqueletização básica via erosões sucessivas (caso ximgproc não exista)."""
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        img = eroded
    return skeleton


def _split_components(skeleton: np.ndarray) -> list[np.ndarray]:
    """Separa o skeleton em componentes conexos retornando as máscaras booleanas."""
    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)
    components = []
    for label_id in range(1, num_labels):
        mask = labels == label_id
        if mask.any():
            components.append(mask)
    return components


def _build_segments(
    components: list[np.ndarray],
    vesselness: np.ndarray,
    ctx: RoiContext,
    polarity: Polarity,
    params: FilamentParams,
) -> list[FilamentSegment]:
    """Aplica filtros geométricos e converte componentes em segmentos finais."""
    segments: list[FilamentSegment] = []
    diameter = 2 * ctx.radius

    for mask in components:
        metrics = _compute_component_metrics(mask, vesselness, ctx)
        if not _passes_acceptance(metrics, diameter, params):
            continue
        if _is_edge_artifact(mask, ctx, params):
            continue

        p1, p2 = metrics["endpoints"]
        global_p1 = (int(p1[0] + ctx.offset_x), int(p1[1] + ctx.offset_y))
        global_p2 = (int(p2[0] + ctx.offset_x), int(p2[1] + ctx.offset_y))
        segments.append(
            FilamentSegment(
                p1=global_p1,
                p2=global_p2,
                length_px=float(metrics["length"]),
                mean_vesselness=float(metrics["mean_vesselness"]),
                polarity=polarity,
            )
        )
    return segments


def _compute_component_metrics(
    mask: np.ndarray,
    vesselness: np.ndarray,
    ctx: RoiContext,
) -> dict:
    """Calcula métricas geométricas e de resposta de um componente do skeleton."""
    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys])
    p1, p2 = _farthest_pair(points)
    bbox_diag = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    length = float(points.shape[0])
    straightness = length / max(bbox_diag, 1e-6)
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))
    dist_to_center = float(np.hypot(centroid_x - ctx.cx_local, centroid_y - ctx.cy_local))
    mean_vesselness = float(np.mean(vesselness[mask])) if vesselness[mask].size else 0.0

    return {
        "endpoints": (p1, p2),
        "length": length,
        "bbox_diag": bbox_diag,
        "straightness": straightness,
        "dist_to_center": dist_to_center,
        "mean_vesselness": mean_vesselness,
    }


def _farthest_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encontra o par de pontos mais distante (otimizado via convex hull)."""
    if points.shape[0] < 2:
        return points[0], points[0]

    if points.shape[0] > HULL_OPTIMIZATION_THRESHOLD:
        hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
        candidates = hull
    else:
        candidates = points

    diffs = candidates[:, None, :] - candidates[None, :, :]
    dists = np.sum(diffs**2, axis=-1)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return candidates[i], candidates[j]


def _passes_acceptance(metrics: dict, diameter: float, params: FilamentParams) -> bool:
    """Verifica se o componente atende aos critérios de aceitação (OR)."""
    long_ok = (
        metrics["length"] > params.min_length_ratio * diameter
        and metrics["mean_vesselness"] > params.min_vesselness
    )
    short_ok = (
        metrics["length"] > params.short_length_ratio * diameter
        and metrics["straightness"] < params.short_max_straightness
        and metrics["dist_to_center"] < params.short_max_center_dist * (diameter / 2)
    )
    return long_ok or short_ok


def _is_edge_artifact(mask: np.ndarray, ctx: RoiContext, params: FilamentParams) -> bool:
    """Rejeita componentes inteiramente periféricos com orientação tangencial."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return True

    distances = np.hypot(xs - ctx.cx_local, ys - ctx.cy_local)
    radius = ctx.radius
    peripheral_ratio = float(np.mean(distances > radius * params.anti_edge_radius_ratio))
    if peripheral_ratio < params.anti_edge_pixel_ratio:
        return False

    angles = np.degrees(np.arctan2(ys - ctx.cy_local, xs - ctx.cx_local))
    angle_std = float(np.std(_unwrap_angles(angles)))
    return angle_std < TANGENTIAL_ANGLE_VARIATION_DEG


def _unwrap_angles(angles_deg: np.ndarray) -> np.ndarray:
    """Desenrola ângulos para evitar descontinuidade em ±180°."""
    radians = np.deg2rad(angles_deg)
    unwrapped = np.unwrap(radians)
    return np.rad2deg(unwrapped)
