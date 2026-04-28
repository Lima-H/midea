"""Pré-processamento de imagens antes da detecção.

Inclui equalização de contraste (CLAHE), detecção automática de polaridade
(claro-em-escuro vs escuro-em-claro) e utilitários de extração de ROI por furo.

A polaridade é decidida comparando a mediana de intensidade no interior do furo
com a mediana no anel periférico — pixels saturados são ignorados para evitar
viés por reflexos especulares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID = (8, 8)
SATURATION_THRESHOLD = 250
INNER_RATIO_FOR_POLARITY = 0.6
RING_INNER_RATIO = 0.9
RING_OUTER_RATIO = 1.1
ROI_MARGIN_RATIO = 0.10

Polarity = Literal["light_on_dark", "dark_on_light"]


@dataclass(frozen=True)
class RoiContext:
    """Recorte da ROI em torno de um furo, em coordenadas locais.

    Attributes:
        roi: Imagem em cinza recortada (uint8).
        cx_local: Coordenada X do centro do furo na ROI.
        cy_local: Coordenada Y do centro do furo na ROI.
        radius: Raio do furo em pixels.
        offset_x: Offset X da ROI na imagem original.
        offset_y: Offset Y da ROI na imagem original.
    """

    roi: np.ndarray
    cx_local: int
    cy_local: int
    radius: int
    offset_x: int
    offset_y: int


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
) -> np.ndarray:
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        gray: Imagem em escala de cinza (uint8).
        clip_limit: Limite de contraste (valores típicos: 2.0–4.0).
        tile_grid: Tamanho do grid de tiles para equalização local.

    Returns:
        Imagem equalizada com mesmo shape e dtype da entrada.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def extract_roi(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    margin_ratio: float = ROI_MARGIN_RATIO,
) -> RoiContext:
    """Recorta uma ROI quadrada ao redor do furo com margem proporcional ao raio.

    Args:
        gray: Imagem em escala de cinza (uint8).
        cx: Centro X do furo na imagem original.
        cy: Centro Y do furo na imagem original.
        radius: Raio do furo em pixels.
        margin_ratio: Margem extra como fração do raio (0.10 = 10% além).

    Returns:
        Contexto com a ROI e coordenadas locais.
    """
    height, width = gray.shape
    margin = int(radius * (1.0 + margin_ratio))
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(width, cx + margin)
    y2 = min(height, cy + margin)
    roi = gray[y1:y2, x1:x2].copy()
    return RoiContext(
        roi=roi,
        cx_local=cx - x1,
        cy_local=cy - y1,
        radius=radius,
        offset_x=x1,
        offset_y=y1,
    )


def detect_polarity(ctx: RoiContext) -> Polarity:
    """Detecta a polaridade dos filamentos esperados dentro do furo.

    Compara a mediana de intensidade do disco central (raio 0.6 r) com a
    mediana do anel externo (entre 0.9 r e 1.1 r). Pixels saturados (>250)
    são ignorados para reduzir o viés por reflexos especulares.

    Args:
        ctx: Contexto da ROI.

    Returns:
        "light_on_dark" se o interior é mais escuro que o anel (filamentos
        aparecem claros sobre fundo escuro). "dark_on_light" caso contrário.
    """
    inner_mask = _disk_mask(ctx, INNER_RATIO_FOR_POLARITY)
    ring_mask = _ring_mask(ctx, RING_INNER_RATIO, RING_OUTER_RATIO)

    valid = ctx.roi <= SATURATION_THRESHOLD
    inner_pixels = ctx.roi[inner_mask & valid]
    ring_pixels = ctx.roi[ring_mask & valid]

    if inner_pixels.size == 0 or ring_pixels.size == 0:
        return "light_on_dark"

    inner_median = float(np.median(inner_pixels))
    ring_median = float(np.median(ring_pixels))
    return "light_on_dark" if inner_median < ring_median else "dark_on_light"


def normalize_polarity(roi: np.ndarray, polarity: Polarity) -> np.ndarray:
    """Inverte a ROI quando necessário para sempre tratar 'claro sobre escuro'.

    Args:
        roi: Imagem em cinza (uint8).
        polarity: Polaridade detectada.

    Returns:
        ROI possivelmente invertida (255 - roi) para que filamentos fiquem
        sempre claros sobre fundo escuro — premissa dos filtros vesselness.
    """
    if polarity == "dark_on_light":
        return cv2.bitwise_not(roi)
    return roi


def _disk_mask(ctx: RoiContext, ratio: float) -> np.ndarray:
    """Constrói máscara booleana do disco central com raio = ratio * r."""
    mask = np.zeros(ctx.roi.shape, dtype=np.uint8)
    cv2.circle(mask, (ctx.cx_local, ctx.cy_local), int(ctx.radius * ratio), 255, -1)
    return mask.astype(bool)


def _ring_mask(ctx: RoiContext, inner_ratio: float, outer_ratio: float) -> np.ndarray:
    """Constrói máscara booleana de um anel entre dois raios fracionários."""
    outer = np.zeros(ctx.roi.shape, dtype=np.uint8)
    inner = np.zeros(ctx.roi.shape, dtype=np.uint8)
    cv2.circle(outer, (ctx.cx_local, ctx.cy_local), int(ctx.radius * outer_ratio), 255, -1)
    cv2.circle(inner, (ctx.cx_local, ctx.cy_local), int(ctx.radius * inner_ratio), 255, -1)
    return ((outer > 0) & (inner == 0))
