"""Composição de overlays para visualização dos resultados.

Mantém a estética do pipeline antigo: círculos verde/vermelho conforme tolerância
do diâmetro, linhas em magenta sobre os filamentos detectados, e rótulo de
diâmetro em mm sobre cada furo.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.filaments import FilamentSegment
from pipeline.holes import HOLE_NOMINAL_DIAMETER_MM, HOLE_TOLERANCE_MM, Hole

COLOR_OK = (0, 255, 0)
COLOR_FAIL = (255, 0, 0)
COLOR_FILAMENT = (255, 0, 255)
COLOR_CENTER_DOT = (255, 0, 0)
COLOR_TEXT_OUTLINE = (0, 0, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_holes(
    rgb: np.ndarray,
    holes: list[Hole],
    pixels_per_mm: float,
    nominal_diameter_mm: float = HOLE_NOMINAL_DIAMETER_MM,
    tolerance_mm: float = HOLE_TOLERANCE_MM,
) -> np.ndarray:
    """Desenha círculos coloridos por tolerância e o diâmetro em mm.

    Args:
        rgb: Imagem RGB de entrada.
        holes: Furos a desenhar.
        pixels_per_mm: Calibração para converter raio em diâmetro (mm).
        nominal_diameter_mm: Diâmetro nominal usado na classificação.
        tolerance_mm: Tolerância em mm para classificar OK/NOK.

    Returns:
        Cópia da imagem com overlay dos furos.
    """
    result = rgb.copy()
    for hole in holes:
        diameter_mm = (hole.radius * 2) / pixels_per_mm
        color = COLOR_OK if abs(diameter_mm - nominal_diameter_mm) <= tolerance_mm else COLOR_FAIL
        _draw_single_hole(result, hole, color, diameter_mm)
    return result


def draw_filaments(
    rgb: np.ndarray,
    holes_with_filaments: list[bool],
    holes: list[Hole],
    segments: list[FilamentSegment],
) -> np.ndarray:
    """Desenha círculos verde/vermelho conforme presença de filamento e linhas magenta.

    Args:
        rgb: Imagem RGB de entrada.
        holes_with_filaments: Flag por furo (mesma ordem de `holes`).
        holes: Furos detectados (apenas para os contornos).
        segments: Filamentos detectados (em coordenadas globais).

    Returns:
        Cópia da imagem com overlay dos filamentos.
    """
    result = rgb.copy()
    for hole, has_filament in zip(holes, holes_with_filaments):
        color = COLOR_FAIL if has_filament else COLOR_OK
        thickness = 3 if has_filament else 2
        cv2.circle(result, hole.center, int(hole.radius), color, thickness)

    for segment in segments:
        cv2.line(result, segment.p1, segment.p2, COLOR_FILAMENT, 3)
    return result


def _draw_single_hole(
    canvas: np.ndarray,
    hole: Hole,
    color: tuple[int, int, int],
    diameter_mm: float,
) -> None:
    """Desenha um único furo (círculo + ponto central + rótulo de diâmetro)."""
    radius_px = int(hole.radius)
    thickness = max(2, radius_px // 30)
    cv2.circle(canvas, hole.center, radius_px, color, thickness)
    cv2.circle(canvas, hole.center, max(3, radius_px // 20), COLOR_CENTER_DOT, -1)

    text = f"{diameter_mm:.1f}"
    font_scale = max(0.5, radius_px / 60)
    font_thickness = max(1, radius_px // 40)
    text_pos = (hole.center[0] - radius_px // 3, hole.center[1] - radius_px - radius_px // 5)
    cv2.putText(
        canvas,
        text,
        text_pos,
        TEXT_FONT,
        font_scale,
        COLOR_TEXT_OUTLINE,
        font_thickness + 1,
        cv2.LINE_AA,
    )
