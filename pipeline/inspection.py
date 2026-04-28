"""Orquestrador do pipeline completo de inspeção.

Encapsula a sequência detecção de furos → calibração → estatísticas →
detecção de filamentos → agregação. Os consumidores (UI Streamlit, script de
galeria, notebook) chamam apenas `run_inspection()` e `compose_overlays()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pipeline.draw import draw_filaments, draw_holes
from pipeline.filaments import FilamentParams, FilamentSegment, detect_filaments
from pipeline.holes import Hole, HoughParams, calibrate_pixels_per_mm, detect_holes
from pipeline.io import to_gray
from pipeline.metrics import HoleStats, aggregate_filaments, compute_hole_stats


@dataclass
class InspectionResult:
    """Resultado consolidado de uma inspeção em uma imagem.

    Attributes:
        rgb: Imagem RGB de entrada (referência, não cópia).
        holes: Furos detectados, em ordem de detecção.
        pixels_per_mm: Calibração obtida pela mediana robusta dos raios.
        calibration_count: Quantos furos entraram no cálculo da mediana.
        hole_stats: Estatísticas agregadas de diâmetro.
        segments_per_hole: Filamentos detectados por furo (mesma ordem de `holes`).
        filament_counts: Contagem de filamentos por furo.
        holes_with_filaments_count: Quantos furos apresentaram >=1 filamento.
        polarities: Polaridade detectada por furo ("light_on_dark", "dark_on_light"
            ou "n/a" quando o furo não teve filamento detectado).
        elapsed_seconds: Tempo total da inspeção.
    """

    rgb: np.ndarray
    holes: list[Hole]
    pixels_per_mm: float
    calibration_count: int
    hole_stats: HoleStats
    segments_per_hole: list[list[FilamentSegment]] = field(default_factory=list)
    filament_counts: list[int] = field(default_factory=list)
    holes_with_filaments_count: int = 0
    polarities: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_filaments(self) -> int:
        """Total de filamentos detectados na imagem."""
        return sum(self.filament_counts)

    @property
    def flat_segments(self) -> list[FilamentSegment]:
        """Lista plana de todos os filamentos detectados."""
        return [seg for segs in self.segments_per_hole for seg in segs]

    @property
    def holes_with_filaments(self) -> list[bool]:
        """Flag por furo indicando se há ao menos um filamento detectado."""
        return [count > 0 for count in self.filament_counts]


def run_inspection(
    rgb: np.ndarray,
    hole_params: HoughParams = HoughParams(),
    filament_params: Optional[FilamentParams] = FilamentParams(),
) -> InspectionResult:
    """Executa o pipeline completo de inspeção em uma imagem.

    Args:
        rgb: Imagem RGB.
        hole_params: Hiperparâmetros do estágio HoughCircles + refinamento.
        filament_params: Hiperparâmetros do detector de filamentos. Use `None`
            para pular a detecção de filamentos (apenas furos + calibração).

    Returns:
        Resultado consolidado com furos, filamentos e estatísticas.
    """
    started = time.time()

    holes = _detect_and_calibrate(rgb, hole_params)
    calibration = calibrate_pixels_per_mm(holes)
    stats = compute_hole_stats(calibration.holes, calibration.pixels_per_mm)
    segments_per_hole = _detect_filaments_for_all(rgb, calibration.holes, filament_params)
    counts, _, with_filaments_count, polarities = aggregate_filaments(segments_per_hole)

    return InspectionResult(
        rgb=rgb,
        holes=calibration.holes,
        pixels_per_mm=calibration.pixels_per_mm,
        calibration_count=calibration.calibration_count,
        hole_stats=stats,
        segments_per_hole=segments_per_hole,
        filament_counts=counts,
        holes_with_filaments_count=with_filaments_count,
        polarities=polarities,
        elapsed_seconds=time.time() - started,
    )


def compose_overlays(result: InspectionResult) -> tuple[np.ndarray, np.ndarray]:
    """Desenha as duas imagens de overlay padrão (furos e filamentos).

    Args:
        result: Resultado de `run_inspection`.

    Returns:
        Par (overlay_holes, overlay_filaments) prontos para exibir.
    """
    overlay_holes = draw_holes(result.rgb, result.holes, result.pixels_per_mm)
    overlay_filaments = draw_filaments(
        result.rgb,
        result.holes_with_filaments,
        result.holes,
        result.flat_segments,
    )
    return overlay_holes, overlay_filaments


def _detect_and_calibrate(rgb: np.ndarray, hole_params: HoughParams) -> list[Hole]:
    """Wrapper para detecção de furos (mantido pequeno por simetria de leitura)."""
    return detect_holes(rgb, hole_params)


def _detect_filaments_for_all(
    rgb: np.ndarray,
    holes: list[Hole],
    filament_params: Optional[FilamentParams],
) -> list[list[FilamentSegment]]:
    """Roda o detector de filamentos em cada furo, ou devolve listas vazias."""
    if filament_params is None or not holes:
        return [[] for _ in holes]
    gray = to_gray(rgb)
    return [detect_filaments(gray, hole, filament_params) for hole in holes]
