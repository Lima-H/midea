"""Agregação de estatísticas por imagem após o pipeline rodar.

Centraliza o cálculo de diâmetros, classificação OK/NOK e contagem de
filamentos para uso tanto na UI quanto no script de galeria.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.filaments import FilamentSegment
from pipeline.holes import HOLE_NOMINAL_DIAMETER_MM, HOLE_TOLERANCE_MM, Hole


@dataclass
class HoleStats:
    """Estatísticas agregadas de diâmetro dos furos.

    Attributes:
        total: Número total de furos válidos.
        ok: Quantidade dentro da tolerância.
        fail: Quantidade fora da tolerância.
        diameters_mm: Lista de diâmetros em mm (mesma ordem dos furos).
        pixels_per_mm: Calibração usada.
        mean_mm: Média dos diâmetros (se houver pelo menos um furo).
        median_mm: Mediana dos diâmetros.
        std_mm: Desvio padrão.
        min_mm: Menor diâmetro.
        max_mm: Maior diâmetro.
        within_tolerance_pct: Percentual dentro da tolerância.
    """

    total: int
    ok: int
    fail: int
    diameters_mm: list[float]
    pixels_per_mm: float
    mean_mm: float = 0.0
    median_mm: float = 0.0
    std_mm: float = 0.0
    min_mm: float = 0.0
    max_mm: float = 0.0
    within_tolerance_pct: float = 0.0


@dataclass
class ImageReport:
    """Relatório consolidado de uma imagem processada.

    Attributes:
        hole_stats: Estatísticas de diâmetro.
        filaments_per_hole: Contagem de filamentos por furo.
        total_filaments: Total de filamentos detectados na imagem.
        holes_with_filaments_count: Furos que apresentaram >=1 filamento.
        polarities: Polaridade detectada para cada furo (mesma ordem).
        elapsed_seconds: Tempo total de processamento.
    """

    hole_stats: HoleStats
    filaments_per_hole: list[int] = field(default_factory=list)
    total_filaments: int = 0
    holes_with_filaments_count: int = 0
    polarities: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def compute_hole_stats(
    holes: list[Hole],
    pixels_per_mm: float,
    nominal_diameter_mm: float = HOLE_NOMINAL_DIAMETER_MM,
    tolerance_mm: float = HOLE_TOLERANCE_MM,
) -> HoleStats:
    """Calcula estatísticas de diâmetro a partir dos furos válidos.

    Args:
        holes: Furos detectados e calibrados.
        pixels_per_mm: Fator de conversão.
        nominal_diameter_mm: Diâmetro nominal para classificação.
        tolerance_mm: Tolerância para OK/NOK.

    Returns:
        Stats agregadas. Campos derivados (mean, median etc.) ficam em zero
        quando não há furos.
    """
    if not holes:
        return HoleStats(
            total=0, ok=0, fail=0, diameters_mm=[], pixels_per_mm=pixels_per_mm,
        )

    diameters = [(hole.radius * 2) / pixels_per_mm for hole in holes]
    ok_count = sum(1 for d in diameters if abs(d - nominal_diameter_mm) <= tolerance_mm)
    fail_count = len(diameters) - ok_count

    return HoleStats(
        total=len(holes),
        ok=ok_count,
        fail=fail_count,
        diameters_mm=diameters,
        pixels_per_mm=pixels_per_mm,
        mean_mm=float(np.mean(diameters)),
        median_mm=float(np.median(diameters)),
        std_mm=float(np.std(diameters)),
        min_mm=float(np.min(diameters)),
        max_mm=float(np.max(diameters)),
        within_tolerance_pct=(ok_count / len(diameters)) * 100,
    )


def aggregate_filaments(
    segments_per_hole: list[list[FilamentSegment]],
) -> tuple[list[int], int, int, list[str]]:
    """Agrega contagens de filamentos por furo e a lista de polaridades.

    Args:
        segments_per_hole: Lista (mesma ordem dos furos) onde cada elemento é
            a lista de segmentos detectados naquele furo.

    Returns:
        Tupla (counts, total, holes_with_at_least_one, polarities).
    """
    counts = [len(segs) for segs in segments_per_hole]
    total = sum(counts)
    holes_with_filaments = sum(1 for c in counts if c > 0)
    polarities: list[str] = []
    for segs in segments_per_hole:
        polarities.append(segs[0].polarity if segs else "n/a")
    return counts, total, holes_with_filaments, polarities
