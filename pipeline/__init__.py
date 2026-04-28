"""Pacote do pipeline de inspeção de furos.

Módulos públicos:
    - io: carregamento e normalização de imagens (HEIC/JPG/PNG, EXIF, downscale).
    - preprocess: CLAHE, detecção de polaridade, recorte de ROI.
    - holes: detecção de furos, refinamento de borda e calibração px/mm.
    - filaments: detector de filamentos (Frangi + black-hat orientado + FLD).
    - draw: composição de overlays para visualização.
    - metrics: agregação de estatísticas por imagem.
    - inspection: orquestrador do pipeline completo (entry point recomendado).
"""

from pipeline import draw, filaments, holes, inspection, io, metrics, preprocess

__all__ = ["draw", "filaments", "holes", "inspection", "io", "metrics", "preprocess"]
