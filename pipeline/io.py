"""Carregamento e normalização de imagens de entrada.

Suporta os formatos JPG, PNG e HEIC (via pillow_heif). Lida com rotação EXIF
automaticamente e oferece downscale opcional para limitar a maior dimensão da
imagem (útil para acelerar o pipeline em fotos de celular ~4032 px).
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - depende do ambiente
    pillow_heif = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MAX_DIMENSION = 4000
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic")


def load_image(
    source: Path | str | bytes | BytesIO,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> np.ndarray:
    """Carrega uma imagem como array RGB normalizado.

    Args:
        source: Caminho do arquivo, bytes da imagem ou file-like (uploads do
            Streamlit). HEIC, JPG e PNG são suportados.
        max_dimension: Limite (px) da maior dimensão. Imagens maiores são
            reduzidas mantendo proporção. Valor 0 desativa o downscale.

    Returns:
        Array RGB com shape (H, W, 3) e dtype uint8.

    Raises:
        ValueError: Se a imagem não puder ser decodificada.
    """
    pil_image = _open_pil(source)
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    rgb = np.array(pil_image)
    if max_dimension > 0:
        rgb = _downscale(rgb, max_dimension)
    return rgb


def list_images(directory: Path | str) -> list[Path]:
    """Lista recursivamente todas as imagens suportadas em um diretório.

    Args:
        directory: Diretório raiz para varredura.

    Returns:
        Lista ordenada de caminhos absolutos.
    """
    root = Path(directory)
    if not root.exists():
        logger.warning("Diretório não encontrado: %s", root)
        return []

    paths: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in paths})


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Converte RGB para escala de cinza preservando dtype uint8."""
    if rgb.ndim == 2:
        return rgb.copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _open_pil(source: Path | str | bytes | BytesIO) -> Image.Image:
    """Resolve a fonte de entrada para um objeto PIL.Image."""
    if isinstance(source, (str, Path)):
        return Image.open(Path(source))
    if isinstance(source, bytes):
        return Image.open(BytesIO(source))
    return Image.open(source)


def _downscale(rgb: np.ndarray, max_dimension: int) -> np.ndarray:
    """Reduz a imagem se a maior dimensão exceder o limite."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return rgb

    scale = max_dimension / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
