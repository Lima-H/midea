"""Aplicação Streamlit de inspeção de furos e filamentos.

Toda a lógica de pipeline vive em `pipeline/`. Este arquivo cuida só da
interface: lê os sliders, chama `run_inspection()` e compõe os painéis.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from pipeline.filaments import FilamentParams
from pipeline.holes import HOLE_NOMINAL_DIAMETER_MM, HOLE_TOLERANCE_MM, HoughParams
from pipeline.inspection import InspectionResult, compose_overlays, run_inspection
from pipeline.io import load_image
from pipeline.metrics import HoleStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

POLARITY_OPTIONS = ["auto", "light_on_dark", "dark_on_light"]


def _hole_params_from_sidebar() -> HoughParams:
    """Constrói HoughParams a partir dos sliders da sidebar."""
    return HoughParams(
        blur_ksize=st.session_state["hole_blur"],
        blur_sigma=3.0,
        median_ksize=st.session_state["hole_median"],
        canny_low=st.session_state["hole_canny_l"],
        canny_high=st.session_state["hole_canny_h"],
        hough_dp=st.session_state["hole_dp"],
        hough_param1=st.session_state["hole_p1"],
        hough_param2=st.session_state["hole_p2"],
        min_radius_px=st.session_state["hole_min_r"],
        max_radius_px=st.session_state["hole_max_r"],
        min_dist_px=st.session_state["hole_min_dist"],
    )


def _filament_params_from_sidebar() -> FilamentParams | None:
    """Lê os sliders de filamentos. Retorna None se a detecção está desativada."""
    if not st.session_state.get("enable_filaments", True):
        return None
    polarity_choice = st.session_state["filament_polarity"]
    force_polarity = None if polarity_choice == "auto" else polarity_choice
    return FilamentParams(
        clahe_clip_limit=st.session_state["filament_clahe"],
        high_pass_sigma=st.session_state["filament_highpass"],
        frangi_sigma_max=st.session_state["filament_sigma_max"],
        score_percentile=st.session_state["filament_score_pct"],
        min_length_ratio=st.session_state["filament_min_len"],
        min_vesselness=st.session_state["filament_min_vess"],
        inner_mask_ratio=st.session_state["filament_inner_mask"],
        force_polarity=force_polarity,
    )


def _render_sidebar() -> None:
    """Renderiza a sidebar com os parâmetros do pipeline."""
    with st.sidebar:
        st.header("Parâmetros do Pipeline")
        _render_hole_sliders()
        st.markdown("---")
        _render_filament_sliders()


def _render_hole_sliders() -> None:
    """Sliders da seção de detecção de furos."""
    st.subheader("Detecção de Furos")
    st.slider("Gaussian Blur kernel", 3, 31, 15, step=2, key="hole_blur")
    st.slider("Median Blur kernel", 3, 15, 7, step=2, key="hole_median")
    st.slider("Canny low (furos)", 10, 150, 70, key="hole_canny_l")
    st.slider("Canny high (furos)", 50, 300, 140, key="hole_canny_h")
    st.slider("Hough dp", 1.0, 3.0, 1.2, step=0.1, key="hole_dp")
    st.slider("Hough param1", 20, 200, 80, key="hole_p1")
    st.slider(
        "Hough param2", 10, 100, 30, key="hole_p2",
        help="Reduza (25-30) se faltam furos. Aumente se aparecem círculos espúrios.",
    )
    st.slider(
        "Distância mínima entre centros (px @ 4032px)", 80, 400, 140, step=10,
        key="hole_min_dist",
        help="Reduza (140-180) se furos adjacentes não são detectados.",
    )
    st.slider("Raio mínimo (px @ 4032px)", 50, 250, 140, step=5, key="hole_min_r")
    st.slider(
        "Raio máximo (px @ 4032px)", 150, 500, 160, step=5, key="hole_max_r",
        help="Reduza (160-220) para evitar que o anel externo seja confundido com o furo.",
    )


def _render_filament_sliders() -> None:
    """Sliders da seção de detecção de filamentos."""
    st.subheader("Detecção de Filamentos")
    st.checkbox("Ativar detecção de filamentos", value=True, key="enable_filaments")
    st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, step=0.5, key="filament_clahe")
    st.slider(
        "High-pass sigma (remove textura)", 0.0, 20.0, 8.0, step=1.0,
        key="filament_highpass",
        help="0 desativa. Valores 6-12 removem textura/baseline antes do Frangi.",
    )
    st.slider("Frangi sigma máximo", 1, 6, 3, key="filament_sigma_max")
    st.slider(
        "Percentil de score (threshold)", 95.0, 99.95, 97.0, step=0.05,
        key="filament_score_pct",
    )
    st.slider(
        "Comprimento mínimo (% diâmetro)", 0.05, 0.50, 0.15, step=0.05,
        key="filament_min_len",
    )
    st.slider(
        "Vesselness média mínima", 0.0, 0.50, 0.10, step=0.05,
        key="filament_min_vess",
    )
    st.slider(
        "Máscara interna (% raio)", 0.70, 0.99, 0.95, step=0.01,
        key="filament_inner_mask",
    )
    st.selectbox("Polaridade", POLARITY_OPTIONS, index=0, key="filament_polarity")


def _render_holes_panel(rgb_overlay, stats: HoleStats) -> None:
    """Painel da coluna 1: imagem com furos e métricas de diâmetro."""
    st.header("1. Análise de Furos")
    st.image(rgb_overlay, use_container_width=True)
    st.markdown(f"### Furos Detectados: {stats.total}")

    if stats.total == 0:
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Diâmetro Médio", f"{stats.mean_mm:.2f} mm")
        st.metric("Diâmetro Mediana", f"{stats.median_mm:.2f} mm")
        st.metric("Desvio Padrão", f"{stats.std_mm:.2f} mm")
    with col_b:
        st.metric("Mín / Máx", f"{stats.min_mm:.2f} / {stats.max_mm:.2f} mm")
        st.metric(
            f"Dentro Tolerância (±{HOLE_TOLERANCE_MM}mm)",
            f"{stats.within_tolerance_pct:.1f}%",
        )
        st.metric("Calibração", f"{stats.pixels_per_mm:.2f} px/mm")

    if stats.fail > 0:
        st.markdown(f"### Fora do padrão: {stats.fail}")
    else:
        st.markdown("### Todos dentro do padrão!")


def _render_filaments_panel(rgb_overlay, result: InspectionResult) -> None:
    """Painel da coluna 2: imagem com filamentos e contagem."""
    st.header("2. Detecção de Filamentos")
    st.image(rgb_overlay, use_container_width=True)
    if result.holes_with_filaments_count > 0:
        st.markdown(
            f"### Furos com filamento: "
            f"{result.holes_with_filaments_count}/{result.hole_stats.total}",
        )
    else:
        st.markdown("### Nenhum filamento detectado!")
    if result.polarities:
        st.caption(f"Polaridades detectadas: {', '.join(result.polarities)}")


def _render_detail_table(result: InspectionResult) -> None:
    """Tabela final com diâmetro e contagem de filamentos por furo."""
    st.markdown("---")
    st.header("Análise Detalhada")

    stats = result.hole_stats
    if stats.total == 0:
        st.info("Nenhum furo detectado")
        return

    diameters = stats.diameters_mm
    has_filament = result.holes_with_filaments
    df = pd.DataFrame({
        "Furo": [f"#{i + 1}" for i in range(len(diameters))],
        "Diâmetro (mm)": [f"{d:.2f}" for d in diameters],
        "Tolerância": [
            "OK" if abs(d - HOLE_NOMINAL_DIAMETER_MM) <= HOLE_TOLERANCE_MM else "NOK"
            for d in diameters
        ],
        "Filamentos": [
            f"{result.filament_counts[i]} fio(s)" if has_filament[i] else "OK"
            for i in range(len(diameters))
        ],
    })
    table_height = min(35 * len(diameters) + 40, 600)
    st.dataframe(df, use_container_width=True, height=table_height)
    total_with = result.holes_with_filaments_count
    st.markdown(
        f"**Resumo:**  \n"
        f"- Total: **{len(diameters)}** furos  \n"
        f"- Diâmetro OK: **{stats.ok}** | NOK: **{stats.fail}**  \n"
        f"- Sem filamento: **{len(diameters) - total_with}** | Com filamento: **{total_with}**",
    )


def _render_results(result: InspectionResult) -> None:
    """Compõe overlays e renderiza todos os painéis de resultado."""
    overlay_holes, overlay_filaments = compose_overlays(result)
    st.caption(
        f"Pipeline executado em {result.elapsed_seconds:.2f}s "
        f"(calibração com {result.calibration_count}/{result.hole_stats.total} furos)",
    )

    col1, col2 = st.columns(2)
    with col1:
        _render_holes_panel(overlay_holes, result.hole_stats)
    with col2:
        _render_filaments_panel(overlay_filaments, result)

    _render_detail_table(result)

    with st.expander("Clique para ver imagem em TAMANHO REAL (zoom)"):
        st.image(overlay_filaments, caption="Detecção de filamentos - Tamanho Real")
        st.image(overlay_holes, caption="Detecção de furos - Tamanho Real")


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    st.set_page_config(layout="wide", page_title="Análise de Placas Midea")
    st.title("Análise de Qualidade - Placas Midea")
    st.markdown("Faça upload da imagem da placa para analisar furos e filamentos.")
    _render_sidebar()

    uploaded_file = st.file_uploader(
        "Escolha uma imagem...", type=["jpg", "jpeg", "png", "heic", "HEIC"],
    )
    if uploaded_file is None:
        st.info("Por favor, faça o upload de uma imagem para começar.")
        return

    try:
        rgb = load_image(uploaded_file.getvalue())
    except Exception as exc:  # pragma: no cover - feedback para o usuário
        st.error(f"Erro ao carregar imagem: {exc}")
        return

    st.subheader("Imagem Original")
    st.image(rgb, use_container_width=True)

    result = run_inspection(
        rgb,
        hole_params=_hole_params_from_sidebar(),
        filament_params=_filament_params_from_sidebar(),
    )
    _render_results(result)


if __name__ == "__main__":
    main()
else:
    main()
