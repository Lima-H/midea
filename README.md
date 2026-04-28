# Análise de Qualidade - Placas Midea

POC de visão computacional para inspecionar furos em peças industriais. Mede o
diâmetro de cada furo (~10mm nominais, tolerância ±1mm) e detecta filamentos
lineares dentro do furo — fios e rebarbas que cruzam o interior.

## Funcionalidades

- **Detecção de furos**: HoughCircles com refinamento por gradiente radial e
  ajuste de elipse. Calibração automática px/mm pela mediana robusta dos raios
  (rejeição de outliers via MAD).
- **Detecção de filamentos**: filtro de vesselness Frangi + black-hat morfológico
  orientado (varredura em 12 ângulos) + análise de componentes conexos do
  esqueleto. Detecta polaridade automaticamente, lidando com vista "fechada"
  (interior escuro com filamento claro) e vista "aberta" (interior claro
  iluminado com filamento escuro).
- **UI Streamlit**: sliders para todos os parâmetros, com avaliação em tempo real.

## Estrutura

```
midea/
├── app.py                       # UI Streamlit
├── pipeline/
│   ├── io.py                    # Carregamento HEIC/JPG/PNG, EXIF, downscale
│   ├── preprocess.py            # CLAHE, polaridade, ROI
│   ├── holes.py                 # Detecção de furos + calibração MAD
│   ├── filaments.py             # Frangi + black-hat orientado
│   ├── draw.py                  # Overlays de visualização
│   └── metrics.py               # Agregação de stats por imagem
├── scripts/
│   └── eval_gallery.py          # Galeria HTML qualitativa (sem labels)
├── data/                        # Imagens de entrada
└── requirements.txt
```

## Instalação

```bash
git clone git@github.com:Lima-H/midea.git
cd midea
pip install -r requirements.txt
```

> A versão `opencv-contrib-python-headless` é necessária para
> `cv2.ximgproc.thinning` (esqueletização Guo-Hall). `scikit-image` traz o
> filtro de vesselness Frangi.

## Como rodar

### UI interativa

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`. Faça upload de uma imagem (`.jpg`, `.jpeg`,
`.png`, `.heic`).

### Galeria qualitativa em todo o dataset

```bash
python scripts/eval_gallery.py --input data --output eval_outputs
open eval_outputs/gallery.html
```

Gera 3 thumbnails por imagem (original, furos, filamentos) em uma única página
HTML para revisão visual rápida.

## Pipeline (resumo técnico)

1. **Carregamento**: HEIC/JPG/PNG com correção EXIF e downscale opcional para
   máx. 2000 px no maior lado.
2. **CLAHE** global (`clipLimit=2.0`, `tileGridSize=8×8`) antes do HoughCircles.
3. **HoughCircles** + ROI + Canny + threshold combinados → contornos →
   `fitEllipse` → refinamento radial multi-raio com filtro IQR.
4. **Calibração**: mediana dos raios após rejeição de outliers via MAD
   (`|r − mediana| ≤ 3 × 1.4826 × MAD`). Assume todos os furos com diâmetro
   nominal igual.
5. **Por furo, detecta filamentos**:
   1. Recorta ROI (margem 10%) e aplica CLAHE local.
   2. Detecta polaridade comparando mediana(interior) vs mediana(anel externo),
      mascarando saturação. Inverte a ROI se necessário para sempre tratar
      "estruturas claras sobre fundo escuro".
   3. Calcula vesselness Frangi (`sigmas=range(1, sigma_max+1)`).
   4. Calcula black-hat com kernel linear varrendo orientações [0°, 180°)
      em passos de 15°.
   5. Funde os mapas (60% Frangi + 40% black-hat) e binariza no
      percentil configurável (default 99.5).
   6. Esqueletiza (Guo-Hall) e separa componentes conexos.
   7. Filtros geométricos: comprimento, retidão, vesselness média e anti-borda
      (rejeita componentes inteiramente periféricos com orientação tangencial).

## Dependências

- `streamlit` — interface web
- `opencv-contrib-python-headless` — processamento de imagem + ximgproc
- `scikit-image` — filtros de vesselness (Frangi)
- `numpy`, `pandas` — dados
- `Pillow` + `pillow-heif` — formatos de imagem (incluindo HEIC)
