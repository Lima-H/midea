# Análise de Qualidade - Placas Midea

Ferramenta de visão computacional para inspeção automatizada de placas Midea. Detecta furos, verifica diâmetros e identifica rebarbas (fios/riscos) dentro dos furos.

## Funcionalidades

### 1. Detecção de Furos
- Detecção automática via HoughCircles + refinamento com fitEllipse e gradiente radial
- Calibração automática de px/mm pela mediana dos raios (todos os furos são 10mm nominais)
- Classificação por tolerância de diâmetro (±1mm)

### 2. Detecção de Rebarbas
- Pipeline baseado em detecção de linhas: Canny → HoughLinesP → merge de segmentos → filtro de borda
- Analisa cada furo individualmente com máscara circular
- Filtra artefatos de borda para evitar falsos positivos

### 3. Parâmetros Ajustáveis
- Sidebar com sliders para todos os parâmetros do pipeline
- Ajuste em tempo real de thresholds de Canny, HoughCircles, merge de linhas, etc.

## Como o Pipeline Funciona

1. **Pré-processamento:** GaussianBlur + MedianBlur na imagem em escala de cinza
2. **Detecção de furos:** HoughCircles → ROI + Canny + Threshold → contorno → fitEllipse → refinamento por gradiente radial
3. **Calibração:** Mediana dos raios detectados define a escala px/mm
4. **Detecção de rebarbas (por furo):**
   - Recorte ROI com máscara circular
   - GaussianBlur → Canny (com máscara interna a 85% do raio)
   - HoughLinesP → merge de segmentos próximos → filtro de linhas de borda
   - Linhas restantes = rebarbas detectadas

## Instalação

```bash
git clone git@github.com:Lima-H/midea.git
cd midea
pip install -r requirements.txt
```

## Como Rodar

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador. Faça upload de uma imagem (.jpg, .jpeg, .png, .heic) para ver a análise.

## Estrutura do Projeto

- `app.py` — Aplicação principal (pipeline + interface Streamlit)
- `requirements.txt` — Dependências Python

## Dependências

- `streamlit` — Interface web
- `opencv-python-headless` — Processamento de imagem
- `numpy` — Cálculos numéricos
- `pandas` — Tabelas de resultados
- `Pillow` + `pillow-heif` — Suporte a formatos de imagem (incluindo HEIC)
