# Análise de Qualidade - Placas Midea

Este projeto é uma ferramenta de visão computacional desenvolvida para automatizar a inspeção de qualidade de placas. O sistema analisa imagens para detectar furos, verificar suas dimensões e identificar imperfeições superficiais como rebarbas ou rugosidades.

## Funcionalidades

O sistema opera através de uma interface web interativa (Streamlit) e oferece duas análises principais simultâneas:

### 1. Análise de Furos
- **Contagem Automática:** Identifica e conta todos os furos presentes na imagem.
- **Verificação de Padrão:** Analisa a área e circularidade de cada furo.
- **Classificação:**
  - 🟢 **OK:** Furos dentro da tolerância de tamanho e formato.
  - 🔴 **DIFF:** Furos que apresentam desvios significativos (muito grandes, muito pequenos ou irregulares).

### 2. Detecção de Rebarbas e Rugosidade
- **Análise de Textura:** Utiliza algoritmos de processamento de imagem para detectar variações de textura que indicam rebarbas ou rugosidade excessiva.
- **Foco no Interior dos Furos:** O algoritmo é otimizado para ignorar as bordas naturais dos furos e focar no centro, onde a superfície deve ser lisa.
- **Mapa de Calor:** Gera uma visualização com mapa de calor (escala TURBO) para destacar as áreas com anomalias.
- **Contagem de Defeitos:** Contabiliza pontos de rugosidade detectados, diferenciando os que estão dentro dos furos (críticos) dos que estão na superfície geral.

## Como Funciona o Algoritmo

1.  **Pré-processamento:** A imagem é convertida para escala de cinza e recebe equalização de histograma (CLAHE) para melhorar o contraste local.
2.  **Detecção de Furos:**
    - Aplica binarização (Otsu) e operações morfológicas para isolar os furos.
    - Filtra contornos baseados em área mínima/máxima e circularidade.
3.  **Detecção de Rebarbas:**
    - Calcula a variância local da imagem para destacar texturas.
    - Aplica um limiar de sensibilidade para separar o fundo liso de áreas rugosas.
    - Verifica a intersecção entre as áreas rugosas e o centro dos furos detectados (ignorando as bordas para evitar falsos positivos).

## Pré-requisitos

Para rodar este projeto, você precisará do **Python 3.8+** instalado.

As principais bibliotecas utilizadas são:
- `streamlit`: Para a interface web.
- `opencv-python`: Para processamento de imagem.
- `numpy`: Para cálculos matemáticos e manipulação de arrays.
- `Pillow`: Para manipulação de imagens.

## Instalação

1.  Clone ou baixe este repositório.
2.  Instale as dependências necessárias executando o comando abaixo no seu terminal:

```bash
pip install streamlit opencv-python numpy Pillow
```

## Como Rodar

1.  Navegue até a pasta do projeto pelo terminal:
    ```bash
    cd /caminho/para/o/projeto/midea
    ```

2.  Recomendado: crie e ative um ambiente virtual (opcional, mas recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Instale as dependências a partir do arquivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  Rode a aplicação com Streamlit:
    ```bash
    streamlit run app.py
    ```

Alternativamente, você pode usar o script de conveniência `run_streamlit.sh` (macOS / Linux / Zsh):

```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

Depois de rodar, abra `http://localhost:8501` no navegador (o Streamlit geralmente abre automaticamente).

5.  Faça o upload de uma imagem da placa (formatos .jpg, .jpeg, .png) para ver a análise em tempo real.

## Como enviar para o GitHub (passo-a-passo)

1.  Inicialize um repositório Git local (se ainda não houver):
    ```bash
    git init
    git add .
    git commit -m "Initial commit - Análise de Placas Midea"
    ```

2.  Crie um repositório no GitHub usando a interface web (anote a URL do repositório, ex: `git@github.com:seu-usuario/midea.git`).

3.  Adicione o remoto e envie o código:
    ```bash
    git remote add origin git@github.com:SEUUSUARIO/midea.git
    git branch -M main
    git push -u origin main
    ```

4.  Depois disso, qualquer alteração pode ser enviada com `git add`, `git commit` e `git push`.

Observação: se usar HTTPS, substitua a URL SSH pela HTTPS (`https://github.com/SEUUSUARIO/midea.git`).

## Estrutura do Projeto

- `app.py`: Arquivo principal da aplicação web.
- `analise_furos.py`, `detectar_rebarbas.py`: Scripts auxiliares com lógicas de detecção.
- `utils/`: Pasta com funções utilitárias.
- `Imagens/`: Pasta com imagens de exemplo.

---
Desenvolvido para auxiliar no controle de qualidade visual.
