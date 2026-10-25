# Projeto 4 — Detecção e classificação de defeitos industriais (MVTec AD · *capsule*)

Disciplina de Processamento Digital de Imagens (UFSCar) — Prof. Dr. Ricardo J. Ferrari.
O enunciado está em [`docs/enunciado_projetos_pdi.pdf`](docs/enunciado_projetos_pdi.pdf).

**Questão de projeto:** é possível detectar e classificar defeitos usando apenas processamento clássico
e atributos interpretáveis?

```
Denoising → Bordas → Morfologia → Segmentação → Atributos → Classificação
```

> Uso didático: a partição é própria e os resultados **não** são benchmark oficial do MVTec AD.

## Plano e andamento

| Etapa | O que faz | Script | Resultados | Situação |
|---|---|---|---|---|
| 1 | Análise exploratória + partição própria | `etapa1_analise_exploratoria.py` | `resultados/etapa1_analise_exploratoria/` | ✅ |
| 2 | Pré-processamento (filtros, cor, bordas, alinhamento, ROI) | `etapa2_preprocessamento.py` | `resultados/etapa2_preprocessamento/` | ✅ |
| 3 | Segmentação — 2 estratégias (modelo estatístico × golden template) | `etapa3_segmentacao.py` | `resultados/etapa3_segmentacao/` | ✅ |
| 4 | Morfologia + avaliação final da segmentação | — | — | ⏳ |
| 5 | Extração de atributos (forma, cor, textura) | — | — | ⏳ |
| 6 | Classificação (SVM, Random Forest, k-NN) | — | — | ⏳ |
| 7 | Relatório | — | — | ⏳ |

Cada pasta de resultados tem um `CONCLUSOES.md` com os números e as decisões da etapa.

## Estrutura

```
pdi-mvtec-capsule/
├── README.md
├── requirements.txt
├── docs/
│   └── enunciado_projetos_pdi.pdf
├── dados/
│   ├── capsule/                  dataset MVTec AD (NÃO versionado — ver abaixo)
│   └── particao.csv              partição própria: referencia / calibracao / avaliacao
├── pdi/                          módulos reutilizáveis
│   ├── preprocessamento.py       leitura, filtros, Sobel/Canny, alinhamento, ROI
│   └── segmentacao.py            estratégias A e B, crescimento de regiões, métricas
├── etapa1_analise_exploratoria.py
├── etapa2_preprocessamento.py
├── etapa3_segmentacao.py
├── resultados/
│   ├── etapa1_analise_exploratoria/
│   ├── etapa2_preprocessamento/
│   └── etapa3_segmentacao/
└── legado/                       primeira versão (golden template + bordas), mantida como referência
```

## Como rodar

1. Baixe a categoria **capsule** do MVTec AD (https://www.mvtec.com/research-teaching/datasets/mvtec-ad,
   licença CC BY-NC-SA 4.0) e extraia em `dados/capsule/`. A pasta deve conter `train/`, `test/` e
   `ground_truth/`.
2. Crie o ambiente e instale as dependências:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Rode as etapas em ordem, a partir da raiz do repositório:
   ```
   python etapa1_analise_exploratoria.py     # ~2 min  -> gera dados/particao.csv
   python etapa2_preprocessamento.py         # ~3 min
   python etapa3_segmentacao.py              # ~5 min
   ```

A partição usa semente fixa (42), então os resultados são reproduzíveis.

## Regras adotadas

- **Partição:** 150 normais de **referência** (modelo do normal), 40 normais de **calibração**
  (limiares) e o restante, 52 normais + 109 defeituosas, para **avaliação**.
- **Nenhum limiar ou parâmetro** é escolhido olhando o conjunto de avaliação.
- **Só técnicas vistas em aula:** filtros Gaussiano/mediana/bilateral, Sobel, Canny, Otsu,
  template matching, crescimento de regiões, morfologia binária e momentos. Os classificadores são
  os pedidos no enunciado.
