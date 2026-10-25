# Etapa 2 — Pré-processamento (conclusões)

Gerado por `etapa2_preprocessamento.py` (código reutilizável em `pdi/preprocessamento.py`).
Números completos em `resumo.txt` e nos CSVs desta pasta.

**Regra usada:** as escolhas usam só cápsulas normais (referência e calibração). Os números
marcados como *descritivos* usam os defeitos do conjunto de avaliação apenas para mostrar o efeito
de cada opção; nenhuma escolha foi feita com eles.

## P1. Alinhamento por template matching (`P1_alinhamento.csv`)

| Alinhamento | \|f − média\| médio | p99 |
|---|---:|---:|
| nenhum | 5,42 | 36,5 |
| template matching (translação) | **5,05** | **32,7** |

- O deslocamento estimado vai de −9 a +21 px em x e de −5 a +4 px em y; a média é 2,7 px.
- Alinhar reduz a variação residual das normais (−11% no p99). Por isso fica no pipeline. O ECC do
  código antigo não é necessário e não foi visto em aula.

## P2. Suavização (`P2_P3_filtros_representacao.csv`, `F1_filtros_zoom.png`, `F2_ruido_vs_bordas.png`)

| Filtro | ruído residual p99 (normais) | preservação das bordas fortes |
|---|---:|---:|
| nenhum | 32,7 | 100% |
| **Gaussiano 5×5** | **31,7** | 93,7% |
| mediana 5×5 | 32,2 | 99,6% |
| bilateral (9, 40, 9) | 32,1 | 104,8% |

- **Escolha: Gaussiano.** É o de menor ruído residual entre os filtros que preservam ≥ 90% das
  bordas.
- **A diferença entre os filtros é pequena (≈ 3%).** O que mais varia entre cápsulas normais não é o
  ruído do sensor: é variação **estrutural** (impressão "500", logo, reflexos, borda). Nenhum
  filtro passa-baixa remove isso, e é por isso que a escolha do filtro pesa pouco no resultado.

## P3. Cinza × canais de cor (descritivo, `F3_representacao_por_tipo.png`)

Defeitos com ≥ 30 px acima do limiar (q99,9% das normais), filtro Gaussiano:

| Representação | crack | faulty_imprint | poke | scratch | squeeze |
|---|---:|---:|---:|---:|---:|
| cinza | 0,70 | 0,00 | 0,33 | 0,00 | 1,00 |
| R | 0,56 | 0,00 | 0,33 | 0,00 | 0,75 |
| G | 0,74 | 0,00 | 0,38 | 0,00 | 1,00 |
| B | 0,61 | 0,00 | 0,43 | 0,00 | 1,00 |
| **RGB (máximo dos 3)** | **0,74** | 0,00 | **0,43** | 0,00 | **1,00** |

- Nenhum canal sozinho é o melhor em todos os tipos (como a etapa 1 indicou).
- Usar os **3 canais e tomar o maior desvio** fica igual ou melhor que o cinza em todos os tipos.
  Por isso o pipeline usa B, G e R.
- Intensidade/cor sozinha **não enxerga** faulty_imprint nem scratch.

## P4. Bordas: Sobel × Canny (`P4_bordas.csv`)

- **Bordas brutas não servem como detector.** Uma cápsula normal tem, no interior, ~2.400 px de
  bordas de Canny e ~3.000 px de Sobel > 60 (impressão, logo, divisão, reflexos). Isso confirma o
  resultado do código antigo.
- **Bordas como canal de diferença em relação à referência funcionam.** Comparando o Sobel ou a
  densidade de bordas do Canny com o que é normal naquele pixel, os arranhões aparecem.
  Descritivo, defeitos detectados:

  | Canal | crack | faulty_imprint | poke | scratch | squeeze |
  |---|---:|---:|---:|---:|---:|
  | Sobel | 0,83 | 0,00 | 0,57 | 0,57 | 1,00 |
  | Canny (densidade) | 0,91 | 0,00 | 0,62 | **0,70** | 1,00 |

- **Conclusão:** cor e bordas são **complementares**. A cor pega poke e squeeze; as bordas pegam
  scratch e crack. A etapa 3 testa as combinações.

## P5. Região de interesse (descritivo, `P5_roi.csv`)

| ROI | área (px) | GT coberto (média / mínimo) |
|---|---:|---:|
| cápsula média com erosão 9 px (código antigo) | 213 mil | 0,956 / 0,42 |
| cápsula média | 221 mil | 0,972 / 0,60 |
| **união das 150 normais de referência + 10 px** | 257 mil | **0,995 / 0,86** |

- **Escolha:** a ROI usa só as normais e cobre a variação natural de posição e forma.
- A erosão do código antigo cortava parte de crack e squeeze, que ficam no contorno (etapa 1).

## Configuração levada para a etapa 3

Alinhamento por template matching, filtro Gaussiano 5×5, canais B/G/R (+ canal de bordas
opcional) e ROI = união das normais + 10 px.
