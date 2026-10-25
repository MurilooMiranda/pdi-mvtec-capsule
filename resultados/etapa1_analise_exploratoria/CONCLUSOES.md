# Etapa 1 — Análise exploratória da categoria *capsule* (conclusões)

Gerado por `etapa1_analise_exploratoria.py`. Os números estão em `10_resumo.txt` e nos CSVs desta pasta.

## Dados

- **351 imagens** 1000×1000 RGB: 242 normais (219 `train/good` + 23 `test/good`) e 109 defeituosas,
  todas com máscara.
- Tipos: crack 23, faulty_imprint 22, poke 21, scratch 23, squeeze 20.

## O que os dados mostram e o que isso implica para o pipeline

| # | Observação (medida) | Figura/arquivo | Consequência para as próximas etapas |
|---|---|---|---|
| 1 | A cápsula quase não se move: deslocamento de −8 a +5 px em x e de −4 a +3 px em y (média 2,4 px); ângulo entre −0,35° e +0,18°; correlação com a média ≥ 0,994. | `08_alinhamento.*` | Basta alinhar por **translação com template matching** (visto em aula). O ECC/rotação do código antigo é desnecessário. |
| 2 | A variação entre cápsulas normais se concentra no **contorno**, na **divisão preta/laranja**, no **"500"** e no **logo "actavis"** (desvio de até ~25 níveis). No resto da cápsula o desvio é ~5 níveis. | `09_media_desvio_normais.png` | Comparar com uma referência exige normalizar pela variação normal de cada pixel (média e desvio das normais). Senão, o contorno e a impressão viram falso positivo. |
| 3 | O contraste do defeito com a vizinhança é **baixo**. Mediana de \|dentro − anel\|: scratch ≈ 4, faulty_imprint ≈ 6, poke ≈ 12, crack ≈ 13, squeeze ≈ 15 níveis, contra um desvio da vizinhança de 12–27. | `06_contraste.*` | Um **limiar simples na intensidade não separa** os defeitos. É preciso comparar com o normal (diferença com referência) e/ou usar textura. Scratch e faulty_imprint serão os mais difíceis. |
| 4 | Em cinza, a metade laranja vira ≈ 93; nos canais ela é R ≈ 133, G ≈ 78, B ≈ 65. A metade preta fica em ≈ 33 em todos os canais. | `07_histogramas_regioes.png` | A conversão para cinza descarta informação. Vale testar **R, G e B separados** contra o cinza. Nenhum canal sozinho é melhor para todos os tipos. |
| 5 | Os defeitos são pequenos: mediana de 0,2–0,6% da imagem (squeeze 2,8%). O menor tem 371 px, e o menor "eixo menor" é ≈ 11 px. | `03_defeitos_geometria.csv`, `04_area_por_tipo.png` | Na **abertura**, um elemento acima de ~7 px apaga os defeitos mais finos. Usar elemento pequeno (3–5 px) e remover componentes pequenos por área. |
| 6 | **Posição**: faulty_imprint fica sempre sobre o "500"; crack e squeeze ficam junto ao contorno (centro a ~33 px da borda); poke e scratch, na maioria na metade laranja. | `05_mapa_posicao_defeitos.png` | Não dá para descartar o contorno com uma erosão grande da ROI, porque perderíamos crack e squeeze. A posição (distância à borda, metade) é um bom **atributo** para separar os tipos. |
| 7 | O squeeze é uma **deformação do formato**, com pouca mudança de cor. A área da cápsula normal varia só de 214 mil a 229 mil px. | `08_alinhamento.csv` | Atributos de **forma da própria cápsula** (área, perímetro, convexidade da ROI) podem detectar o squeeze melhor que a diferença de intensidade. |

## Partição própria (`dados/particao.csv`)

Sorteio com semente 42, só entre as normais:

| Conjunto | Imagens | Uso |
|---|---|---|
| referência | 150 normais | construir o "modelo do normal" (média/desvio, template) |
| calibração | 40 normais | escolher limiares e parâmetros, sem ver nenhum defeito |
| avaliação | 52 normais + 109 defeituosas | medir segmentação (Dice/IoU) e classificar com validação cruzada |

**Regra:** nenhum parâmetro é escolhido olhando o conjunto de avaliação. Os números do
`legado/resultados/avaliacao.txt` (código antigo) foram calibrados no `test/` inteiro e precisam ser refeitos dentro dessa
regra.

## Ressalva

As observações 3, 5 e 6 usam as máscaras dos defeitos. Isso é aceitável numa análise
exploratória: serve para entender o problema e escolher *quais técnicas* testar. Os valores
numéricos dos parâmetros (limiares etc.) devem sair do conjunto de calibração.
