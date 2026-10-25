# Etapa 3 — Segmentação do defeito (conclusões)

Gerado por `etapa3_segmentacao.py` (código reutilizável em `pdi/segmentacao.py`).

## Protocolo

1. O modelo do "normal" é ajustado com as **150 normais de referência**.
2. Os limiares vêm das **40 normais de calibração**:
   - T_baixo = quantil 99% dos pixels;
   - T_alto = percentil 80 do pico de cada imagem normal.
3. A segmentação é por **crescimento de regiões**: sementes > T_alto, crescendo por vizinhos
   8-conectados > T_baixo.
4. A avaliação é nas **161 imagens de avaliação** (52 normais + 109 defeituosas), com Dice, IoU,
   precisão e recall pixel a pixel contra o GT.
5. **Ainda sem morfologia**, que é a etapa 4.

## As duas estratégias

- **A — modelo estatístico global.** Para cada pixel e canal, a média e o desvio das 150 normais;
  z = maior \|f − média\| / desvio entre os canais.
- **B — golden template com K = 5 vizinhos** (evolução da ideia do código antigo). Escolhem-se as
  5 normais mais parecidas com a imagem (menor diferença média numa versão reduzida). Pixel a pixel,
  usa-se a **menor** diferença entre a imagem e essas 5, dividida pelo desvio normal. A ideia: se o
  "500" ou o logo aparece igual em alguma vizinha, aquela região não acusa nada.

**Duas melhorias aplicadas às duas estratégias**, justificadas só com normais e com a etapa 1:
- **Tolerância a desalinhamento de 3 px.** O desvio de cada pixel passa a ser o maior desvio numa
  janela 7×7 (dilatação em tons de cinza, ou seja, filtro de máximo). Motivo: a cápsula gira até
  0,5°, o que desloca ~3 px nas pontas, e a translação não corrige isso.
- **Suavização 15×15 do mapa z.** Defeitos são áreas (o mais fino tem ~11 px); picos isolados de
  1–3 px não são defeito.

## Resultados (`resumo_por_config.csv`, `S1_*.png`, `S2_*.png`)

| Config | Descrição | Detecção | Dice | IoU | Precisão | Recall | Normais c/ alarme |
|---|---|---:|---:|---:|---:|---:|---:|
| base | golden template antigo (1 vizinho, sem normalizar) | 0,39 | 0,135 | 0,093 | 0,26 | 0,12 | 17% |
| A0 | A sem tolerância, suavização 5×5 | 0,39 | 0,205 | 0,155 | 0,25 | 0,21 | 13% |
| A | A completa, cor | **0,48** | 0,240 | 0,181 | 0,30 | 0,25 | 17% |
| A+S | A + canal Sobel | 0,47 | 0,252 | 0,192 | 0,32 | 0,26 | 23% |
| A+C | A + canal Canny | 0,44 | 0,239 | 0,180 | 0,32 | 0,23 | 23% |
| B | B completa, cor | 0,45 | 0,256 | 0,199 | 0,29 | **0,28** | 17% |
| **B+S** | B + canal Sobel | 0,46 | **0,264** | **0,205** | 0,30 | 0,28 | 23% |
| B+C | B + canal Canny | 0,44 | 0,248 | 0,188 | 0,32 | 0,25 | 23% |

Dice médio por imagem defeituosa. "Detecção" = defeito com pelo menos um pixel previsto dentro do GT.

**Por tipo (Dice | detecção):**

| Config | crack | faulty_imprint | poke | scratch | squeeze |
|---|---|---|---|---|---|
| base | 0,28 \| 0,70 | 0,00 \| 0,04 | 0,13 \| 0,33 | 0,00 \| 0,04 | 0,28 \| 0,90 |
| A | 0,45 \| 0,83 | 0,00 \| 0,00 | 0,35 \| 0,48 | 0,01 \| 0,13 | 0,41 \| 1,00 |
| B+S | 0,46 \| 0,74 | 0,00 \| 0,00 | 0,36 \| 0,43 | 0,09 \| 0,17 | 0,43 \| 1,00 |
| B+C | 0,36 \| 0,56 | 0,00 \| 0,00 | 0,35 \| 0,48 | **0,22 \| 0,56** | 0,31 \| 0,60 |

## O que isso mostra

1. **As duas estratégias novas são bem melhores que o código antigo.** O Dice dobra (0,135 → 0,26) e
   a detecção sobe de 39% para 45–48%. Só a tolerância a desalinhamento e a suavização (A0 → A) já
   levam a detecção de 0,39 para 0,48.
2. **B é só um pouco melhor que A** no Dice (0,256 × 0,240; global 0,394 × 0,355). A vantagem
   esperada (explicar o logo e a impressão, que mudam de posição) aparece só em parte. A escolha
   das vizinhas pela diferença média da imagem inteira nem sempre pega cápsulas com o logo na
   mesma posição. Na cápsula normal `train/good/006` (logo visível), as 5 vizinhas escolhidas não
   têm o logo, e ele continua com z alto nas duas estratégias. A máscara final não marca nada
   nesse caso, porque o pico fica abaixo do T_alto, mas a margem é pequena. **Melhoria possível:**
   escolher as vizinhas pela semelhança **local** (por exemplo, template matching só na região do
   logo/impressão).
3. **Cor e bordas se complementam, mas com um conflito.** O canal Canny faz o scratch passar de
   9–22% para 56% de detecção. Porém o limiar alto calibrado sobe muito (de ~9 para ~46), porque os
   picos de borda das normais dominam, e o squeeze cai de 100% para 60%. A saída natural é
   segmentar **cor e bordas separadamente, cada uma com o próprio limiar calibrado, e unir as
   máscaras**. Isso fica para a etapa 4.
4. **faulty_imprint não é detectado por nenhuma estratégia.** Dentro do defeito z ≈ 1–2, contra picos
   normais de ≈ 6–13. Uma falha na impressão "500" é menor que a variação normal da própria
   impressão (a cápsula gira e o "500" muda de posição). É uma limitação real da comparação pixel a
   pixel, e entra na discussão do relatório. Na classificação, atributos de textura e posição podem
   ajudar.
5. **As máscaras são parciais.** Precisão ≈ 0,30 e recall ≈ 0,28: marcamos só pedaços do defeito
   (bordas do poke, extremidades do squeeze). Na etapa 4, o fechamento e o preenchimento de região
   devem aumentar o recall; a abertura e a remoção de componentes pequenos devem reduzir o alarme
   nas normais.

## Configurações levadas para a etapa 4

- **Estratégia A: A+S.**
- **Estratégia B: B+S** (melhor Dice).
- Variante a testar: união com a máscara do canal Canny.

## Cuidado metodológico

Os limiares vêm só das normais de calibração. A comparação entre configurações é feita no conjunto
de avaliação: é o experimento que o enunciado pede ("comparar duas estratégias"), e não há defeitos
suficientes para separar mais um conjunto de validação. Por isso, os números da configuração
escolhida podem estar levemente otimistas. Isso deve ser declarado no relatório.
