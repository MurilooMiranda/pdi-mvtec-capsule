# Legado — primeira versão da localização de defeitos

Primeira versão do pipeline (commit de 23/10/2026), mantida para referência e comparação. Ela foi
substituída pelas etapas 1–3 da raiz do repositório.

- `pipeline_etapa1.py`: bilateral → ROI (Otsu) → bordas (Canny/Sobel) + abertura → golden template
  (1 referência, alinhamento ECC).
- `avaliar_etapa1.py`: avaliação no `test/` inteiro; gera `resultados/avaliacao.txt`.
- `resultados/`: tabelas e figuras de exemplo dessa versão.

Única alteração feita na reorganização: o caminho do dataset (`DATASET_DIR`) passou a apontar para
`../dados/capsule`. Para rodar, entre na pasta `legado/`:

```
cd legado
python pipeline_etapa1.py --defeito crack --indice 0
```

**Por que foi substituída** (detalhes em `../resultados/etapa3_segmentacao/CONCLUSOES.md`):
- os parâmetros foram escolhidos no próprio `test/` (vazamento); não havia partição própria;
- o alinhamento por ECC e a máscara HSV não fazem parte do conteúdo das aulas;
- bordas brutas não separam defeito de estrutura normal (impressão, logo, reflexos);
- na partição nova, esta abordagem (recalibrada, configuração "base") tem Dice 0,135, contra 0,26
  das estratégias novas.
