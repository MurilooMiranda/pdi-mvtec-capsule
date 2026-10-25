"""
Projeto 4 (PDI - UFSCar) - ETAPA 3: comparação das estratégias de segmentação

Configurações comparadas (todas com o filtro escolhido na etapa 2):
  base   golden template do código antigo: 1 vizinho, sem normalização (cor B, G, R)
  A0     modelo estatístico global, cor, SEM tolerância a desalinhamento e suavização 5x5
  A      modelo estatístico global, cor (B, G, R), tolerância 3 px e suavização 15x15
  A+S    idem A + canal de bordas Sobel
  A+C    idem A + canal de bordas Canny (densidade local)
  B      golden template com 5 vizinhos, cor (B, G, R), normalizado, tolerância 3 px, 15x15
  B+S    idem B + Sobel
  B+C    idem B + Canny

Protocolo:
  1. modelo ajustado com as 150 normais de REFERÊNCIA;
  2. limiares (T_baixo, T_alto) calibrados com as 40 normais de CALIBRAÇÃO;
  3. segmentação por crescimento de regiões nas 161 imagens de AVALIAÇÃO (52 normais +
     109 defeituosas), SEM morfologia (a morfologia é a etapa 4);
  4. máscara levada de volta ao referencial original e comparada com o GT.

Métricas por tipo: Dice, IoU, precisão, recall (pixel a pixel), fração de defeitos detectados
(algum pixel previsto dentro do GT), e nas normais: fração com falso alarme e pixels FP.

Saídas em resultados/etapa3_segmentacao/.   Uso: python etapa3_segmentacao.py (~5 min)
"""

from __future__ import annotations

import csv
import gc
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pdi import preprocessamento as pp  # noqa: E402
from pdi import segmentacao as sg  # noqa: E402

FILTRO = "gaussiano"     # escolhido na etapa 2 (resultados/etapa2_preprocessamento/CONCLUSOES.md)
SAIDA = pp.RAIZ / "resultados" / "etapa3_segmentacao"
SAIDA.mkdir(parents=True, exist_ok=True)
CONFIGS = {
    "base": lambda: sg.ModeloVizinhos(FILTRO, None, k=1, normalizar=False, tolerancia=0, suav=5),
    "A0": lambda: sg.ModeloGlobal(FILTRO, None, tolerancia=0, suav=5),
    "A": lambda: sg.ModeloGlobal(FILTRO, None),
    "A+S": lambda: sg.ModeloGlobal(FILTRO, "sobel"),
    "A+C": lambda: sg.ModeloGlobal(FILTRO, "canny"),
    "B": lambda: sg.ModeloVizinhos(FILTRO, None, k=5),
    "B+S": lambda: sg.ModeloVizinhos(FILTRO, "sobel", k=5),
    "B+C": lambda: sg.ModeloVizinhos(FILTRO, "canny", k=5),
}
CORES = {"base": "#52514e", "A0": "#c9c8c2", "A": "#9cc2ef", "A+S": "#2a78d6", "A+C": "#0d4a96",
         "B": "#f3b08f", "B+S": "#eb6834", "B+C": "#a8401a"}
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
resumo: list[str] = []


def log(msg=""):
    print(msg, flush=True)
    resumo.append(msg)


def salvar_csv(nome, linhas):
    with open(SAIDA / nome, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


dados = pp.carregar_dados()
REF, CAL, AVA = (dados.do_conjunto(c) for c in ("referencia", "calibracao", "avaliacao"))
ROI = dados.roi
TIPOS = ["good"] + pp.TIPOS_DEFEITO

por_imagem, resumo_cfg, limiares, EXEMPLOS = [], [], {}, {}
ex_ids = {t: next(a.caminho for a in AVA if a.tipo == t) for t in TIPOS}
for nome, fabrica in CONFIGS.items():
    modelo = fabrica().ajustar([a.img for a in REF], ROI)
    tb, ta = sg.calibrar([modelo.mapa(a.img) for a in CAL], ROI)
    limiares[nome] = dict(t_baixo=round(tb, 3), t_alto=round(ta, 3))
    linhas = []
    for a in AVA:
        z = modelo.mapa(a.img)
        pred_al = sg.crescimento_regioes(z, tb, ta)
        pred = pp.mascara_volta(pred_al, *a.desloc)
        m = sg.metricas(pred, a.gt)
        linha = dict(config=nome, imagem=a.caminho, tipo=a.tipo, detectou=int(m["tp"] > 0),
                     marcou_algo=int(pred.any()),
                     n_componentes=cv2.connectedComponents(pred.astype(np.uint8))[0] - 1,
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        linhas.append(linha)
        if a.caminho in ex_ids.values():
            EXEMPLOS[(nome, a.tipo)] = (z, pred_al, a)
    por_imagem += linhas
    # resumo da configuração
    lin = dict(config=nome, descricao=modelo.nome + (f" + {modelo.bordas}" if modelo.bordas else "")
               + f" | tol={modelo.tolerancia} suav={modelo.suav}",
               t_baixo=round(tb, 2), t_alto=round(ta, 2))
    normais = [l for l in linhas if l["tipo"] == "good"]
    defeit = [l for l in linhas if l["tipo"] != "good"]
    lin["fp_normais_frac"] = round(np.mean([l["marcou_algo"] for l in normais]), 3)
    lin["fp_normais_px"] = round(np.mean([l["fp"] for l in normais]), 1)
    lin["deteccao"] = round(np.mean([l["detectou"] for l in defeit]), 3)
    for met in ("dice", "iou", "precisao", "recall"):
        lin[met] = round(np.mean([l[met] for l in defeit]), 3)
    tp, fp, fn = (sum(l[k] for l in defeit) for k in ("tp", "fp", "fn"))
    lin["dice_global"] = round(2 * tp / (2 * tp + fp + fn), 3)
    lin["precisao_global"] = round(tp / max(tp + fp, 1), 3)
    for t in pp.TIPOS_DEFEITO:
        lt = [l for l in defeit if l["tipo"] == t]
        lin[f"dice_{t}"] = round(np.mean([l["dice"] for l in lt]), 3)
        lin[f"detec_{t}"] = round(np.mean([l["detectou"] for l in lt]), 3)
    resumo_cfg.append(lin)
    log(f"{nome:<5} T=({tb:.2f}, {ta:.2f}) | detecção {lin['deteccao']:.2f} | Dice {lin['dice']:.3f} "
        f"(global {lin['dice_global']:.3f}) | prec {lin['precisao']:.2f} rec {lin['recall']:.2f} | "
        f"normais c/ alarme {lin['fp_normais_frac']:.2f} ({lin['fp_normais_px']:.0f} px)")
    del modelo
    gc.collect()

salvar_csv("segmentacao_por_imagem.csv", por_imagem)
salvar_csv("resumo_por_config.csv", resumo_cfg)
json.dump(dict(filtro=FILTRO, limiares=limiares), open(SAIDA / "limiares.json", "w"), indent=2)

log("\nDice médio por tipo | fração de defeitos detectados")
log(f"{'config':<6}" + "".join(f"{t:>20}" for t in pp.TIPOS_DEFEITO))
for lin in resumo_cfg:
    log(f"{lin['config']:<6}" + "".join(f"{lin[f'dice_{t}']:>12.3f} | {lin[f'detec_{t}']:.2f}"
                                       for t in pp.TIPOS_DEFEITO))

# =============================================================================
# FIGURAS
# =============================================================================
# S1: Dice e detecção por tipo
fig, ax = plt.subplots(1, 2, figsize=(15, 3.9))
w = 0.8 / len(CONFIGS)
for i, nome in enumerate(CONFIGS):
    lin = next(l for l in resumo_cfg if l["config"] == nome)
    for j, met in enumerate(("dice", "detec")):
        v = [lin[f"{met}_{t}"] for t in pp.TIPOS_DEFEITO]
        b = ax[j].bar(np.arange(5) + (i - len(CONFIGS) / 2 + 0.5) * w, v, w * 0.92,
                      color=CORES[nome], label=f"{nome}")
ax[0].set_title("Dice médio por tipo (sem morfologia)")
ax[1].set_title("Fração de defeitos detectados (algum pixel dentro do GT)")
for a in ax:
    a.set_xticks(range(5), pp.TIPOS_DEFEITO)
    a.set_ylim(0, 1.05)
ax[1].legend(frameon=False, ncol=5, loc="upper left")
fig.tight_layout()
fig.savefig(SAIDA / "S1_dice_deteccao_por_tipo.png", dpi=120)
plt.close(fig)

# S2: compromisso detecção x falso alarme nas normais
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for lin in resumo_cfg:
    ax.scatter(lin["fp_normais_frac"], lin["deteccao"], s=110, color=CORES[lin["config"]], zorder=3)
    ax.annotate(f"{lin['config']}  (Dice {lin['dice']:.2f})", (lin["fp_normais_frac"], lin["deteccao"]),
                xytext=(7, -3), textcoords="offset points")
ax.set_xlabel("cápsulas NORMAIS com falso alarme")
ax.set_ylabel("defeitos detectados")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title("Detecção × falso alarme (avaliação)")
fig.tight_layout()
fig.savefig(SAIDA / "S2_deteccao_vs_falso_alarme.png", dpi=120)
plt.close(fig)

# S3: exemplos (uma imagem por tipo): GT, z e máscara de base, A+S e B+S
melhor = lambda pref: max((l for l in resumo_cfg if l["config"].startswith(pref) and l["config"] != "A0"),
                          key=lambda l: l["dice"])["config"]
mostrar = ["base", melhor("A"), melhor("B")]
fig, ax = plt.subplots(len(TIPOS), 1 + 2 * len(mostrar), figsize=(22, 2.9 * len(TIPOS)))
crop = (slice(230, 770), slice(60, 940))
for i, t in enumerate(TIPOS):
    z0, _, a = EXEMPLOS[(mostrar[0], t)]
    vis = cv2.cvtColor(a.img, cv2.COLOR_BGR2RGB).copy()
    gt_al = pp.transladar(a.gt.astype(np.uint8), *a.desloc, borda=cv2.BORDER_CONSTANT)
    cont, _ = cv2.findContours(gt_al, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cont, -1, (0, 220, 0), 3)
    ax[i, 0].imshow(vis[crop])
    ax[i, 0].set_title(f"{a.caminho} (GT verde)", fontsize=8)
    for j, nome in enumerate(mostrar):
        z, pred, _ = EXEMPLOS[(nome, t)]
        ta = limiares[nome]["t_alto"]
        ax[i, 1 + 2 * j].imshow(z[crop], cmap="inferno", vmin=0, vmax=ta * 1.5)
        ax[i, 1 + 2 * j].set_title(f"z — {nome}", fontsize=8)
        ov = vis.copy()
        ov[pred] = (0.4 * ov[pred] + 0.6 * np.array([235, 104, 52])).astype(np.uint8)
        ax[i, 2 + 2 * j].imshow(ov[crop])
        ax[i, 2 + 2 * j].set_title(f"máscara — {nome}", fontsize=8)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("Exemplos: mapa de anomalia z e máscara do crescimento de regiões (laranja) — sem morfologia")
fig.tight_layout()
fig.savefig(SAIDA / "S3_exemplos.png", dpi=90)
plt.close(fig)

(SAIDA / "resumo.txt").write_text("\n".join(resumo), encoding="utf-8")
print(f"\nArquivos em {SAIDA}")
