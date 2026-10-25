"""
Projeto 4 (PDI - UFSCar) - ETAPA 2: avaliação do pré-processamento

Perguntas e como cada uma é respondida:
  P1. Alinhar ajuda?            -> variação residual |f - média| das normais de calibração,
                                   com e sem alinhamento por template matching.
  P2. Qual suavização usar?     -> para cada filtro (nenhum, Gaussiano, mediana, bilateral):
                                   ruído residual nas normais de calibração (quanto menor, menos
                                   falso positivo) x preservação das bordas fortes (quanto o
                                   gradiente de Sobel nas bordas reais da cápsula se mantém).
                                   REGRA DE ESCOLHA (só normais): menor ruído residual entre os
                                   filtros que preservam >= 90% da força das bordas.
  P3. Cinza ou cor?             -> mesmo ruído residual por canal + (descritivo) quanto do defeito
                                   passa do limiar calibrado nas normais, por tipo.
  P4. Sobel ou Canny?           -> bordas brutas nas normais (o que o código antigo fazia) e
                                   bordas como canal de diferença em relação à referência.
  P5. Qual ROI?                 -> cobertura das máscaras GT e área da ROI.

Medidas "descritivas" usam os defeitos do conjunto de avaliação SÓ para mostrar o efeito;
nenhuma escolha de parâmetro é feita com elas (ver CONCLUSOES.md).

Saídas em resultados/etapa2_preprocessamento/.   Uso: python etapa2_preprocessamento.py (~3 min)
"""

from __future__ import annotations

import csv

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pdi import preprocessamento as pp  # noqa: E402

SAIDA = pp.RAIZ / "resultados" / "etapa2_preprocessamento"
SAIDA.mkdir(parents=True, exist_ok=True)
REPRESENTACOES = ("cinza", "R", "G", "B")
PRESERVACAO_MIN = 0.90
Q_ALARME = 0.999          # limiar = quantil 99,9% de z nas normais de calibração
MIN_PX_DETECCAO = 30
CORES = {"nenhum": "#52514e", "gaussiano": "#2a78d6", "mediana": "#1baf7a", "bilateral": "#eb6834",
         "cinza": "#52514e", "R": "#e34948", "G": "#1baf7a", "B": "#2a78d6", "rgb (máx.)": "#6250d6",
         "sobel": "#eda100", "canny": "#6250d6", "cinza + sobel": "#eb6834"}
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


class Estatistica:
    """Média e desvio padrão pixel a pixel acumulados (sem guardar todas as imagens)."""

    def __init__(self):
        self.n, self.s, self.s2 = 0, None, None

    def add(self, x):
        x = x.astype(np.float64)
        if self.s is None:
            self.s, self.s2 = np.zeros_like(x), np.zeros_like(x)
        self.n += 1
        self.s += x
        self.s2 += x * x

    def media_desvio(self, roi):
        mu = self.s / self.n
        sd = np.sqrt(np.maximum(self.s2 / self.n - mu * mu, 0))
        piso = np.array([max(np.percentile(sd[..., c][roi], 25), 1.0) for c in range(sd.shape[-1])])
        return mu.astype(np.float32), np.maximum(sd, piso).astype(np.float32)


# =============================================================================
dados = pp.carregar_dados()
ROI = dados.roi
REF, CAL, AVA = (dados.do_conjunto(c) for c in ("referencia", "calibracao", "avaliacao"))
DEF = [a for a in AVA if a.defeituosa]
GT_AL = {a.caminho: pp.transladar(a.gt.astype(np.uint8), *a.desloc, borda=cv2.BORDER_CONSTANT) > 0
         for a in DEF}   # GT levado ao referencial alinhado


def recall_descritivo(z_de, limiar):
    """Para cada defeito: fração dos pixels do GT com z > limiar e se >= 30 px passaram."""
    out = {}
    for a in DEF:
        z = z_de(a)
        acima = (z > limiar) & GT_AL[a.caminho]
        out.setdefault(a.tipo, []).append((acima.sum() / GT_AL[a.caminho].sum(),
                                           acima.sum() >= MIN_PX_DETECCAO))
    return {t: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for t, vs in out.items()}


# =============================================================================
# P1. ALINHAMENTO
# =============================================================================
log("=== P1. Alinhamento por template matching ===")
desl = np.array([a.desloc for a in dados.amostras])
log(f"deslocamentos estimados: dx {desl[:, 0].min()}..{desl[:, 0].max()} px, "
    f"dy {desl[:, 1].min()}..{desl[:, 1].max()} px (média |d| = {np.hypot(*desl.T).mean():.1f} px)")
linhas_al = []
for alinhado in (False, True):
    est = Estatistica()
    for a in REF:
        img = a.img if alinhado else pp.transladar(a.img, *a.desloc, inverso=True)
        est.add(pp.canais(img, "cinza"))
    mu, _ = est.media_desvio(ROI)
    res = []
    for a in CAL:
        img = a.img if alinhado else pp.transladar(a.img, *a.desloc, inverso=True)
        res.append(np.abs(pp.canais(img, "cinza").astype(np.float32) - mu)[ROI])
    res = np.concatenate(res)
    linhas_al.append(dict(alinhamento="template matching" if alinhado else "nenhum",
                          residuo_medio=round(float(res.mean()), 2),
                          residuo_p99=round(float(np.percentile(res, 99)), 1)))
    log(f"{linhas_al[-1]['alinhamento']:<18} |f - média| nas normais de calibração: "
        f"média {res.mean():.2f}  p99 {np.percentile(res, 99):.1f} níveis")
salvar_csv("P1_alinhamento.csv", linhas_al)

# =============================================================================
# P2 + P3. FILTROS x REPRESENTAÇÃO
# =============================================================================
log("\n=== P2/P3. Suavização x representação (ruído nas normais e preservação de bordas) ===")
# bordas fortes "verdadeiras": 2% maiores gradientes da média SEM filtro dentro da cápsula
media_bruta = np.mean([pp.canais(a.img, "cinza")[..., 0].astype(np.float32) for a in REF], axis=0)
mag_media = pp.sobel_magnitude(media_bruta)
BORDAS_FORTES = dados.roi_media & (mag_media > np.percentile(mag_media[dados.roi_media], 98))

linhas_f, MODELOS = [], {}
for filtro in pp.FILTROS:
    est = Estatistica()
    for a in REF:
        f = pp.suavizar(a.img, filtro)
        cinza = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        est.add(np.dstack([cinza, f[..., 2], f[..., 1], f[..., 0]]))   # cinza, R, G, B
    mu, sd = est.media_desvio(ROI)
    MODELOS[filtro] = (mu, sd)
    res = {r: [] for r in REPRESENTACOES}
    zcal = {r: [] for r in REPRESENTACOES + ("rgb (máx.)",)}
    preserv = []
    for a in CAL:
        f = pp.suavizar(a.img, filtro)
        cinza = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x = np.dstack([cinza, f[..., 2], f[..., 1], f[..., 0]]).astype(np.float32)
        d = np.abs(x - mu)
        z = d / sd
        for k, r in enumerate(REPRESENTACOES):
            res[r].append(d[..., k][ROI])
            zcal[r].append(z[..., k][ROI])
        zcal["rgb (máx.)"].append(z[..., 1:].max(-1)[ROI])
        bruto = cv2.cvtColor(a.img, cv2.COLOR_BGR2GRAY)
        preserv.append(pp.sobel_magnitude(cinza)[BORDAS_FORTES].mean() /
                       pp.sobel_magnitude(bruto)[BORDAS_FORTES].mean())
    limiares = {r: float(np.quantile(np.concatenate(v), Q_ALARME)) for r, v in zcal.items()}
    # descritivo: quanto de cada defeito passa do limiar calibrado (uma passada por defeito)
    reps = REPRESENTACOES + ("rgb (máx.)",)
    acum = {r: {} for r in reps}
    for a in DEF:
        f = pp.suavizar(a.img, filtro)
        cinza = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        z = np.abs(np.dstack([cinza, f[..., 2], f[..., 1], f[..., 0]]).astype(np.float32) - mu) / sd
        g = GT_AL[a.caminho]
        for k, r in enumerate(reps):
            zz = z[..., k] if k < 4 else z[..., 1:].max(-1)
            n = int(((zz > limiares[r]) & g).sum())
            acum[r].setdefault(a.tipo, []).append((n / g.sum(), n >= MIN_PX_DETECCAO))
    for r in reps:
        rec = {t: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for t, vs in acum[r].items()}
        linha = dict(filtro=filtro, representacao=r,
                     residuo_medio=round(float(np.concatenate(res[r]).mean()), 2) if r in res else "",
                     residuo_p99=round(float(np.percentile(np.concatenate(res[r]), 99)), 1) if r in res else "",
                     preservacao_bordas=round(float(np.mean(preserv)), 3),
                     limiar_z=round(limiares[r], 2))
        for t in pp.TIPOS_DEFEITO:
            linha[f"frac_gt_{t}"] = round(rec[t][0], 3)
            linha[f"detec_{t}"] = round(rec[t][1], 3)
        linha["frac_gt_media"] = round(np.mean([rec[t][0] for t in pp.TIPOS_DEFEITO]), 3)
        linha["detec_media"] = round(np.mean([rec[t][1] for t in pp.TIPOS_DEFEITO]), 3)
        linhas_f.append(linha)
    c = next(l for l in linhas_f if l["filtro"] == filtro and l["representacao"] == "cinza")
    log(f"{filtro:<10} ruído (cinza) média {c['residuo_medio']:5.2f} p99 {c['residuo_p99']:5.1f} | "
        f"preservação das bordas {c['preservacao_bordas']:.1%}")
salvar_csv("P2_P3_filtros_representacao.csv", linhas_f)

# --- escolha do filtro (só com normais) -------------------------------------------
cinza_f = [l for l in linhas_f if l["representacao"] == "cinza"]
aptos = [l for l in cinza_f if l["preservacao_bordas"] >= PRESERVACAO_MIN]
ESCOLHIDO = min(aptos, key=lambda l: l["residuo_p99"])["filtro"]
log(f"\nREGRA: menor ruído p99 entre os filtros com preservação >= {PRESERVACAO_MIN:.0%} "
    f"-> filtro escolhido: {ESCOLHIDO.upper()}")

log("\nDescritivo (defeitos da avaliação): fração do GT acima do limiar calibrado | "
    "defeitos com >= 30 px acima")
log(f"{'filtro':<10}{'repr.':<11}" + "".join(f"{t[:9]:>16}" for t in pp.TIPOS_DEFEITO) + f"{'média':>14}")
for l in linhas_f:
    log(f"{l['filtro']:<10}{l['representacao']:<11}" +
        "".join(f"{l[f'frac_gt_{t}']:>9.2f} | {l[f'detec_{t}']:>4.2f}" for t in pp.TIPOS_DEFEITO) +
        f"{l['frac_gt_media']:>7.2f} | {l['detec_media']:.2f}")

# =============================================================================
# P4. BORDAS: SOBEL x CANNY
# =============================================================================
log("\n=== P4. Bordas ===")
# (a) bordas brutas nas normais de calibração (dentro da cápsula sem o contorno)
roi_int = cv2.erode(dados.roi_media.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
brutas = {"Canny 15/40": [], "Sobel |G|>60": []}
for a in CAL:
    cinza = cv2.cvtColor(pp.suavizar(a.img, ESCOLHIDO), cv2.COLOR_BGR2GRAY)
    brutas["Canny 15/40"].append(int((pp.canny(cinza) > 0)[roi_int].sum()))
    brutas["Sobel |G|>60"].append(int((pp.sobel_magnitude(cinza) > 60)[roi_int].sum()))
for k, v in brutas.items():
    log(f"{k:<14} pixels de borda por cápsula NORMAL (interior): média {np.mean(v):.0f}  "
        f"-> todos seriam falso positivo se borda = defeito")

# (b) bordas como canal de diferença com a referência
linhas_b = []
for tipo_b in ("sobel", "canny"):
    est = Estatistica()
    for a in REF:
        est.add(pp.canal_bordas(pp.suavizar(a.img, ESCOLHIDO), tipo_b)[..., None])
    mu, sd = est.media_desvio(ROI)
    zc = [(np.abs(pp.canal_bordas(pp.suavizar(a.img, ESCOLHIDO), tipo_b)[..., None] - mu) / sd)[..., 0][ROI]
          for a in CAL]
    lim = float(np.quantile(np.concatenate(zc), Q_ALARME))
    rec = recall_descritivo(lambda a: (np.abs(pp.canal_bordas(pp.suavizar(a.img, ESCOLHIDO), tipo_b)[..., None]
                                              - mu) / sd)[..., 0], lim)
    linha = dict(canal=tipo_b, limiar_z=round(lim, 2))
    for t in pp.TIPOS_DEFEITO:
        linha[f"frac_gt_{t}"] = round(rec[t][0], 3)
        linha[f"detec_{t}"] = round(rec[t][1], 3)
    linha["frac_gt_media"] = round(np.mean([rec[t][0] for t in pp.TIPOS_DEFEITO]), 3)
    linha["detec_media"] = round(np.mean([rec[t][1] for t in pp.TIPOS_DEFEITO]), 3)
    linhas_b.append(linha)
    log(f"canal {tipo_b:<6} (diferença c/ referência): fração do GT acima do limiar média "
        f"{linha['frac_gt_media']:.2f} | defeitos detectados {linha['detec_media']:.2f}  "
        + " ".join(f"{t[:6]}={rec[t][1]:.2f}" for t in pp.TIPOS_DEFEITO))
salvar_csv("P4_bordas.csv", linhas_b)

# =============================================================================
# P5. ROI
# =============================================================================
log("\n=== P5. ROI (cobertura do GT, descritivo) ===")
opcoes = {"cápsula média, erosão 9 px (código antigo)":
          cv2.erode(dados.roi_media.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0,
          "cápsula média": dados.roi_media,
          f"união das normais de referência + {pp.ROI_MARGEM} px (escolhida)": ROI}
linhas_r = []
for nome, r in opcoes.items():
    cob = [(GT_AL[a.caminho] & r).sum() / GT_AL[a.caminho].sum() for a in DEF]
    por_tipo = {t: np.mean([c for c, a in zip(cob, DEF) if a.tipo == t]) for t in pp.TIPOS_DEFEITO}
    linhas_r.append(dict(roi=nome, area_px=int(r.sum()), cobertura_media=round(np.mean(cob), 3),
                         cobertura_min=round(np.min(cob), 3),
                         **{f"cob_{t}": round(v, 3) for t, v in por_tipo.items()}))
    log(f"{nome:<52} área {r.sum():>7} px | GT coberto: média {np.mean(cob):.3f} mín {np.min(cob):.3f} | "
        + " ".join(f"{t[:6]}={v:.2f}" for t, v in por_tipo.items()))
salvar_csv("P5_roi.csv", linhas_r)

# =============================================================================
# FIGURAS
# =============================================================================
# F1: efeito dos filtros num arranhão (zoom) + gradiente de Sobel
ex = next(a for a in DEF if a.tipo == "scratch")
ys, xs = np.nonzero(GT_AL[ex.caminho])
cy, cx = int(ys.mean()), int(xs.mean())
jan = (slice(cy - 90, cy + 90), slice(cx - 130, cx + 130))
fig, ax = plt.subplots(3, 4, figsize=(15, 8.4))
for j, filtro in enumerate(pp.FILTROS):
    f = pp.suavizar(ex.img, filtro)
    cinza = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    ax[0, j].imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[jan])
    ax[0, j].set_title(filtro)
    ax[1, j].imshow(cinza[jan], cmap="gray", vmin=40, vmax=170)
    ax[1, j].set_title("tons de cinza")
    ax[2, j].imshow(pp.sobel_magnitude(cinza)[jan], cmap="magma", vmin=0, vmax=150)
    ax[2, j].set_title("|∇| Sobel")
for a in ax.ravel():
    a.axis("off")
fig.suptitle(f"Efeito da suavização em um arranhão ({ex.caminho})")
fig.tight_layout()
fig.savefig(SAIDA / "F1_filtros_zoom.png", dpi=110)
plt.close(fig)

# F2: compromisso ruído x preservação de bordas
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for l in cinza_f:
    ax.scatter(l["preservacao_bordas"], l["residuo_p99"], s=90, color=CORES[l["filtro"]], zorder=3)
    ax.annotate(l["filtro"], (l["preservacao_bordas"], l["residuo_p99"]), xytext=(6, 4),
                textcoords="offset points")
ax.axvline(PRESERVACAO_MIN, ls="--", color="#999")
ax.set_xlabel("preservação das bordas fortes (|∇| filtrado / sem filtro)")
ax.set_ylabel("ruído residual nas normais (p99 de |f − média|)")
ax.set_title("Suavização: ruído × preservação (tons de cinza)")
fig.tight_layout()
fig.savefig(SAIDA / "F2_ruido_vs_bordas.png", dpi=120)
plt.close(fig)

# F3: fração do defeito acima do limiar, por representação (filtro escolhido) e bordas
reps = REPRESENTACOES + ("rgb (máx.)",)
sel = {l["representacao"]: l for l in linhas_f if l["filtro"] == ESCOLHIDO}
fig, ax = plt.subplots(figsize=(11, 3.8))
w = 0.8 / (len(reps) + 2)
series = [(r, sel[r]) for r in reps] + [(l["canal"], l) for l in linhas_b]
for i, (nome, l) in enumerate(series):
    v = [l[f"detec_{t}"] for t in pp.TIPOS_DEFEITO]
    ax.bar(np.arange(5) + (i - len(series) / 2 + 0.5) * w, v, w * 0.92, color=CORES[nome], label=nome)
ax.set_xticks(range(5), pp.TIPOS_DEFEITO)
ax.set_ylim(0, 1.05)
ax.set_ylabel(f"defeitos com ≥ {MIN_PX_DETECCAO} px acima do limiar")
ax.set_title(f"Descritivo: quanto cada representação 'enxerga' o defeito (filtro {ESCOLHIDO}; "
             f"limiar = q{Q_ALARME * 100:g}% das normais)")
ax.legend(ncol=7, frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(SAIDA / "F3_representacao_por_tipo.png", dpi=120)
plt.close(fig)

# F4: mapas z de um exemplo por tipo: cinza x rgb(máx) x sobel
mu, sd = MODELOS[ESCOLHIDO]
est = Estatistica()
for a in REF:
    est.add(pp.canal_bordas(pp.suavizar(a.img, ESCOLHIDO), "sobel")[..., None])
mu_s, sd_s = est.media_desvio(ROI)
fig, ax = plt.subplots(5, 4, figsize=(14, 15))
for i, t in enumerate(pp.TIPOS_DEFEITO):
    a = next(x for x in DEF if x.tipo == t)
    f = pp.suavizar(a.img, ESCOLHIDO)
    cinza = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    z = np.abs(np.dstack([cinza, f[..., 2], f[..., 1], f[..., 0]]).astype(np.float32) - mu) / sd
    zs = (np.abs(pp.canal_bordas(f, "sobel")[..., None] - mu_s) / sd_s)[..., 0]
    vis = cv2.cvtColor(a.img, cv2.COLOR_BGR2RGB).copy()
    cont, _ = cv2.findContours(GT_AL[a.caminho].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cont, -1, (0, 220, 0), 3)
    crop = (slice(230, 770), slice(60, 940))
    for j, (img, tt) in enumerate([(vis, f"{a.caminho} (GT verde)"), (z[..., 0], "z cinza"),
                                   (z[..., 1:].max(-1), "z rgb (máx.)"), (zs, "z Sobel")]):
        ax[i, j].imshow(img[crop], cmap=None if img.ndim == 3 else "inferno",
                        **({} if img.ndim == 3 else dict(vmin=0, vmax=12)))
        ax[i, j].set_title(tt, fontsize=9)
        ax[i, j].axis("off")
fig.suptitle("Mapas de desvio em relação ao normal: z = |f − média| / desvio (escala 0–12)")
fig.tight_layout()
fig.savefig(SAIDA / "F4_mapas_z_por_tipo.png", dpi=100)
plt.close(fig)

(SAIDA / "resumo.txt").write_text("\n".join(resumo), encoding="utf-8")
print(f"\nArquivos em {SAIDA}")
