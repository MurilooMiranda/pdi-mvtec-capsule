"""
Projeto 4 (PDI - UFSCar) - ETAPA 1: análise exploratória da categoria capsule + partição própria

Objetivo: conhecer os dados ANTES de escolher filtros, limiares e morfologia.
Tudo aqui usa só técnicas vistas em aula: tons de cinza / canais RGB, histogramas,
média e desvio padrão, Otsu, morfologia, componentes conexos, momentos e
casamento de padrões (template matching).

Saídas (em resultados/etapa1_analise_exploratoria/):
    01_inventario.csv               quantas imagens/máscaras por pasta
    02_exemplos_por_tipo.png        3 exemplos de cada classe com o GT em verde + zoom
    03_defeitos_geometria.csv       área, posição, eixos, excentricidade, orientação de cada defeito
    04_area_por_tipo.png            distribuição da área dos defeitos por tipo
    05_mapa_posicao_defeitos.png    onde os defeitos aparecem (soma das máscaras por tipo)
    06_contraste.csv / .png         intensidade dentro do defeito x anel ao redor (cinza e R, G, B)
    07_histogramas_regioes.png      histograma do fundo, metade preta e metade laranja
    08_alinhamento.csv / .png       deslocamento de cada cápsula (template matching)
    09_media_desvio_normais.png     imagem média e mapa de desvio padrão das cápsulas normais
    10_resumo.txt                   todos os números importantes em texto
e em dados/:
    particao.csv                    partição própria (referencia / calibracao / avaliacao)

Uso:  python etapa1_analise_exploratoria.py          (~1-2 min)
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "dados" / "capsule"
PARTICAO = RAIZ / "dados" / "particao.csv"
SAIDA = RAIZ / "resultados" / "etapa1_analise_exploratoria"
TIPOS_DEFEITO = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
TODOS = ["good"] + TIPOS_DEFEITO
CORES = {"good": "#52514e", "crack": "#2a78d6", "faulty_imprint": "#eb6834", "poke": "#1baf7a",
         "scratch": "#eda100", "squeeze": "#6250d6"}

# --- partição -----------------------------------------------------------------
SEMENTE = 42
N_REFERENCIA = 150   # normais usadas só para construir o "modelo do normal"
N_CALIBRACAO = 40    # normais usadas só para escolher limiares (nunca avaliadas)
# o restante das normais + TODAS as defeituosas formam o conjunto de avaliação

# --- alinhamento ------------------------------------------------------------------
MARGEM_TEMPLATE = 60  # recorte central da imagem média (sem 60 px de cada lado)
ESCALA_MEDIA = 0.5    # média/desvio calculados em 500x500 (mais rápido, mesmo padrão)

SAIDA.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
resumo: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    resumo.append(msg)


def ler(p: Path, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(p), flag)
    if img is None:
        raise FileNotFoundError(p)
    return img


def mascara_gt(tipo: str, nome: str) -> np.ndarray:
    return ler(DATASET / "ground_truth" / tipo / nome.replace(".png", "_mask.png"),
               cv2.IMREAD_GRAYSCALE) > 127


def roi_capsula(cinza: np.ndarray) -> np.ndarray:
    """Cápsula = Otsu invertido (fundo claro) + fechamento + maior componente preenchido."""
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cont, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(cinza)
    cv2.drawContours(m, [max(cont, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return m > 0


# =============================================================================
# 1. INVENTÁRIO
# =============================================================================
log("=== 1. Inventário ===")
inventario, amostras = [], []
for origem in ("train", "test"):
    for tipo in sorted(p.name for p in (DATASET / origem).iterdir() if p.is_dir()):
        arquivos = sorted((DATASET / origem / tipo).glob("*.png"))
        n_masc = len(list((DATASET / "ground_truth" / tipo).glob("*.png"))) if tipo != "good" else 0
        inventario.append(dict(pasta=f"{origem}/{tipo}", imagens=len(arquivos), mascaras=n_masc))
        amostras += [dict(caminho=p, origem=origem, tipo=tipo) for p in arquivos]
        log(f"{origem}/{tipo:<15} {len(arquivos):4d} imagens  {n_masc:4d} máscaras")
tamanhos = {ler(a["caminho"], cv2.IMREAD_UNCHANGED).shape for a in amostras[::25]}
log(f"tamanhos (amostra): {tamanhos}  | total: {len(amostras)} imagens")
with open(SAIDA / "01_inventario.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pasta", "imagens", "mascaras"])
    w.writeheader()
    w.writerows(inventario)

# =============================================================================
# 2. EXEMPLOS VISUAIS
# =============================================================================
fig, ax = plt.subplots(len(TODOS), 6, figsize=(16, 2.9 * len(TODOS)))
for i, tipo in enumerate(TODOS):
    arqs = sorted((DATASET / "test" / tipo).glob("*.png"))[:3]
    for j, p in enumerate(arqs):
        img = cv2.cvtColor(ler(p), cv2.COLOR_BGR2RGB)
        vis, zoom = img.copy(), img[380:620, 300:700]
        if tipo != "good":
            gt = mascara_gt(tipo, p.name).astype(np.uint8)
            cont, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cont, -1, (0, 220, 0), 3)
            ys, xs = np.nonzero(gt)
            cy, cx = int(ys.mean()), int(xs.mean())
            zoom = vis[max(cy - 110, 0):cy + 110, max(cx - 110, 0):cx + 110]
        ax[i, 2 * j].imshow(vis[200:800, 50:950])
        ax[i, 2 * j + 1].imshow(zoom)
        ax[i, 2 * j].set_title(f"{tipo}/{p.name}", fontsize=8)
        ax[i, 2 * j + 1].set_title("zoom", fontsize=8)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("Exemplos por classe (contorno verde = máscara ground truth)")
fig.tight_layout()
fig.savefig(SAIDA / "02_exemplos_por_tipo.png", dpi=110)
plt.close(fig)

# =============================================================================
# 3-6. DEFEITOS: GEOMETRIA, POSIÇÃO E CONTRASTE
# =============================================================================
log("\n=== 3. Geometria dos defeitos (máscaras GT) ===")
geo, contraste = [], []
mapa_pos = {t: np.zeros((1000, 1000), np.float32) for t in TIPOS_DEFEITO}
anel_ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # anel de ~15 px ao redor
for tipo in TIPOS_DEFEITO:
    for p in sorted((DATASET / "test" / tipo).glob("*.png")):
        img = ler(p)
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = mascara_gt(tipo, p.name)
        roi = roi_capsula(cinza)
        mapa_pos[tipo] += gt
        g8 = gt.astype(np.uint8)
        n_comp = cv2.connectedComponents(g8)[0] - 1
        m = cv2.moments(g8, binaryImage=True)
        A = m["m00"]
        cx, cy = m["m10"] / A, m["m01"] / A
        mu20, mu02, mu11 = m["mu20"] / A, m["mu02"] / A, m["mu11"] / A
        d = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
        eixo_maior = 4 * np.sqrt((mu20 + mu02) / 2 + d)
        eixo_menor = 4 * np.sqrt(max((mu20 + mu02) / 2 - d, 1e-9))
        fronteira = g8 - cv2.erode(g8, np.ones((3, 3), np.uint8))
        # metade da cápsula: a fronteira preta/laranja fica perto de x = 500
        dentro_roi = (gt & roi).sum() / A
        # distância do centro do defeito à borda da cápsula (transformada de distância)
        dist_borda = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)[int(cy), int(cx)]
        geo.append(dict(tipo=tipo, imagem=p.name, area_px=int(A), n_componentes=n_comp,
                        perimetro_px=int(fronteira.sum()), centro_x=round(cx, 1), centro_y=round(cy, 1),
                        eixo_maior=round(eixo_maior, 1), eixo_menor=round(eixo_menor, 1),
                        excentricidade=round(eixo_maior / eixo_menor, 2),
                        orientacao_graus=round(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)), 1),
                        metade="preta" if cx < 500 else "laranja",
                        dist_borda_capsula_px=round(float(dist_borda), 1),
                        fracao_dentro_capsula=round(dentro_roi, 3)))
        # contraste: média dentro do GT x média num anel ao redor (restrito à cápsula)
        anel = (cv2.dilate(g8, anel_ee) > 0) & ~gt & roi
        linha = dict(tipo=tipo, imagem=p.name)
        for nome, canal in (("cinza", cinza), ("B", img[..., 0]), ("G", img[..., 1]), ("R", img[..., 2])):
            dentro, fora = canal[gt].astype(float), canal[anel].astype(float)
            linha[f"{nome}_dentro"] = round(dentro.mean(), 1)
            linha[f"{nome}_anel"] = round(fora.mean(), 1) if fora.size else np.nan
            linha[f"{nome}_dif"] = round(abs(dentro.mean() - fora.mean()), 1) if fora.size else np.nan
            linha[f"{nome}_desvio_anel"] = round(fora.std(), 1) if fora.size else np.nan
        contraste.append(linha)

for nome, linhas in (("03_defeitos_geometria.csv", geo), ("06_contraste.csv", contraste)):
    with open(SAIDA / nome, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

log(f"{'tipo':<15}{'n':>4}{'área mín':>10}{'mediana':>9}{'máx':>8}{'% img':>7}"
    f"{'excentr.':>10}{'compon.':>9}{'na preta':>10}{'dist borda':>12}")
for tipo in TIPOS_DEFEITO:
    g = [r for r in geo if r["tipo"] == tipo]
    a = np.array([r["area_px"] for r in g])
    log(f"{tipo:<15}{len(g):>4}{a.min():>10}{int(np.median(a)):>9}{a.max():>8}"
        f"{100 * np.median(a) / 1e6:>6.2f}%{np.median([r['excentricidade'] for r in g]):>10.1f}"
        f"{np.mean([r['n_componentes'] for r in g]):>9.1f}"
        f"{np.mean([r['metade'] == 'preta' for r in g]):>9.0%}"
        f"{np.median([r['dist_borda_capsula_px'] for r in g]):>10.0f}px")
menor = min(geo, key=lambda r: r["area_px"])
log(f"menor defeito: {menor['tipo']}/{menor['imagem']} com {menor['area_px']} px "
    f"(eixo menor ~{menor['eixo_menor']:.0f} px) -> limite para o tamanho da abertura")
log(f"menor eixo menor entre todos os defeitos: {min(r['eixo_menor'] for r in geo):.1f} px")

# figura: área por tipo
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, tipo in enumerate(TIPOS_DEFEITO):
    a = [r["area_px"] for r in geo if r["tipo"] == tipo]
    ax.scatter(np.full(len(a), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(a)), a,
               s=22, color=CORES[tipo], alpha=0.85)
    ax.hlines(np.median(a), i - 0.3, i + 0.3, color="#0b0b0b", lw=2)
ax.set_xticks(range(len(TIPOS_DEFEITO)), TIPOS_DEFEITO)
ax.set_yscale("log")
ax.set_ylabel("área do defeito (px, escala log)")
ax.set_title("Tamanho dos defeitos por tipo (traço = mediana; imagem tem 1.000.000 px)")
fig.tight_layout()
fig.savefig(SAIDA / "04_area_por_tipo.png", dpi=120)
plt.close(fig)

# figura: mapa de posição
media_ref = np.mean([cv2.cvtColor(ler(p), cv2.COLOR_BGR2GRAY)
                     for p in sorted((DATASET / "train" / "good").glob("*.png"))[:20]], axis=0)
fig, ax = plt.subplots(1, len(TIPOS_DEFEITO), figsize=(18, 3.4))
for a, tipo in zip(ax, TIPOS_DEFEITO):
    a.imshow(media_ref[200:800, 50:950], cmap="gray")
    m = mapa_pos[tipo][200:800, 50:950]
    a.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.75)
    a.set_title(f"{tipo} (n={sum(r['tipo'] == tipo for r in geo)})")
    a.axis("off")
fig.suptitle("Onde ficam os defeitos: soma das máscaras GT sobre uma cápsula média")
fig.tight_layout()
fig.savefig(SAIDA / "05_mapa_posicao_defeitos.png", dpi=110)
plt.close(fig)

# figura e resumo: contraste
log("\n=== 4. Contraste: |média dentro do defeito - média do anel ao redor| ===")
log("(comparar com o desvio padrão do anel: se |dif| < desvio, um limiar simples não separa)")
log(f"{'tipo':<15}" + "".join(f"{c:>16}" for c in ("cinza", "B", "G", "R")))
for tipo in TIPOS_DEFEITO:
    c = [r for r in contraste if r["tipo"] == tipo]
    log(f"{tipo:<15}" + "".join(
        f"{np.nanmedian([r[f'{k}_dif'] for r in c]):>8.1f} (σ{np.nanmedian([r[f'{k}_desvio_anel'] for r in c]):>4.1f})"
        for k in ("cinza", "B", "G", "R")))
fig, ax = plt.subplots(figsize=(10, 3.8))
largura = 0.2
for k, (canal, cor) in enumerate((("cinza", "#52514e"), ("B", "#2a78d6"), ("G", "#1baf7a"), ("R", "#e34948"))):
    vals = [[r[f"{canal}_dif"] for r in contraste if r["tipo"] == t] for t in TIPOS_DEFEITO]
    bp = ax.boxplot(vals, positions=np.arange(len(TIPOS_DEFEITO)) + (k - 1.5) * largura,
                    widths=largura * 0.85, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=cor, alpha=0.75)
    ax.plot([], [], color=cor, lw=6, label=canal)
ax.set_xticks(range(len(TIPOS_DEFEITO)), TIPOS_DEFEITO)
ax.set_ylabel("|dentro − anel| (níveis de intensidade)")
ax.set_title("Contraste do defeito com a vizinhança, por canal")
ax.legend(ncol=4, frameon=False)
fig.tight_layout()
fig.savefig(SAIDA / "06_contraste.png", dpi=120)
plt.close(fig)

# =============================================================================
# 7. HISTOGRAMAS DAS REGIÕES DE UMA CÁPSULA NORMAL
# =============================================================================
img = ler(DATASET / "train" / "good" / "000.png")
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi = roi_capsula(cinza)
xx = np.arange(1000)[None, :].repeat(1000, 0)
regioes = {"fundo": ~cv2.dilate(roi.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool),
           "metade preta": roi & (xx < 480), "metade laranja": roi & (xx > 520)}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
log("\n=== 5. Intensidade por região (train/good/000) ===")
for nome, m in regioes.items():
    ax[0].hist(cinza[m], bins=64, range=(0, 256), alpha=0.6, label=nome, density=True)
    log(f"{nome:<15} cinza média={cinza[m].mean():6.1f}  desvio={cinza[m].std():5.1f}  "
        f"R={img[..., 2][m].mean():6.1f} G={img[..., 1][m].mean():6.1f} B={img[..., 0][m].mean():6.1f}")
ax[0].set_title("Histograma em tons de cinza por região")
ax[0].set_xlabel("intensidade")
ax[0].legend(frameon=False)
for k, (cor, nome) in enumerate(((2, "R"), (1, "G"), (0, "B"))):
    ax[1].hist(img[..., cor][regioes["metade laranja"]], bins=64, range=(0, 256), alpha=0.6,
               label=nome, color=("#e34948", "#1baf7a", "#2a78d6")[k], density=True)
ax[1].set_title("Metade laranja: histograma de cada canal")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(SAIDA / "07_histogramas_regioes.png", dpi=120)
plt.close(fig)

# =============================================================================
# 8-9. VARIAÇÃO ENTRE CÁPSULAS NORMAIS
# =============================================================================
log("\n=== 6. Variação entre cápsulas normais ===")
normais = sorted((DATASET / "train" / "good").glob("*.png")) + sorted((DATASET / "test" / "good").glob("*.png"))
cinzas = [cv2.cvtColor(ler(p), cv2.COLOR_BGR2GRAY) for p in normais]
media0 = np.mean(cinzas[:30], axis=0).astype(np.uint8)
template = media0[MARGEM_TEMPLATE:-MARGEM_TEMPLATE, MARGEM_TEMPLATE:-MARGEM_TEMPLATE]
alinh = []
for p, c in zip(normais, cinzas):
    res = cv2.matchTemplate(c, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    r = roi_capsula(c)
    m = cv2.moments(r.astype(np.uint8), binaryImage=True)
    ang = np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    alinh.append(dict(imagem=f"{p.parent.parent.name}/{p.name}", dx=loc[0] - MARGEM_TEMPLATE,
                      dy=loc[1] - MARGEM_TEMPLATE, correlacao=round(score, 4),
                      area_capsula=int(r.sum()), angulo_graus=round(ang, 2)))
dx = np.array([a["dx"] for a in alinh])
dy = np.array([a["dy"] for a in alinh])
log(f"{len(normais)} cápsulas normais | deslocamento (template matching): "
    f"dx {dx.min()}..{dx.max()} px, dy {dy.min()}..{dy.max()} px, "
    f"|d| médio {np.hypot(dx, dy).mean():.1f} px")
log(f"ângulo do eixo maior: {min(a['angulo_graus'] for a in alinh):.2f}° a "
    f"{max(a['angulo_graus'] for a in alinh):.2f}° | área da cápsula: "
    f"{min(a['area_capsula'] for a in alinh)} a {max(a['area_capsula'] for a in alinh)} px")
log(f"correlação com a média: mín {min(a['correlacao'] for a in alinh):.3f}, "
    f"mediana {np.median([a['correlacao'] for a in alinh]):.3f}")
with open(SAIDA / "08_alinhamento.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(alinh[0].keys()))
    w.writeheader()
    w.writerows(alinh)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(dx, dy, s=14, color="#2a78d6", alpha=0.6)
ax.axhline(0, color="#999", lw=0.8)
ax.axvline(0, color="#999", lw=0.8)
ax.set_xlabel("dx (px)")
ax.set_ylabel("dy (px)")
ax.set_title("Deslocamento de cada cápsula normal\nem relação à média (template matching)")
fig.tight_layout()
fig.savefig(SAIDA / "08_alinhamento.png", dpi=120)
plt.close(fig)

# média e desvio padrão pixel a pixel (após alinhar por translação)
alinhadas = []
for c, a in zip(cinzas, alinh):
    M = np.float32([[1, 0, -a["dx"]], [0, 1, -a["dy"]]])
    al = cv2.warpAffine(c, M, (1000, 1000), borderMode=cv2.BORDER_REPLICATE)
    alinhadas.append(cv2.resize(al, None, fx=ESCALA_MEDIA, fy=ESCALA_MEDIA, interpolation=cv2.INTER_AREA))
pilha = np.stack(alinhadas).astype(np.float32)
media, desvio = pilha.mean(0), pilha.std(0)
roi_m = roi_capsula(media.astype(np.uint8))
log(f"desvio padrão entre normais dentro da cápsula: mediana {np.median(desvio[roi_m]):.1f}, "
    f"p90 {np.percentile(desvio[roi_m], 90):.1f}, p99 {np.percentile(desvio[roi_m], 99):.1f} níveis")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].imshow(media[100:400, 25:475], cmap="gray")
ax[0].set_title(f"Média de {len(normais)} cápsulas normais alinhadas")
im = ax[1].imshow(desvio[100:400, 25:475], cmap="magma")
ax[1].set_title("Desvio padrão pixel a pixel (claro = varia muito entre normais)")
fig.colorbar(im, ax=ax[1], fraction=0.03)
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(SAIDA / "09_media_desvio_normais.png", dpi=120)
plt.close(fig)

# =============================================================================
# 10. PARTIÇÃO PRÓPRIA
# =============================================================================
log("\n=== 7. Partição própria ===")
rng = np.random.default_rng(SEMENTE)
idx = rng.permutation(len(normais))
conjunto = {}
for k, i in enumerate(idx):
    rel = normais[i].relative_to(DATASET).as_posix()
    conjunto[rel] = ("referencia" if k < N_REFERENCIA else
                     "calibracao" if k < N_REFERENCIA + N_CALIBRACAO else "avaliacao")
linhas = []
for a in amostras:
    rel = a["caminho"].relative_to(DATASET).as_posix()
    tipo = a["tipo"]
    linhas.append(dict(caminho=rel, tipo=tipo, rotulo="normal" if tipo == "good" else "defeituoso",
                       conjunto=conjunto.get(rel, "avaliacao"),
                       mascara="" if tipo == "good" else f"ground_truth/{tipo}/{Path(rel).stem}_mask.png"))
with open(PARTICAO, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
    w.writeheader()
    w.writerows(linhas)
for c in ("referencia", "calibracao", "avaliacao"):
    sel = [l for l in linhas if l["conjunto"] == c]
    por_tipo = {t: sum(l["tipo"] == t for l in sel) for t in TODOS}
    log(f"{c:<11} {len(sel):4d} imagens  " + "  ".join(f"{t}={n}" for t, n in por_tipo.items() if n))
log("regra: referência e calibração só têm cápsulas normais e nunca são avaliadas;")
log("       nenhum limiar/parâmetro pode ser escolhido olhando o conjunto de avaliação.")

(SAIDA / "10_resumo.txt").write_text("\n".join(resumo), encoding="utf-8")
print(f"\nArquivos gravados em {SAIDA} e {PARTICAO}")
