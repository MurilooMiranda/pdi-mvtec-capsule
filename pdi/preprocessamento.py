"""
Projeto 4 (PDI - UFSCar) - pré-processamento (módulo reutilizável; avaliado na etapa 2)

Funções usadas por todas as etapas seguintes. Só técnicas vistas em aula:
  - leitura e partição própria (dados/particao.csv, gerado na etapa 1)
  - representação: tons de cinza ou canais B, G, R separados
  - suavização: Gaussiano, mediana, bilateral
  - bordas: magnitude do gradiente de Sobel e detector de Canny
  - alinhamento: casamento de padrões (cv2.matchTemplate) -> translação
  - ROI da cápsula: Otsu + fechamento + maior componente preenchido

Convenção: imagens BGR uint8 (padrão OpenCV); máscaras booleanas.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent      # raiz do repositório
DATASET = RAIZ / "dados" / "capsule"
PARTICAO = RAIZ / "dados" / "particao.csv"
TIPOS_DEFEITO = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]

# --- parâmetros dos filtros (os mesmos em todas as comparações) -------------------
GAUSS_K = 5                 # Gaussiano 5x5 (sigma automático pelo OpenCV: 1,1)
MEDIANA_K = 5               # mediana 5x5
BILATERAL = (9, 40, 9)      # d, sigmaColor, sigmaSpace (valores do Daniel, etapa 1 antiga)
FILTROS = ("nenhum", "gaussiano", "mediana", "bilateral")

# --- bordas ------------------------------------------------------------------------
CANNY_BAIXO, CANNY_ALTO = 15, 40   # razão ~1:2,7 (slides: 1:2 a 1:3)

# --- alinhamento -----------------------------------------------------------------------
MARGEM_TEMPLATE = 60               # recorte central da imagem média usado como padrão


# =============================================================================
# DADOS
# =============================================================================
@dataclass
class Amostra:
    caminho: str            # relativo à pasta capsule/
    tipo: str               # good | crack | ...
    conjunto: str           # referencia | calibracao | avaliacao
    mascara: str            # caminho relativo da máscara GT ('' para good)
    img: np.ndarray | None = field(default=None, repr=False)   # BGR alinhada
    gt: np.ndarray | None = field(default=None, repr=False)    # bool, referencial ORIGINAL
    desloc: tuple[int, int] = (0, 0)                             # (dx, dy) estimado

    @property
    def defeituosa(self) -> bool:
        return self.tipo != "good"


def ler(caminho: Path, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(caminho), flag)
    if img is None:
        raise FileNotFoundError(caminho)
    return img


def carregar_particao(conjuntos=("referencia", "calibracao", "avaliacao")) -> list[Amostra]:
    with open(PARTICAO, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return [Amostra(l["caminho"], l["tipo"], l["conjunto"], l["mascara"])
            for l in linhas if l["conjunto"] in conjuntos]


def ler_gt(a: Amostra) -> np.ndarray:
    if not a.mascara:
        return np.zeros((1000, 1000), bool)
    return ler(DATASET / a.mascara, cv2.IMREAD_GRAYSCALE) > 127


# =============================================================================
# REPRESENTAÇÃO E SUAVIZAÇÃO
# =============================================================================
def canais(img_bgr: np.ndarray, representacao: str) -> np.ndarray:
    """'cinza' -> (H, W, 1); 'rgb' -> (H, W, 3) na ordem B, G, R; 'R'/'G'/'B' -> 1 canal."""
    if representacao == "cinza":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    if representacao == "rgb":
        return img_bgr
    return img_bgr[..., {"B": 0, "G": 1, "R": 2}[representacao]][..., None]


def suavizar(img: np.ndarray, filtro: str) -> np.ndarray:
    """Aplica o filtro a cada canal (o OpenCV trata BGR canal a canal no Gaussiano e na
    mediana; o bilateral em imagem colorida usa a distância de cor entre os pixels)."""
    if filtro == "nenhum":
        return img
    if filtro == "gaussiano":
        return cv2.GaussianBlur(img, (GAUSS_K, GAUSS_K), 0)
    if filtro == "mediana":
        return cv2.medianBlur(img, MEDIANA_K)
    if filtro == "bilateral":
        return cv2.bilateralFilter(img, *BILATERAL)
    raise ValueError(filtro)


# =============================================================================
# BORDAS
# =============================================================================
def sobel_magnitude(cinza: np.ndarray) -> np.ndarray:
    """Magnitude do gradiente de Sobel 3x3: sqrt(gx² + gy²) (float32, escala bruta)."""
    g = cinza.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def canny(cinza: np.ndarray, baixo: int = CANNY_BAIXO, alto: int = CANNY_ALTO) -> np.ndarray:
    """Canny (supressão não máxima + histerese). Retorna mapa binário {0, 1} float32."""
    return (cv2.Canny(cinza, baixo, alto, L2gradient=True) > 0).astype(np.float32)


def canal_bordas(img_bgr: np.ndarray, tipo: str) -> np.ndarray:
    """Canal de bordas para entrar na comparação com a referência:
    'sobel' -> magnitude; 'canny' -> densidade local de bordas (Canny suavizado 9x9),
    porque um mapa de 1 px de largura não pode ser comparado pixel a pixel."""
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if tipo == "sobel":
        return sobel_magnitude(cinza)
    if tipo == "canny":
        return cv2.GaussianBlur(canny(cinza), (9, 9), 0) * 255
    raise ValueError(tipo)


# =============================================================================
# ALINHAMENTO E ROI
# =============================================================================
def criar_template(cinzas_referencia: list[np.ndarray]) -> np.ndarray:
    media = np.mean(cinzas_referencia, axis=0).astype(np.uint8)
    m = MARGEM_TEMPLATE
    return media[m:-m, m:-m]


def estimar_deslocamento(cinza: np.ndarray, template: np.ndarray) -> tuple[int, int]:
    """Posição do máximo da correlação normalizada -> deslocamento (dx, dy) em relação à média."""
    res = cv2.matchTemplate(cinza, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    return loc[0] - MARGEM_TEMPLATE, loc[1] - MARGEM_TEMPLATE


def transladar(img: np.ndarray, dx: int, dy: int, inverso: bool = False,
               borda=cv2.BORDER_REPLICATE) -> np.ndarray:
    """Leva a imagem ao referencial da média (inverso=False) ou de volta (inverso=True)."""
    s = 1 if inverso else -1
    M = np.float32([[1, 0, s * dx], [0, 1, s * dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=borda)


def mascara_volta(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Máscara estimada no referencial alinhado -> referencial original (para comparar com o GT)."""
    return transladar(mask.astype(np.uint8), dx, dy, inverso=True,
                      borda=cv2.BORDER_CONSTANT) > 0


def roi_capsula(cinza: np.ndarray) -> np.ndarray:
    """Otsu invertido (fundo claro) + fechamento 15x15 (tapa o '500' branco e reflexos)
    + maior componente conexo preenchido."""
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cont, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(cinza)
    cv2.drawContours(m, [max(cont, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return m > 0


def dilatar(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), ee) > 0


# =============================================================================
# CARREGAMENTO COMPLETO (lê, alinha e guarda tudo em memória)
# =============================================================================
@dataclass
class Dados:
    amostras: list[Amostra]
    template: np.ndarray
    roi: np.ndarray              # ROI de avaliação (referencial alinhado)
    roi_media: np.ndarray        # cápsula na imagem média (sem dilatação)

    def do_conjunto(self, conjunto: str) -> list[Amostra]:
        return [a for a in self.amostras if a.conjunto == conjunto]


ROI_MARGEM = 10   # px além da união das cápsulas normais de referência


def carregar_dados(alinhar: bool = True, verbose: bool = True) -> Dados:
    """Lê as 351 imagens, estima o deslocamento por template matching (padrão = média das
    normais de REFERÊNCIA) e guarda as imagens já alinhadas.

    ROI de avaliação = união das cápsulas de todas as normais de referência + 10 px. Usa
    só normais (nada de máscara de defeito) e cobre a variação natural de posição/forma."""
    amostras = carregar_particao()
    refs_cinza = []
    for a in amostras:
        bruta = ler(DATASET / a.caminho)
        a.img = bruta
        a.gt = ler_gt(a)
        if a.conjunto == "referencia":
            refs_cinza.append(cv2.cvtColor(bruta, cv2.COLOR_BGR2GRAY))
    template = criar_template(refs_cinza)
    uniao = np.zeros((1000, 1000), bool)
    for a in amostras:
        if alinhar:
            a.desloc = estimar_deslocamento(cv2.cvtColor(a.img, cv2.COLOR_BGR2GRAY), template)
            a.img = transladar(a.img, *a.desloc)
        if a.conjunto == "referencia":
            uniao |= roi_capsula(cv2.cvtColor(a.img, cv2.COLOR_BGR2GRAY))
    media = np.mean([cv2.cvtColor(a.img, cv2.COLOR_BGR2GRAY) for a in amostras
                     if a.conjunto == "referencia"], axis=0).astype(np.uint8)
    if verbose:
        print(f"{len(amostras)} imagens carregadas e {'alinhadas' if alinhar else 'NÃO alinhadas'}")
    return Dados(amostras, template, dilatar(uniao, ROI_MARGEM), roi_capsula(media))
