"""
Projeto 4 (PDI - UFSCar) - segmentação do defeito (módulo reutilizável; avaliado na etapa 3)

Duas estratégias de localização, ambas do tipo "diferença em relação ao normal" (o enunciado
cita "diferença em relação a uma referência"), mas com referências diferentes:

  A) MODELO ESTATÍSTICO GLOBAL
     Com as 150 normais de referência alinhadas, calcula-se a média mu e o desvio sigma de
     cada pixel em cada canal. Para uma imagem f:
         z(x, y) = max_c |f_c(x, y) - mu_c(x, y)| / sigma_c(x, y)
     Simples, mas a média "borra" o '500' e o logo, que mudam de posição (a cápsula gira
     no próprio eixo), então sigma fica alto ali e defeitos nessas regiões somem.

  B) GOLDEN TEMPLATE COM K VIZINHOS (evolução da ideia do Daniel)
     Para cada imagem, escolhem-se as K normais de referência mais parecidas com ela
     (menor diferença média absoluta numa versão reduzida) e, pixel a pixel, usa-se a MENOR
     diferença entre f e essas K referências:
         d_c(x, y) = min_k |f_c(x, y) - r_k,c(x, y)| ;  z = max_c d_c / sigma_c
     Se a impressão/logo de f aparece igual em pelo menos uma vizinha, a diferença ali é
     pequena; um defeito não aparece em nenhuma normal e continua com diferença alta.

Canais: B, G, R (cor) e, opcionalmente, um canal de bordas: magnitude do gradiente de Sobel
ou densidade local de bordas do Canny (etapa 2). As bordas capturam textura/descontinuidade,
que é como arranhões e rachaduras aparecem, mesmo quando mudam pouco a cor média.

Limiarização por CRESCIMENTO DE REGIÕES (slides de Segmentação): sementes = pixels com
z > T_alto; a região cresce para vizinhos 8-conectados com z > T_baixo. Os dois limiares vêm
SÓ das normais de calibração.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import preprocessamento as pp

SUAV_Z = 15           # Gaussiano 15x15 no mapa z: junta a evidência de pixels vizinhos do mesmo
                      # defeito. 15 px ~ largura do defeito mais fino (eixo menor ~11 px, etapa 1),
                      # então não "apaga" defeitos, mas dilui picos isolados de 1-3 px.
TOLERANCIA = 3        # px. A cápsula gira até ~0,5° (etapa 1); a 350 px do centro isso desloca
                      # ~3 px, que o alinhamento por translação não corrige. O desvio sigma de cada
                      # pixel é substituído pelo MAIOR sigma numa janela 7x7 (dilatação em tons de
                      # cinza = filtro de máximo): uma borda deslocada de até 3 px não vira alarme.
MINIATURA = 125       # lado da imagem reduzida usada para escolher os vizinhos (estratégia B)


def canais_modelo(img_bgr: np.ndarray, bordas: str | None) -> np.ndarray:
    """(H, W, C) float32: B, G, R e, opcionalmente, o canal de bordas ('sobel' ou 'canny')."""
    x = [img_bgr.astype(np.float32)]
    if bordas:
        x.append(pp.canal_bordas(img_bgr, bordas)[..., None])
    return np.concatenate(x, axis=-1)


def _desvio_com_piso(soma, soma2, n, roi, tolerancia):
    mu = soma / n
    sd = np.sqrt(np.maximum(soma2 / n - mu * mu, 0)).astype(np.float32)
    if tolerancia > 0:
        ee = np.ones((2 * tolerancia + 1, 2 * tolerancia + 1), np.uint8)
        sd = np.dstack([cv2.dilate(sd[..., c], ee) for c in range(sd.shape[-1])])
    piso = np.array([max(np.percentile(sd[..., c][roi], 25), 1.0) for c in range(sd.shape[-1])])
    return mu.astype(np.float32), np.maximum(sd, piso).astype(np.float32)


def _combinar(dev: np.ndarray, sd: np.ndarray, roi: np.ndarray, suav: int) -> np.ndarray:
    z = (dev / sd).max(axis=-1)
    return cv2.GaussianBlur(z, (suav, suav), 0) * roi


class ModeloGlobal:
    """Estratégia A."""

    nome = "A: modelo estatístico global"

    def __init__(self, filtro: str, bordas: str | None = None, tolerancia: int = TOLERANCIA,
                 suav: int = SUAV_Z):
        self.filtro, self.bordas = filtro, bordas
        self.tolerancia, self.suav = tolerancia, suav

    def ajustar(self, refs: list[np.ndarray], roi: np.ndarray) -> "ModeloGlobal":
        s = s2 = None
        for img in refs:
            x = canais_modelo(pp.suavizar(img, self.filtro), self.bordas).astype(np.float64)
            s = x if s is None else s + x
            s2 = x * x if s2 is None else s2 + x * x
        self.mu, self.sd = _desvio_com_piso(s, s2, len(refs), roi, self.tolerancia)
        self.roi = roi
        return self

    def mapa(self, img: np.ndarray) -> np.ndarray:
        x = canais_modelo(pp.suavizar(img, self.filtro), self.bordas)
        return _combinar(np.abs(x - self.mu), self.sd, self.roi, self.suav)


class ModeloVizinhos(ModeloGlobal):
    """Estratégia B (usa o sigma global de A só para pôr os canais na mesma escala)."""

    nome = "B: golden template, K vizinhos"

    def __init__(self, filtro: str, bordas: str | None = None, k: int = 5, normalizar: bool = True,
                 tolerancia: int = TOLERANCIA, suav: int = SUAV_Z):
        super().__init__(filtro, bordas, tolerancia, suav)
        self.k, self.normalizar = k, normalizar

    def _mini(self, img_suave):
        g = cv2.cvtColor(img_suave, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (MINIATURA, MINIATURA), interpolation=cv2.INTER_AREA).astype(np.float32)

    def ajustar(self, refs, roi):
        super().ajustar(refs, roi)
        self.refs = [pp.suavizar(r, self.filtro) for r in refs]
        self.minis = np.stack([self._mini(r) for r in self.refs])
        self.roi_mini = cv2.resize(roi.astype(np.uint8), (MINIATURA, MINIATURA),
                                   interpolation=cv2.INTER_NEAREST) > 0
        if not self.normalizar:
            self.sd = np.ones_like(self.sd)
        return self

    def vizinhos(self, img_suave) -> np.ndarray:
        dif = np.abs(self.minis - self._mini(img_suave))[:, self.roi_mini].mean(axis=1)
        return np.argsort(dif)[:self.k]

    def mapa(self, img):
        f = pp.suavizar(img, self.filtro)
        x = canais_modelo(f, self.bordas)
        dev = None
        for i in self.vizinhos(f):
            d = np.abs(x - canais_modelo(self.refs[i], self.bordas))
            dev = d if dev is None else np.minimum(dev, d)
        return _combinar(dev, self.sd, self.roi, self.suav)


# =============================================================================
# LIMIARES E CRESCIMENTO DE REGIÕES
# =============================================================================
def calibrar(mapas_normais: list[np.ndarray], roi: np.ndarray,
             q_baixo: float = 0.99, p_alto: float = 80) -> tuple[float, float]:
    """Limiares a partir de imagens NORMAIS (conjunto de calibração):
    T_baixo = quantil q_baixo de todos os pixels da ROI (critério de similaridade);
    T_alto  = percentil p_alto do MÁXIMO de z de cada imagem normal, ou seja, a semente só
              dispara em ~(100 - p_alto)% das cápsulas normais (taxa-alvo de falso alarme)."""
    vals = np.concatenate([m[roi] for m in mapas_normais])
    t_baixo = float(np.quantile(vals, q_baixo))
    t_alto = float(np.percentile([m[roi].max() for m in mapas_normais], p_alto))
    return t_baixo, max(t_alto, t_baixo)


def crescimento_regioes(z: np.ndarray, t_baixo: float, t_alto: float) -> np.ndarray:
    """Componentes 8-conectados de {z > T_baixo} que contêm ao menos uma semente {z > T_alto}."""
    n, rot = cv2.connectedComponents((z > t_baixo).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(z.shape, bool)
    sementes = np.unique(rot[(z > t_alto) & (rot > 0)])
    return np.isin(rot, sementes)


# =============================================================================
# MÉTRICAS PIXEL A PIXEL
# =============================================================================
def metricas(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    if gt.sum() == 0:   # imagem normal: só faz sentido contar falso positivo
        return dict(tp=tp, fp=fp, fn=fn, dice=np.nan, iou=np.nan, precisao=np.nan, recall=np.nan)
    return dict(tp=tp, fp=fp, fn=fn,
                dice=2 * tp / (2 * tp + fp + fn), iou=tp / (tp + fp + fn),
                precisao=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn))
