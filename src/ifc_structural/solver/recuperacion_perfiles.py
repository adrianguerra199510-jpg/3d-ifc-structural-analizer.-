"""Esfuerzos de las barras: corrección de extremos y perfiles en I (secciones compuestas).

**Extremos.** CalculiX entrega en las estaciones extremas de cada barra el
valor del elemento (hasta ~5 % por debajo del esfuerzo de flexión real, justo
donde el momento suele ser máximo). Se extrapola linealmente el esfuerzo
axial de cada sección desde las dos estaciones interiores más próximas (exacto
si el momento varía linealmente).

**Perfiles en I.**

Un perfil en I se analiza como tres barras rectangulares (alas y alma) unidas
por tramos rígidos (ver ``generador_inp``). La flexión y el esfuerzo axial que
entrega CalculiX para esas barras son los correctos (se comprobó contra
M·c/I), pero el cortante no: ccx lo reparte por igual entre alas y alma
(τ = V / área total en todos los nodos), cuando en un perfil en I lo lleva
casi todo el alma. Aquí se rehace el tensor de cada nodo con las hipótesis de
la teoría de vigas:

* esfuerzo axial σ: el de CalculiX (tensor proyectado sobre el eje de la barra);
* cortante en la dirección del peralte: ``V / (tw·hw)`` en los nodos del alma;
* cortante en la dirección del ancho: ``V / (2·bf·tf)`` en los nodos de las alas;
* el resto de componentes, nulas.

Las fuerzas cortantes se obtienen del equilibrio de cada rebanada entre dos
estaciones: ``V = d(∫σ·x dA)/ds``, con ``∫σ·x dA`` calculada con el esfuerzo
axial de los nodos de cada parte (lineal en la sección). No incluye el
cortante por torsión.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.geometria import ejes_seccion, es_perfil_compuesto, partes_seccion
from ..core.modelo import Proyecto
from .generador_inp import MallaAnalisis
from .lector_resultados import MallaTensiones

_DECIMALES_ESTACION = 6
#: Protección: la extrapolación al extremo se descarta si cambia el esfuerzo en más
#: de esta fracción del máximo de la barra (el error habitual de ccx es de 5 a 20 %).
TOLERANCIA_EXTRAPOLACION = 0.5


def hay_perfiles_compuestos(proyecto: Proyecto) -> bool:
    return any(es_perfil_compuesto(proyecto.perfiles[b.perfil]) for b in proyecto.todas_las_barras().values())


def hay_barras(proyecto: Proyecto) -> bool:
    return bool(proyecto.todas_las_barras())


def _extrapolacion_cuadratica(coef, s_est, i):
    """Extrapolación parabólica al extremo ``i`` desde las tres estaciones
    siguientes, o None si la curvatura no es la misma en las tres estaciones y
    las dos siguientes (hay un quiebre del momento y solo vale la lineal)."""

    n = len(s_est)
    if n < 5:
        return None
    if i == 0:
        e = [1, 2, 3, 4]
    else:
        e = [n - 2, n - 3, n - 4, n - 5]
    s1, s2, s3, s4 = (s_est[j] for j in e)

    def curvatura(a, b, c, sa, sb, sc):  # segunda diferencia dividida
        return ((c - b) / (sc - sb) - (b - a) / (sb - sa)) / (sc - sa)

    resultado = {}
    discrepancia, magnitud = 0.0, 0.0
    for k, matriz in coef.items():
        v1, v2, v3, v4 = (matriz[j] for j in e)
        c1 = curvatura(v1, v2, v3, s1, s2, s3)
        c2 = curvatura(v2, v3, v4, s2, s3, s4)
        discrepancia = max(discrepancia, float(np.abs(c1 - c2).max()))
        magnitud = max(magnitud, float(np.abs(c1).max()), float(np.abs(c2).max()))
        x0 = s_est[i]
        # polinomio de Newton por (s1, s2, s3)
        resultado[k] = v1 + (v2 - v1) / (s2 - s1) * (x0 - s1) + c1 * (x0 - s1) * (x0 - s2)
    if discrepancia > 0.02 * magnitud:
        return None
    return resultado


def _tensor(valores: np.ndarray) -> np.ndarray:
    m = np.zeros((len(valores), 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = valores[:, 0], valores[:, 1], valores[:, 2]
    m[:, 0, 1] = m[:, 1, 0] = valores[:, 3]
    m[:, 1, 2] = m[:, 2, 1] = valores[:, 4]
    m[:, 2, 0] = m[:, 0, 2] = valores[:, 5]
    return m


def _a_seis(m: np.ndarray) -> np.ndarray:
    return np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 1, 2], m[:, 2, 0]], axis=1)


def recuperar_perfiles(
    proyecto: Proyecto, malla: MallaAnalisis, malla_tensiones: MallaTensiones, tensiones: np.ndarray
) -> np.ndarray:
    """Copia de ``tensiones`` (N, 6) con el cortante rehecho en los nodos de
    las barras con perfil en I."""

    resultado = np.array(tensiones, dtype=float, copy=True)
    barra_de_celda = np.array([malla.elementos[int(e)].barra_id for e in malla_tensiones.ids_elementos])
    parte_de_celda = np.array([malla.elementos[int(e)].parte for e in malla_tensiones.ids_elementos])

    for barra_id, barra in proyecto.todas_las_barras().items():
        perfil = proyecto.perfiles[barra.perfil]
        compuesto = es_perfil_compuesto(perfil)
        celdas = np.where(barra_de_celda == barra_id)[0]
        if len(celdas) == 0:
            continue
        ejes = ejes_seccion(proyecto, barra)
        partes = partes_seccion(proyecto, barra)
        origen = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas, dtype=float)
        if ejes.profundidad_en_e1:
            d_dir, w_dir = ejes.e1, ejes.e2
        else:
            d_dir, w_dir = ejes.e2, ejes.e1

        # Nodos de cada parte y su posición (s, profundidad, ancho)
        nodos_de_parte: Dict[int, np.ndarray] = {
            k: np.unique(malla_tensiones.conectividad[celdas[parte_de_celda[celdas] == k]]) for k in range(len(partes))
        }
        todos = np.unique(np.concatenate(list(nodos_de_parte.values())))
        rel = malla_tensiones.coordenadas[todos] - origen
        longitud = np.linalg.norm(np.array(proyecto.nodos[barra.nodo_fin].coordenadas) - origen)
        s_nodo = {int(n): float(r @ ejes.t) for n, r in zip(todos, rel)}
        estaciones = sorted({round(v / longitud, _DECIMALES_ESTACION) for v in s_nodo.values()})
        indice_estacion = {e: i for i, e in enumerate(estaciones)}
        s_est = np.array(estaciones) * longitud

        # Esfuerzo axial de cada parte en cada estación: plano σ = a + b·u + c·v
        coef: Dict[int, np.ndarray] = {k: np.zeros((len(estaciones), 3)) for k in range(len(partes))}
        nodos_estacion: Dict[int, Dict[int, List[int]]] = {}
        for k in range(len(partes)):
            por_estacion: Dict[int, List[int]] = {}
            for n in nodos_de_parte[k]:
                por_estacion.setdefault(indice_estacion[round(s_nodo[int(n)] / longitud, _DECIMALES_ESTACION)], []).append(int(n))
            nodos_estacion[k] = por_estacion
            for i, ns in por_estacion.items():
                pos = malla_tensiones.coordenadas[ns] - origen
                sigma = np.einsum("j,njk,k->n", ejes.t, _tensor(tensiones[ns]), ejes.t)
                matriz = np.column_stack([np.ones(len(ns)), pos @ d_dir, pos @ w_dir])
                coef[k][i] = np.linalg.lstsq(matriz, sigma, rcond=None)[0]
        if len(estaciones) < 2:
            continue
        extremos_conservados: set = set()
        # CalculiX entrega en los extremos una media de elemento (hasta ~5 % baja
        # donde el momento es máximo): se extrapola linealmente desde el interior.
        if len(estaciones) >= 3:
            escala = max(
                (abs(np.einsum("j,njk,k->n", ejes.t, _tensor(tensiones[nodos_de_parte[k]]), ejes.t)).max()
                 for k in coef if len(nodos_de_parte[k])), default=0.0)
            for i, vecina, siguiente in ((0, 1, 2), (-1, -2, -3)):
                # Las estaciones 1 y 2 de cada extremo están siempre en el mismo tramo
                # lineal del momento que la estación extrema: cada tramo entre cargas se
                # divide al menos en dos elementos (ver ``construir_malla``).
                razon = (s_est[vecina] - s_est[i]) / (s_est[siguiente] - s_est[vecina])
                extrapolado = {k: coef[k][vecina] + (coef[k][vecina] - coef[k][siguiente]) * razon for k in coef}
                cuadratico = _extrapolacion_cuadratica(coef, s_est, i)
                if cuadratico is not None:
                    extrapolado = cuadratico
                diferencia = 0.0
                for k in coef:
                    ns = nodos_estacion[k].get(i % len(estaciones), [])
                    if not ns:
                        continue
                    pos = malla_tensiones.coordenadas[ns] - origen
                    matriz = np.column_stack([np.ones(len(ns)), pos @ d_dir, pos @ w_dir])
                    diferencia = max(diferencia, float(np.abs(matriz @ (extrapolado[k] - coef[k][i])).max()))
                if diferencia <= TOLERANCIA_EXTRAPOLACION * escala:
                    for k in coef:
                        coef[k][i] = extrapolado[k]
                else:
                    extremos_conservados.add(i % len(estaciones))

        if not compuesto:  # sección rectangular: solo se corrige el esfuerzo axial de los extremos
            if len(estaciones) < 3:
                continue
            for i in (0, len(estaciones) - 1):
                ns = nodos_estacion[0].get(i, [])
                if not ns or i in extremos_conservados:
                    continue
                pos = malla_tensiones.coordenadas[ns] - origen
                a, b, c = coef[0][i]
                nuevo = a + b * (pos @ d_dir) + c * (pos @ w_dir)
                actual = np.einsum("j,njk,k->n", ejes.t, _tensor(resultado[ns]), ejes.t)
                resultado[ns] += _a_seis((nuevo - actual)[:, None, None] * np.outer(ejes.t, ejes.t)[None])
            continue

        q_prof = np.zeros(len(estaciones))
        q_anc = np.zeros(len(estaciones))
        for k, parte in enumerate(partes):
            if ejes.profundidad_en_e1:
                uc, vc, dp, da = parte.desplazamiento1, parte.desplazamiento2, parte.dim1, parte.dim2
            else:
                uc, vc, dp, da = parte.desplazamiento2, parte.desplazamiento1, parte.dim2, parte.dim1
            area = dp * da
            a, b, c = coef[k][:, 0], coef[k][:, 1], coef[k][:, 2]
            q_prof += area * (a * uc + b * (uc**2 + dp**2 / 12.0) + c * uc * vc)
            q_anc += area * (a * vc + b * uc * vc + c * (vc**2 + da**2 / 12.0))

        ds = np.diff(s_est)
        v_prof_elem = np.diff(q_prof) / ds
        v_anc_elem = np.diff(q_anc) / ds
        v_prof = np.array([np.mean(v_prof_elem[max(i - 1, 0): i + 1]) for i in range(len(estaciones))])
        v_anc = np.array([np.mean(v_anc_elem[max(i - 1, 0): i + 1]) for i in range(len(estaciones))])

        for k, parte in enumerate(partes):
            area_parte = parte.dim1 * parte.dim2
            nodos = nodos_de_parte[k]
            estaciones_nodo = np.array([indice_estacion[round(s_nodo[int(n)] / longitud, _DECIMALES_ESTACION)] for n in nodos])
            sigma = np.einsum("j,njk,k->n", ejes.t, _tensor(tensiones[nodos]), ejes.t)
            pos = malla_tensiones.coordenadas[nodos] - origen
            en_extremo = ((estaciones_nodo == 0) | (estaciones_nodo == len(estaciones) - 1)) & ~np.isin(
                estaciones_nodo, list(extremos_conservados))
            if len(estaciones) >= 3 and en_extremo.any():
                a, b, c = coef[k][estaciones_nodo].T
                sigma = np.where(en_extremo, a + b * (pos @ d_dir) + c * (pos @ w_dir), sigma)
            if parte.en_el_eje:
                tau_prof = v_prof[estaciones_nodo] / area_parte
                tau_anc = np.zeros(len(nodos))
            else:
                tau_prof = np.zeros(len(nodos))
                tau_anc = v_anc[estaciones_nodo] / (2.0 * area_parte)
            local = (
                sigma[:, None, None] * np.outer(ejes.t, ejes.t)[None]
                + tau_prof[:, None, None] * (np.outer(ejes.t, d_dir) + np.outer(d_dir, ejes.t))[None]
                + tau_anc[:, None, None] * (np.outer(ejes.t, w_dir) + np.outer(w_dir, ejes.t))[None]
            )
            resultado[nodos] = _a_seis(local)
    return resultado
