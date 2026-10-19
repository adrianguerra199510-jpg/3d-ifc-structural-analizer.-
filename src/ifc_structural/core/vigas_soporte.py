"""Detección automática de las vigas que soportan cada lado de una losa.

La transferencia de cargas de área (``core.transferencia``) necesita saber
qué viga recibe la carga de cada lado del contorno de la losa. Un IFC
exportado desde Revit no trae esa relación, así que se infiere
geométricamente: una viga soporta un lado si

1. es (casi) paralela al lado (menos de ~10° de diferencia),
2. su eje pasa cerca del lado, medido en el plano de la losa (media anchura
   de la viga más una tolerancia) y en la dirección normal (no más lejos
   que media altura de la viga + medio espesor de la losa), y
3. cubre al menos la mitad del largo del lado.

Si varias vigas cumplen, se elige la que cubre más largo del lado. Limitación
conocida: si un lado está soportado por varias vigas en serie (p. ej. dos
vigas colineales), la carga de ese lado se asigna solo a la que más lo cubre.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometria import dimensiones_seccion, normal_losa
from .modelo import Losa, Proyecto

#: Coseno del ángulo máximo permitido entre el lado de la losa y la viga.
_COS_PARALELISMO = 0.985
#: Fracción mínima del lado que debe cubrir la viga.
_COBERTURA_MINIMA = 0.5


def vigas_soporte_de_losa(
    proyecto: Proyecto, losa: Losa, tolerancia_en_planta: float = 0.15, tolerancia_vertical: float = 0.15
) -> List[Optional[str]]:
    """Lista con la viga que soporta cada lado del contorno (o ``None``)."""

    contorno = np.array([proyecto.nodos[n].coordenadas for n in losa.nodos_contorno], dtype=float)
    normal = normal_losa(contorno)
    n_lados = len(contorno)
    resultado: List[Optional[str]] = [None] * n_lados

    for i in range(n_lados):
        a, b = contorno[i], contorno[(i + 1) % n_lados]
        largo = float(np.linalg.norm(b - a))
        if largo < 1e-9:
            continue
        d = (b - a) / largo
        medio = (a + b) / 2.0

        mejor = None  # (cobertura, -distancia_en_planta, id)
        for id_, viga in proyecto.vigas.items():
            p0 = np.array(proyecto.nodos[viga.nodo_inicio].coordenadas, dtype=float)
            p1 = np.array(proyecto.nodos[viga.nodo_fin].coordenadas, dtype=float)
            eje = p1 - p0
            largo_viga = float(np.linalg.norm(eje))
            if largo_viga < 1e-9:
                continue
            t = eje / largo_viga
            if abs(float(np.dot(d, t))) < _COS_PARALELISMO:
                continue

            altura, ancho = dimensiones_seccion(proyecto.perfiles[viga.perfil])
            # Punto del eje de la viga más cercano al punto medio del lado.
            q = p0 + np.dot(medio - p0, t) * t
            separacion = q - medio
            vertical = float(np.dot(separacion, normal))
            en_planta = float(np.linalg.norm(separacion - vertical * normal))
            if en_planta > ancho / 2.0 + tolerancia_en_planta:
                continue
            if abs(vertical) > altura / 2.0 + losa.espesor / 2.0 + tolerancia_vertical:
                continue

            s0, s1 = sorted((float(np.dot(p0 - a, d)), float(np.dot(p1 - a, d))))
            cobertura = max(0.0, min(largo, s1) - max(0.0, s0)) / largo
            if cobertura < _COBERTURA_MINIMA:
                continue

            candidato = (cobertura, -en_planta, id_)
            if mejor is None or candidato[:2] > mejor[:2]:
                mejor = candidato

        if mejor is not None:
            resultado[i] = mejor[2]

    return resultado


def asignar_vigas_soporte(proyecto: Proyecto, **kwargs) -> int:
    """Detecta y asigna las vigas de soporte de todas las losas. Devuelve
    cuántos lados quedaron con viga asignada."""

    total = 0
    for losa in proyecto.losas.values():
        losa.vigas_soporte = vigas_soporte_de_losa(proyecto, losa, **kwargs)
        total += sum(1 for v in losa.vigas_soporte if v)
    return total
