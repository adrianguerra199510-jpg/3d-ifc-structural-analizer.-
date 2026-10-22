"""Proceso hijo que malla con Gmsh los sólidos triangulados de un análisis.

Gmsh puede quedarse colgado o caerse con superficies complicadas; ejecutarlo en un proceso aparte permite ponerle un
límite de tiempo y seguir con otro tipo de elemento sin congelar la aplicación. Un solo proceso malla **todas** las
piezas, una tras otra (lanzar un proceso por pieza hace saltar a los antivirus). Uso (lo lanza ``tetraedros``):

    python -m ifc_structural.solido.proceso_gmsh --lote entrada.npz carpeta [--desde k]

``entrada.npz`` lleva ``n``, ``tamano``, ``angulo`` y, por pieza ``i``, ``v{i}`` (vértices), ``t{i}`` (triángulos) y
``d{i}`` (y, si la pieza lleva otro tamaño de elemento, ``h{i}``): 3 para rellenar la superficie cerrada con tetraedros, 2 para volver a mallar una superficie (la superficie
media de una lámina) con triángulos casi equiláteros, 4 para mallar con cuadriláteros una tapa plana (la sección de una
pieza prismática, que después se extruye en hexaedros). Antes de empezar cada pieza el proceso escribe su número en
``carpeta/actual.txt`` (así el proceso padre sabe cuál se colgó) y al terminarla deja ``res_{i}.npz`` (``coordenadas``
y ``tetraedros`` en el orden de CalculiX, o ``triangulos``, o ``cuadrilateros``) o ``err_{i}.txt`` con el motivo. Termina con código 0 si
llegó al final.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

#: Orden de CalculiX a partir del de Gmsh para ``Tetrahedron10`` (cambian los dos últimos nodos medios).
DE_GMSH = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
#: La malla de tetraedros debe tener casi el volumen de la superficie de partida; si no (la superficie vuelta a mallar
#: con elementos grandes «corta» las curvas), se prueba con elementos más pequeños.
VOLUMEN_MINIMO, VOLUMEN_MAXIMO = 0.9, 1.1


def _mallar_en_sesion(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float,
                      por_vuelta: int = 0, netgen: bool = True):
    """Malla una pieza con Gmsh ya inicializado (el modelo se borra al empezar); ``por_vuelta`` > 0 refina las
    superficies curvas con ese número de elementos por vuelta; ``netgen`` optimiza la malla con Netgen."""

    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("solido")
    n = len(vertices)
    superficie = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, superficie, list(range(1, n + 1)), np.asarray(vertices, float).ravel().tolist())
    gmsh.model.mesh.addElementsByType(superficie, 2, list(range(1, len(triangulos) + 1)),
                                      (np.asarray(triangulos, np.int64).ravel() + 1).tolist())
    gmsh.model.mesh.classifySurfaces(angulo * np.pi / 180.0, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    superficies = [e[1] for e in gmsh.model.getEntities(2)]
    lazo = gmsh.model.geo.addSurfaceLoop(superficies)
    gmsh.model.geo.addVolume([lazo])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", tamano)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * tamano)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", por_vuelta)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if netgen else 0)  # quita tetraedros planos («astillas»)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)  # nodos medios en el punto medio de la arista
    gmsh.model.mesh.generate(3)
    tipos, _, nodos_por_tipo = gmsh.model.mesh.getElements(3)
    if 11 not in list(tipos):
        raise RuntimeError("Gmsh no generó tetraedros")
    conectividad = np.asarray(nodos_por_tipo[list(tipos).index(11)], dtype=np.int64).reshape(-1, 10)
    etiquetas, coordenadas, _ = gmsh.model.mesh.getNodes()
    etiquetas = np.asarray(etiquetas, dtype=np.int64)
    coordenadas = np.asarray(coordenadas, dtype=float).reshape(-1, 3)
    orden = np.argsort(etiquetas)
    indices = orden[np.searchsorted(etiquetas[orden], conectividad)]
    return coordenadas, indices[:, list(DE_GMSH)]


def _remallar_en_sesion(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float, algoritmo: int):
    """Superficie (abierta o cerrada) vuelta a mallar con triángulos de lado ``tamano``: (coordenadas, triángulos)."""

    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("superficie")
    superficie = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, superficie, list(range(1, len(vertices) + 1)), np.asarray(vertices, float).ravel().tolist())
    gmsh.model.mesh.addElementsByType(superficie, 2, list(range(1, len(triangulos) + 1)),
                                      (np.asarray(triangulos, np.int64).ravel() + 1).tolist())
    gmsh.model.mesh.classifySurfaces(angulo * np.pi / 180.0, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    gmsh.option.setNumber("Mesh.MeshSizeMax", tamano)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * tamano)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", algoritmo)
    gmsh.model.mesh.generate(2)
    tipos, _, nodos_por_tipo = gmsh.model.mesh.getElements(2)
    if 2 not in list(tipos):
        raise RuntimeError("Gmsh no generó triángulos")
    tri = np.asarray(nodos_por_tipo[list(tipos).index(2)], dtype=np.int64).reshape(-1, 3)
    etiquetas, coordenadas, _ = gmsh.model.mesh.getNodes()
    etiquetas = np.asarray(etiquetas, dtype=np.int64)
    coordenadas = np.asarray(coordenadas, dtype=float).reshape(-1, 3)
    orden = np.argsort(etiquetas)
    indices = orden[np.searchsorted(etiquetas[orden], tri)]
    usados, inverso = np.unique(indices, return_inverse=True)
    return coordenadas[usados], inverso.reshape(-1, 3)


def _triangulos_de_tapa(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float):
    """Tapa plana mallada con triángulos casi equiláteros de lado ``tamano`` (Frontal-Delaunay), con el contorno partido
    en sus esquinas vivas (si no, Gmsh las recorta: es una sola curva cerrada): (coordenadas, triángulos)."""

    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("seccion")
    superficie = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, superficie, list(range(1, len(vertices) + 1)), np.asarray(vertices, float).ravel().tolist())
    gmsh.model.mesh.addElementsByType(superficie, 2, list(range(1, len(triangulos) + 1)),
                                      (np.asarray(triangulos, np.int64).ravel() + 1).tolist())
    gmsh.model.mesh.classifySurfaces(angulo * np.pi / 180.0, True, True, angulo * np.pi / 180.0)
    gmsh.model.mesh.createGeometry()
    gmsh.option.setNumber("Mesh.MeshSizeMax", tamano)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * tamano)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MinimumCurvePoints", 2)  # un lado corto puede quedar sin partir
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    tipos, _, nodos_por_tipo = gmsh.model.mesh.getElements(2)
    if 2 not in list(tipos):
        raise RuntimeError("Gmsh no generó triángulos")
    tri = np.asarray(nodos_por_tipo[list(tipos).index(2)], dtype=np.int64).reshape(-1, 3)
    etiquetas, coordenadas, _ = gmsh.model.mesh.getNodes()
    etiquetas = np.asarray(etiquetas, dtype=np.int64)
    coordenadas = np.asarray(coordenadas, dtype=float).reshape(-1, 3)
    orden = np.argsort(etiquetas)
    indices = orden[np.searchsorted(etiquetas[orden], tri)]
    usados, inverso = np.unique(indices, return_inverse=True)
    return coordenadas[usados], inverso.reshape(-1, 3)


def triangulos_a_cuadrilateros(coordenadas: np.ndarray, triangulos: np.ndarray):
    """Cada triángulo partido en 3 cuadriláteros por los puntos medios de sus lados y su centro (siempre convexos y con
    la orientación del triángulo): (coordenadas, cuadriláteros (3·m, 4))."""

    t = np.asarray(triangulos, dtype=np.int64)
    aristas = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    unicas, inverso = np.unique(aristas, axis=0, return_inverse=True)
    inverso = inverso.reshape(3, -1).T  # (m, 3): medio de a-b, b-c, c-a
    n, e = len(coordenadas), len(unicas)
    medios = 0.5 * (coordenadas[unicas[:, 0]] + coordenadas[unicas[:, 1]])
    centros = coordenadas[t].mean(axis=1)
    m_ab, m_bc, m_ca = (n + inverso[:, k] for k in range(3))
    g = n + e + np.arange(len(t))
    a, b, c = t.T
    quads = np.vstack([np.stack([a, m_ab, g, m_ca], 1), np.stack([b, m_bc, g, m_ab], 1), np.stack([c, m_ca, g, m_bc], 1)])
    return np.vstack([coordenadas, medios, centros]), quads


#: Seno mínimo de los ángulos de un cuadrilátero de la sección (unos 6°; casi llano o casi nulo da jacobianos malos).
SENO_MINIMO_CUADRILATERO = 0.1


def _peor_seno(coordenadas: np.ndarray, quads: np.ndarray, normal: np.ndarray) -> float:
    """Menor seno de los ángulos interiores de los cuadriláteros (negativo si alguno no es convexo)."""

    p = coordenadas[quads]
    area = sum(np.cross(p[:, i], p[:, (i + 1) % 4]) for i in range(4)) @ normal
    signo = np.where(area < 0, -1.0, 1.0)
    peor = np.inf
    for i in range(4):
        a, b = p[:, (i + 1) % 4] - p[:, i], p[:, (i - 1) % 4] - p[:, i]
        seno = signo * (np.cross(a, b) @ normal) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        peor = min(peor, float(seno.min()))
    return peor


def cuadrilateros(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0):
    """Cuadriláteros de lado ≈ ``tamano`` de una tapa plana con la misma área: triángulos casi equiláteros de 2,5 veces
    ese lado partidos en 3 cuadriláteros cada uno (el algoritmo de cuadriláteros de Gmsh deja triángulos o cuadriláteros
    casi planos con elementos grandes frente a la sección); si no conservan el área o la forma, con elementos menores."""

    area0, _ = _area_y_calidad(vertices, triangulos)
    p = np.asarray(vertices, float)[np.asarray(triangulos, np.int64)]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).sum(axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-300)
    ultimo = "Gmsh no pudo mallar la sección"
    for h in (2.5 * tamano, 1.25 * tamano, 0.6 * tamano):
        gmsh.initialize(["gmsh", "-nopopup"], interruptible=False)
        try:
            coordenadas, tri = _triangulos_de_tapa(gmsh, vertices, triangulos, h, angulo)
        except Exception as exc:  # noqa: BLE001 - Gmsh lanza Exception genéricas
            ultimo = str(exc) or exc.__class__.__name__
            continue
        finally:
            gmsh.finalize()
        coordenadas, quads = triangulos_a_cuadrilateros(coordenadas, tri)
        area, _ = _area_y_calidad(coordenadas, np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
        if abs(area / max(area0, 1e-30) - 1.0) > 0.03:
            ultimo = "la sección mallada con cuadriláteros no conserva el área"
        elif _peor_seno(coordenadas, quads, normal) < SENO_MINIMO_CUADRILATERO:
            ultimo = "la sección tiene cuadriláteros deformados"
        else:
            return coordenadas, quads
    raise RuntimeError(ultimo)


def _area_y_calidad(vertices: np.ndarray, triangulos: np.ndarray):
    """(área total, peor calidad 4√3·A/Σl² de los triángulos: 1 equilátero, 0 degenerado)."""

    p = np.asarray(vertices, float)[np.asarray(triangulos, np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    l2 = sum(((p[:, (k + 1) % 3] - p[:, k]) ** 2).sum(axis=1) for k in range(3))
    return float(area.sum()), float((4.0 * np.sqrt(3.0) * area / np.maximum(l2, 1e-30)).min()) if len(p) else 0.0


def remallar(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0):
    """Prueba MeshAdapt, Delaunay y Frontal-Delaunay (con el tamaño pedido y con la mitad) hasta que uno deja
    triángulos sanos con la misma área; Gmsh se reinicia en cada intento."""

    area0, _ = _area_y_calidad(vertices, triangulos)
    ultimo = "Gmsh no pudo volver a mallar la superficie"
    for h in (tamano, 0.5 * tamano):
        for algoritmo in (1, 5, 6):
            gmsh.initialize(["gmsh", "-nopopup"], interruptible=False)
            try:
                coordenadas, tri = _remallar_en_sesion(gmsh, vertices, triangulos, h, angulo, algoritmo)
            except Exception as exc:  # noqa: BLE001 - Gmsh lanza Exception genéricas
                ultimo = str(exc) or exc.__class__.__name__
                continue
            finally:
                gmsh.finalize()
            area, calidad = _area_y_calidad(coordenadas, tri)
            if calidad >= 0.05 and abs(area / max(area0, 1e-30) - 1.0) <= 0.03:
                return coordenadas, tri
            ultimo = "la superficie vuelta a mallar no conserva la forma"
    raise RuntimeError(ultimo)


def _volumen_superficie(vertices: np.ndarray, triangulos: np.ndarray) -> float:
    p = np.asarray(vertices, float)[np.asarray(triangulos, np.int64)]
    return abs(float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum())) / 6.0


#: Peor calidad 6√2·V/l³ admitida (1 = tetraedro regular): por debajo CalculiX da «nonpositive jacobian».
CALIDAD_MINIMA = 1e-3


def _peor_calidad(coordenadas: np.ndarray, tetraedros: np.ndarray) -> float:
    p = coordenadas[tetraedros[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0
    largo = np.max([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=0)
    return float((6.0 * np.sqrt(2.0) * v / np.maximum(largo, 1e-300) ** 3).min()) if len(p) else 0.0


def _volumen_tetraedros(coordenadas: np.ndarray, tetraedros: np.ndarray) -> float:
    p = coordenadas[tetraedros[:, :4]]
    return float(np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])).sum()) / 6.0


def rellenar(gmsh, vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0, provisional=None):
    """Tetraedros de una superficie cerrada. Si Gmsh falla con el tamaño pedido (p. ej. «Invalid boundary mesh
    (overlapping facets)» en una arandela con elementos mayores que su espesor) o la malla no conserva el volumen (la
    superficie vuelta a mallar con elementos grandes recorta las curvas de una pieza pequeña), se prueba con la mitad
    y la cuarta parte del tamaño (refinar por curvatura costaría hasta 50 veces más tetraedros en las arandelas). Gmsh se reinicia en cada intento: tras un error su estado queda corrompido.

    El optimizador Netgen (que quita los tetraedros planos) solo se usa si la malla los tiene: con algunas piezas
    pequeñas (pernos) cierra Gmsh con una violación de segmento. ``provisional(coordenadas, tetraedros)`` recibe cada
    mejor malla aceptable antes de seguir probando: si Gmsh se cierra después, la pieza conserva la que ya tenía."""

    objetivo = _volumen_superficie(vertices, triangulos)
    primero, mejor = None, None
    for h, por_vuelta in ((tamano, 0), (0.5 * tamano, 0), (0.25 * tamano, 0)):
        malla = None
        for netgen in (False, True):
            gmsh.initialize(["gmsh", "-nopopup"], interruptible=False)
            try:
                coordenadas, tetraedros = _mallar_en_sesion(gmsh, vertices, triangulos, h, angulo, por_vuelta, netgen)
            except Exception as exc:  # noqa: BLE001 - Gmsh lanza Exception genéricas
                primero = primero or (str(exc) or exc.__class__.__name__)
                break
            finally:
                gmsh.finalize()
            if _peor_calidad(coordenadas, tetraedros) >= CALIDAD_MINIMA:
                malla = coordenadas, tetraedros
                break
            primero = primero or "Gmsh dejó tetraedros planos"
            if not 0.6 <= _volumen_tetraedros(coordenadas, tetraedros) / max(objetivo, 1e-300) <= 1.4:
                break  # no serviría ni sin astillas: no vale la pena optimizarla (Netgen es lo que cierra Gmsh)
        if malla is None:
            continue
        coordenadas, tetraedros = malla
        relacion = _volumen_tetraedros(coordenadas, tetraedros) / max(objetivo, 1e-300)
        if VOLUMEN_MINIMO <= relacion <= VOLUMEN_MAXIMO:
            return coordenadas, tetraedros
        if mejor is None or abs(relacion - 1.0) < abs(mejor[0] - 1.0):
            mejor = (relacion, coordenadas, tetraedros)
            if provisional is not None and 0.6 <= relacion <= 1.4:
                provisional(coordenadas, tetraedros)
        primero = primero or f"la malla de Gmsh tiene el {relacion:.0%} del volumen de la pieza"
    # aunque recorte algo (una arandela de 10 mm con elementos de 6 cm), mejor que miles de vóxeles: se avisa al mallar
    if mejor is not None and 0.6 <= mejor[0] <= 1.4:
        return mejor[1], mejor[2]
    raise RuntimeError(primero)


def mallar(vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0):
    """Tetraedros cuadráticos de una pieza (coordenadas, (M, 10) en el orden de CalculiX), en este mismo proceso."""

    import gmsh

    return rellenar(gmsh, vertices, triangulos, tamano, angulo)


def _escribir(ruta: Path, texto: str) -> None:
    """Escritura atómica: el proceso padre nunca lee un archivo a medio escribir."""

    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(texto, encoding="utf-8")
    os.replace(temporal, ruta)


def mallar_lote(entrada: str, carpeta: str, desde: int = 0) -> int:
    import gmsh

    salida = Path(carpeta)
    with np.load(entrada) as datos:
        n, tamano, angulo = int(datos["n"]), float(datos["tamano"]), float(datos["angulo"])
        piezas = [(np.array(datos[f"v{i}"]), np.array(datos[f"t{i}"]), int(datos[f"d{i}"]) if f"d{i}" in datos else 3,
                   float(datos[f"h{i}"]) if f"h{i}" in datos else tamano)
                  for i in range(n)]
    for i in range(desde, n):
        _escribir(salida / "actual.txt", str(i))
        vertices, triangulos, dimension, tamano_pieza = piezas[i]
        if dimension == 4:
            try:
                coordenadas, quads = cuadrilateros(gmsh, vertices, triangulos, tamano_pieza, angulo)
            except Exception as exc:  # noqa: BLE001
                _escribir(salida / f"err_{i}.txt", str(exc) or exc.__class__.__name__)
                continue
            resultado = {"coordenadas": coordenadas, "cuadrilateros": quads}
        elif dimension == 2:
            try:
                coordenadas, tri = remallar(gmsh, vertices, triangulos, tamano, angulo)
            except Exception as exc:  # noqa: BLE001
                _escribir(salida / f"err_{i}.txt", str(exc) or exc.__class__.__name__)
                continue
            resultado = {"coordenadas": coordenadas, "triangulos": tri}
        else:
            def provisional(coordenadas, tetraedros, i=i):
                temporal = salida / f"res_{i}.tmp.npz"
                np.savez(temporal, coordenadas=coordenadas, tetraedros=tetraedros)
                os.replace(temporal, salida / f"res_{i}.npz")

            try:
                coordenadas, tetraedros = rellenar(gmsh, vertices, triangulos, tamano, angulo, provisional)
            except Exception as exc:  # noqa: BLE001
                _escribir(salida / f"err_{i}.txt", str(exc) or exc.__class__.__name__)
                continue
            resultado = {"coordenadas": coordenadas, "tetraedros": tetraedros}
        temporal = salida / f"res_{i}.tmp.npz"
        np.savez(temporal, **resultado)
        os.replace(temporal, salida / f"res_{i}.npz")
    _escribir(salida / "actual.txt", str(n))
    return 0


def main(argumentos=None) -> int:
    argumentos = list(sys.argv[1:] if argumentos is None else argumentos)
    try:
        if argumentos[:1] == ["--lote"] and len(argumentos) in (3, 5):
            desde = int(argumentos[4]) if len(argumentos) == 5 and argumentos[3] == "--desde" else 0
            return mallar_lote(argumentos[1], argumentos[2], desde)
    except Exception as exc:  # noqa: BLE001 - Gmsh no disponible, entrada ilegible...
        print(f"ERROR: {exc}")
        return 1
    print("ERROR: uso: proceso_gmsh --lote entrada.npz carpeta [--desde k]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
