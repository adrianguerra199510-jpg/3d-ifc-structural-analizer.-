"""Proceso hijo que malla con Gmsh un sólido triangulado.

Gmsh puede quedarse colgado o caerse con superficies complicadas; ejecutarlo en un proceso aparte permite ponerle un
límite de tiempo y seguir con otro tipo de elemento sin congelar la aplicación. Uso (lo lanza ``tetraedros``):

    python -m ifc_structural.solido.proceso_gmsh entrada.npz salida.npz

``entrada.npz`` lleva ``vertices``, ``triangulos``, ``tamano`` y ``angulo``; ``salida.npz`` recibe ``coordenadas`` y
``tetraedros`` (orden de nodos de CalculiX). Termina con código 0 si todo salió bien.
"""

from __future__ import annotations

import sys

import numpy as np

#: Orden de CalculiX a partir del de Gmsh para ``Tetrahedron10`` (cambian los dos últimos nodos medios).
DE_GMSH = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)


def mallar(vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0):
    """Tetraedros cuadráticos (coordenadas, (M, 10) en el orden de Gmsh convertido al de CalculiX)."""

    import gmsh

    gmsh.initialize(["gmsh", "-nopopup"], interruptible=False)
    try:
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
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
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
    finally:
        gmsh.finalize()
    orden = np.argsort(etiquetas)
    indices = orden[np.searchsorted(etiquetas[orden], conectividad)]
    return coordenadas, indices[:, list(DE_GMSH)]


def main(argumentos=None) -> int:
    argumentos = list(sys.argv[1:] if argumentos is None else argumentos)
    if len(argumentos) != 2:
        print("ERROR: uso: proceso_gmsh entrada.npz salida.npz")
        return 2
    try:
        datos = np.load(argumentos[0])
        coordenadas, tetraedros = mallar(datos["vertices"], datos["triangulos"], float(datos["tamano"]), float(datos["angulo"]))
        np.savez(argumentos[1], coordenadas=coordenadas, tetraedros=tetraedros)
    except Exception as exc:  # noqa: BLE001 - Gmsh lanza Exception genéricas
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
