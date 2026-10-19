"""Ventana principal de la aplicación: une el visor 3D, los paneles de
elementos y resultados, y los diálogos de cargas/apoyos/casos en un único
flujo: Importar IFC -> asignar apoyos y cargas -> definir casos y
combinaciones -> Analizar -> ver resultados."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QFileDialog, QMainWindow, QMessageBox

from ..combinaciones.motor_superposicion import superponer_todas_las_combinaciones
from ..config import Configuracion, cargar_configuracion
from ..core.cargas import Apoyo, CargaArea, CargaLineal
from ..core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ..core.modelo import Proyecto
from ..core.persistencia import cargar_proyecto, guardar_proyecto
from ..core.transferencia import cargas_area_peso_propio_losas, peso_propio_barras, transferir_carga_area
from ..ifc.importador import importar_ifc
from ..solver.ejecutor_ccx import ejecutar_ccx_todos_los_casos
from ..solver.generador_inp import construir_malla, generar_inp_todos_los_casos
from ..solver.lector_resultados import leer_resultados_caso
from .dialogo_apoyos import DialogoApoyo
from .dialogo_cargas import DialogoCargas
from .dialogo_casos import DialogoCasos
from .panel_elementos import PanelElementos
from .panel_resultados import PanelResultados
from .visor_3d import Visor3D

logger = logging.getLogger(__name__)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador Estructural 3D (IFC + CalculiX) — Fase 1")
        self.resize(1400, 900)

        self.configuracion: Configuracion = cargar_configuracion()
        self.proyecto = Proyecto()
        self.apoyos: List[Apoyo] = []
        self.casos: Dict[str, CasoCarga] = {}
        self.combinaciones: Dict[str, Combinacion] = {}
        self.cargas_area_por_caso: Dict[str, list] = {}
        self.cargas_lineales_por_caso: Dict[str, list] = {}
        self.cargas_puntuales_por_caso: Dict[str, list] = {}
        self._contador_cargas = 0
        self._contador_apoyos = 0

        self.malla = None
        self.resultados_por_caso: dict = {}
        self.resultados_por_combinacion: dict = {}

        self.visor = Visor3D(self)
        self.setCentralWidget(self.visor)

        self.panel_elementos = PanelElementos(self)
        dock_elementos = QDockWidget("Elementos", self)
        dock_elementos.setWidget(self.panel_elementos)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_elementos)

        self.panel_resultados = PanelResultados(self)
        dock_resultados = QDockWidget("Resultados", self)
        dock_resultados.setWidget(self.panel_resultados)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_resultados)

        self.panel_elementos.elemento_seleccionado.connect(self._al_seleccionar_elemento)
        self.visor.elemento_seleccionado.connect(self._al_seleccionar_elemento)

        self._crear_menus()
        self.statusBar().showMessage("Listo. Comience importando un modelo IFC (menú Archivo).")

    # -- Menús ------------------------------------------------------------
    def _crear_menus(self) -> None:
        menu_archivo = self.menuBar().addMenu("&Archivo")
        menu_archivo.addAction("Importar IFC...", self._importar_ifc)
        menu_archivo.addSeparator()
        menu_archivo.addAction("Abrir proyecto...", self._abrir_proyecto)
        menu_archivo.addAction("Guardar proyecto como...", self._guardar_proyecto)

        menu_cargas = self.menuBar().addMenu("&Cargas")
        menu_cargas.addAction("Asignar carga...", self._asignar_carga)

        menu_apoyos = self.menuBar().addMenu("A&poyos")
        menu_apoyos.addAction("Asignar apoyo...", self._asignar_apoyo)

        menu_casos = self.menuBar().addMenu("Cas&os")
        menu_casos.addAction("Gestionar casos y combinaciones...", self._gestionar_casos)

        menu_analisis = self.menuBar().addMenu("&Analizar")
        menu_analisis.addAction("Ejecutar análisis...", self._ejecutar_analisis)

    # -- Sincronización visor <-> tabla ------------------------------------
    def _al_seleccionar_elemento(self, tipo: str, id_: str) -> None:
        self.visor.resaltar(tipo, id_)

    # -- Archivo ------------------------------------------------------------
    def _importar_ifc(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Importar IFC", "", "Archivos IFC (*.ifc)")
        if not ruta:
            return
        try:
            self.proyecto = importar_ifc(ruta)
        except Exception as exc:
            logger.exception("Error al importar el IFC")
            QMessageBox.critical(self, "Error al importar", str(exc))
            return

        self.visor.mostrar_proyecto(self.proyecto)
        self.panel_elementos.actualizar(self.proyecto)
        QMessageBox.information(
            self,
            "Importación completa",
            f"Vigas: {len(self.proyecto.vigas)}\n"
            f"Columnas: {len(self.proyecto.columnas)}\n"
            f"Losas: {len(self.proyecto.losas)}\n"
            f"Muros: {len(self.proyecto.muros)}",
        )

    def _guardar_proyecto(self) -> None:
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        guardar_proyecto(
            Path(ruta),
            self.proyecto,
            self.apoyos,
            self.casos,
            self.combinaciones,
            self.cargas_area_por_caso,
            self.cargas_lineales_por_caso,
            self.cargas_puntuales_por_caso,
        )
        self.statusBar().showMessage(f"Proyecto guardado en {ruta}")

    def _abrir_proyecto(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        (
            self.proyecto,
            self.apoyos,
            self.casos,
            self.combinaciones,
            self.cargas_area_por_caso,
            self.cargas_lineales_por_caso,
            self.cargas_puntuales_por_caso,
        ) = cargar_proyecto(Path(ruta))
        self.visor.mostrar_proyecto(self.proyecto)
        self.panel_elementos.actualizar(self.proyecto)
        self.statusBar().showMessage(f"Proyecto cargado desde {ruta}")

    # -- Apoyos y cargas ----------------------------------------------------
    def _asignar_apoyo(self) -> None:
        if not self.proyecto.nodos:
            QMessageBox.warning(self, "Sin modelo", "Importe un IFC primero.")
            return
        dialogo = DialogoApoyo(self.proyecto, parent=self)
        if dialogo.exec():
            self._contador_apoyos += 1
            self.apoyos.append(dialogo.apoyo_resultante(f"AP{self._contador_apoyos}"))
            self.statusBar().showMessage(f"Apoyo agregado en el nodo {self.apoyos[-1].nodo_id}")

    def _asignar_carga(self) -> None:
        if not self.casos:
            QMessageBox.warning(self, "Sin casos de carga", "Defina al menos un caso de carga primero (menú Casos).")
            return
        if not self.proyecto.vigas and not self.proyecto.losas:
            QMessageBox.warning(self, "Sin modelo", "Importe un IFC primero.")
            return
        dialogo = DialogoCargas(self.proyecto, list(self.casos), parent=self)
        if dialogo.exec():
            self._contador_cargas += 1
            carga = dialogo.resultado(f"CARGA{self._contador_cargas}")
            if isinstance(carga, CargaArea):
                self.cargas_area_por_caso.setdefault(carga.caso_id, []).append(carga)
            elif isinstance(carga, CargaLineal):
                self.cargas_lineales_por_caso.setdefault(carga.caso_id, []).append(carga)
            else:
                self.cargas_puntuales_por_caso.setdefault(carga.caso_id, []).append(carga)
            self.statusBar().showMessage("Carga agregada.")

    def _gestionar_casos(self) -> None:
        dialogo = DialogoCasos(self.casos, self.combinaciones, parent=self)
        if dialogo.exec():
            self.casos, self.combinaciones = dialogo.resultado()

    # -- Análisis -----------------------------------------------------------
    def _ejecutar_analisis(self) -> None:
        if not self.proyecto.vigas and not self.proyecto.columnas:
            QMessageBox.warning(self, "Sin modelo", "Importe un IFC primero.")
            return
        if not self.casos:
            QMessageBox.warning(self, "Sin casos de carga", "Defina al menos un caso de carga.")
            return
        if not self.apoyos:
            respuesta = QMessageBox.question(
                self, "Sin apoyos", "No hay apoyos definidos; el análisis fallará. ¿Continuar igualmente?"
            )
            if respuesta != QMessageBox.Yes:
                return

        try:
            self.statusBar().showMessage("Preparando cargas (peso propio y transferencia de losas)...")
            cargas_lineales_por_caso = self._preparar_cargas_lineales()

            self.statusBar().showMessage("Construyendo la malla de análisis...")
            self.malla = construir_malla(self.proyecto, cargas_lineales_por_caso, self.cargas_puntuales_por_caso)

            self.statusBar().showMessage("Generando los archivos .inp...")
            directorio = self.configuracion.directorio_trabajo_absoluto
            rutas_inp = generar_inp_todos_los_casos(
                self.proyecto,
                self.malla,
                cargas_lineales_por_caso,
                self.cargas_puntuales_por_caso,
                self.apoyos,
                directorio,
            )

            self.statusBar().showMessage("Ejecutando CalculiX (ccx)...")
            resultados_ejecucion = ejecutar_ccx_todos_los_casos(rutas_inp, self.configuracion.ruta_ccx)

            casos_fallidos = [c for c, r in resultados_ejecucion.items() if not r.exitoso]
            if casos_fallidos:
                detalle = "\n".join(f"- {c}" for c in casos_fallidos)
                QMessageBox.critical(
                    self, "ccx falló", f"Los siguientes casos no terminaron con éxito:\n{detalle}"
                )
                self.statusBar().showMessage("El análisis falló.")
                return

            self.resultados_por_caso = {
                caso_id: leer_resultados_caso(r.ruta_dat, r.ruta_frd) for caso_id, r in resultados_ejecucion.items()
            }
            self.resultados_por_combinacion = superponer_todas_las_combinaciones(
                self.resultados_por_caso, list(self.combinaciones.values())
            )

            self.panel_resultados.establecer_malla(self.malla)
            resultados_para_mostrar = dict(self.resultados_por_caso)
            resultados_para_mostrar.update(self.resultados_por_combinacion)
            self.panel_resultados.actualizar_resultados(resultados_para_mostrar)

            self.statusBar().showMessage("Análisis completo.")
            QMessageBox.information(self, "Análisis completo", "El análisis terminó correctamente.")
        except Exception as exc:
            logger.exception("Error durante el análisis")
            QMessageBox.critical(self, "Error durante el análisis", str(exc))
            self.statusBar().showMessage("El análisis falló.")

    def _preparar_cargas_lineales(self) -> Dict[str, list]:
        """Combina, por cada caso de carga: las cargas lineales manuales,
        el peso propio de vigas/columnas/losas (aplicado al caso de tipo
        "D", carga muerta, si existe) y la transferencia por áreas
        tributarias de cada CargaArea a sus vigas de soporte."""

        resultado: Dict[str, list] = {caso_id: list(cargas) for caso_id, cargas in self.cargas_lineales_por_caso.items()}

        caso_muerta = next((c.id for c in self.casos.values() if c.tipo == TipoCaso.D), None)
        if caso_muerta:
            resultado.setdefault(caso_muerta, [])
            resultado[caso_muerta].extend(peso_propio_barras(self.proyecto, caso_muerta))
            for carga_area_pp in cargas_area_peso_propio_losas(self.proyecto, caso_muerta):
                resultado[caso_muerta].extend(transferir_carga_area(self.proyecto, carga_area_pp))

        for caso_id, cargas_area in self.cargas_area_por_caso.items():
            resultado.setdefault(caso_id, [])
            for carga_area in cargas_area:
                resultado[caso_id].extend(transferir_carga_area(self.proyecto, carga_area))

        for caso_id in self.casos:
            resultado.setdefault(caso_id, [])

        return resultado
