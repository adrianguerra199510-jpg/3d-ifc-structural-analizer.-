# Analizador Estructural 3D desde IFC (Fase 1)

Aplicación de escritorio en Python para análisis estructural lineal a partir de
modelos IFC (exportados desde Revit), usando **CalculiX (ccx)** como solver y
una experiencia de uso inspirada en Autodesk Inventor Nastran.

## Alcance de la Fase 1

- Importar un IFC con `ifcopenshell` y listar vigas, columnas y losas (ID,
  nombre, nodos, perfil, material, espesor).
- Visualizar la geometría en 3D con PyVista y seleccionar elementos/caras.
- Asignar **cargas de área** (kPa) sobre losas, transferidas a vigas por
  **áreas tributarias** (modo por defecto). Asignar cargas lineales y
  puntuales sobre vigas.
- Asignar apoyos en nodos.
- Definir casos de carga y al menos una combinación.
- Generar un `.inp` de CalculiX con elementos **B31** para vigas/columnas,
  material elástico lineal, secciones y cargas.
- Ejecutar `ccx` mediante `subprocess` y verificar que terminó con éxito.
- Leer el `.dat` con N, V, M por elemento y mostrarlos en tabla y diagrama.

Fuera de alcance en esta fase: modo shell para losas (placas S4/S3), no
linealidad, sismo, mallado de sólidos, reportes PDF. La arquitectura ya deja
espacio para esas extensiones (ver `core/modelo.py::Losa.modo_analisis`).

## Transferencia de cargas de área → vigas (decisión de diseño)

Las losas en una dirección se reparten como carga lineal uniforme sobre las
vigas de apoyo, usando el ancho tributario clásico de la losa (mitad de la
luz hacia cada lado, o hasta el borde de la losa).

Para losas en dos direcciones se usa el **método de líneas de reparto a 45°
(áreas tributarias)**, no el de líneas de fluencia ("yield-line"). Se eligió
áreas tributarias a 45° porque:

1. Es el método que usan en la práctica herramientas como SAP2000/ETABS para
   repartir cargas de losa en vigas de un modelo de pórtico, y da resultados
   conservadores y predecibles para diseño elástico.
2. Las líneas de fluencia requieren conocer la relación de momentos
   resistentes entre franjas, lo que no está disponible en esta fase (no hay
   diseño de refuerzo todavía) y añadiría una no linealidad conceptual a un
   análisis que en Fase 1 es puramente elástico lineal.
3. El algoritmo de áreas tributarias a 45° es geométrico y determinista: se
   calcula recortando el polígono de la losa con bisectrices a 45° desde cada
   borde, lo que es robusto de implementar con `shapely` para losas de
   geometría arbitraria (no solo rectángulos).

El resultado de la transferencia se muestra en el visor como cargas
lineales equivalentes sobre cada viga (uniformes o trapezoidales simplificadas
a un valor promedio en Fase 1).

## Peso propio

Se calcula a partir de la densidad del material y el área de la sección (o el
espesor para losas/muros) y se aplica como `*DLOAD` tipo `GRAV` en el caso de
carga `D` (carga muerta).

## Instalación

```bash
pip install -r requirements.txt
```

CalculiX (`ccx`) debe estar instalado en el sistema. La ruta al ejecutable se
configura en `config.json`:

```json
{
  "ruta_ccx": "ccx"
}
```

## Ejecutar la aplicación

```bash
python -m ifc_structural.app
```

## Ejecutar las pruebas

```bash
pytest
```

## Estructura del proyecto

```
src/ifc_structural/
├── app.py                        # punto de entrada de la GUI
├── config.py                     # carga de config.json
├── core/
│   ├── modelo.py                 # Proyecto, Nodo, Material, Perfil, Viga, Columna, Losa, Muro
│   ├── cargas.py                  # CargaPuntual, CargaLineal, CargaArea, Apoyo
│   ├── casos_carga.py             # CasoCarga, Combinacion
│   └── transferencia.py           # áreas tributarias (1 y 2 direcciones) + peso propio
├── ifc/
│   └── importador.py              # lectura de IFC con ifcopenshell
├── solver/
│   ├── generador_inp.py           # escritura del .inp de CalculiX
│   ├── ejecutor_ccx.py            # ejecución de ccx vía subprocess
│   └── lector_resultados.py       # parseo del .dat (N, V, M, desplazamientos, reacciones)
├── combinaciones/
│   └── motor_superposicion.py     # superposición lineal de casos → combinaciones
└── gui/
    ├── ventana_principal.py
    ├── visor_3d.py                 # widget PyVista + selección
    ├── panel_elementos.py
    ├── dialogo_cargas.py
    ├── dialogo_apoyos.py
    ├── dialogo_casos.py
    └── panel_resultados.py
```
