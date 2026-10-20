# Analizador Estructural 3D desde IFC (Fase 1)

Aplicación de escritorio en Python para análisis estructural lineal a partir de
modelos IFC (exportados desde Revit), usando **CalculiX (ccx)** como solver y
una experiencia de uso inspirada en Autodesk Inventor Nastran.

## Alcance de la Fase 1

- Importar un IFC con `ifcopenshell` y listar vigas, columnas y losas. Los
  extremos de vigas y columnas que quedan a menos de 0,4 m se conectan
  automáticamente y las vigas que sostienen cada losa se detectan solas.
- Visualizar la geometría en 3D (sólidos reales) y **seleccionar con el mouse**
  elementos completos o **caras** de los elementos.
- Asignar **apoyos y cargas sobre lo seleccionado** (ver «Cómo se usa»).
- Casos de carga D y L y combinaciones ACI 318 creados por defecto (editables).
- Generar el `.inp` de CalculiX con elementos **B31** y ejecutar `ccx`.
- Mostrar **solo esfuerzos** (von Mises, σ1, σ3, τmax) en MPa como mapa de
  colores sobre el sólido de cada barra, por caso, por combinación o como
  envolvente de combinaciones. No se muestran fuerzas, desplazamientos ni
  reacciones.

- **ANALIZAR 3D**: analiza todo el modelo como **sólido** (ver más abajo), con la geometría
  real de cada pieza, sin instalar nada más.

Fuera de alcance en esta fase: modo shell para losas (placas S4/S3), no
linealidad, sismo, reportes PDF.

## Cómo se usa

1. **Importar IFC** (barra de herramientas).
2. **Seleccionar** en el visor con clic izquierdo. Modo **Elemento** (toda la
   viga/columna/losa) o modo **Cara** (una cara del sólido). `Ctrl`/`Shift` +
   clic suma o quita elementos de la selección; clic en vacío deselecciona.
   Para alcanzar la base de una columna use *Vistas ▾ → Inferior*.
3. **Asignar apoyo**: con las caras extremas seleccionadas (base de columnas),
   elija empotramiento, articulado o personalizado.
4. **Asignar carga**: sobre la cara superior/inferior de una losa (presión en
   kPa), sobre una viga o columna completa (kN/m, con dirección) o sobre sus
   caras (presión; las caras laterales dan carga lineal, las extremas una
   fuerza axial). Elija el caso de carga en el diálogo.
5. **▶ ANALIZAR** (o `F5`): valida el modelo (estabilidad, barras sueltas,
   losas sin vigas, etc.), ejecuta CalculiX por cada caso y abre la pestaña
   **Esfuerzos**.
6. **Casos y combinaciones** (botón de la barra o menú *Asignar*): formulario con
   dos pestañas. En *Casos de carga* se crean con identificador, nombre y
   tipo (D, L, W, E…). En *Combinaciones* se escribe el nombre y el **factor de
   cada caso** (0 = no participa), o se parte de una plantilla ACI 318 /
   AASHTO y se edita; se muestra la fórmula resultante («1.2·D + 1.6·L»).
   «Guardar» confirma el formulario en la lista y «OK» aplica todo; «Cancelar»
   no cambia nada. Al eliminar un caso se quita de las combinaciones y se
   borran sus cargas (con aviso previo). Al asignar una carga se elige el caso.
7. En **Esfuerzos** elija resultado (caso, combinación o envolvente), la
   magnitud, la escala (automática o manual en MPa) y haga clic en el modelo
   para leer el valor en un punto. «Volver al modelo» regresa a la edición;
   si cambia algo hay que volver a analizar.

Limitaciones conocidas: CalculiX no detecta estructuras inestables, por eso la
aplicación comprueba el equilibrio de cuerpo rígido antes de calcular; los
esfuerzos en los extremos de las barras son extrapolados y pueden ser
inexactos en los nudos (concentraciones); para barras inclinadas con giros
restringidos CalculiX puede fallar y se muestra un diagnóstico; las cargas
sobre muros no se transfieren y los apoyos solo se asignan en extremos de
vigas y columnas.

## Interfaz

- **Pantalla de inicio** con los tres pasos (importar, asignar, analizar); los paneles
  aparecen al cargar un modelo y se reabren desde *Ver*.
- **Panel único a la izquierda en forma de árbol** con todo el proyecto: elementos (con sus asignaciones colgando
  de cada uno), apoyos, conexiones, cargas por caso, materiales, casos, combinaciones y resultados. Seleccionar
  en el árbol selecciona en el visor y viceversa; clic derecho abre el menú de lo que se puede hacer, *Supr*
  elimina asignaciones, doble clic en un resultado lo muestra, y hay un buscador. El panel de resultados
  (a la derecha) aparece al analizar.
- **Barra de herramientas agrupada** con iconos propios: archivo, modo de selección
  (elemento / cara), caso activo, asignaciones (apoyo, carga, material, conexión),
  materiales y casos, y el botón verde **ANALIZAR** (F5).
- **Barra de vista** sobre el visor: isométrica, planta, frontal, lateral, inferior,
  encuadrar, visibilidad por tipo de elemento (los iconos llevan el color del elemento)
  y el cambio Modelo / Esfuerzos.
- **Clic derecho** sobre un elemento: carga, apoyo, material, conexión, propiedades.
- Barra de estado con el modo de selección, el caso activo y la selección actual.

El logo y todos los iconos son SVG dibujados en código (`gui/iconos.py`); la paleta y
la hoja de estilos están en `gui/estilo.py`. Para regenerar `recursos/logo.svg`,
`logo.png` y `logo.ico` (icono del `.exe`): `python recursos/generar_logo.py`.

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

## Análisis 3D (modelo sólido)

El botón **ANALIZAR 3D** (F6) malla cada viga, columna, losa y muro con su geometría real
en **hexaedros** (CalculiX `C3D8I`) y resuelve el modelo completo como un sólido. Es la
misma idea que la versión portable: el mallador es propio (barrido de la sección de cada
barra, sin Gmsh ni otras dependencias), así que no hay nada que instalar.

- **Geometría.** Un perfil en I se malla con sus alas y su alma; un rectángulo, con
  su sección. Las barras alineadas con el mismo perfil (la columna partida por una unión)
  se funden en una sola pieza y el extremo de una viga se **recorta hasta la superficie**
  de la columna que atraviesa, de modo que su cara extrema queda apoyada en ella. A las
  losas y muros se les quitan los hexaedros que solapan con vigas y columnas.
- **Uniones.** Las piezas se unen con `*TIE` por las caras que se miran y se tocan (cara
  extrema de la viga contra el ala, losa sobre viga, tramos de columna): equivale a una
  soldadura continua. Una losa que el IFC coloca con holgura o solapada se une a las caras
  cercanas y se avisa. No hay pernos, contacto con separación ni fricción.
- **Apoyos y cargas.** Se aplican sobre la **sección o la cara** elegida: un apoyo
  empotrado fija todos los nodos de la cara; uno con giros libres (pasador, rodillo)
  mueve la cara como cuerpo rígido con un nodo de referencia. Las fuerzas y momentos puntuales
  se reparten por el área de la sección, las cargas lineales a lo largo de ella y las presiones
  sobre las caras de la losa. El peso propio es `*DLOAD,GRAV` con el peso específico del
  material (caso de tipo D). Las conexiones manuales (articulaciones) no se aplican.
- **Tamaño de elemento.** El diálogo propone el más fino que da unos 40 000 hexaedros y
  estima la cantidad (máximo 400 000). Con 1 elemento en el espesor de un ala, `C3D8I`
  flexiona bien; las concentraciones junto a apoyos, cargas puntuales y esquinas
  interiores dependen del tamaño y no convergen del todo (son singularidades).
- **Resultados.** Los mismos de siempre (von Mises, σ1, σ3, τmax, casos, combinaciones y
  envolvente), pero sobre el sólido: se ven las alas, el alma y las uniones. Comprobado
  contra `M·c/I`, `wL²/8` y el equilibrio (`tests/test_solido.py`): error < 3 %.

## Estructuras de acero: perfiles en I y uniones

- **Perfiles reales.** Los perfiles `IfcIShapeProfileDef` (IPE, HEA, HEB…), los
  rectangulares y los circulares se leen del IFC (del conjunto de perfiles del
  material o de la extrusión del elemento) con su peralte orientado como está
  colocado en el modelo. Un perfil en I se analiza con su **forma real**: CalculiX
  2.21 no tiene sección en I para vigas B31, así que se descompone en alas y alma
  (tres barras rectangulares unidas en cada nudo por tramos rígidos). La flexión
  fuerte y débil y la torsión de sección abierta salen del modelo; el cortante
  (que ccx reparte por igual entre alas y alma) se rehace: `V/(tw·hw)` en el alma.
  Se comprobó contra `M·c/I` y `V/(tw·hw)` (error < 1 %, ver `tests/test_perfiles_i.py`).
  Los radios de acuerdo alma–ala no se modelan (~3 % menos de área que la tabla).
- **Extremos de barra.** ccx entrega en los extremos de cada barra esfuerzos de
  flexión hasta un 5–20 % bajos (justo donde el momento suele ser máximo): se
  extrapolan desde el interior de la barra.
- **Uniones en T.** Una viga que llega a media altura de una columna continua
  (o una secundaria contra una principal) se conecta: su extremo pasa al eje de la otra
  barra y esta se parte en el punto de unión. Los `IfcMember` se importan como viga o
  columna según su inclinación.
- **No se modelan** placas de extremo, soldaduras, pernos ni rigidizadores (se
  avisa al importar): las barras se unen por sus ejes. Tampoco se incluye el cortante
  por torsión de los perfiles en I. Los perfiles huecos, en L o en U no están
  soportados (se aproximan con un rectángulo macizo y se avisa); el círculo se
  analiza como un cuadrado de igual inercia.
- **Cargas en los extremos.** En modo **Cara** se pueden cargar las caras extremas
  de cada barra: una presión (× el área real del perfil) o una fuerza puntual en la
  dirección elegida, aplicada en el nodo del extremo.

## Peso propio

Se calcula a partir de la densidad del material y el área de la sección (o el
espesor para losas/muros) y se aplica como `*DLOAD` tipo `GRAV` en el caso de
carga `D` (carga muerta).

## Versión portable para Windows (incluye CalculiX)

La carpeta portable **no necesita instalar Python ni CalculiX**: trae el
solver `ccx.exe` (CalculiX 2.21, licencia GPL, ver `herramientas/ccx/LEEME.txt`)
y la aplicación lo encuentra sola.

- **Descargarla ya compilada**: pestaña [**Releases**](../../releases/tag/ejecutable-windows),
  archivo `AnalizadorEstructuralIFC-portable.zip` (la genera GitHub Actions con
  `.github/workflows/compilar-ejecutable-windows.yml`, ejecutando el mismo
  script de abajo y una autoprueba de CalculiX).
- **Compilarla uno mismo**: con Python 3.10–3.13 instalado, doble clic en
  `COMPILAR_PORTABLE.bat`. Crea un entorno aislado, instala dependencias,
  compila con PyInstaller, agrega `herramientas/ccx`, ejecuta la autoprueba y
  deja la carpeta `C:\AnalizadorEstructuralIFC_portable\AnalizadorEstructuralIFC\` y, junto al `.bat`,
  `AnalizadorEstructuralIFC-portable.zip`. Compila en rutas cortas de `C:\` porque
  Windows limita las rutas a 260 caracteres y PySide6 tiene rutas muy profundas.

`AnalizadorEstructuralIFC.exe --autoprueba` resuelve una viga en voladizo y la
compara con la teoría de vigas (escribe `autoprueba.txt`): sirve para comprobar
CalculiX en cualquier PC.

## Instalación

```bash
pip install -r requirements.txt
```

CalculiX (`ccx`) debe estar disponible: con `"ruta_ccx": "auto"` (valor por
defecto, en `config.json`) se usa el de `herramientas/ccx` (Windows) o el que
esté en el `PATH`. Para usar otro, escriba su ruta completa:

```json
{
  "ruta_ccx": "auto"
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
│   ├── transferencia.py           # áreas tributarias (1 y 2 direcciones) + peso propio
│   ├── geometria.py               # sólidos y caras de cada elemento
│   ├── asignacion.py              # selección → apoyos y cargas
│   ├── vigas_soporte.py           # detección de vigas bajo cada losa
│   └── conectividad.py            # unión de nodos cercanos
├── ifc/
│   └── importador.py              # lectura de IFC con ifcopenshell
├── solver/
│   ├── generador_inp.py           # escritura del .inp de CalculiX (perfiles en I como alas + alma)
│   ├── recuperacion_perfiles.py   # cortante de perfiles en I y esfuerzos en los extremos de barra
│   ├── ejecutor_ccx.py            # ejecución de ccx vía subprocess
│   ├── lector_resultados.py       # malla expandida y tensor de esfuerzos del .frd
│   ├── analisis.py                # validación, análisis y combinaciones
│   └── mecanismos.py              # detección de mecanismos
├── solido/                        # análisis 3D como modelo sólido
│   ├── seccion2d.py               # malla de cuadriláteros de la sección (rectángulo, doble T)
│   ├── miembros.py                # barras fundidas y recortadas en las uniones
│   ├── malla.py                   # hexaedros de barras, losas y muros; caras frontera
│   ├── contactos.py               # *TIE entre caras coplanares enfrentadas
│   ├── generador.py               # .inp (C3D8I, TIE, apoyos, cargas)
│   └── analisis_solido.py         # orquestación y resultados
├── combinaciones/
│   ├── motor_superposicion.py     # superposición lineal de tensores → combinaciones
│   └── esfuerzos.py               # von Mises, principales, cortante máximo
└── gui/
    ├── ventana_principal.py
    ├── visor_3d.py                 # widget PyVista + selección
    ├── dialogo_cargas.py
    ├── dialogo_apoyos.py
    ├── dialogo_casos.py
    ├── dialogo_analisis_3d.py      # tamaño de elemento del análisis 3D
    ├── panel_resultados.py         # panel de esfuerzos
    ├── trabajador_analisis.py      # hilo del análisis
    ├── iconos.py                   # logo e iconos SVG
    ├── estilo.py                   # paleta, estilo y hoja de estilos
    ├── pagina_bienvenida.py        # pantalla de inicio
    └── barra_vista.py              # vistas y visibilidad sobre el visor
```
