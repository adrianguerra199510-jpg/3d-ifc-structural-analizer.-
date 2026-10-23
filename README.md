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
- Mostrar esfuerzos (von Mises, σ1, σ3, τmax) en MPa como mapa de
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
   losas sin vigas, etc.), pregunta **qué casos de carga se resuelven** (casillas; dice qué combinaciones se
   podrán calcular con ellos: una combinación necesita todos sus casos), ejecuta CalculiX por cada caso elegido y
   abre la pestaña **Esfuerzos**.
6. **Casos y combinaciones** (botón de la barra o menú *Asignar*): formulario con
   dos pestañas. En *Casos de carga* se crean con identificador, nombre y
   tipo (D, L, W, E…); los de carga muerta (D) tienen la casilla **Incluir el peso propio** (marcada por defecto;
   desmárquela en un caso de cargas muertas sobreimpuestas o si otro caso D ya lleva el peso propio). Los
   proyectos guardados antes de la casilla la llevan marcada solo en su primer caso D, como hasta ahora. En *Combinaciones* se escribe el nombre y el **factor de
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
- **Símbolos de apoyo** según los grados de libertad: bloque (empotrado), pirámide (articulado), pirámide
  con ruedas orientada según la traslación (rodillo, uno por cada traslación restringida) y un anillo
  por cada giro restringido, con su etiqueta.
- **Barra de vista** sobre el visor: isométrica, planta, frontal, lateral, inferior,
  encuadrar, visibilidad por tipo de elemento (los iconos llevan el color del elemento)
  y el cambio Modelo / Esfuerzos.
- **Ratón en el visor** (como SAP2000, AutoCAD o Revit: el visor está quieto en modo selección y la vista solo
  se mueve cuando se pide):

  | Acción | Qué hace |
  |---|---|
  | Clic izquierdo | selecciona (Ctrl o Mayús suma o quita) |
  | Arrastrar con el izquierdo | rectángulo de selección: de izquierda a derecha, lo que queda dentro; de derecha a izquierda, lo que toca |
  | Rueda | zoom hacia el cursor (la cámara se acerca sin cambiar el ángulo de visión, con límites) |
  | Botón central (rueda pulsada) arrastrando | desplaza la vista |
  | Ctrl o Mayús + botón central arrastrando | gira la vista alrededor del punto enfocado, con el eje Z siempre vertical |
  | Clic derecho | menú del elemento; sobre el fondo, el menú de la vista (estilo, vistas de cámara, qué se muestra, escala) |

  Sin botón central (touchpad): el grupo **Ratón** de la barra de vista cambia lo que hace el arrastre
  izquierdo a *Mover* o *Rotar* (un clic sin arrastrar sigue seleccionando); *Seleccionar* es el modo normal.
- **Editar:** *Deshacer* (Ctrl+Z) y *Rehacer* (Ctrl+Y o Ctrl+Mayús+Z) para todos los cambios del modelo
  (asignaciones, materiales, casos, constraints y eliminaciones; los últimos 60). *Eliminar* (**Supr** con el
  visor activo, o clic derecho → «Eliminar del modelo») quita los elementos seleccionados con sus apoyos,
  cargas, conexiones y constraints, y los nodos que quedan sin uso; Ctrl+Z lo recupera.
- **Estilo de vista** (*Ver → Estilo de vista* o clic derecho sobre el fondo): **sólido**, **transparente** (se ve a través
  de las piezas, que se siguen pudiendo elegir) y **alambre** (solo aristas).
- **Tamaño de los símbolos:** los apoyos, las flechas de las cargas, los constraints y las etiquetas se dimensionan
  según el modelo (y, si hay barras, la longitud típica). *Ver → Escala de símbolos…* cambia cada parámetro al instante,
  como **porcentaje** del tamaño automático o como **tamaño en píxeles** de pantalla (que se mantiene al acercar o
  alejar la vista), y se recuerda entre sesiones. Los interruptores de la barra de vista (solo el símbolo; el nombre
  sale al pasar el cursor) muestran u ocultan apoyos, cargas y constraints.
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

## Deformada y video

Además de los esfuerzos, el panel **Resultados** tiene el grupo **Deformada** (en el análisis de barras
y en el 3D). CalculiX entrega los desplazamientos de cada nodo (`*NODE FILE, U`), que se combinan igual
que los esfuerzos (la envolvente usa la combinación que más desplaza):

- **Mostrar deformada**: la malla coloreada por el esfuerzo elegido se dibuja desplazada, con el contorno
  sin deformar en gris. Apoyos y cargas se ocultan mientras tanto.
- **Escala ×N**: ampliación de los desplazamientos; **Auto** la ajusta para que el mayor mida ~8 % del
  tamaño del modelo.
- **▶ Animar**: la deformada va de 0 a 100 % y vuelve, en bucle, con el visor libre para girar y acercar.
- **Guardar video…**: graba un ciclo con la vista actual en **MP4** (H.264, con `imageio-ffmpeg`, que viaja
  dentro de la carpeta portable) o **GIF**.

La flecha de un voladizo coincide con `PL³/3EI` (barras: +1,7 %; sólido: +6 %, que incluye la deformación
por cortante), ver `tests/test_solido.py`.

## Análisis 3D (modelo sólido)

Un solo botón **ANALIZAR** (F5): el programa decide qué análisis corresponde. Si el modelo tiene sólidos de
forma libre, apoyos o cargas sobre superficies, constraints o ninguna barra, lo resuelve como **sólido 3D**
(el diálogo de ANALIZAR dice por qué y con qué malla; el tamaño de elemento se elige con **Mallar**, y si no se
malló antes se usa el tamaño sugerido); si son solo barras (con o sin losas), como barras.
*Analizar → Modo de análisis* permite forzar uno de los dos. El análisis 3D usa tres elementos de CalculiX:
**tetraedros cuadráticos `C3D10`** (los de *Placa Base Pro*) en las piezas macizas, **hexaedros cuadráticos `C3D20R`
barridos** en las piezas prismáticas (casilla del diálogo, activada por defecto) y **láminas `S6`** en las piezas que
son láminas. Las vigas, columnas, losas y muros se mallan con su geometría real (un barrido de la sección de cada
barra) en hexaedros que al final se parten en 5 o 6 tetraedros cuadráticos conformes; los sólidos de forma libre, con
Gmsh. La versión portable lleva CalculiX y Gmsh incluidos, así que no hay nada que instalar.

**Mallar** (F7, junto a ANALIZAR): genera la malla del análisis sólido sin resolverlo y la dibuja (sólidos en gris,
láminas en ocre, con las aristas de cada elemento), para revisarla y ajustar el tamaño de elemento antes de calcular.
La vista *Malla* de la barra del visor vuelve a mostrarla. ANALIZAR usa esa misma malla si el tamaño es el mismo y la
geometría, los perfiles, los materiales y los apoyos o cargas puntuales en barras no cambiaron; si cambió algo, vuelve
a mallar sola.

- **Hexaedros barridos (`C3D20R`).** Una pieza es prismática cuando es la extrusión recta de una sección constante
  (columnas, pernos, tubos, barras, placas gruesas): sus caras son dos tapas perpendiculares a un eje y caras laterales
  paralelas a él. Su sección se malla con cuadriláteros (Gmsh, con al menos unos 3 elementos a lo ancho) y se extruye en
  capas (al menos 2, y de largo ≤ 4 veces el lado de la sección): cada cuadrilátero de cada capa es un hexaedro de 20
  nodos con integración reducida, que da mejores esfuerzos con menos nodos (voladizo de 1 m: flecha y esfuerzo de
  flexión iguales a la teoría con 320 hexaedros). Las piezas que solapan con otras mayores o con barras, y las secciones
  que Gmsh no puede pasar a cuadriláteros sanos, siguen con `C3D10` (se avisa). Se unen a las demás con `*TIE` como el
  resto; las cargas de superficie se reparten con los pesos consistentes de una cara de 8 nodos.

- **¿Cuándo es lámina?** Cuando su espesor medio `t = 2·V/A` (V el volumen y A el área de toda su superficie) es como
  mucho **1/20 de su tamaño en planta** `L = √(A/2)` (el criterio habitual de placa o cáscara delgada): el cuerpo y el
  cono de un balde, el anillo inferior, una tolva, tubos de pared delgada, chapas. Una losa o un muro, si su espesor es
  ≤ 1/20 de √(área de su contorno). Los bloques, ejes, pernos, engranajes y placas gruesas van con `C3D10`; una barra
  redonda esbelta (que el cociente confundiría con una lámina) se reconoce como maciza al medir su espesor con rayos.
  En un cajón (el anillo inferior del balde, un tubo rectangular) cada pared conserva una de sus pieles y donde se juntan
  pieles de lados distintos las superficies medias terminan a ~0,7 espesores la una de la otra: esas grietas de las
  esquinas se **cosen** (cada vértice del borde se lleva al otro borde, partiendo su arista), porque una esquina abierta
  deja el cajón sin rigidez (el anillo del balde bajaba 85 mm con su peso propio).

- **Geometría.** Un perfil en I se malla con sus alas y su alma; un rectángulo, con
  su sección. Las barras alineadas con el mismo perfil (la columna partida por una unión)
  se funden en una sola pieza y el extremo de una viga se **recorta hasta la superficie**
  de la columna que atraviesa, de modo que su cara extrema queda apoyada en ella. A las
  losas y muros se les quitan los elementos que solapan con vigas y columnas.
- **Uniones.** Las piezas se unen con `*TIE` por las caras que se miran y se tocan (cara
  extrema de la viga contra el ala, losa sobre viga, tramos de columna): equivale a una
  soldadura continua. Una losa que el IFC coloca con holgura o solapada se une a las caras
  cercanas y se avisa. No hay pernos, contacto con separación ni fricción.
- **Apoyos y cargas.** Se aplican sobre la **sección o la cara** elegida: un apoyo
  empotrado fija todos los nodos de la cara; uno con giros libres (pasador, rodillo)
  mueve la cara como cuerpo rígido con un nodo de referencia. Las fuerzas y momentos puntuales
  se reparten por el área de la sección, las cargas lineales a lo largo de ella y las presiones
  sobre las caras de la losa. El peso propio es `*DLOAD,GRAV` con el peso específico del
  material (en los casos de tipo D con la casilla del peso propio). Las conexiones manuales (articulaciones) no se aplican.
- **Tamaño de elemento.** El diálogo de Mallar propone el más fino que da unos 60 000 elementos (tetraedros + láminas) y
  estima la cantidad y la memoria que usará CalculiX (no hay máximo: si no cabe en la memoria libre, avisa). La estimación cuenta los triángulos de superficie que hará Gmsh (las piezas
  pequeñas, como pernos y tuercas, no bajan de un mínimo aunque crezca el tamaño): si ni el tamaño más grueso cumple
  el objetivo, propone el más fino que da casi lo mismo que el más grueso. Con 1 elemento en el espesor de un ala,
  `C3D10` flexiona bien (IPE300 en voladizo: +0,15 % frente a la teoría); las concentraciones junto a apoyos, cargas
  puntuales y esquinas interiores dependen del tamaño y no convergen del todo (son singularidades).
- **Resultados.** Los mismos de siempre (von Mises, σ1, σ3, τmax, casos, combinaciones y
  envolvente), pero sobre el sólido: se ven las alas, el alma y las uniones. Comprobado
  contra `M·c/I`, `wL²/8` y el equilibrio (`tests/test_solido.py`): error < 3 %.

## Reporte del análisis (PDF o Word)

Tras analizar, **Archivo → Exportar reporte de análisis…** (Ctrl+R) o el botón *Exportar reporte (PDF / Word)…* del panel
de esfuerzos genera un documento con: datos del modelo y materiales, apoyos, casos y combinaciones, la **malla** de
elementos finitos (número de elementos y nodos, imagen), una tabla resumen con el **Von Mises máximo** (valor, ubicación y
elemento) y el **desplazamiento máximo** de cada caso y combinación, y por cada uno la imagen del mapa de Von Mises [MPa] y
de la **deformada** ampliada y coloreada por el desplazamiento [mm] (la ampliación y los colores son los del panel), más las
advertencias del análisis. Se elige el formato al guardar: **PDF** (`reportlab`) o **Word** `.docx` (`python-docx`); los
dos llevan el mismo contenido. Las imágenes se dibujan aparte, sin cambiar la vista del visor. Sirve para el análisis de
barras y para el sólido 3D (`tests/test_reporte.py`). Si alguna pieza se desplaza más que el 10 % del tamaño del modelo, el
análisis avisa de que forma un mecanismo (no está unida a las demás ni apoyada) y sus resultados no son válidos.

## Geometría de cualquier elemento y apoyos/cargas en cualquier superficie

Además de vigas, columnas, losas y muros, el importador lee la **geometría real de cualquier
otro elemento del IFC** (placas, pernos, cartelas, piezas a medida, `IfcBuildingElementProxy`…)
y la conserva como **sólido de forma libre**:

- **Superficies.** La malla de triángulos de cada elemento se agrupa en superficies: las
  caras planas (triángulos coplanares, tolerancia 4°) y las superficies curvas (un cilindro
  son 3: dos tapas y la lateral; una esfera es 1). Se numeran de arriba abajo y se eligen con un
  clic en modo **Cara**, igual que las caras de una viga. Los sólidos aparecen en el árbol, con
  su propio interruptor de visibilidad.
- **Apoyos.** *Asignar → Apoyo* sobre una superficie de un sólido (o, con clic derecho, «Asignar
  apoyo en la superficie…» sobre la cara de una viga, losa o muro): empotrada, deslizante en X, Y o Z,
  cuerpo rígido o cualquier combinación de Ux, Uy, Uz, Rx, Ry, Rz.
  **Pasador** (el agujero de una oreja o cartela): en vez de fijar cada nodo de la superficie (que pega
  el agujero a un perno infinitamente rígido), fija el **centro** de la superficie, el promedio de sus
  desplazamientos ponderado por el área (acople distribuido, como el RBE3): el agujero se deforma y la
  pieza gira alrededor del perno. Con todos los apoyos pasadores alineados el modelo puede girar y se avisa.
- **Material elastoplástico (bilineal).** En *Materiales* se indica la fluencia **fy** y, si se quiere, el
  módulo tangente **Et** (por defecto E/100; las plantillas de acero y aluminio traen su fy). En el diálogo de
  ANALIZAR, la casilla «Material elastoplástico (análisis no lineal)» escribe `*PLASTIC` (endurecimiento
  isótropo lineal) y aplica la carga en incrementos: el acero no pasa de fy (más el endurecimiento) y los
  picos locales se reparten. Con plasticidad los casos no se pueden sumar, así que **cada combinación se
  resuelve como un análisis propio** con sus cargas mayoradas. El visor y el reporte muestran la
  **deformación plástica equivalente (PEEQ, %)**, con aviso por encima del 5 % (EN 1993-1-5, anexo C).
  Tarda varias veces más que el análisis elástico. Si no converge, la estructura plastifica hasta formar un
  mecanismo (se supera su capacidad).
- **Cargas.** *Asignar → Carga* sobre una superficie: **presión** (kPa, por defecto normal a la
  superficie y hacia dentro), **fuerza total** (kN repartida por área) o **momento total** (kN·m,
  repartido con fuerzas proporcionales a la distancia al centro), en la dirección elegida (normal,
  ejes globales o un vector).
- **Análisis.** Solo con **ANALIZAR 3D**. Los sólidos libres se mallan con **tetraedros cuadráticos
  `C3D10`** generados por Gmsh a partir de su superficie triangulada (el mismo elemento del modelo sólido
  de *Placa Base Pro*): siguen las superficies curvas e inclinadas sin escalonarlas. Gmsh corre en **un solo proceso
  aparte para todas las piezas** (lanzar uno por pieza hacía saltar a los antivirus), con límite de tiempo por pieza
  (60 s): si una pieza lo cuelga, se termina y se relanza desde la siguiente (como mucho 3 veces). Si falla con una
  pieza, lo vuelve a intentar con la mitad y la cuarta parte del tamaño; las que aun así no puede mallar
  (superficies que se tocan por aristas) se rellenan por vóxeles, que también se parten en `C3D10`, y se avisa.
  Cada cara de borde se asocia a la superficie más cercana, y de ahí salen los
  nodos donde se aplican apoyos y cargas (en un triángulo cuadrático la presión va a los nodos medios,
  1/3 del área cada uno). Donde dos sólidos (o un sólido y una viga o columna) se
  solapan, la pieza menor cede el volumen común, para no contarlo dos veces; las piezas se unen
  por las superficies que se tocan (`*TIE`, sin mover nodos: `ADJUST=NO`; si se solapaban, como una tuerca metida en
  la placa, el borde cortado no tiene caras planas enfrentadas y se une por proximidad) y las que no tocan a ninguna otra
  con apoyo se detectan antes de resolver. Los sólidos deben ser superficies cerradas (se avisa si no lo son;
  se admiten aristas compartidas por 4 triángulos, típicas de Revit). Un elemento IFC con varias piezas sin
  conexión (un equipo exportado como un solo `IfcBuildingElementProxy`) se importa como un sólido por pieza.
- **Láminas.** Las piezas que son láminas (ver arriba) se modelan con **cáscaras `S6`** de CalculiX (triángulo
  cuadrático con 6 grados de libertad por nodo): se obtiene su superficie media lanzando un rayo desde cada triángulo de
  una piel hasta la piel opuesta (el espesor es la distancia) y Gmsh la vuelve a mallar, en el mismo proceso que los
  tetraedros, con triángulos casi equiláteros del tamaño de elemento (las superficies de Revit vienen en abanico, con
  triángulos larguísimos); si no puede, se refina por bisección. Cada elemento lleva su espesor (`*SHELL SECTION`) y
  las normales son coherentes entre vecinos. Funciona con las superficies que se tocan por aristas, que Gmsh rechaza.
  Los apoyos sobre una lámina restringen directamente los 6 grados (un empotramiento de verdad); las cargas de superficie
  van a los nodos medios. Las láminas se acoplan a las piezas vecinas con `*EQUATION` entre los nodos que están a menos de
  0,6 veces el tamaño de elemento más medio espesor (traslaciones; también giros entre dos láminas; con un sólido, el
  giro de cada nodo de la lámina sigue al del parche de nodos del sólido que lo rodea, ajustado por mínimos cuadrados,
  así que la unión transmite también momento). Un voladizo de chapa da ~95 % de la flecha de la teoría
  (`tests/test_solidos_libres.py`, `tests/test_c3d10_laminas.py`).
- **Conectar** (botón de la barra, menú *Asignar → Conectar elementos* o clic derecho) ofrece dos formas:
  **manualmente** (los dos elementos o superficies seleccionados) y **automáticamente** (detecta los
  elementos que se tocan, se solapan o están a menos de una separación elegida, los lista para marcar y crea un
  constraint por pareja; no repite las que ya tienen uno). Se mide sobre la **geometría real** de cada elemento: se
  reparten puntos sobre su superficie y se mide su distancia con signo a la superficie del otro (antes se medía sobre
  una malla gruesa de vóxeles y se perdían los contactos inclinados o curvos). Con «cuerpo rígido», todas las piezas que
  quedan conectadas entre sí forman un solo cuerpo rígido (el diálogo lo advierte).
- **Constraints entre superficies o elementos 3D completos.** Con dos superficies (modo **Cara**) o dos
  elementos (modo **Elemento**) se elige a qué se aplica: *las superficies más cercanas* (la distancia se mide entre
  las superficies de las caras, no entre sus vértices; a igual distancia, la de mayor área en contacto y la que se mira)
  o *los elementos completos* (soldadura por donde se tocan o a menos de la separación máxima;
  cuerpo rígido o articulación de los dos cuerpos enteros). Tipos, como los *joint constraints* de SAP:
  **soldadura** (`*TIE`, une punto a punto y cierra la holgura; sustituye a la unión automática entre
  esas dos piezas), **cuerpo rígido** (las dos superficies se mueven juntas, `*RIGID BODY`),
  **igualar movimientos** (cada nodo de la superficie más fina sigue al punto más cercano de la otra, interpolado en su
  cara, en Ux, Uy, Uz marcados, `*EQUATION`) y **articulación** (superficies rígidas con los centros de su área
  igualados en las tres traslaciones y en los giros marcados). Aparecen en el árbol (*Constraints*), se dibujan en el visor
  y se guardan en el proyecto. Solo se aplican en el análisis 3D. Entre dos barras, *Conectar manualmente*
  pregunta si se quiere la conexión del modelo de barras (articulación o enlace rígido) o el constraint 3D.
  Un grado de libertad solo puede depender de una cosa: los nodos con apoyo propio y los esclavos de una soldadura
  (`*TIE`, también las uniones automáticas) quedan fuera de los cuerpos rígidos y de las igualaciones, y los cuerpos
  rígidos que comparten nodos (una pieza unida a dos con «cuerpo rígido») se funden en uno; antes, CalculiX no avisaba y
  daba desplazamientos absurdos. La tolerancia de una soldadura entre superficies sale de la holgura del 95 % de los
  nodos (no de la mínima) y se avisa si parte de la superficie queda sin unir. Una soldadura con una lámina entre
  superficies elegidas solo acopla esas superficies.
- **Selección por zona.** Arrastre un rectángulo con el botón izquierdo (Ctrl o Mayús suma a la selección): de izquierda a derecha (azul, continuo)
  se eligen solo los elementos —o superficies, en modo Cara— que quedan por completo dentro; de derecha
  a izquierda (verde, discontinuo), todo lo que toca. En modo Cara solo cuentan las caras que miran a
  la cámara. Con Ctrl/Mayús se suma a la selección actual.
- **Apoyos y cargas.** Los interruptores de apoyos, cargas y constraints de la barra de vista (y *Ver → Mostrar…*)
  los ocultan o muestran; el tamaño se cambia en *Ver → Escala de símbolos…*; en las cargas sobre una superficie la flecha nunca es mayor que ella.
- **Archivos reales (Revit y similares).** Se respeta la unidad del archivo (mm, cm, pies): la geometría se importa siempre
  en metros. Los materiales sin propiedades cuyo nombre (o el del elemento o su tipo) dice *acero* se importan como
  acero estructural y se avisa. Las placas delgadas clasificadas como viga, columna o miembro (rigidizadores) y los muros
  o losas que son chapas se mallan con **una capa** de elementos en el espesor y celdas de hasta 12 veces el espesor en
  su plano (un alma de 10 mm y 6 m no se convierte en cientos de miles de celdas). Las piezas sueltas (sin tocar a
  nada, sin apoyos ni cargas) no abortan el análisis: se dejan fijas y se avisa. Probado con una viga armada en cajón
  exportada de Revit en milímetros (`Lifting Device`).
- **Qué no hace.** Los sólidos no participan en el análisis de barras (se avisa) y no se modelan
  pernos con pretensión, contacto con separación ni fricción. Comprobado con CalculiX contra la
  teoría (voladizo `PL³/3EI`, compresión `σ = p`, perno en tracción `FL/EA`) en
  `tests/test_solidos_libres.py` y con clics reales en `tests/guion_gui_libre_xvfb.py`.

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
- **Placas, pernos y rigidizadores** no son barras: en el análisis de barras no participan
  (se importan como sólidos de forma libre y solo se analizan con ANALIZAR 3D; las barras se
  unen por sus ejes). Tampoco se incluye el cortante
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
│   ├── asignacion_superficie.py   # apoyos y cargas en una superficie cualquiera
│   ├── solidos.py                 # sólidos de forma libre: triángulos, superficies y persistencia
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
│   ├── malla.py                   # malla de todo el modelo (C3D10 + C3D20R + S6); caras frontera; estimación
│   ├── hex_a_tet.py               # hexaedros -> tetraedros cuadráticos C3D10 conformes
│   ├── libres.py                  # voxelización de un sólido de forma libre (celdas dentro + superficies)
│   ├── tetraedros.py              # C3D10 de los sólidos libres: un solo proceso de Gmsh para todas las piezas
│   ├── barridos.py                # C3D20R barridos de las piezas prismáticas: detección, extrusión, caras
│   ├── laminas.py                 # láminas S6: criterio, superficie media, remallado con Gmsh, espesor
│   ├── proceso_gmsh.py            # proceso hijo que malla con Gmsh (límite de tiempo)
│   ├── superficies.py             # caras de hexaedro de una superficie; reparto de presión, fuerza y momento
│   ├── contactos.py               # *TIE entre caras coplanares enfrentadas
│   ├── generador.py               # .inp (C3D10, C3D20R y S6, TIE, EQUATION, apoyos, cargas)
│   └── analisis_solido.py         # orquestación y resultados
├── combinaciones/
│   ├── motor_superposicion.py     # superposición lineal de tensores → combinaciones
│   └── esfuerzos.py               # von Mises, principales, cortante máximo
└── gui/
    ├── ventana_principal.py
    ├── visor_3d.py                 # widget PyVista + selección
    ├── dialogo_cargas.py
    ├── dialogo_apoyos.py
    ├── dialogo_superficie.py       # apoyos y cargas en una superficie
    ├── dialogo_casos.py
    ├── dialogo_analisis_3d.py      # tamaño de elemento del análisis 3D
    ├── panel_resultados.py         # panel de esfuerzos
    ├── trabajador_analisis.py      # hilo del análisis
    ├── iconos.py                   # logo e iconos SVG
    ├── estilo.py                   # paleta, estilo y hoja de estilos
    ├── pagina_bienvenida.py        # pantalla de inicio
    └── barra_vista.py              # vistas y visibilidad sobre el visor
```
