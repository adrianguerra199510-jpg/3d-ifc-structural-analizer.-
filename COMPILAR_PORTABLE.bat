@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem Algunos equipos no tienen System32 en el PATH (xcopy, powershell, etc. no se encuentran)
set "PATH=%PATH%;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
title Analizador Estructural IFC - Compilar version portable

echo ===================================================
echo  Analizador Estructural IFC
echo  Compilacion de la version PORTABLE (con CalculiX)
echo ===================================================
echo.
echo Este script crea un entorno aislado, instala las
echo dependencias, compila el programa con PyInstaller,
echo le agrega CalculiX (ccx) y genera una carpeta portable
echo y un .zip. La primera vez descarga varios cientos de MB
echo y puede tardar 10-20 minutos. Se necesita internet.
echo.
echo Todo se compila en una carpeta CORTA de la unidad del sistema
echo (%SystemDrive%\ifcbuild) porque Windows no admite rutas de mas
echo de 260 caracteres y PySide6 tiene rutas muy profundas.
echo.

set "BUILD=%SystemDrive%\ifcbuild"
set "SALIDA=%SystemDrive%\AnalizadorEstructuralIFC_portable"
set "CARPETA=%SALIDA%\AnalizadorEstructuralIFC"

rem ---- 1. Buscar un Python compatible (3.10 a 3.13) -------------------
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo ERROR: no se encontro un Python 3.10 - 3.13.
    echo Instale Python 3.12 desde https://www.python.org/downloads/
    echo marcando "Add python.exe to PATH" y vuelva a ejecutar este script.
    echo ^(Las versiones 3.14 o superiores todavia no tienen todos los paquetes.^)
    echo.
    pause
    exit /b 1
)
echo Se usara: %PY%
%PY% --version
echo.

if not exist "herramientas\ccx\ccx.exe" (
    echo ERROR: falta herramientas\ccx\ccx.exe ^(CalculiX^).
    echo Vuelva a descomprimir el zip completo, sin mover carpetas.
    goto :error
)

rem ---- 2. Entorno virtual e instalacion de dependencias ---------------
echo [1/5] Creando el entorno aislado ^(%BUILD%\venv^)...
if exist "%~dp0.venv_build" rmdir /s /q "%~dp0.venv_build" >nul 2>nul
if not exist "%BUILD%" mkdir "%BUILD%"
if not exist "%BUILD%\venv\Scripts\python.exe" (
    %PY% -m venv "%BUILD%\venv"
    if errorlevel 1 goto :error
)
set "VPY=%BUILD%\venv\Scripts\python.exe"

echo.
echo [2/5] Instalando dependencias ^(puede tardar^)...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error
"%VPY%" -m pip install pyinstaller
if errorlevel 1 goto :error

rem Libreria nativa de Gmsh (tetraedros C3D10 de los solidos): debe ir en la raiz de la carpeta portable
set "GMSHLIB="
if exist "%BUILD%\gmshlib.txt" del "%BUILD%\gmshlib.txt"
"%VPY%" -c "import gmsh; open(r'%BUILD%\gmshlib.txt', 'w').write(gmsh.libpath)"
if exist "%BUILD%\gmshlib.txt" set /p GMSHLIB=<"%BUILD%\gmshlib.txt"
if not defined GMSHLIB (
    echo ERROR: no se encontro la libreria de Gmsh ^(pip install gmsh^).
    goto :error
)
echo Libreria de Gmsh: %GMSHLIB%

rem ---- 3. Compilar -----------------------------------------------------
echo.
echo [3/5] Compilando con PyInstaller ^(varios minutos^)...
if exist "%SALIDA%" rmdir /s /q "%SALIDA%"
if exist "%BUILD%\work" rmdir /s /q "%BUILD%\work"
"%VPY%" -m PyInstaller --name AnalizadorEstructuralIFC --onedir --windowed --noconfirm ^
    --distpath "%SALIDA%" --workpath "%BUILD%\work" --specpath "%BUILD%" ^
    --icon "%~dp0recursos\logo.ico" ^
    --paths "%~dp0src" ^
    --collect-all ifcopenshell ^
    --collect-data pyvista ^
    --collect-all pyvistaqt ^
    --collect-all shapely ^
    --collect-all imageio ^
    --collect-all imageio_ffmpeg ^
    --collect-all reportlab ^
    --collect-data docx ^
    --hidden-import gmsh ^
    --add-binary "%GMSHLIB%;." ^
    "%~dp0iniciar_app.py"
if errorlevel 1 goto :error
if not exist "%CARPETA%\AnalizadorEstructuralIFC.exe" goto :error

rem ---- 4. Agregar CalculiX y archivos de apoyo -------------------------
echo.
echo [4/5] Agregando CalculiX a la carpeta portable...
xcopy /E /I /Y /Q "%~dp0herramientas\ccx" "%CARPETA%\ccx" >nul
if errorlevel 1 goto :error
copy /Y "%~dp0config.example.json" "%CARPETA%\config.json" >nul
copy /Y "%~dp0LEEME_PORTABLE.txt" "%CARPETA%\LEEME.txt" >nul

rem ---- 5. Autoprueba: resuelve una viga con el ccx incluido -------------
echo.
echo [5/5] Autoprueba del solver ^(CalculiX^)...
if exist "%CARPETA%\autoprueba.txt" del "%CARPETA%\autoprueba.txt"
start "" /wait "%CARPETA%\AnalizadorEstructuralIFC.exe" --autoprueba
set "RC=%ERRORLEVEL%"
if exist "%CARPETA%\autoprueba.txt" type "%CARPETA%\autoprueba.txt"
if not "%RC%"=="0" (
    echo.
    echo AVISO: la autoprueba de CalculiX fallo ^(codigo %RC%^).
    echo El programa se compilo, pero revise el mensaje de arriba.
    echo Si dice que falta una DLL, instale "Visual C++ Redistributable 2015-2022 x64".
    echo.
) else (
    echo CalculiX funciona correctamente.
)
if exist "%CARPETA%\autoprueba.txt" del "%CARPETA%\autoprueba.txt"
if exist "%CARPETA%\trabajo_ccx" rmdir /s /q "%CARPETA%\trabajo_ccx"

echo.
echo Creando el archivo .zip portable...
if exist "%~dp0AnalizadorEstructuralIFC-portable.zip" del "%~dp0AnalizadorEstructuralIFC-portable.zip"
powershell -NoProfile -Command "Compress-Archive -Path '%CARPETA%' -DestinationPath '%~dp0AnalizadorEstructuralIFC-portable.zip'"

echo.
echo ===================================================
echo  LISTO
echo  Carpeta portable : %CARPETA%
echo  Ejecutable       : %CARPETA%\AnalizadorEstructuralIFC.exe
echo  Zip para copiar  : %~dp0AnalizadorEstructuralIFC-portable.zip
echo.
echo  Copie la carpeta ^(o el zip^) a cualquier PC con Windows 10/11
echo  de 64 bits: no necesita instalar Python ni CalculiX.
echo ===================================================
echo.
set /p RESPUESTA=Desea ejecutarlo ahora? (S/N): 
if /I "%RESPUESTA%"=="S" start "" "%CARPETA%\AnalizadorEstructuralIFC.exe"
start "" explorer "%SALIDA%"
pause
exit /b 0

:error
echo.
echo ===================================================
echo  Hubo un error. Revise los mensajes de arriba.
echo ===================================================
pause
exit /b 1
