@echo off
setlocal
title Analizador Estructural IFC - Compilacion automatica

echo ===================================================
echo  Analizador Estructural IFC - Compilacion automatica
echo ===================================================
echo.
echo Este script instala las dependencias y compila el
echo ejecutable con PyInstaller. Puede tardar varios
echo minutos (descarga PySide6, PyVista/VTK, etc.).
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: no se encontro "python" en el PATH.
    echo Instale Python 3.12 desde https://www.python.org/downloads/
    echo marcando la opcion "Add python.exe to PATH" en el instalador,
    echo y vuelva a ejecutar este script.
    echo.
    pause
    exit /b 1
)

echo [1/3] Actualizando pip e instalando dependencias del proyecto...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python -m pip install pyinstaller
if errorlevel 1 goto :error

echo.
echo [2/3] Compilando el ejecutable con PyInstaller...
python -m PyInstaller --name AnalizadorEstructuralIFC --onedir --windowed --noconfirm ^
    --paths src ^
    --collect-all ifcopenshell ^
    --collect-data pyvista ^
    --collect-all pyvistaqt ^
    --collect-all shapely ^
    iniciar_app.py
if errorlevel 1 goto :error

echo.
echo [3/3] Listo.
if not exist dist\AnalizadorEstructuralIFC\AnalizadorEstructuralIFC.exe goto :error

echo.
echo ===================================================
echo  Compilacion exitosa.
echo  Ejecutable: dist\AnalizadorEstructuralIFC\AnalizadorEstructuralIFC.exe
echo ===================================================
echo.
set /p RESPUESTA=¿Desea ejecutarlo ahora? (S/N):
if /I "%RESPUESTA%"=="S" start "" "dist\AnalizadorEstructuralIFC\AnalizadorEstructuralIFC.exe"

pause
exit /b 0

:error
echo.
echo ===================================================
echo  Hubo un error. Revise los mensajes de arriba.
echo ===================================================
pause
exit /b 1
