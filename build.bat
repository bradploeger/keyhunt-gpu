@echo off
REM Direct nvcc build for Windows. Run from an
REM "x64 Native Tools Command Prompt for VS 2022".
REM
REM   build.bat            uses ARCH below
REM   build.bat 89         overrides the architecture

setlocal
set ARCH=86
if not "%~1"=="" set ARCH=%~1

set HSIZE=256
set FILTER=19

where nvcc >nul 2>&1 || (echo ERROR: nvcc not found. Is the CUDA Toolkit installed and on PATH? & exit /b 1)
where cl   >nul 2>&1 || (echo ERROR: cl.exe not found. Use the x64 Native Tools Command Prompt. & exit /b 1)

echo Building for sm_%ARCH% ...
nvcc -O3 -std=c++14 -arch=sm_%ARCH% ^
     -DHSIZE=%HSIZE% -DFILTER_LOG2_BITS=%FILTER% -D_CRT_SECURE_NO_WARNINGS ^
     -Xptxas -O3,-v ^
     -o keyhunt-gpu.exe src\search.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)

echo.
echo Building host tests ...
for %%T in (test_math test_kernel_logic test_targets test_progress test_ptx_sequence) do (
  cl /nologo /O2 /EHsc /std:c++14 /D_CRT_SECURE_NO_WARNINGS ^
     /DHSIZE=%HSIZE% /DFILTER_LOG2_BITS=%FILTER% ^
     test\%%T.cpp /Fe:%%T.exe /Fo:%%T.obj >nul
  if errorlevel 1 (echo FAILED building %%T & exit /b 1)
)
del *.obj >nul 2>&1

echo.
echo Running host tests ...
test_math.exe          || exit /b 1
echo.
test_kernel_logic.exe  || exit /b 1
echo.
test_targets.exe       || exit /b 1
echo.
test_progress.exe      || exit /b 1
echo.
test_ptx_sequence.exe  || exit /b 1

echo.
echo Build OK. Now run:  keyhunt-gpu.exe --selftest
endlocal
