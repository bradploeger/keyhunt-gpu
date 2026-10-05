@echo off
REM Direct nvcc build for Windows. Run from an
REM "x64 Native Tools Command Prompt for VS 2022".
REM
REM   build_all.bat      Builds for all supported CUDA Architectures

setlocal

set HSIZE=1024
set FILTER=17

REM OpenSSL (libcrypto) is required to seal found private keys. Point OPENSSL_DIR
REM at your OpenSSL install (the vcpkg layout is the default). Override it by
REM setting OPENSSL_DIR in the environment before running this script.
if not defined OPENSSL_DIR set OPENSSL_DIR=C:\vcpkg\installed\x64-windows
if not exist "%OPENSSL_DIR%\include\openssl\evp.h" (
  echo ERROR: OpenSSL headers not found under "%OPENSSL_DIR%".
  echo        Install OpenSSL ^(e.g. vcpkg install openssl:x64-windows^) and set OPENSSL_DIR.
  exit /b 1
)
set OSSL_INC=-I"%OPENSSL_DIR%\include"
set OSSL_LIB="%OPENSSL_DIR%\lib\libcrypto.lib"
set OSSL_SYS=crypt32.lib ws2_32.lib advapi32.lib user32.lib

where nvcc >nul 2>&1 || (echo ERROR: nvcc not found. Is the CUDA Toolkit installed and on PATH? & exit /b 1)
where cl   >nul 2>&1 || (echo ERROR: cl.exe not found. Use the x64 Native Tools Command Prompt. & exit /b 1)

echo Building for all supported architectures ...
nvcc -O3 -std=c++14 -arch=all ^
     -DHSIZE=%HSIZE% -DFILTER_LOG2_BITS=%FILTER% -D_CRT_SECURE_NO_WARNINGS ^
     -Xptxas -O3,-v ^
     %OSSL_INC% ^
     -o keyhunt-gpu-cuda.exe src\search.cu %OSSL_LIB% %OSSL_SYS%
if errorlevel 1 (echo BUILD FAILED & exit /b 1)

echo.
echo Building host tests ...
for %%T in (test_math test_kernel_logic test_targets test_progress test_ptx_sequence) do (
  cl /nologo /O2 /EHsc /std:c++14 /D_CRT_SECURE_NO_WARNINGS ^
     /DHSIZE=%HSIZE% /DFILTER_LOG2_BITS=%FILTER% ^
     test\%%T.cpp /Fe:%%T.exe /Fo:%%T.obj >nul
  if errorlevel 1 (echo FAILED building %%T & exit /b 1)
)
cl /nologo /O2 /EHsc /std:c++14 /D_CRT_SECURE_NO_WARNINGS ^
   %OSSL_INC% test\test_seal.cpp /Fe:test_seal.exe /Fo:test_seal.obj ^
   %OSSL_LIB% %OSSL_SYS% >nul
if errorlevel 1 (echo FAILED building test_seal & exit /b 1)
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
test_seal.exe          || exit /b 1

echo.
echo NOTE: libcrypto-3-x64.dll (from %OPENSSL_DIR%\bin) must be on PATH at runtime.
echo Build OK. Now run:  keyhunt-gpu-cuda.exe --selftest
endlocal