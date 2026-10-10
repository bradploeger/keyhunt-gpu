@echo off
setlocal enabledelayedexpansion

REM   Direct nvcc build for Windows. Run from an
REM   "x64 Native Tools Command Prompt for VS 2022".

REM --- 1. INITIALIZE GLOBAL DEFAULTS ---
set "CLI_ARCH="
set "CLI_HSIZE="
set "CLI_FILTER="
set "IS_BARE_OR_DEFAULT=0"

REM --- 2. PARSE COMMAND LINE ARGUMENTS ---
:PARSE_ARGS
if "%~1"=="" goto :ARGS_DONE
set "ARG=%~1"

if /I "%ARG%"=="-hsize" (set "CLI_HSIZE=%~2" & shift & shift & goto :PARSE_ARGS)
if /I "%ARG%"=="-filter" (set "CLI_FILTER=%~2" & shift & shift & goto :PARSE_ARGS)

if not defined CLI_ARCH (
    set "CLI_ARCH=%~1"
    if not "%~2"==="" (
        for /f "delims=0123456789" %%A in ("%~2") do goto :NOT_NUM1
        set "CLI_HSIZE=%~2"
        shift
    )
    :NOT_NUM1
    if not "%~2"==="" (
        for /f "delims=0123456789" %%A in ("%~2") do goto :NOT_NUM2
        set "CLI_FILTER=%~2"
        shift
    )
    :NOT_NUM2
    shift
    goto :PARSE_ARGS
)
shift
goto :PARSE_ARGS
:ARGS_DONE

REM --- 3. APPLY RULES FOR MISSING ARCH WITH CUSTOM METRICS ---
if not defined CLI_ARCH (
    if defined CLI_HSIZE (set "CLI_ARCH=all")
    if defined CLI_FILTER (set "CLI_ARCH=all")
)

REM --- 4. CONDITIONAL HARDWARE DETECTION OR EXPLICIT EVALUATION ---
if defined CLI_ARCH (
    echo CLI argument detected. Skipping hardware auto-detection...
    set "ARCH=%CLI_ARCH%"
    
    if /I "!ARCH!"=="default" (
        set "IS_BARE_OR_DEFAULT=1"
        set "ARCH_TARGETS=all"
        set "HSIZE=1024"
        set "FILTER=17"
        goto :VALIDATION_PASSED
    )
    if /I "!ARCH!"=="all" (
        set "ARCH_TARGETS=sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_90a sm_100 sm_101 sm_120 sm_121"
        goto :VALIDATION_PASSED
    )
    
    set "ARCH_TARGETS=!ARCH!"
    goto :STRICT_VALIDATION
)

REM --- RUN HARDWARE AUTO-DETECTION MATRIX ---
set "IS_BARE_OR_DEFAULT=1"
:: Use nvidia-smi to query the GPU name
for /f "tokens=2 delims=:" %%a in ('nvidia-smi -L 2^>nul') do (
    :: Extract out the GPU name string (strips out the trailing UUID)
    for /f "tokens=1 delims=(" %%b in ("%%a") do (
        set "GPU_NAME=%%b"
    )
)

:: Trim leading/trailing spaces
if defined GPU_NAME (
    set "GPU_NAME=%GPU_NAME:~1%"
    echo [INFO] Detected via driver: %GPU_NAME%
) else (
    echo [ERROR] nvidia-smi command failed or no NVIDIA card found.
)


if "%GPU_NAME%"=="" (
    echo WARNING: No NVIDIA GPU detected. Defaulting to baseline parameters.
    set "ARCH_TARGETS=all"
    set "HSIZE=1024"
    set "FILTER=17"
    goto :VALIDATION_PASSED
)

echo Found: %GPU_NAME%
echo %GPU_NAME% | findstr /I "5090 5080 5070 Blackwell" >nul && (set "ARCH_TARGETS=sm_100" & set "HSIZE=2048" & set "FILTER=19" & goto :VALIDATION_PASSED)
echo %GPU_NAME% | findstr /I "H100 Hopper" >nul             && (set "ARCH_TARGETS=sm_90a" & set "HSIZE=2048" & set "FILTER=19" & goto :VALIDATION_PASSED)
echo %GPU_NAME% | findstr /I "4090 4080 4070 4060 Ada" >nul    && (set "ARCH_TARGETS=sm_89"  & set "HSIZE=1536" & set "FILTER=18" & goto :VALIDATION_PASSED)
echo %GPU_NAME% | findstr /I "3090 3080 3070 3060 Ampere" >nul  && (set "ARCH_TARGETS=sm_86"  & set "HSIZE=1536" & set "FILTER=18" & goto :VALIDATION_PASSED)
echo %GPU_NAME% | findstr /I "2080 2070 2060 Titan Turing" >nul && (set "ARCH_TARGETS=sm_75"  & set "HSIZE=1024" & set "FILTER=17" & goto :VALIDATION_PASSED)
echo %GPU_NAME% | findstr /I "1080 1070 1060 Pascal" >nul       && (set "ARCH_TARGETS=sm_61"  & set "HSIZE=512"  & set "FILTER=16" & goto :VALIDATION_PASSED)

set "ARCH_TARGETS=sm_75"
set "HSIZE=1024"
set "FILTER=17"
goto :VALIDATION_PASSED

:STRICT_VALIDATION
set "PREFIX=!ARCH_TARGETS:~0,3!"
if /I not "!PREFIX!"=="sm_" (
    echo ERROR: Invalid architecture format "!ARCH_TARGETS!". Must be "all", "default", or "sm_XX".
    exit /b 1
)
set "ARCH_RAW_NUM=!ARCH_TARGETS:~3!"
set "ARCH_NUM=!ARCH_RAW_NUM!"
if /I "!ARCH_RAW_NUM:~-1!"=="a" set "ARCH_NUM=!ARCH_RAW_NUM:~0,-1!"

for /f "delims=0123456789" %%A in ("!ARCH_NUM!") do (
    echo ERROR: Architecture version suffix "!ARCH_TARGETS!" contains invalid characters.
    exit /b 1
)
if !ARCH_NUM! LSS 75 (echo ERROR: Architecture is below the sm_75 threshold. & exit /b 1)
if !ARCH_NUM! GTR 121 (echo ERROR: Architecture exceeds the maximum sm_121 limit. & exit /b 1)

:VALIDATION_PASSED
if defined CLI_HSIZE set "HSIZE=%CLI_HSIZE%"
if defined CLI_FILTER set "FILTER=%CLI_FILTER%"

REM --- 5. ENVIRONMENT & TOOL VERIFICATION ---
if not defined OPENSSL_DIR set OPENSSL_DIR=C:\vcpkg\installed\x64-windows
if not exist "%OPENSSL_DIR%\include\openssl\evp.h" (
  echo ERROR: OpenSSL headers not found under "%OPENSSL_DIR%".
  exit /b 1
)
set OSSL_INC=-I"%OPENSSL_DIR%\include"
set OSSL_LIB="%OPENSSL_DIR%\lib\libcrypto.lib"
set OSSL_SYS=crypt32.lib ws2_32.lib advapi32.lib user32.lib

where nvcc >nul 2>&1 || (echo ERROR: nvcc missing from PATH. & exit /b 1)
where cl   >nul 2>&1 || (echo ERROR: cl.exe missing. Run inside VS Native Tools Command Prompt. & exit /b 1)
if not exist "%OSSL_LIB%" (echo ERROR: OpenSSL library not found at "%OSSL_LIB%". & exit /b 1)

REM --- 6. MAIN MULTI-TARGET COMPILATION LOOP ---
for %%A in (%ARCH_TARGETS%) do (
    set "CURRENT_ARCH=%%A"
    
    if defined CLI_HSIZE goto :SkipArchMapping

    :: 2. If CLI_ARCH is blank OR it is explicitly set to "default", skip the block
    if "%CLI_ARCH%"=="" goto :SkipArchMapping
    if /I "%CLI_ARCH%"=="default" goto :SkipArchMapping

    :: 3. Now safely run your mapping checks without unstable nested parentheses
    if /I "%%A"=="sm_121" (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_120" (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_101" (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_100" (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_90a" (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_90"  (set "HSIZE=2048" & set "FILTER=19")
    if /I "%%A"=="sm_89"  (set "HSIZE=1536" & set "FILTER=18")
    if /I "%%A"=="sm_87"  (set "HSIZE=1536" & set "FILTER=18")
    if /I "%%A"=="sm_86"  (set "HSIZE=1536" & set "FILTER=18")
    if /I "%%A"=="sm_80"  (set "HSIZE=1536" & set "FILTER=18")
    if /I "%%A"=="sm_75"  (set "HSIZE=1024" & set "FILTER=17")

    :SkipArchMapping
    if defined CLI_FILTER set "FILTER=%CLI_FILTER%"
    
    if "!IS_BARE_OR_DEFAULT!"=="1" (
        set "OUT_DIR=bin\win64"
    ) else (
        set "OUT_DIR=bin\win64\!CURRENT_ARCH!"
    )
    
    echo ===================================================
    echo Target Code: !CURRENT_ARCH! ^[HSIZE=!HSIZE!, FILTER=!FILTER!^]
    echo Output Path: !OUT_DIR!\keyhunt-gpu.exe
    echo ===================================================
    
    if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

    nvcc -O3 -std=c++14 -arch=!CURRENT_ARCH! ^
         -DHSIZE=!HSIZE! -DFILTER_LOG2_BITS=!FILTER! -D_CRT_SECURE_NO_WARNINGS ^
         -Xptxas -O2,-v ^
         %OSSL_INC% ^
         -o "!OUT_DIR!\keyhunt-gpu.exe" src\search.cu %OSSL_LIB% %OSSL_SYS%
    if errorlevel 1 (echo COMPILATION FOR !CURRENT_ARCH! FAILED & exit /b 1)
)

REM --- 7. BUILD HOST TESTS (With Pre-Build Target Isolation Cleanup) ---
echo.
set "TEST_DIR=bin\win64\tests"
set "LOG_FILE=!TEST_DIR!\test_results.log"

if exist "!TEST_DIR!" (
    echo Cleaning up previous test directory artifacts...
    del /q "!TEST_DIR!\*.exe" >nul 2>&1
    del /q "!TEST_DIR!\*.obj" >nul 2>&1
    del /q "!LOG_FILE!" >nul 2>&1
) else (
    mkdir "!TEST_DIR!"
)

echo Building host tests ...
for %%T in (test_math test_kernel_logic test_targets test_progress test_ptx_sequence test_glv) do (
  cl /nologo /O2 /EHsc /std:c++14 /D_CRT_SECURE_NO_WARNINGS ^
     /DHSIZE=%HSIZE% /DFILTER_LOG2_BITS=%FILTER% ^
     test\%%T.cpp /Fe:"!TEST_DIR!\%%T.exe" /Fo:"!TEST_DIR!\%%T.obj" >nul
  if errorlevel 1 (echo FAILED building %%T & exit /b 1)
)

cl /nologo /O2 /EHsc /std:c++14 /D_CRT_SECURE_NO_WARNINGS ^
   %OSSL_INC% test\test_seal.cpp /Fe:"!TEST_DIR!\test_seal.exe" /Fo:"!TEST_DIR!\test_seal.obj" ^
   %OSSL_LIB% %OSSL_SYS% >nul
if errorlevel 1 (echo FAILED building test_seal & exit /b 1)
del "!TEST_DIR!\*.obj" >nul 2>&1

REM --- 8. COLLECT DETAILED GPU DEVICE PROPERTIES FOR THE SUMMARY LOG ---
echo Gathering hardware device profile...
set "PROP_NAME=N/A"
set "PROP_RAM=N/A"
set "PROP_DRV=N/A"

for /f "tokens=2 delims==" %%I in ('wmic path win32_videocontroller get caption /format:list 2^nul') do (
    echo %%I | findstr /I "NVIDIA" >nul && set "PROP_NAME=%%I"
)
for /f "tokens=2 delims==" %%I in ('wmic path win32_videocontroller get AdapterRAM /format:list 2^nul') do (
    if not "%%I"=="" set "PROP_RAM=%%I"
)
for /f "tokens=2 delims==" %%I in ('wmic path win32_videocontroller get DriverVersion /format:list 2^nul') do (
    if not "%%I"=="" set "PROP_DRV=%%I"
)

REM Convert RAM bytes to Megabytes safely using raw math string manipulation
if not "!PROP_RAM!"=="N/A" (
    set /a "PROP_RAM_MB=!PROP_RAM! / 1048576"
) else (
    set "PROP_RAM_MB=N/A"
)

REM --- 9. RUN EXECUTION TESTS WITH AUTO-REDIRECT TO LOG FILE ---
echo Running host tests and exporting logs...
echo =================================================== > "!LOG_FILE!"
echo               SYSTEM BUILD SUMMATION REPORT        >> "!LOG_FILE!"
echo =================================================== >> "!LOG_FILE!"
echo Timestamp:          %DATE% %TIME% >> "!LOG_FILE!"
echo Build Architecture: %ARCH_TARGETS% >> "!LOG_FILE!"
echo Settings Applied:   HSIZE=%HSIZE%, FILTER=%FILTER% >> "!LOG_FILE!"
echo --------------------------------------------------- >> "!LOG_FILE!"
echo GPU Device Name:    !PROP_NAME! >> "!LOG_FILE!"
echo Dedicated VRAM:     !PROP_RAM_MB! MB >> "!LOG_FILE!"
echo Windows Driver Ver: !PROP_DRV! >> "!LOG_FILE!"
echo =================================================== >> "!LOG_FILE!"
echo. >> "!LOG_FILE!"

for %%X in (test_math test_kernel_logic test_targets test_progress test_ptx_sequence test_seal test_glv) do (
    echo Running %%X... >> "!LOG_FILE!"
    "!TEST_DIR!\%%X.exe" >> "!LOG_FILE!" 2>&1
    if errorlevel 1 (
        echo.
        echo ❌ TEST CRITICAL FAILURE: %%X failed execution.
        echo ---------------------------------------------------
        echo Dumping log snippet below:
        echo ---------------------------------------------------
        type "!LOG_FILE!"
        exit /b 1
    )
    echo %%X completed successfully. >> "!LOG_FILE!"
    echo ----------------------------------------- >> "!LOG_FILE!"
)

echo.
echo Build process complete. All tests passed.
echo Summary log exported to: !LOG_FILE!
endlocal
