@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - vllm Server ^& Client
echo  OpenAI-compatible GPU inference server
echo  GPU REQUIRED (CUDA 11.8+)
echo ============================================================
echo.

REM --- Check vllm ---
python -c "import vllm" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vllm is not installed.
    echo         Run install-dep-vllm.bat first.
    pause
    exit /b 1
)

echo  Select action:
echo.
echo  [1] Start vllm Server (background API server)
echo  [2] Chat via Python client (requires server already running)
echo  [3] Start server AND open chat client (combined)
echo  [4] Show server status / test connection
echo  [0] Exit
echo.
set /p ACTION_CHOICE="Enter choice: "

if "%ACTION_CHOICE%"=="0" exit /b 0
if "%ACTION_CHOICE%"=="1" goto :START_SERVER
if "%ACTION_CHOICE%"=="2" goto :START_CLIENT
if "%ACTION_CHOICE%"=="3" goto :START_BOTH
if "%ACTION_CHOICE%"=="4" goto :TEST_CONNECTION
echo [ERROR] Invalid choice.
pause
exit /b 1

REM ============================================================
:MODEL_SELECT
echo.
echo  Select Gemma 4 model:
echo.
echo  [1] google/gemma-4-E2B-it    (~5 GB,  needs ~8 GB VRAM)
echo  [2] google/gemma-4-E4B-it    (~8 GB,  needs ~12 GB VRAM)
echo  [3] google/gemma-4-26B-A4B-it (~27 GB, needs ~32 GB VRAM)
echo  [4] google/gemma-4-31B-it    (~31 GB, needs ~48 GB VRAM)
echo  [5] Custom model ID
echo  [0] Back
echo.
set /p MODEL_CHOICE="Enter choice: "

if "%MODEL_CHOICE%"=="0" exit /b 0
if "%MODEL_CHOICE%"=="1" set "MODEL_ID=google/gemma-4-E2B-it"     & set "MAX_LEN=8192"
if "%MODEL_CHOICE%"=="2" set "MODEL_ID=google/gemma-4-E4B-it"     & set "MAX_LEN=8192"
if "%MODEL_CHOICE%"=="3" set "MODEL_ID=google/gemma-4-26B-A4B-it" & set "MAX_LEN=16384"
if "%MODEL_CHOICE%"=="4" set "MODEL_ID=google/gemma-4-31B-it"     & set "MAX_LEN=16384"
if "%MODEL_CHOICE%"=="5" (
    set /p MODEL_ID="Enter HuggingFace model ID: "
    set "MAX_LEN=8192"
)
if not defined MODEL_ID goto :MODEL_SELECT
goto :eof

REM ============================================================
:START_SERVER
call :MODEL_SELECT
if not defined MODEL_ID exit /b 1

set "PORT=8000"
set /p PORT="Server port [default: 8000]: "
if "%PORT%"=="" set "PORT=8000"

set "DTYPE=bfloat16"
set /p DTYPE="dtype [bfloat16 / float16, default: bfloat16]: "
if "%DTYPE%"=="" set "DTYPE=bfloat16"

set "GPU_UTIL=0.90"
set /p GPU_UTIL="GPU memory utilization [default: 0.90]: "
if "%GPU_UTIL%"=="" set "GPU_UTIL=0.90"

echo.
echo [INFO] Starting vllm server...
echo [INFO] Model:    %MODEL_ID%
echo [INFO] Port:     %PORT%
echo [INFO] dtype:    %DTYPE%
echo [INFO] Max len:  %MAX_LEN%
echo [INFO] GPU util: %GPU_UTIL%
echo.
echo [INFO] Server will be available at: http://localhost:%PORT%/v1
echo [INFO] Press Ctrl+C to stop the server.
echo.

python -m vllm.entrypoints.openai.api_server ^
    --model %MODEL_ID% ^
    --port %PORT% ^
    --dtype %DTYPE% ^
    --max-model-len %MAX_LEN% ^
    --gpu-memory-utilization %GPU_UTIL% ^
    --served-model-name %MODEL_ID%

goto :eof

REM ============================================================
:START_CLIENT
set "SERVER_PORT=8000"
set /p SERVER_PORT="Server port [default: 8000]: "
if "%SERVER_PORT%"=="" set "SERVER_PORT=8000"

REM Get model from server
for /f "delims=" %%M in ('python -c "import urllib.request,json;r=urllib.request.urlopen('http://localhost:%SERVER_PORT%/v1/models');data=json.load(r);print(data['data'][0]['id'] if data['data'] else 'unknown')" 2^>nul') do set "ACTIVE_MODEL=%%M"
if "%ACTIVE_MODEL%"=="" (
    echo [ERROR] No running vllm server found at port %SERVER_PORT%.
    echo         Start the server first with option [1].
    pause
    exit /b 1
)
echo [OK] Connected to server. Active model: %ACTIVE_MODEL%
echo.

set "TEMP_SCRIPT=%TEMP%\gemma4_vllm_client_%RANDOM%.py"
(
echo # Gemma 4 vllm OpenAI client chat
echo from openai import OpenAI
echo.
echo client = OpenAI(base_url="http://localhost:%SERVER_PORT%/v1", api_key="not-needed")
echo MODEL = r"%ACTIVE_MODEL%"
echo.
echo messages = [{"role": "system", "content": "You are a helpful assistant."}]
echo print(f"Gemma 4 vllm Chat (model: {MODEL})")
echo print(f"Server: http://localhost:%SERVER_PORT%/v1")
echo print("Type 'quit' to exit, 'reset' to clear history")
echo print("=" * 60)
echo.
echo while True:
echo     try:
echo         user_input = input("You: ").strip()
echo     except (EOFError, KeyboardInterrupt):
echo         print("\nExiting.")
echo         break
echo     if not user_input:
echo         continue
echo     if user_input.lower() in ("quit", "exit", "q"):
echo         print("Goodbye!")
echo         break
echo     if user_input.lower() == "reset":
echo         messages = [{"role": "system", "content": "You are a helpful assistant."}]
echo         print("[INFO] Conversation history cleared.")
echo         continue
echo.
echo     messages.append({"role": "user", "content": user_input})
echo     try:
echo         stream = client.chat.completions.create(
echo             model=MODEL,
echo             messages=messages,
echo             temperature=1.0,
echo             top_p=0.95,
echo             max_tokens=1024,
echo             stream=True,
echo         )
echo         print("Gemma: ", end="", flush=True)
echo         full_response = ""
echo         for chunk in stream:
echo             delta = chunk.choices[0].delta.content or ""
echo             print(delta, end="", flush=True)
echo             full_response += delta
echo         print()
echo         messages.append({"role": "assistant", "content": full_response})
echo     except Exception as e:
echo         print(f"\n[ERROR] {e}")
echo         messages.pop()
) > "%TEMP_SCRIPT%"

python "%TEMP_SCRIPT%"
del "%TEMP_SCRIPT%" >nul 2>&1
goto :eof

REM ============================================================
:START_BOTH
call :MODEL_SELECT
if not defined MODEL_ID exit /b 1

set "PORT=8000"
set /p PORT="Server port [default: 8000]: "
if "%PORT%"=="" set "PORT=8000"

echo.
echo [INFO] Starting vllm server in background window...
start "Gemma4 vllm Server" cmd /c "python -m vllm.entrypoints.openai.api_server --model %MODEL_ID% --port %PORT% --dtype bfloat16 --max-model-len %MAX_LEN% --gpu-memory-utilization 0.90 --served-model-name %MODEL_ID%"

echo [INFO] Waiting 30 seconds for server to initialize...
timeout /t 30 /nobreak >nul

echo [INFO] Testing server connection...
python -c "import urllib.request; urllib.request.urlopen('http://localhost:%PORT%/health')" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Server may still be starting. Proceeding to client anyway.
) else (
    echo [OK] Server is ready.
)
echo.

REM Start client
set "SERVER_PORT=%PORT%"
set "ACTIVE_MODEL=%MODEL_ID%"

set "TEMP_SCRIPT=%TEMP%\gemma4_vllm_client_%RANDOM%.py"
(
echo from openai import OpenAI
echo client = OpenAI(base_url="http://localhost:%PORT%/v1", api_key="not-needed")
echo MODEL = r"%MODEL_ID%"
echo messages = [{"role": "system", "content": "You are a helpful assistant."}]
echo print(f"Gemma 4 vllm Chat (model: {MODEL})")
echo print("Type 'quit' to exit, 'reset' to clear history")
echo print("=" * 60)
echo while True:
echo     try:
echo         user_input = input("You: ").strip()
echo     except (EOFError, KeyboardInterrupt):
echo         print("\nExiting."); break
echo     if not user_input: continue
echo     if user_input.lower() in ("quit","exit","q"): print("Goodbye!"); break
echo     if user_input.lower() == "reset":
echo         messages = [{"role": "system", "content": "You are a helpful assistant."}]
echo         print("[INFO] History cleared."); continue
echo     messages.append({"role": "user", "content": user_input})
echo     try:
echo         stream = client.chat.completions.create(model=MODEL, messages=messages, temperature=1.0, top_p=0.95, max_tokens=1024, stream=True)
echo         print("Gemma: ", end="", flush=True)
echo         full = ""
echo         for chunk in stream:
echo             d = chunk.choices[0].delta.content or ""; print(d, end="", flush=True); full += d
echo         print(); messages.append({"role": "assistant", "content": full})
echo     except Exception as e:
echo         print(f"\n[ERROR] {e}"); messages.pop()
) > "%TEMP_SCRIPT%"

python "%TEMP_SCRIPT%"
del "%TEMP_SCRIPT%" >nul 2>&1
goto :eof

REM ============================================================
:TEST_CONNECTION
set "PORT=8000"
set /p PORT="Server port to test [default: 8000]: "
if "%PORT%"=="" set "PORT=8000"

echo.
echo [INFO] Testing vllm server at http://localhost:%PORT%...
python -c "import urllib.request,json; r=urllib.request.urlopen('http://localhost:%PORT%/v1/models'); data=json.load(r); [print(f'  Model: {m[\"id\"]}') for m in data['data']]; print('[OK] Server is running.')" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot connect to http://localhost:%PORT%
    echo         Make sure the server is running (option [1]).
)
echo.
pause
goto :eof
