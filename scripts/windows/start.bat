@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Interactive Chat
echo  Start an interactive chat session with a Gemma 4 model
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Run install-dep.bat first.
    pause
    exit /b 1
)

REM --- Model selection ---
echo  Select a model:
echo.
echo  [1] E2B   - google/gemma-4-E2B-it   (small, fast, audio+image)
echo  [2] E4B   - google/gemma-4-E4B-it   (medium, fast, audio+image)
echo  [3] 26B-A4B - google/gemma-4-26B-A4B-it (large MoE, image)
echo  [4] 31B   - google/gemma-4-31B-it   (largest dense, image)
echo  [5] Custom model path or HuggingFace ID
echo  [0] Exit
echo.
set /p MODEL_CHOICE="Enter choice [1-5, 0 to exit]: "

if "%MODEL_CHOICE%"=="0" exit /b 0
if "%MODEL_CHOICE%"=="1" set "MODEL_ID=google/gemma-4-E2B-it"
if "%MODEL_CHOICE%"=="2" set "MODEL_ID=google/gemma-4-E4B-it"
if "%MODEL_CHOICE%"=="3" set "MODEL_ID=google/gemma-4-26B-A4B-it"
if "%MODEL_CHOICE%"=="4" set "MODEL_ID=google/gemma-4-31B-it"
if "%MODEL_CHOICE%"=="5" (
    set /p MODEL_ID="Enter model ID or local path: "
)
if not defined MODEL_ID (
    echo [ERROR] Invalid choice.
    pause
    exit /b 1
)

REM --- Thinking mode ---
echo.
echo  Enable thinking/reasoning mode?
echo  [1] No  - Direct answer (faster, recommended for most tasks)
echo  [2] Yes - Step-by-step reasoning (slower, better for math/logic)
echo.
set /p THINK_CHOICE="Enter choice [1/2, default=1]: "
if "%THINK_CHOICE%"=="2" (
    set "ENABLE_THINKING=True"
    echo [INFO] Thinking mode: ENABLED
) else (
    set "ENABLE_THINKING=False"
    echo [INFO] Thinking mode: DISABLED
)

REM --- System prompt ---
echo.
set /p SYSTEM_PROMPT="System prompt [default: 'You are a helpful assistant.']: "
if "%SYSTEM_PROMPT%"=="" set "SYSTEM_PROMPT=You are a helpful assistant."

REM --- Max new tokens ---
echo.
set /p MAX_TOKENS="Max new tokens per response [default: 2048]: "
if "%MAX_TOKENS%"=="" set "MAX_TOKENS=2048"

REM --- Write and run the inference script ---
echo.
echo [INFO] Loading model: %MODEL_ID%
echo [INFO] Max tokens: %MAX_TOKENS%
echo [INFO] Thinking mode: %ENABLE_THINKING%
echo [INFO] This may take several minutes on first load (model download + initialization).
echo.

REM Write a temporary Python chat script
set "TEMP_SCRIPT=%TEMP%\gemma4_chat_%RANDOM%.py"

(
echo import sys
echo from transformers import AutoProcessor, AutoModelForCausalLM
echo import torch
echo.
echo MODEL_ID = r"%MODEL_ID%"
echo ENABLE_THINKING = %ENABLE_THINKING%
echo MAX_NEW_TOKENS = %MAX_TOKENS%
echo SYSTEM_PROMPT = r"%SYSTEM_PROMPT%"
echo.
echo print(f"[INFO] Loading model: {MODEL_ID}")
echo print("[INFO] Please wait...")
echo.
echo processor = AutoProcessor.from_pretrained(MODEL_ID)
echo model = AutoModelForCausalLM.from_pretrained(
echo     MODEL_ID,
echo     dtype="auto",
echo     device_map="auto",
echo )
echo.
echo print(f"[OK] Model loaded on device: {next(model.parameters()).device}")
echo print("=" * 60)
echo print(" Gemma 4 Interactive Chat")
echo print(f" Model: {MODEL_ID}")
echo print(f" Thinking mode: {'ON' if ENABLE_THINKING else 'OFF'}")
echo print(" Type 'quit' or 'exit' to stop.")
echo print(" Type 'clear' to reset conversation history.")
echo print("=" * 60)
echo print()
echo.
echo history = []
echo.
echo while True:
echo     try:
echo         user_input = input("You: ").strip()
echo     except (EOFError, KeyboardInterrupt):
echo         print("\n[INFO] Session ended.")
echo         break
echo.
echo     if user_input.lower() in ("quit", "exit"):
echo         print("[INFO] Goodbye!")
echo         break
echo.
echo     if user_input.lower() == "clear":
echo         history = []
echo         print("[INFO] Conversation history cleared.")
echo         continue
echo.
echo     if not user_input:
echo         continue
echo.
echo     history.append({"role": "user", "content": user_input})
echo.
echo     messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
echo.
echo     text = processor.apply_chat_template(
echo         messages,
echo         tokenize=False,
echo         add_generation_prompt=True,
echo         enable_thinking=ENABLE_THINKING,
echo     )
echo.
echo     inputs = processor(text=text, return_tensors="pt").to(model.device)
echo     input_len = inputs["input_ids"].shape[-1]
echo.
echo     with torch.inference_mode():
echo         outputs = model.generate(
echo             **inputs,
echo             max_new_tokens=MAX_NEW_TOKENS,
echo             temperature=1.0,
echo             top_p=0.95,
echo             top_k=64,
echo             do_sample=True,
echo         )
echo.
echo     raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
echo     parsed = processor.parse_response(raw_response)
echo.
echo     if ENABLE_THINKING and isinstance(parsed, dict) and "thinking" in parsed:
echo         print(f"\n[Thinking]\n{parsed['thinking']}\n")
echo         answer = parsed.get("response", raw_response)
echo     elif isinstance(parsed, dict):
echo         answer = parsed.get("response", str(parsed))
echo     else:
echo         answer = str(parsed)
echo.
echo     print(f"\nGemma 4: {answer}\n")
echo.
echo     # Add only the final answer to history (no thinking traces)
echo     history.append({"role": "assistant", "content": answer})
) > "%TEMP_SCRIPT%"

python "%TEMP_SCRIPT%"
set EXIT_CODE=%errorlevel%

REM Clean up temp file
del "%TEMP_SCRIPT%" >nul 2>&1

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Chat session ended with errors.
    echo         Make sure all dependencies are installed (run install-dep.bat).
)

pause
endlocal
