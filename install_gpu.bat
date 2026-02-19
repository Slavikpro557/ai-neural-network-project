@echo off
echo.
echo ================================================================
echo          Installing PyTorch with GPU Support
echo ================================================================
echo.

echo Checking GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if errorlevel 1 (
    echo.
    echo ERROR: No NVIDIA GPU detected!
    echo Make sure drivers are installed.
    pause
    exit /b 1
)
echo.
echo GPU found! Proceeding with installation...
echo.

echo Step 1: Removing old PyTorch (CPU version)...
python -m pip uninstall -y torch torchvision torchaudio
echo Done
echo.

echo Step 2: Installing PyTorch with CUDA 12.4...
echo This will take 5-10 minutes...
echo.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

if errorlevel 1 (
    echo.
    echo CUDA 12.4 failed, trying CUDA 12.1...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

if errorlevel 1 (
    echo.
    echo CUDA 12.1 failed, trying CUDA 11.8...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

if errorlevel 1 (
    echo.
    echo ERROR: All CUDA versions failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 3: Verifying installation...
echo ================================================================
echo.
python -c "import torch; print('PyTorch version:', torch.__version__); cuda=torch.cuda.is_available(); print('CUDA available:', cuda); print('GPU name:', torch.cuda.get_device_name(0) if cuda else 'NONE'); print('GPU memory:', str(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))+'GB' if cuda else 'N/A')"
echo.

python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ================================================================
    echo ERROR: GPU still not detected!
    echo ================================================================
    pause
    exit /b 1
)

echo ================================================================
echo SUCCESS! GPU is ready to use!
echo ================================================================
echo.
echo You can now run: start.bat
echo.
pause
