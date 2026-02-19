@echo off
echo.
echo ================================================================
echo           AZR Model Trainer - Starting
echo ================================================================
echo.

echo [1/4] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo OK - Python found
echo.

echo [2/4] Checking PyTorch...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo PyTorch not installed - installing now...
    echo.
    
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo No GPU detected - installing CPU version
        echo WARNING: Training will be SLOW!
        echo.
        python -m pip install torch torchvision torchaudio
    ) else (
        echo GPU detected! Installing PyTorch with CUDA...
        nvidia-smi --query-gpu=name --format=csv,noheader
        echo.
        echo This takes 5-10 minutes, please wait...
        echo.
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        if errorlevel 1 (
            echo Trying CUDA 12.1...
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        )
        if errorlevel 1 (
            echo Trying CUDA 11.8...
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        )
        if errorlevel 1 (
            echo CUDA install failed - using CPU version
            python -m pip install torch torchvision torchaudio
        )
    )
    echo.
    echo PyTorch installation complete!
    echo.
) else (
    echo OK - PyTorch already installed
)
echo.

echo Installing other dependencies...
python -m pip install -q -r requirements.txt
echo OK - Dependencies ready
echo.

echo [3/4] Checking files...
if not exist "templates\index_complete.html" (
    echo Interface file not found - building...
    python build_complete_interface.py
)
echo OK - Files ready
echo.

echo [4/4] Device info:
python -c "import torch; gpu=torch.cuda.is_available(); print('Device:', torch.cuda.get_device_name(0) if gpu else 'CPU'); print('Memory:', str(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))+'GB' if gpu else 'N/A')"
echo.

echo ================================================================
echo   Server starting at: http://localhost:8000
echo.
echo   Browser will open automatically
echo   Press Ctrl+C to stop
echo ================================================================
echo.

start http://localhost:8000
python server_with_datasets.py

echo.
echo ================================================================
echo   Server stopped
echo ================================================================
pause
