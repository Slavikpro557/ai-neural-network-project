@echo off
chcp 65001 >nul 2>&1
cls
echo.
echo ================================================================
echo        AZR Model Trainer - Install Dependencies
echo ================================================================
echo.

:: Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo [OK] Python found
echo.

:: Check pip
echo [2/4] Checking pip...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found!
    pause
    exit /b 1
)
echo [OK] pip found
echo.

:: Check GPU
echo [3/4] Checking GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU found!
    echo.
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
    set INSTALL_CUDA=1
) else (
    echo [INFO] NVIDIA GPU not found - will use CPU
    echo [WARNING] Training will be SLOW without GPU!
    set INSTALL_CUDA=0
)
echo.

:: Install dependencies
echo [4/4] Installing dependencies...
echo.

:: Remove old PyTorch
echo Removing old PyTorch...
python -m pip uninstall -y torch torchvision torchaudio 2>nul

:: Install PyTorch
if %INSTALL_CUDA%==1 (
    echo.
    echo Installing PyTorch with CUDA 12.4 (for GPU)...
    echo This may take several minutes...
    echo.
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    if %errorlevel% neq 0 (
        echo.
        echo [WARNING] Failed to install CUDA 12.4, trying CUDA 12.1...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        if %errorlevel% neq 0 (
            echo.
            echo [WARNING] Failed to install CUDA 12.1, trying CUDA 11.8...
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            
            if %errorlevel% neq 0 (
                echo.
                echo [ERROR] Failed to install PyTorch with CUDA!
                echo Installing CPU version...
                python -m pip install torch torchvision torchaudio
            )
        )
    )
) else (
    echo.
    echo Installing PyTorch (CPU version)...
    echo [WARNING] Without GPU training will be very slow!
    echo.
    python -m pip install torch torchvision torchaudio
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

:: Install other packages
echo Installing other packages...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] All dependencies installed
echo.

:: Check installation
echo ================================================================
echo                    CHECKING INSTALLATION
echo ================================================================
echo.
python -c "import torch; print('PyTorch version:', torch.__version__); cuda = torch.cuda.is_available(); print('CUDA available:', cuda); print('Device:', torch.cuda.get_device_name(0) if cuda else 'CPU'); print('GPU memory:', str(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))+'GB' if cuda else 'N/A')"
echo.

:: Summary
echo ================================================================
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %errorlevel% equ 0 (
    echo INSTALLATION COMPLETE! GPU READY TO USE!
    echo.
    echo Training will be FAST!
) else (
    echo INSTALLATION COMPLETE (CPU mode)
    echo.
    echo [WARNING] GPU not detected - training will be SLOW!
    echo If you have NVIDIA GPU:
    echo   1. Install drivers: https://www.nvidia.com/Download/index.aspx
    echo   2. Run install.bat again
)
echo ================================================================
echo.
echo To start server use: start.bat
echo.
pause
