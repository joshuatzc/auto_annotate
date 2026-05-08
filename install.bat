@echo off
setlocal

set ENV_NAME=auto_anno

echo Checking conda...
conda --version >nul 2>&1 || (
    echo ERROR: conda not found.
    echo   Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

conda env list | findstr /R "^%ENV_NAME% " >nul 2>&1
if %errorlevel% == 0 (
    conda run -n %ENV_NAME% python --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Conda environment '%ENV_NAME%' already exists -- updating packages...
    ) else (
        echo Conda environment '%ENV_NAME%' exists but has no Python -- reinstalling...
        conda env remove -n %ENV_NAME% -y
        conda create -n %ENV_NAME% python=3.9 -y
    )
) else (
    echo Creating conda environment '%ENV_NAME%' with Python 3.9...
    conda create -n %ENV_NAME% python=3.9 -y
)

echo Installing dependencies...
conda run -n %ENV_NAME% python -m pip install --upgrade pip --quiet
conda run -n %ENV_NAME% python -m pip install -r requirements.txt --quiet

echo.
echo Done. To use the tool:
echo   conda activate %ENV_NAME%
echo   python -m auto_annotate.cli
