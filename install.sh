#!/usr/bin/env bash
set -e

ENV_NAME="auto_anno"

echo "Checking conda..."
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found."
    echo "  Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if env exists AND has Python installed inside it (not a system fallback)
ENV_HAS_PYTHON=false
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    PYTHON_PATH=$(conda run -n "${ENV_NAME}" python3 -c "import sys; print(sys.executable)" 2>/dev/null || true)
    if echo "${PYTHON_PATH}" | grep -q "/envs/${ENV_NAME}/"; then
        ENV_HAS_PYTHON=true
    fi
fi

if [ "${ENV_HAS_PYTHON}" = true ]; then
    echo "Conda environment '${ENV_NAME}' already exists — updating packages..."
elif conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment '${ENV_NAME}' exists but has no Python — reinstalling..."
    conda env remove -n "${ENV_NAME}" -y
    conda create -n "${ENV_NAME}" python=3.9 -y
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.9..."
    conda create -n "${ENV_NAME}" python=3.9 -y
fi

echo "Installing dependencies..."
conda run -n "${ENV_NAME}" python3 -m pip install --upgrade pip --quiet
conda run -n "${ENV_NAME}" python3 -m pip install -r requirements.txt --quiet

echo ""
echo "Done. To use the tool:"
echo "  conda activate ${ENV_NAME}"
echo "  python3 -m auto_annotate.cli"
