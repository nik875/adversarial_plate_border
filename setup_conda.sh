#!/bin/bash

set -e

# Configuration
ENV_NAME="${1:-adversarial-plate}"
PYTHON_VERSION="${2:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Detect OS and architecture
    OS=$(uname -s)
    ARCH=$(uname -m)

    if [ "$OS" = "Linux" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        if [ "$ARCH" = "aarch64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        fi
    elif [ "$OS" = "Darwin" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        if [ "$ARCH" = "arm64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        fi
    else
        echo "Error: Unsupported OS: $OS"
        exit 1
    fi

    echo "Downloading from: $MINICONDA_URL"
    curl -fsSL -o /tmp/miniconda.sh "$MINICONDA_URL"
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash

    echo "Miniconda installed to $HOME/miniconda3"
    echo "Please run: source ~/.bashrc"
    echo "Then re-run this script."
    exit 0
else
    echo "Conda found: $(conda --version)"
fi

echo ""

# Step 2: Create conda environment
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Step 3: Install dependencies
echo ""
echo "Activating environment and installing packages..."

# Use conda run to execute commands in the environment
echo "Installing PyTorch with CUDA 12.8 support..."
conda run -n "$ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installing remaining dependencies..."
conda run -n "$ENV_NAME" pip install \
    matplotlib \
    pandas \
    tqdm \
    scipy \
    scikit-learn \
    cma \
    kornia \
    shapely \
    fast-alpr \
    open_image_models[onnx] \
    onnx2torch \
    pillow_heif \
    seaborn \
    python-Levenshtein \
    diffusers \
    transformers \
    accelerate \
    b2 \
    datasets \
    sentencepiece \
    h5py

echo ""
echo "Installation complete!"
echo ""

# Step 4: Export environment
echo "Exporting environment to environment.yml..."
conda env export -n "$ENV_NAME" > "$PROJECT_DIR/environment.yml"
echo "Saved to: $PROJECT_DIR/environment.yml"

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To use this environment on another machine, run:"
echo "  conda env create -f environment.yml"
