# Conda Environment Setup

This guide explains how to use the conda-based setup instead of pip and requirements.txt.

## Initial Setup on Fresh Linux Instance

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
./setup_conda.sh
```

**Optional parameters:**
- Environment name (default: `adversarial-plate`):
  ```bash
  ./setup_conda.sh myenv
  ```
- Python version (default: `3.11`):
  ```bash
  ./setup_conda.sh myenv 3.10
  ```

The script will:
1. Download and install Miniconda if conda is not present
2. Create a new conda environment with the specified Python version
3. Install all dependencies including PyTorch with CUDA 12.8 support
4. Generate `environment.yml` for reproducible environments

### Option 2: Manual Setup

```bash
# Install Miniconda (Linux x86_64)
curl -fsSL -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate

# Create environment
conda create -y -n adversarial-plate python=3.11

# Activate environment
conda activate adversarial-plate

# Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```

## Using the Environment

### Activate the environment:
```bash
conda activate adversarial-plate
```

### Deactivate:
```bash
conda deactivate
```

### Run scripts within the environment:
```bash
conda run -n adversarial-plate python script.py
```

## Sharing and Reproducing

### Export your environment:
```bash
conda env export -n adversarial-plate > environment.yml
```

The setup script automatically generates `environment.yml` after installation.

### Recreate environment on another machine:
```bash
conda env create -f environment.yml
conda activate adversarial-plate
```

### Update environment.yml after installing new packages:
```bash
conda activate adversarial-plate
# ... install new package with pip or conda ...
conda env export > environment.yml
```

## Installing Additional Packages

While the environment is activated:

```bash
# Using pip (for packages not in conda)
pip install package-name

# Using conda (when available)
conda install package-name
```

Then update `environment.yml`:
```bash
conda env export > environment.yml
```

## Environment Information

- **Default environment name:** `adversarial-plate`
- **Python version:** 3.11 (default, customizable)
- **CUDA support:** 12.8 (PyTorch wheels)
- **Location:** `~/miniconda3/envs/adversarial-plate`

## Troubleshooting

### Conda command not found after Miniconda installation:
```bash
source ~/.bashrc
# Or restart your terminal
```

### Update Miniconda:
```bash
conda update -n base -c defaults conda
```

### Remove environment:
```bash
conda env remove -n adversarial-plate
```

### Check installed packages:
```bash
conda activate adversarial-plate
conda list
```

## Comparison: pip vs Conda

| Aspect | pip + requirements.txt | Conda |
|--------|------------------------|-------|
| **Reproducibility** | Version pinning needed | environment.yml includes exact versions |
| **Binary dependencies** | May need system packages | Handles dependencies automatically |
| **PyTorch CUDA** | Extra index URL required | Integrated index management |
| **Cross-platform** | Often fails on different systems | Better cross-platform support |
| **Sharing** | Text file | YAML file with full environment |

## Notes

- The setup script uses Miniconda (lightweight) instead of full Anaconda
- PyTorch is installed via pip with the CUDA 12.8 wheel index to ensure GPU support
- Other packages are installed via pip as specified in `requirements.txt`
- The `environment.yml` captures the exact versions of all installed packages
