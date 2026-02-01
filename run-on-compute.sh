#!/bin/bash
# Run latest changes on compute machine
# Usage: ./run-on-compute.sh [command] [args...]

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="foundationmodel"

echo "==== Adversarial Plate Compute Runner ===="
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"
echo ""

# Pull latest changes
echo "Pulling latest changes from git..."
cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull origin "$BRANCH"
echo "✓ Repository updated"
echo ""

# Show help if no command provided
if [ $# -eq 0 ]; then
    echo "Available commands:"
    echo ""
    echo "  ./run-on-compute.sh download-datasets"
    echo "    Extract datasets from local opensourcedata.tar"
    echo ""
    echo "  ./run-on-compute.sh download-datasets-b2"
    echo "    Download datasets from B2"
    echo ""
    echo "  ./run-on-compute.sh python <script> [args...]"
    echo "    Run a Python script"
    echo ""
    echo "  ./run-on-compute.sh bash <script> [args...]"
    echo "    Run a Bash script"
    echo ""
    echo "Example:"
    echo "  ./run-on-compute.sh python train.py --epochs 100"
    echo ""
    exit 0
fi

COMMAND=$1
shift

case "$COMMAND" in
    download-datasets)
        echo "Extracting datasets from local tarball..."
        "$REPO_DIR/foundationmodel/dataset/download_datasets.sh" --local
        ;;
    download-datasets-b2)
        echo "Downloading datasets from B2..."
        "$REPO_DIR/foundationmodel/dataset/download_datasets.sh"
        ;;
    python)
        echo "Running Python script: $@"
        python3 "$@"
        ;;
    bash)
        echo "Running Bash script: $@"
        bash "$@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Use './run-on-compute.sh' with no arguments for help"
        exit 1
        ;;
esac

echo ""
echo "✓ Command completed"
