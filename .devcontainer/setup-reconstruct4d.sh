#!/bin/bash
set -e

echo "Setting up reconstruct4D environment..."

# Check if we're in the right directory
if [ -f "/workspace/pyproject.toml" ]; then
    cd /workspace
    
    # Initialize git submodules
    if [ -d ".git" ]; then
        echo "Initializing git submodules..."
        git submodule update --init --recursive || true
    fi
    
    # Run uv sync to set up the Python environment
    echo "Running uv sync to install Python dependencies..."
    uv sync
    
    echo "reconstruct4D environment setup complete!"
    echo "To activate the environment, run: source .venv/bin/activate"
else
    echo "Note: reconstruct4D project files not found. Run 'uv sync' manually after cloning the repository."
fi