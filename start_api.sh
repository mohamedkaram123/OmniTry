#!/bin/bash
# Script to start OmniTry API server
# اسكريبت لتشغيل سيرفر OmniTry

set -e

echo "🚀 Starting OmniTry API Server..."
echo ""

# Check if checkpoints exist
if [ ! -d "checkpoints/FLUX.1-Fill-dev" ]; then
    echo "⚠️  Warning: Checkpoints not found!"
    echo "Please download the models first:"
    echo ""
    echo "mkdir -p checkpoints"
    echo "cd checkpoints"
    echo "git clone https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
    echo "wget https://huggingface.co/Kunbyte/OmniTry/resolve/main/omnitry_v1_unified.safetensors"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo "📦 Checking dependencies..."
pip show fastapi > /dev/null 2>&1 || {
    echo "Installing API dependencies..."
    pip install -r requirements_api.txt
}

# Check CUDA
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Start server
echo ""
echo "✅ Starting server on http://0.0.0.0:8000"
echo "📖 API docs will be available at http://0.0.0.0:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python api_server.py
