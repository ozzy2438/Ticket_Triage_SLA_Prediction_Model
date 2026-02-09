#!/bin/bash

# =============================================================================
# QUICK START SCRIPT - AI Ticket Triage System
# =============================================================================
# This script sets up and starts the production-ready API
# =============================================================================

set -e  # Exit on error

echo "======================================================================="
echo "  🎫 AI-POWERED TICKET TRIAGE SYSTEM - QUICK START"
echo "======================================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "   ✅ Dependencies installed"

# Check if model is trained
echo ""
echo "🤖 Checking model status..."
if [ -f "checkpoints/best_model.pt" ] && [ -f "checkpoints/label_encoder.pkl" ] && [ -f "checkpoints/scaler.pkl" ]; then
    echo "   ✅ Model artifacts found - Production mode ready"
    MODEL_STATUS="PRODUCTION"
else
    echo "   ⚠️  Model artifacts not found - API will run in DEMO mode"
    echo ""
    echo "   To train the model:"
    echo "   1. Open notebooks/04_evaluation.ipynb"
    echo "   2. Run all cells"
    echo "   3. Model will be saved to checkpoints/"
    echo ""
    MODEL_STATUS="DEMO"
fi

# Start API
echo ""
echo "======================================================================="
echo "  🚀 STARTING API SERVER"
echo "======================================================================="
echo "   Mode: $MODEL_STATUS"
echo "   Host: http://localhost:8000"
echo ""
echo "   📖 Documentation: http://localhost:8000/docs"
echo "   💚 Health Check:  http://localhost:8000/health"
echo "   📊 Statistics:    http://localhost:8000/stats"
echo "======================================================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd api
python main.py
