#!/bin/bash

echo "🐕 Launching DogArmor..."
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./setup_dogarmor.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ Dependencies not installed. Please run setup first:"
    echo "   ./setup_dogarmor.sh"
    exit 1
fi

# Launch the application
echo "🚀 Starting DogArmor server..."
echo "🌐 Visit: http://localhost:5000"
echo "📱 Demo: http://localhost:5000/demo"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 run.py