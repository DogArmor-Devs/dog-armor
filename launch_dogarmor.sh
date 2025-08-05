#!/bin/bash

echo "🐕 DogArmor Launcher"
echo "===================="

# Test the application first
echo "🧪 Testing application..."
python3 test_dogarmor.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Launching DogArmor..."
    echo "📱 Visit: http://localhost:5000"
    echo "🎯 Demo: http://localhost:5000/demo"
    echo "⏹️  Press Ctrl+C to stop"
    echo ""
    python3 run.py
else
    echo "❌ Tests failed. Please fix the issues before launching."
    exit 1
fi