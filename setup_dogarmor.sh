#!/bin/bash

echo "🐕 DogArmor Setup Script"
echo "=========================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install required packages
echo "📦 Installing dependencies..."
pip3 install --break-system-packages Flask pandas Pillow Werkzeug

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p app/static/uploads

# Test the application
echo "🧪 Testing application..."
python3 -c "from app import app; print('✅ App imports successfully')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 DogArmor is ready!"
    echo "=========================="
    echo "🚀 To start the application:"
    echo "   python3 run.py"
    echo ""
    echo "📱 Visit: http://localhost:5000"
    echo "🎯 Demo: http://localhost:5000/demo"
    echo ""
    echo "🔧 Optional: Train AI model"
    echo "   python3 train_breed_model.py"
    echo ""
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi