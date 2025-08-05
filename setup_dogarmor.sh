#!/bin/bash

echo "🐕 Setting up DogArmor - AI-Powered Dog Gear Recommendations"
echo "================================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p app/static/uploads
mkdir -p models/retrained_models

# Create gear_data.csv if it doesn't exist
if [ ! -f "gear_data.csv" ]; then
    echo "📊 Creating sample gear data..."
    cat > gear_data.csv << 'EOF'
breed,size,pulls,budget,gear_type,recommendation
labrador,large,yes,medium,harness,Active dog harness
beagle,medium,no,low,collar,Comfortable collar
bulldog,large,yes,high,leash,Strong leash
poodle,medium,no,medium,toys,Interactive toys
golden_retriever,large,yes,high,harness,Premium harness
german_shepherd,large,yes,high,leash,Heavy-duty leash
EOF
fi

echo ""
echo "✅ Setup complete! DogArmor is ready to run."
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python3 run.py"
echo ""
echo "🌐 Then visit: http://localhost:5000"
echo ""
echo "📝 Optional: Train the AI model for better predictions:"
echo "   python3 train_breed_model.py"
echo ""