from app import app

if __name__ == '__main__':
    print("🚀 Starting DogArmor...")
    print("📱 Visit: http://localhost:5000")
    print("🎯 Demo: http://localhost:5000/demo")
    app.run(debug=True, host='0.0.0.0', port=5000)
