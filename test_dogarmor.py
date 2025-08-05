#!/usr/bin/env python3
"""
DogArmor Comprehensive Test Script
Tests all major functionality after the complete fixes
"""

import sys
from app import app

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    try:
        from app import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_routes():
    """Test all web routes"""
    print("\n🌐 Testing web routes...")
    client = app.test_client()
    
    routes = [
        ('/', 'Home'),
        ('/demo', 'Demo'),
        ('/features', 'Features'),
        ('/about', 'About'),
        ('/team', 'Team'),
        ('/how-it-works', 'How it Works'),
        ('/why-us', 'Why Us')
    ]
    
    all_passed = True
    for route, name in routes:
        try:
            response = client.get(route)
            if response.status_code == 200:
                print(f"✅ {name}: {response.status_code}")
            else:
                print(f"❌ {name}: {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_passed = False
    
    return all_passed

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔌 Testing API endpoints...")
    client = app.test_client()
    
    # Test /recommend endpoint
    try:
        response = client.post('/recommend', json={
            'breed': 'labrador',
            'size': 'medium',
            'puller': 'yes',
            'budget': 'medium'
        })
        if response.status_code == 200:
            data = response.get_json()
            print(f"✅ /recommend: {response.status_code}")
            print(f"   Response: {data['recommendations']}")
        else:
            print(f"❌ /recommend: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ /recommend: Error - {e}")
        return False
    
    return True

def test_upload_directory():
    """Test that upload directory exists"""
    print("\n📁 Testing upload directory...")
    import os
    upload_dir = 'app/static/uploads'
    if os.path.exists(upload_dir):
        print(f"✅ Upload directory exists: {upload_dir}")
        return True
    else:
        print(f"❌ Upload directory missing: {upload_dir}")
        return False

def test_static_files():
    """Test that static files are accessible"""
    print("\n📄 Testing static files...")
    import os
    static_dir = 'app/static'
    if os.path.exists(static_dir):
        files = os.listdir(static_dir)
        if len(files) > 0:
            print(f"✅ Static files found: {len(files)} files")
            return True
        else:
            print("❌ No static files found")
            return False
    else:
        print("❌ Static directory missing")
        return False

def test_templates():
    """Test that templates are accessible"""
    print("\n🎨 Testing templates...")
    import os
    template_dir = 'app/templates'
    if os.path.exists(template_dir):
        files = os.listdir(template_dir)
        required_templates = ['base.html', 'index.html', 'demo.html']
        missing = [t for t in required_templates if t not in files]
        if not missing:
            print(f"✅ All required templates found: {len(files)} files")
            return True
        else:
            print(f"❌ Missing templates: {missing}")
            return False
    else:
        print("❌ Template directory missing")
        return False

def main():
    """Run all tests"""
    print("🐕 DogArmor Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_routes,
        test_api_endpoints,
        test_upload_directory,
        test_static_files,
        test_templates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! DogArmor is ready for production!")
        print("\n🚀 To launch the application:")
        print("   python3 run.py")
        print("\n📱 Visit: http://localhost:5000")
        print("🎯 Demo: http://localhost:5000/demo")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())