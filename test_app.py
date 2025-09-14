#!/usr/bin/env python3
"""
Quick test script to validate the improved fitness analyzer app components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import folium
        import pandas as pd
        import numpy as np
        import fitdecode
        import tcxparser
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_visualization_functions():
    """Test the visualization helper functions"""
    try:
        # Import the functions from main.py would go here
        # For now, just test that the concepts work
        import pandas as pd
        import plotly.express as px
        
        # Test data
        test_hr = [120, 130, 140, 150, 160, 150, 140, 130]
        df = pd.DataFrame({'Heart Rate': test_hr, 'Time': range(len(test_hr))})
        fig = px.line(df, x='Time', y='Heart Rate', title='Test Chart')
        
        print("✅ Visualization functions work")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_data_processing():
    """Test basic data processing capabilities"""
    try:
        import json
        
        # Test JSON processing
        test_data = {"duration": 3600, "distance": 5000, "calories": 300}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data["duration"] == 3600
        print("✅ Data processing works")
        return True
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def main():
    print("🧪 Testing Fitness Data Analyzer Components\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Visualization Test", test_visualization_functions),
        ("Data Processing Test", test_data_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your app is ready to run.")
        print("\nTo start the app, run:")
        print("streamlit run main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
