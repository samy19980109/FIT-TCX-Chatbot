#!/bin/bash

# Fitness Data Analyzer - Startup Script
echo "🏃‍♂️ Starting Fitness Data Analyzer..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your Google Cloud credentials before running."
fi

# Start the Streamlit app
echo "🚀 Launching Streamlit application..."
streamlit run main.py

echo "👋 Thanks for using Fitness Data Analyzer!"
