#!/bin/bash

# ISL NMF MVP - Quick Start Script
# This script starts the local web server and opens the browser

echo "🚀 Starting ISL NMF MVP..."
echo ""

# Change to web directory
cd "$(dirname "$0")/web"

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✓ Python 3 found"
    echo "Starting server on http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "================================================"
    echo ""
    
    # Open browser after a short delay
    (sleep 2 && open http://localhost:8000) &
    
    # Start server
    python3 -m http.server 8000
else
    echo "❌ Python 3 not found"
    echo "Please install Python 3 or start the server manually:"
    echo "  cd web"
    echo "  python3 -m http.server 8000"
    exit 1
fi
