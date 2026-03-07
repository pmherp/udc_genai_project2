#!/bin/bash

# Ensure the script stops on errors
set -e

# Create and activate a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please ensure it exists in the root directory."
    exit 1
fi

# Notify the user
echo "Setup complete. Virtual environment is ready and dependencies are installed."