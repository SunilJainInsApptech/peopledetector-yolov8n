#!/bin/bash

set -eu

PROJECT_DIR=$(cd $(dirname $0) && pwd)
cd $PROJECT_DIR

# Create a virtual environment to run our code
VENV_NAME=".venv-build"
PYTHON="$PROJECT_DIR/$VENV_NAME/bin/python"

export PATH=$PATH:$HOME/.local/bin

# Check if virtual environment exists, if not create it
if [ ! -d "$VENV_NAME" ]; then
    if [ ! "$(command -v uv)" ]; then
        if [ ! "$(command -v curl)" ]; then
            echo "curl is required to install UV. please install curl on this system to continue."
            exit 1
        fi
        echo "Installing uv command"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    
    if ! uv venv $VENV_NAME; then
        echo "unable to create required virtual environment"
        exit 1
    fi
    
    source $VENV_NAME/bin/activate
    
    if ! uv pip install -r requirements.txt; then
        echo "unable to sync requirements to venv"
        exit 1
    fi
else
    source $VENV_NAME/bin/activate
fi

# Run the main Python module
cd src
exec $PYTHON main.py "$@"