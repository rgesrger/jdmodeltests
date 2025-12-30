#!/bin/bash
set -e

# ONNX Runtime version
ORT_VERSION=1.16.3
# C++ SDK package (Linux x64)
ORT_TAR=onnxruntime-linux-x64-${ORT_VERSION}.tgz
ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TAR}
DEST_DIR=onnxruntime-linux-x64-${ORT_VERSION}

# Check if already exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Downloading ONNX Runtime C++ SDK..."
    # Try wget first, fallback to curl if wget is not available
    if command -v wget >/dev/null 2>&1; then
        wget $ORT_URL -O $ORT_TAR
    elif command -v curl >/dev/null 2>&1; then
        curl -L $ORT_URL -o $ORT_TAR
    else
        echo "Error: Neither wget nor curl is installed."
        exit 1
    fi

    echo "Extracting..."
    tar -xzf $ORT_TAR
    echo "Done."
else
    echo "ONNX Runtime C++ SDK already exists."
fi
