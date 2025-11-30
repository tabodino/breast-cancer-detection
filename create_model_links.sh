#!/bin/bash

# Script to create symbolic links for best models
# Usage: ./create_model_links.sh

set -e  # Exit on error

echo "Creating symbolic links for best models..."
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "   Error: models/ directory not found"
    echo "   Please run this script from the project root"
    exit 1
fi

cd models/

# Array of model names
models=("efficientnet_b3" "resnet50" "mobilenet_v3" "unet" "cnn")

success_count=0
fail_count=0

for model in "${models[@]}"; do
    echo "Processing: $model"
    
    # Find the best model file with run_id (most recent)
    best_file=$(ls -t best_model_${model}_*.keras 2>/dev/null | head -1)
    
    if [ -n "$best_file" ]; then
        link_name="best_${model}.keras"
        
        # Remove old symlink if exists
        if [ -L "$link_name" ] || [ -e "$link_name" ]; then
            rm -f "$link_name"
            echo "  Removed old: $link_name"
        fi
        
        # Create new symbolic link
        ln -s "$best_file" "$link_name"
        
        # Verify the link works
        if [ -L "$link_name" ] && [ -e "$link_name" ]; then
            # Get file size
            size=$(stat -f%z "$best_file" 2>/dev/null || stat -c%s "$best_file" 2>/dev/null)
            size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc)
            
            echo "  Created: $link_name → $best_file ($size_mb MB)"
            ((success_count++))
        else
            echo "  Link created but verification failed: $link_name"
            ((fail_count++))
        fi
    else
        echo "  No model found matching: best_model_${model}_*.keras"
        ((fail_count++))
    fi
    
    echo ""
done