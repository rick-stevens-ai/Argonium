#!/bin/bash

# Script to create symbolic links for HR-*-GOOD-RT.json files
# Links are created in ~/Dropbox/for-Ozan directory

SOURCE_DIR="/Users/stevens/Dropbox/SS-new/LUCID"
TARGET_DIR="$HOME/Dropbox/for-Ozan"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Change to source directory
cd "$SOURCE_DIR" || exit 1

echo "Creating links for HR-*-GOOD-RT.json files..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "---"

# Counter for files processed
count=0

# Create links for all HR-GOOD-*-RT.json files
for file in HR-GOOD-*-RT.json; do
    if [ -f "$file" ]; then
        echo "Linking: $file"
        ln -sf "$SOURCE_DIR/$file" "$TARGET_DIR/"
        ((count++))
    fi
done

echo "---"
echo "Created $count symbolic links in $TARGET_DIR"

# List the created links
echo
echo "Current links in target directory:"
ls -la "$TARGET_DIR"/*.json 2>/dev/null | grep "RT.json"
