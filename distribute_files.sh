#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory_name> <number_of_new_directories>"
    exit 1
fi

SOURCE_DIR="$1"
NUM_DIRS="$2"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' does not exist"
    exit 1
fi

if ! [[ "$NUM_DIRS" =~ ^[0-9]+$ ]] || [ "$NUM_DIRS" -le 0 ]; then
    echo "Error: Number of directories must be a positive integer"
    exit 1
fi

echo "Creating $NUM_DIRS new directories..."
for ((i=1; i<=NUM_DIRS; i++)); do
    NEW_DIR="${SOURCE_DIR}-${i}"
    mkdir -p "$NEW_DIR"
    echo "Created directory: $NEW_DIR"
done

echo "Collecting files from $SOURCE_DIR..."
FILES=()
while IFS= read -r -d '' file; do
    FILES+=("$file")
done < <(find "$SOURCE_DIR" -type f -print0)

TOTAL_FILES=${#FILES[@]}
echo "Found $TOTAL_FILES files to distribute"

if [ $TOTAL_FILES -eq 0 ]; then
    echo "No files found in $SOURCE_DIR"
    exit 0
fi

echo "Randomly distributing files..."
for file in "${FILES[@]}"; do
    RANDOM_DIR_NUM=$((RANDOM % NUM_DIRS + 1))
    TARGET_DIR="${SOURCE_DIR}-${RANDOM_DIR_NUM}"
    
    RELATIVE_PATH="${file#$SOURCE_DIR/}"
    TARGET_PATH="$TARGET_DIR/$RELATIVE_PATH"
    TARGET_PARENT_DIR=$(dirname "$TARGET_PATH")
    
    mkdir -p "$TARGET_PARENT_DIR"
    
    cp "$file" "$TARGET_PATH"
    echo "Copied: $file -> $TARGET_PATH"
done

echo "Distribution complete!"
echo "Files distributed across $NUM_DIRS directories:"
for ((i=1; i<=NUM_DIRS; i++)); do
    TARGET_DIR="${SOURCE_DIR}-${i}"
    FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
    echo "  $TARGET_DIR: $FILE_COUNT files"
done