#!/bin/bash

# Usage: ./remove_small_files.sh <min_byte_size> [directory]
# Example: ./remove_small_files.sh 1000 ./data

# Get the minimum size in bytes and target directory
MIN_SIZE=$1
TARGET_DIR=${2:-.}  # Default to current directory if not specified

if [[ -z "$MIN_SIZE" || ! "$MIN_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <min_byte_size> [directory]"
  exit 1
fi

# Find and delete files smaller than MIN_SIZE bytes
find "$TARGET_DIR" -type f -size -"${MIN_SIZE}c" -print -delete
