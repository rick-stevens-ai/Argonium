#!/bin/bash

# Usage: ./validate_pdf.sh /path/to/file.pdf

FILE="$1"
GOOD_DIR="GOOD"
BAD_DIR="BAD"

mkdir -p "$GOOD_DIR" "$BAD_DIR"

if qpdf --check "$FILE" &> /dev/null; then
    echo "Valid PDF: $FILE"
    mv "$FILE" "$GOOD_DIR/"
else
    echo "Invalid PDF: $FILE"
    mv "$FILE" "$BAD_DIR/"
fi
