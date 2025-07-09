#!/usr/bin/env python3
"""
remove_literal_star.py
Remove the literal text "(*)" from every string in a JSON file.

Usage
-----
# overwrite the original file in-place
python remove_literal_star.py data.json

# write cleaned JSON to a new file
python remove_literal_star.py data.json -o cleaned.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

LITERAL = "(*)"


def strip_literal(obj: Any) -> Any:
    """Recursively remove the literal text '(*)' from strings inside obj."""
    if isinstance(obj, str):
        return obj.replace(LITERAL, "")
    if isinstance(obj, list):
        return [strip_literal(item) for item in obj]
    if isinstance(obj, dict):
        # Clean both keys and values
        return {strip_literal(k): strip_literal(v) for k, v in obj.items()}
    # Numbers, booleans, None, etc. are returned unchanged
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Remove the literal text "(*)" from a JSON file.'
    )
    parser.add_argument("input", help="Path to the input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path for the cleaned JSON (default: overwrite input file)",
        default=None,
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    cleaned = strip_literal(data)

    out_path = Path(args.output) if args.output else in_path
    out_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f'Clean JSON written to "{out_path}"')


if __name__ == "__main__":
    main()
