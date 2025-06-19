#!/usr/bin/env python3

import json
import sys

def main():
    # Collect file paths from the command line
    input_files = sys.argv[1:]
    
    if not input_files:
        print("Usage: merge_json.py file1.json file2.json ...", file=sys.stderr)
        sys.exit(1)

    all_data = []

    # Read and combine data from each file
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {file_path} does not contain a JSON list. Skipping.", file=sys.stderr)
        except FileNotFoundError:
            print(f"Error: {file_path} not found.", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not valid JSON.", file=sys.stderr)

    # Output merged data to stdout
    print(json.dumps(all_data, indent=4))

if __name__ == "__main__":
    main()
