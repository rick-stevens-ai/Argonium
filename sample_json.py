#!/usr/bin/env python3

import sys
import json
import random

def main():
    if len(sys.argv) < 3:
        print("Usage: python sample_json.py <input_file> <n_samples>")
        sys.exit(1)

    input_file = sys.argv[1]
    n_samples = int(sys.argv[2])

    # Read the JSON data from the file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If the requested number of samples is larger than the data size,
    # sample the entire data set.
    n_samples = min(n_samples, len(data))

    # Randomly sample the data
    sampled_records = random.sample(data, n_samples)

    # Output the sampled list as JSON
    # "indent=2" is optional, but it makes the output more readable.
    print(json.dumps(sampled_records, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
