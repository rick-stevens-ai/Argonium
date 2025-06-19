import json
import random

def randomize_json_rows(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Ensure the data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("JSON file does not contain a list of objects.")

    # Shuffle the list
    random.shuffle(data)

    # Write the randomized data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
input_file = "input.json"  # Replace with your input file name
output_file = "output.json"  # Replace with your desired output file name
randomize_json_rows(input_file, output_file)

print(f"Randomized rows saved to {output_file}.")

