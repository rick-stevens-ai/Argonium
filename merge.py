import json

def merge_text(question, answer):
    """
    Merge the question and answer into a single coherent text passage.
    Modify this function as needed if you want to apply GPT-4 or
    further text processing. For now, we're just concatenating them.
    """
    # Example of a simple Q&A format
    merged = f"Question: {question}\nAnswer: {answer}"
    return merged

def main(input_file, output_file):
    # Read the input JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data_str = f.read().strip()
        # Because the file may have trailing commas or multiple objects,
        # we’ll wrap the objects in a list if needed
        # This attempt is to handle your sample structure of repeated { } blocks.
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            # If not valid JSON array, we try a trick:
            # 1. Remove trailing commas
            # 2. Wrap in brackets
            fixed_str = f"[{data_str}]"
            fixed_str = fixed_str.replace("},\n]", "}]")  # handle trailing commas
            data = json.loads(fixed_str)

    # data should now be a list of question/answer objects
    new_records = []
    for record in data:
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()
        merged_text = merge_text(question, answer)
        new_records.append({"text": merged_text})

    # Write out the new JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Adjust the file paths as needed
    input_file_path = "input.json"   # your input file with Q/A records
    output_file_path = "output.json" # your desired output file

    main(input_file_path, output_file_path)
