#!/usr/bin/env python3

"""
Example script to:
1. Parse a JSON file containing Q&A pairs.
2. Generate a nicely formatted Markdown (MD) file.
3. Convert each text field into wrapped paragraphs (no code blocks).

Usage:
    python render_json_to_md.py input.json output.md
"""

import json
import sys
import textwrap

def paragraphify(text: str, width: int = 80) -> str:
    """
    Convert a raw string into wrapped paragraphs of a given line width.
    - We strip leading/trailing whitespace.
    - We fill each paragraph (split by blank lines) with 'width' characters max.
    """
    # Split on blank lines to preserve paragraph structure.
    paragraphs = text.strip().split("\n\n")

    wrapped_paragraphs = []
    for para in paragraphs:
        # textwrap.fill will wrap each paragraph at the given width.
        wrapped_paragraph = textwrap.fill(para.strip(), width=width)
        wrapped_paragraphs.append(wrapped_paragraph)

    # Join paragraphs with a blank line in between
    return "\n\n".join(wrapped_paragraphs)

def create_markdown_content(data: dict, wrap_width: int = 80) -> str:
    """
    Given parsed JSON data (which contains QA pairs and their details),
    return a string containing Markdown-formatted text with paragraph-wrapped lines.
    """
    md_lines = []

    # We'll assume top-level structure is data["files"] → each file → "qa_pairs"
    files_data = data.get("files", {})

    for filename, file_content in files_data.items():
        md_lines.append(f"# Q&A Pairs for {filename}\n")

        qa_pairs = file_content.get("qa_pairs", [])
        chunks_processed = file_content.get("chunks_processed", "N/A")

        md_lines.append(f"**Chunks processed**: {chunks_processed}")
        md_lines.append("---\n")

        for idx, qa in enumerate(qa_pairs, start=1):
            question         = qa.get("question", "")
            original_thought = qa.get("original_thought", "")
            original_answer  = qa.get("original_answer", "")
            analysis         = qa.get("analysis", "")
            updated_thought  = qa.get("updated_thought", "")
            updated_answer   = qa.get("updated_answer", "")
            final_check      = qa.get("final_check", "")

            md_lines.append(f"## Q&A Pair {idx}\n")

            # Question
            md_lines.append("**Question**:\n")
            md_lines.append(paragraphify(question, width=wrap_width))
            md_lines.append("")

            # Original Thought
            md_lines.append("**Original Thought**:\n")
            md_lines.append(paragraphify(original_thought, width=wrap_width))
            md_lines.append("")

            # Original Answer
            md_lines.append("**Original Answer**:\n")
            md_lines.append(paragraphify(original_answer, width=wrap_width))
            md_lines.append("")

            # Analysis
            md_lines.append("**Analysis**:\n")
            md_lines.append(paragraphify(analysis, width=wrap_width))
            md_lines.append("")

            # Updated Thought
            md_lines.append("**Updated Thought**:\n")
            md_lines.append(paragraphify(updated_thought, width=wrap_width))
            md_lines.append("")

            # Updated Answer
            md_lines.append("**Updated Answer**:\n")
            md_lines.append(paragraphify(updated_answer, width=wrap_width))
            md_lines.append("")

            # Final Check
            md_lines.append("**Final Check**:\n")
            md_lines.append(paragraphify(final_check, width=wrap_width))
            md_lines.append("")

            # Separator line
            md_lines.append("---\n")

    # Join all lines into a single Markdown string
    return "\n".join(md_lines)

def main():
    if len(sys.argv) < 3:
        print("Usage: python render_json_to_md.py input.json output.md")
        sys.exit(1)

    input_json = sys.argv[1]
    output_md  = sys.argv[2]

    # 1. Read the input JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Create the Markdown text (set desired wrap width as needed)
    md_content = create_markdown_content(data, wrap_width=80)

    # 3. Write the Markdown content to the output file
    with open(output_md, 'w', encoding='utf-8') as md_file:
        md_file.write(md_content)

    print(f"Markdown file successfully created: {output_md}")


if __name__ == "__main__":
    main()
