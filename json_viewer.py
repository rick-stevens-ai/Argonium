#!/usr/bin/env python3
"""
Simple JSON Viewer for multiple-choice questions
Generates an HTML file from a JSON file containing questions and answers
"""

import json
import sys
from pathlib import Path


def generate_html(json_file):
    """Generate an HTML viewer from a JSON file"""

    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the base name for the output file
    output_file = Path(json_file).stem + "_viewer.html"

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Viewer: {Path(json_file).name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .stats {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .question-card {{
            background-color: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        .question-number {{
            font-size: 18px;
            font-weight: bold;
            color: #3498db;
        }}
        .question-type {{
            background-color: #3498db;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .question-text {{
            font-size: 16px;
            margin-bottom: 20px;
            white-space: pre-wrap;
            line-height: 1.8;
        }}
        .answer-section {{
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .answer-label {{
            font-weight: bold;
            color: #2e7d32;
            margin-bottom: 8px;
        }}
        .answer-text {{
            color: #1b5e20;
            font-size: 15px;
        }}
        .reference-section {{
            background-color: #f9f9f9;
            border-left: 4px solid #9e9e9e;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .reference-label {{
            font-weight: bold;
            color: #616161;
            margin-bottom: 8px;
        }}
        .reference-text {{
            font-size: 13px;
            color: #424242;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            line-height: 1.5;
        }}
        .toggle-button {{
            background-color: #9e9e9e;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            margin-top: 10px;
        }}
        .toggle-button:hover {{
            background-color: #757575;
        }}
        .hidden {{
            display: none;
        }}
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .nav-button {{
            display: block;
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .nav-button:hover {{
            background-color: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Question Viewer</h1>
        <div class="stats">
            <strong>Source:</strong> {Path(json_file).name}<br>
            <strong>Total Questions:</strong> {len(data)}
        </div>
    </div>
"""

    # Add each question
    for idx, item in enumerate(data, 1):
        question = item.get('question', 'No question text')
        answer = item.get('answer', 'No answer provided')
        text = item.get('text', 'No reference text')
        q_type = item.get('type', 'unknown')

        html_content += f"""
    <div class="question-card" id="q{idx}">
        <div class="question-header">
            <span class="question-number">Question {idx}</span>
            <span class="question-type">{q_type}</span>
        </div>

        <div class="question-text">{question}</div>

        <div class="answer-section">
            <div class="answer-label">✓ Correct Answer:</div>
            <div class="answer-text">{answer}</div>
        </div>

        <div class="reference-section">
            <div class="reference-label">📚 Reference Text:</div>
            <button class="toggle-button" onclick="toggleReference({idx})">
                Show/Hide Reference
            </button>
            <div id="ref{idx}" class="reference-text hidden">{text}</div>
        </div>
    </div>
"""

    # Add navigation and closing tags
    html_content += """
    <script>
        function toggleReference(num) {
            const ref = document.getElementById('ref' + num);
            ref.classList.toggle('hidden');
        }

        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Add scroll to top button when scrolling
        window.addEventListener('scroll', function() {
            const btn = document.getElementById('topBtn');
            if (btn && window.pageYOffset > 300) {
                btn.style.display = 'block';
            } else if (btn) {
                btn.style.display = 'none';
            }
        });
    </script>

    <button id="topBtn" class="nav-button" onclick="scrollToTop()"
            style="position: fixed; bottom: 20px; right: 20px; display: none; z-index: 999;">
        ↑ Top
    </button>
</body>
</html>
"""

    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML viewer generated: {output_file}")
    print(f"Total questions: {len(data)}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_viewer.py <json_file>")
        print("\nExample: python json_viewer.py DLB-MC.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)

    output = generate_html(json_file)
    print(f"\nOpen {output} in your browser to view the questions")
