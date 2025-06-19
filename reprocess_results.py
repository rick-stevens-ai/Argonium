#!/usr/bin/env python3

"""
Reprocess existing benchmark results to extract accuracy and confidence correctly
"""

import json
import sys
from run_all_models import extract_accuracy_from_output, extract_confidence_from_output, print_summary_table

def reprocess_results(filename):
    """Reprocess an existing results file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Update summary data with correct parsing
    for i, result in enumerate(data['detailed_results']):
        if result['success']:
            accuracy = extract_accuracy_from_output(result['stdout'])
            confidence = extract_confidence_from_output(result['stdout'])
            
            # Update the summary
            data['summary'][i]['accuracy_percent'] = accuracy
            data['summary'][i]['average_confidence'] = confidence
    
    # Re-sort by accuracy
    data['summary'].sort(key=lambda x: (x['success'], x['accuracy_percent'] or -1), reverse=True)
    
    # Save updated file
    updated_filename = filename.replace('.json', '_updated.json')
    with open(updated_filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated results saved to: {updated_filename}")
    
    # Print the summary table
    print_summary_table(data['summary'])
    
    return data['summary']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reprocess_results.py <results_file.json>")
        sys.exit(1)
    
    reprocess_results(sys.argv[1])