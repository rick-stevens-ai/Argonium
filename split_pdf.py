import sys
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_pdf, num_parts):
    # Open the input PDF file
    with open(input_pdf, 'rb') as file:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        # Calculate pages per part (assuming equal distribution)
        pages_per_part = total_pages // num_parts
        
        # Initialize a writer for each part
        writers = [PdfWriter() for _ in range(num_parts)]
        
        # Determine the base name and extension of the input file
        base, ext = input_pdf.rsplit('.', 1)
        file_number = 1
        
        # Distribute pages into parts
        for page_num, page in enumerate(reader.pages):
            # Add the page to the appropriate writer
            writers[page_num // pages_per_part].add_page(page)
            
            # If we've reached the end of a part, write it out
            if (page_num + 1) % pages_per_part == 0:
                writers[page_num // pages_per_part].write(f"{base}_part_{file_number}.{ext}")
                file_number += 1

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python split_pdf.py <input_pdf> <num_parts>")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    num_parts = int(sys.argv[2])
    
    split_pdf(input_pdf, num_parts)
