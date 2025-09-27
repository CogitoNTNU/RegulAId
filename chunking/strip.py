import PyPDF2
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from PIL import Image
import tempfile
import os

def clean_aiact_pdf(input_pdf_path, output_pdf_path):
    """
    Clean the AIACT PDF by removing unnecessary headers, footers, and initial content
    while preserving page numbers and overall page structure.
    """
    
    # Read the original PDF
    with open(input_pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_writer = PyPDF2.PdfWriter()
        
        # Process each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            
            # Extract text from the page
            text = page.extract_text()
            
            # For the first page, we need special handling
            if page_num == 0:
                # Find where the "Whereas:" section starts
                whereas_match = re.search(r'Whereas:', text)
                if whereas_match:
                    # Get the position of "Whereas:"
                    start_pos = whereas_match.start()
                    
                    # Extract only the text from "Whereas:" onward
                    cleaned_text = text[start_pos:]
                else:
                    # If "Whereas:" not found, use the entire text
                    cleaned_text = text
            else:
                # For other pages, we'll process to remove headers/footers
                cleaned_text = text
            
            # Remove headers and footers using regex patterns
            # Header pattern: "EN OJ L, 12.7.2024" and similar
            header_pattern = r'EN\s*OJ\s*L,\s*12\.7\.2024\s*'
            cleaned_text = re.sub(header_pattern, '', cleaned_text)
            
            # Footer pattern: ELI URL
            footer_pattern = r'ELI:\s*http://data\.europa\.eu/eli/reg/2024/1689/oj\s*'
            cleaned_text = re.sub(footer_pattern, '', cleaned_text)
            
            # Remove page number patterns like "2/144", "3/144", etc.
            # But be careful not to remove actual content numbers
            page_num_pattern = r'^\d+/\d+\s*$'  # Only matches lines with just page numbers
            lines = cleaned_text.split('\n')
            cleaned_lines = []
            for line in lines:
                if not re.match(page_num_pattern, line.strip()):
                    cleaned_lines.append(line)
            cleaned_text = '\n'.join(cleaned_lines)
            
            # Create a new page with the cleaned text
            # We'll use reportlab to create a new page that mimics the original structure
            
            # Create a temporary PDF with the cleaned text
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            
            # Set font and size (approximate the original)
            can.setFont("Helvetica", 10)
            
            # Add the cleaned text to the page
            # We'll use text object for better formatting control
            text_object = can.beginText(40, 750)  # Starting position
            
            # Split text into lines and add them
            lines = cleaned_text.split('\n')
            for line in lines:
                if line.strip():  # Only add non-empty lines
                    text_object.textLine(line.strip())
            
            can.drawText(text_object)
            
            # Add page number at the bottom (preserve original page numbering)
            can.setFont("Helvetica", 8)
            can.drawString(500, 30, f"{page_num + 1}/{len(pdf_reader.pages)}")
            
            can.save()
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            
            # Create a new PDF page from the temporary PDF
            temp_pdf = PyPDF2.PdfReader(packet)
            new_page = temp_pdf.pages[0]
            
            # Add the new page to the writer
            pdf_writer.add_page(new_page)
        
        # Write the cleaned PDF to file
        with open(output_pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)

def clean_aiact_pdf_alternative(input_pdf_path, output_pdf_path):
    """
    Alternative approach that preserves more of the original formatting
    by using page cropping and text extraction.
    """
    
    with open(input_pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_writer = PyPDF2.PdfWriter()
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            
            # For the first page, we need to crop out the content before "Whereas:"
            if page_num == 0:
                # Extract text to find the position of "Whereas:"
                text = page.extract_text()
                whereas_match = re.search(r'Whereas:', text)
                
                if whereas_match:
                    # Create a new page by cropping (this is a simplified approach)
                    # We'll lower the page to remove top content
                    page.cropbox.upper_left = (50, 700)  # Adjust these values as needed
                    page.cropbox.lower_right = (550, 50)
            
            # Add the (possibly modified) page to the writer
            pdf_writer.add_page(page)
        
        # Write the output PDF
        with open(output_pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)

def simple_text_extraction_cleaner(input_pdf_path, output_pdf_path):
    """
    A simpler approach that extracts text and recreates the PDF with clean formatting.
    This loses some formatting but provides clean text.
    """
    
    with open(input_pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Create a new PDF using reportlab
        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Clean the text
            if page_num == 0:
                # Remove content before "Whereas:"
                whereas_match = re.search(r'Whereas:', text)
                if whereas_match:
                    text = text[whereas_match.start():]
            
            # Remove headers and footers
            text = re.sub(r'EN\s*OJ\s*L,\s*12\.7\.2024\s*', '', text)
            text = re.sub(r'ELI:\s*http://data\.europa\.eu/eli/reg/2024/1689/oj\s*', '', text)
            
            # Start a new page (except for the first iteration)
            if page_num > 0:
                c.showPage()
            
            # Set font
            c.setFont("Helvetica", 10)
            
            # Add the cleaned text
            text_object = c.beginText(40, 750)
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip() and not re.match(r'^\d+/\d+\s*$', line.strip()):
                    if i > 0:
                        text_object.textLine("")  # Add line break
                    text_object.textLine(line.strip())
            
            c.drawText(text_object)
            
            # Add page number
            c.setFont("Helvetica", 8)
            c.drawString(500, 30, f"{page_num + 1}/{len(pdf_reader.pages)}")
        
        c.save()

# Main execution
if __name__ == "__main__":
    # 1. Get the absolute directory of the current script (paragraphs.py)
    # '.../RegulAId/chunking'
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one level (from 'chunking' to 'RegulAId')
    parent_dir = os.path.dirname(script_dir)

    # 3. Construct the full, absolute path to the target file
    # This joins '.../RegulAId' with 'data/raw/AIACT.md'
    file_path = os.path.join(parent_dir, 'data', 'raw', 'AIACT.pdf')
    
    input_pdf = file_path  # Replace with your actual file path

    # 3. Construct the full, absolute path to the target file
    # This joins '.../RegulAId' with 'data/processed/AIACT.md'
    output_pdf = os.path.join(parent_dir, 'data', 'processed', 'AIACT_clean.pdf')
    
    try:
        # Try the simple text extraction approach first
        simple_text_extraction_cleaner(input_pdf, output_pdf)
        print(f"Successfully created cleaned PDF: {output_pdf}")
    except Exception as e:
        print(f"Error with simple approach: {e}")
        print("Trying alternative approach...")
        
        try:
            clean_aiact_pdf(input_pdf, output_pdf)
            print(f"Successfully created cleaned PDF: {output_pdf}")
        except Exception as e2:
            print(f"Error with alternative approach: {e2}")
            print("Please check the input PDF path and try again.")