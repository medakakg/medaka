# Import necessary libraries
import re
import requests
import os
import argparse


def extract_unique_pdfs(rtf_path, output_folder):
    """
    Extracts unique PDF file names from the given RTF file, downloads each PDF 
    from the HPRA website, and saves them to the specified output folder.

    Parameters:
    - rtf_path (str): Path to the input RTF file containing the HTML source.
    - output_folder (str): Path to the folder where the PDFs should be saved.
    """
    base_url = "https://www.hpra.ie/img/uploaded/swedocuments/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(rtf_path, 'r', encoding='utf-8') as file:                 # Open the RTF file
        content = file.read()                                           # Read the content      
    pdf_files = re.findall(r'([a-zA-Z0-9\-_]+\.pdf)', content)          # Extract PDF file names
    unique_pdf_files = set(pdf_files)  # Remove duplicates
    print(unique_pdf_files)
    for pdf_name in unique_pdf_files:
        pdf_url = base_url + pdf_name                                   # Construct the full URL for each PDF
        pdf_path = os.path.join(output_folder, pdf_name)
        try:
            response = requests.get(pdf_url)                            # Download the PDF
            response.raise_for_status()                                 
            with open(pdf_path, 'wb') as pdf_file:                      # Save the PDF   
                pdf_file.write(response.content)
            print(f"Downloaded and saved: {pdf_name}")
        except requests.RequestException as e:
            print(f"Failed to download {pdf_url}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and download PDFs from an RTF file.")
    parser.add_argument("--rtf_path", required=True, help="Path to the input RTF file")
    parser.add_argument("--output_folder", required=True, help="Directory to save downloaded PDFs")
    args = parser.parse_args()
    extract_unique_pdfs(args.rtf_path, args.output_folder)
