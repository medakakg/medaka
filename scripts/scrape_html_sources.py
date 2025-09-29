"""
This script scrapes HTML source code from the HPRA medicine search results pages
and saves the content into an RTF file.

- Iterates through pages 1 to 499 of the HPRA search results
- Retrieves each page's HTML using the requests library
- Parses the HTML with BeautifulSoup
- Saves the formatted HTML source into 'sourcecode.rtf' using RTF syntax
"""

# Import necessary libraries
from bs4 import BeautifulSoup
import requests

base_url = 'https://www.hpra.ie/homepage/medicines/medicines-information/find-a-medicine/results?page='

with open('sourcecode.rtf', 'w', encoding='utf-8') as file:                                # Output file to save the HTML source code
    for page_num in range(1, 500):
        url = f"{base_url}{page_num}"                                                           # Construct URL for the current page
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html')
        
        file.write(f"{{\\rtf1\\ansi\\deff0 {{\\b Page {page_num} HTML Source Code }}\\par\n")   # Add header of the page scraped
        file.write(str(soup).replace('\n', '\\par\n'))                                          # Replace newlines with RTF line breaks
        file.write("\\par\n\\par\n")                                                            
        print(f"HTML source code for page {page_num} appended to 'sourcecode.rtf'")

print("HTML source code for all pages saved in 'sourcecode.rtf'.")
