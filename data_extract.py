import handle_pdf

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract PDF names and create output files.")
    parser.add_argument('--input', required=True, help="Input folder containing PDFs")
    parser.add_argument('--output', required=False, help="Output folder path (default: create 'output' next to input folder)")
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output

    
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return
    
    # Create output folder
    output_folder = os.path.join(os.path.dirname(input_folder), "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through PDF files
    pdfFiles = os.listdir(input_folder)
    for filename in pdfFiles:
        if filename.lower().endswith(".pdf"):
            name = os.path.splitext(filename)[0]
            folder_name = f"{name}"
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            handle_pdf(pdfFiles, folder_path)

if __name__ == "__main__":
    main()
