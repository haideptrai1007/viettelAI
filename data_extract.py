import fitz  # PyMuPDF
import tempfile
import os
from unsloth import FastVisionModel
import torch
from transformers import AutoModel
import re
import argparse

def index_img(filePath):
    with open(filePath, "r+", encoding="utf-8") as f:
        content = f.read()
        
        # Counter for images
        counter = 1
        def replace_image(match):
            nonlocal counter
            replacement = f"<|image_{counter}|>"
            counter += 1
            return replacement

        # Replace all image references
        new_content = re.sub(r'!\[\]\(images/\d+\.jpg\)', replace_image, content)

        # Go to the beginning and overwrite
        f.seek(0)
        f.write(new_content)
        f.truncate()

def handle_pdf(filePath, outPath, model, tokenizer):
    prompt = """
    <image>\n<|grounding|>Convert the document to markdown, and you must ignore the watermark, and use placeholder <|image|> for Image in document.
    """

    imgs = []
    markdownPath = os.path.join(outPath, "main.md")
    with tempfile.TemporaryDirectory() as path:
        doc = fitz.open(filePath)
        for page_number in range(len(doc)):
            page = doc.load_page(page_number) 
            pix = page.get_pixmap(dpi=100)    
            tempPath = os.path.join(path + f"page_{page_number}.png")
            pix.save(tempPath)
            imgs.append(tempPath)

            img_idx = 1
            image_list = page.get_images(full=True)
            for img in image_list[1:]:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"image{img_idx}.{image_ext}"
                image_path = os.path.join(outPath, "images", image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                imgs.append(image_path)
        
        doc.close()

        for img in imgs:
            resultPath = os.path.join(path + "/result.mmd")
            with torch.inference_mode():
                _ = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img,
                    output_path=path,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=True,
                    test_compress=True
                )

            
            with open(resultPath, "r", encoding="utf-8") as rf:
                result_content = rf.read()

            with open(markdownPath, "a", encoding="utf-8") as mf:
                mf.write("\n") 
                mf.write(result_content)
            
            # Delete File
            os.remove(resultPath)
    
    index_img(markdownPath)


# Main                        
def main():
    parser = argparse.ArgumentParser(description="Extract PDF names and create output files.")
    parser.add_argument('--input', required=True, help="Input folder containing PDFs")
    parser.add_argument('--output', required=False, help="Output folder path (default: create 'output' next to input folder)")
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output

    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
    from huggingface_hub import snapshot_download
    modelPath = snapshot_download("unsloth/DeepSeek-OCR", local_dir = "deepseek_ocr")

    model, tokenizer = FastVisionModel.from_pretrained(
        modelPath,
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        auto_model = AutoModel,
        trust_remote_code = True,
        unsloth_force_compile = True,
        use_gradient_checkpointing = "unsloth"
    )

    
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

            handle_pdf(pdfFiles, folder_path, model, tokenizer)

if __name__ == "__main__":
    main()
