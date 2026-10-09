import os
import json
import fitz

DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
PDF_DIR = os.path.join(DATA_DIR, "institutional_pdfs")
PAGE_DIR = os.path.join(DATA_DIR, "website_pages")

os.makedirs(OUTPUT_DIR, exist_ok=True)

documents = []

# Extract PDF text page by page
for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(PDF_DIR, filename)

    try:
        pdf = fitz.open(path)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "source": filename,
                    "source_type": "pdf",
                    "page": page_num,
                    "text": text
                })

        pdf.close()

    except Exception as error:
        print(f"PDF ERROR: {filename}: {error}")

# Load scraped webpage text
for filename in os.listdir(PAGE_DIR):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(PAGE_DIR, filename)

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    documents.append({
        "source": filename,
        "source_type": "html",
        "page": None,
        "text": content
    })

# Save extracted documents
output_path = os.path.join(OUTPUT_DIR, "documents.jsonl")

with open(output_path, "w", encoding="utf-8") as file:
    for document in documents:
        file.write(json.dumps(document, ensure_ascii=False) + "\n")

print(f"Documents extracted: {len(documents)}")
print(f"Saved to: {output_path}")