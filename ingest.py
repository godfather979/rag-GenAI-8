import os
import json
import hashlib
import pymupdf
import pytesseract
import faiss

from PIL import Image
from io import BytesIO

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


# =====================================================
# CONFIGURATION
# =====================================================

PDF_DIR = "data/institutional_pdfs"
PAGES_DIR = "data/website_pages"
OUTPUT_DIR = "vector_store"

MAX_DOCUMENTS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# EMBEDDINGS AND CHUNKING
# =====================================================

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.node_parser = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=75
)

# =====================================================
# METADATA
# =====================================================

def load_metadata(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


pdf_metadata = load_metadata("metadata/pdf_metadata.json")
page_metadata = load_metadata("metadata/pages_metadata.json")

metadata_map = {}

for item in pdf_metadata + page_metadata:
    local_path = item.get("local_path", "")
    if local_path:
        metadata_map[os.path.basename(local_path)] = item


# =====================================================
# PDF TEXT EXTRACTION + OCR FALLBACK
# =====================================================

def extract_pdf(path):
    extracted_pages = []

    with pymupdf.open(path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # OCR only when native text is absent
            if not text:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(3, 3),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                ).strip()

            if text:
                extracted_pages.append({
                    "page": page_num,
                    "text": text
                })

    return extracted_pages


# =====================================================
# LOAD FIRST 100 DOCUMENTS
# =====================================================

documents = []
processed = 0

# PDFs first, followed by scraped webpage text
pdf_files = sorted(
    f for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
)

page_files = sorted(
    f for f in os.listdir(PAGES_DIR)
    if f.lower().endswith(".txt")
)

for filename in pdf_files:
    if processed >= MAX_DOCUMENTS:
        break

    path = os.path.join(PDF_DIR, filename)

    try:
        pages = extract_pdf(path)
        source = metadata_map.get(filename, {})

        for item in pages:
            documents.append(
                Document(
                    text=item["text"],
                    metadata={
                        "source": filename,
                        "source_url": source.get("url", ""),
                        "source_title": source.get("title", filename),
                        "source_type": "pdf",
                        "page": item["page"]
                    }
                )
            )

        processed += 1
        print(f"[{processed}/{MAX_DOCUMENTS}] PDF: {filename}")

    except Exception as e:
        print(f"PDF ERROR: {filename}: {e}")


for filename in page_files:
    if processed >= MAX_DOCUMENTS:
        break

    path = os.path.join(PAGES_DIR, filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        source = metadata_map.get(filename, {})

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": filename,
                    "source_url": source.get("url", ""),
                    "source_title": source.get("title", filename),
                    "source_type": "html"
                }
            )
        )

        processed += 1
        print(f"[{processed}/{MAX_DOCUMENTS}] PAGE: {filename}")

    except Exception as e:
        print(f"PAGE ERROR: {filename}: {e}")


print(f"\nDocuments processed: {processed}")
print(f"Text-bearing documents: {len(documents)}")

if not documents:
    raise RuntimeError("No text extracted. Cannot build the index.")


# =====================================================
# BUILD FAISS INDEX
# =====================================================

dimension = len(
    Settings.embed_model.get_text_embedding("test")
)

# Inner-product search with normalized vectors
faiss_index = faiss.IndexFlatIP(dimension)

vector_store = FaissVectorStore(
    faiss_index=faiss_index
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# =====================================================
# SAVE INDEX
# =====================================================

index.storage_context.persist(
    persist_dir=OUTPUT_DIR
)

print("\nIngestion complete!")
print(f"Documents processed: {processed}")
print(f"Chunks indexed: {faiss_index.ntotal}")
print(f"Index saved to: {OUTPUT_DIR}")