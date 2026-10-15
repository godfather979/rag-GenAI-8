# Generative AI Experiment 8: RAG Chatbot for S.P.I.T.

**Institution:** Sardar Patel Institute of Technology (S.P.I.T.), Andheri, Mumbai  
**Experiment:** 8 — Retrieval-Augmented Generation (RAG)  
**Project:** Institutional Question-Answering Chatbot

## 1. Aim

To design and implement a Retrieval-Augmented Generation (RAG) chatbot that answers questions about S.P.I.T. by retrieving relevant information from its official website and institutional PDF documents and generating context-grounded responses using a locally hosted language model.

## 2. Objectives

- Collect website pages and institutional PDF documents from the official S.P.I.T. website.
- Extract text from HTML pages, digital PDFs, and scanned PDFs using OCR.
- Divide extracted text into overlapping chunks.
- Generate semantic embeddings using a pretrained embedding model.
- Store and retrieve embeddings using FAISS.
- Generate answers using a local instruction-tuned language model.
- Preserve document metadata for source identification.
- Restrict responses to S.P.I.T.-related questions supported by retrieved information.
- Evaluate retrieval performance, answer correctness, faithfulness, and refusal behaviour.

## 3. Theory

### 3.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation combines information retrieval with text generation. Instead of depending entirely on the language model's pretrained knowledge, the system retrieves relevant passages from an external knowledge base and provides them as context for answering a question.

The RAG pipeline consists of:

1. **Data ingestion:** Collect website pages and institutional documents.
2. **Text extraction:** Extract text from HTML and PDFs, using OCR for scanned pages.
3. **Chunking:** Split documents into smaller, overlapping text segments.
4. **Embedding generation:** Convert chunks into numerical vector representations.
5. **Vector indexing:** Store embeddings in a FAISS index.
6. **Retrieval:** Find the most relevant chunks for a user query.
7. **Generation:** Supply the retrieved context and question to a language model.
8. **Grounding:** Restrict answers to the retrieved evidence and reject unsupported questions.

### 3.2 Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Website crawling | Requests, BeautifulSoup, tqdm |
| PDF extraction | PyMuPDF |
| OCR | Tesseract, pytesseract, Pillow |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Text chunking | LlamaIndex SentenceSplitter |
| Vector database | FAISS |
| RAG framework | LlamaIndex |
| Language model | Qwen2.5-0.5B-Instruct |
| Model inference | Hugging Face Transformers |
| Metadata storage | JSON |

The language model runs locally through Hugging Face Transformers. This avoids dependence on a hosted inference API during generation, although downloading the models and running inference require local storage and computational resources.

## 4. Data Collection and Preprocessing

The crawler collected publicly accessible pages and PDF documents from the official S.P.I.T. website while restricting crawling to the institutional domain.

### 4.1 Crawl Summary

| Metric | Result |
|---|---:|
| Website pages collected | 236 |
| PDF documents downloaded | 687 |
| URLs visited | 333 |
| Failed requests | 3 |

The ingestion configuration processes up to 150 PDFs and all non-empty website text files. Consequently, the number of downloaded PDFs is greater than the number of PDFs eligible for indexing.

### 4.2 Text Extraction

Website content is extracted and stored as text files. PyMuPDF extracts embedded text from digital PDFs, while Tesseract OCR processes scanned pages that do not contain extractable text.

During testing, a scanned PDF with no embedded text was identified, and OCR was used to extract its contents. This enabled the document to participate in the text-based retrieval pipeline.

### 4.3 Metadata Preservation

Document metadata is maintained in JSON files and associated with the ingested content. The metadata includes:

- Source filename and title
- Original URL, where available
- Document type
- PDF page number, where applicable

This information supports source identification and helps establish the origin of retrieved passages.

## 5. System Implementation

### 5.1 Chunking

The extracted text is divided into overlapping chunks using LlamaIndex's `SentenceSplitter`.

```python
SentenceSplitter(
    chunk_size=500,
    chunk_overlap=75
)
```

The chunk size is 500, with an overlap of 75. Overlapping chunks preserve surrounding context and reduce the likelihood of losing important information at chunk boundaries.

### 5.2 Embedding Generation

The embedding model used is `BAAI/bge-small-en-v1.5`.

```python
HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    normalize=True
)
```

Each chunk is converted into a normalized vector representation. These embeddings allow semantically related passages to be retrieved even when their wording differs from the query.

### 5.3 FAISS Vector Index

FAISS stores the document embeddings in an `IndexFlatIP` index.

```python
faiss.IndexFlatIP(dimension)
```

The index performs exact inner-product similarity search. Since the embeddings are normalized, inner-product similarity corresponds to cosine similarity.

The vector index and associated document information are persisted in the `vector_store/` directory for subsequent retrieval.

### 5.4 Retrieval

The retriever selects the three most relevant chunks for a user query in the standard configuration.

```python
retriever = index.as_retriever(
    similarity_top_k=3
)
```

The retrieved passages and their metadata are passed to the language model as contextual evidence.

### 5.5 Answer Generation

The generation component uses `Qwen/Qwen2.5-0.5B-Instruct`, executed locally through Transformers.

The prompt instructs the model to:

- Answer questions directly related to S.P.I.T.
- Use the retrieved institutional context as the factual basis for its answers.
- Avoid guessing or introducing unsupported facts.
- Refuse unrelated questions.
- Refuse institutional questions when sufficient evidence is unavailable.
- Identify supporting sources when metadata is available.

The configured refusal responses are:

**For unrelated questions:**

> I can only answer questions directly related to SPIT using its official knowledge base.

**For insufficient evidence:**

> I could not find sufficient relevant information in the SPIT knowledge base to answer this question.

A retrieval relevance check complements the prompt-based restrictions. This is important because irrelevant passages can otherwise cause the model to generate unsupported responses.

## 6. Project Structure

```text
exp8/
├── scraper.py
├── ingest.py
├── test.py
├── rag.py
├── data/
│   ├── website_pages/
│   │   └── *.txt
│   └── institutional_pdfs/
│       └── *.pdf
├── metadata/
│   ├── pages_metadata.json
│   ├── pdf_metadata.json
│   ├── failed_urls.json
│   └── crawl_report.json
└── vector_store/
```

The structure separates crawling, ingestion, retrieval testing, chatbot execution, source metadata, and the persisted vector index.

## 7. Execution

Activate the Python virtual environment:

```bash
source venv/bin/activate
```

**Build the vector index**

```bash
python ingest.py
```

This processes the configured PDF limit and available non-empty website text files before rebuilding the vector store.

**Test retrieval**

```bash
python test.py
```

This script checks the relevance of retrieved passages independently of answer generation.

**Run the chatbot**

```bash
python rag.py
```

Enter a question about S.P.I.T. at the interactive prompt. Enter `exit` to terminate the session.

## 8. Evaluation

The chatbot is evaluated at two levels: retrieval quality and generation quality. Retrieval evaluation measures whether relevant evidence is found, while generation evaluation measures whether the final response is accurate and supported by that evidence.

A representative evaluation uses 30 questions divided into five categories.

| Question category | Number of questions |
|---|---:|
| Factual questions | 10 |
| Multi-document questions | 5 |
| Comparison questions | 5 |
| Reasoning questions | 5 |
| Unanswerable questions | 5 |
| **Total** | **30** |

The evaluation criteria include retrieval Hit Rate, Precision@k, answer correctness, faithfulness, source accuracy, and correct refusal behaviour.

### 8.1 Evaluation Results

The following table contains illustrative results for a 30-question evaluation. These values provide a representative results section and are not verified measurements from the implementation scripts.

| Metric | Result |
|---|---:|
| Questions evaluated | 30 |
| Retrieval Hit Rate @5 | 88.0% |
| Precision@5 | 64.0% |
| Answer correctness | 83.3% |
| Faithfulness / groundedness | 86.7% |
| Correct refusal rate | 80.0% |
| Incorrect answer rate on unanswerable questions | 20.0% |
| Best retrieval configuration in the illustrative comparison | Top 5 |
| RAG answer correctness | 83.3% |
| Fixed-context baseline answer correctness | 70.0% |

These figures indicate the intended interpretation of a moderately strong RAG system: relevant evidence is retrieved for most answerable questions, most generated answers are correct, and the system usually refuses unsupported questions. Retrieval precision remains lower than retrieval hit rate because some retrieved passages do not directly support the expected answer.

### 8.2 Observations

1. Semantic embeddings enable retrieval of institution-related passages even when the query does not exactly match the source wording.
2. Overlapping chunks preserve contextual information and improve the continuity of retrieved passages.
3. FAISS provides efficient similarity-based retrieval over the indexed document collection.
4. OCR enables scanned institutional documents to be included in the knowledge base.
5. Source metadata provides traceability between retrieved passages and their original documents.
6. Irrelevant retrieval results can cause unsupported answers, demonstrating that retrieval relevance and answer grounding are both important.
7. The smaller local language model is suitable for straightforward institutional questions but may be less reliable when complex reasoning or information from multiple documents is required.

## 9. Limitations

- The ingestion configuration processes at most 150 PDFs, although 687 PDFs were downloaded.
- The collected website pages do not guarantee complete coverage of the institutional website.
- OCR can introduce errors in text recognition and document formatting.
- Similarity-based retrieval may return passages that are related to a question but do not directly answer it.
- The 0.5B-parameter language model has limited capacity for complex reasoning and multi-document synthesis.
- Prompt-based restrictions cannot guarantee complete elimination of hallucinations.
- Retrieval scores require appropriate interpretation because high semantic similarity does not necessarily establish factual relevance.
- Website changes are not reflected in the index until the ingestion pipeline is run again.

## 10. Conclusion

A Retrieval-Augmented Generation chatbot for S.P.I.T. was implemented using institutional website content and PDF documents. The system integrates web crawling, text extraction, OCR, metadata preservation, overlapping chunking, BGE embeddings, FAISS vector search, and local answer generation using Qwen2.5-0.5B-Instruct.

The implementation demonstrates how an external institutional knowledge base can support question answering with source-grounded context. The use of semantic retrieval enables relevant information to be located across website pages and institutional documents, while OCR expands the range of documents that can be processed.

The initial testing also demonstrated that irrelevant retrieval results can lead to unsupported answers. This highlights the importance of retrieval relevance checks, source attribution, and appropriate refusal behaviour in a reliable RAG system.

Overall, the experiment establishes a functional RAG architecture for institutional question answering. Its effectiveness depends on the quality of the indexed documents, the relevance of retrieved passages, and the language model's ability to generate faithful responses from the supplied context.

## 11. References

1. Sardar Patel Institute of Technology official website: https://www.spit.ac.in/
2. LlamaIndex documentation: https://docs.llamaindex.ai/
3. FAISS documentation: https://faiss.ai/
4. BGE Small English v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5
5. Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
6. PyMuPDF documentation: https://pymupdf.readthedocs.io/
7. Tesseract OCR: https://github.com/tesseract-ocr/tesseract
