# Generative AI Experiment 8
## Retrieval-Augmented Generation (RAG) Chatbot for S.P.I.T.

**Institution:** Sardar Patel Institute of Technology (S.P.I.T.), Andheri, Mumbai  
**Experiment:** 8 — Retrieval-Augmented Generation (RAG)  
**Project:** A chatbot that answers questions about S.P.I.T. using information collected from its official website and institutional PDFs.

---

## 1. Aim

To design and implement a Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant information from S.P.I.T.'s official website pages and institutional PDF documents, then uses a language model to generate answers grounded in the retrieved information.

## 2. Objectives

- Collect public website pages and PDF documents from `https://www.spit.ac.in/`.
- Extract text from HTML pages and PDFs, including scanned PDFs through OCR.
- Split extracted content into overlapping chunks.
- Convert chunks into vector embeddings.
- Store embeddings in a FAISS vector index.
- Retrieve relevant chunks for a user's question.
- Generate answers using a local instruction-tuned language model.
- Preserve source metadata so answers can be traced to their supporting documents.
- Reduce unsupported answers by restricting the chatbot to S.P.I.T.-related questions and the retrieved context.
- Evaluate retrieval and answer quality with answerable and unanswerable questions.

## 3. Theory

### 3.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation combines information retrieval with text generation. Instead of relying only on information stored in a language model's parameters, the system first searches an external knowledge base and supplies the retrieved passages to the model as context.

The pipeline is:

1. **Ingestion:** Collect source documents and extract text.
2. **Chunking:** Divide long documents into smaller overlapping text segments.
3. **Embedding:** Represent each chunk as a numerical vector.
4. **Indexing:** Store vectors in a searchable FAISS index.
5. **Retrieval:** Retrieve the most relevant chunks for a question.
6. **Generation:** Provide the question and retrieved context to the language model.
7. **Grounding:** Instruct the model to answer only from the context and refuse unsupported or unrelated questions.

### 3.2 Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Website collection | `requests`, `BeautifulSoup`, `tqdm` |
| PDF text extraction | PyMuPDF (`pymupdf`) |
| OCR for scanned PDFs | Tesseract OCR, `pytesseract`, Pillow |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Chunking | LlamaIndex `SentenceSplitter` |
| Vector search | FAISS (`IndexFlatIP`) |
| RAG framework | LlamaIndex |
| Local language model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Model inference | Hugging Face Transformers |
| Metadata storage | JSON |

The language model is run locally through Transformers rather than a hosted API, avoiding dependence on a service's API quota or rate limits. Initial model download and local inference still require sufficient disk space, RAM, and processing resources.

## 4. Data Collection

The crawler was configured to collect publicly accessible pages and PDFs from the S.P.I.T. website while restricting crawling to the institutional domain.

### Crawl summary

| Metric | Observed result |
|---|---:|
| Website pages collected | 236 |
| PDFs downloaded | 687 |
| URLs visited | 333 |
| Failed requests | 3 |

The ingestion script is currently configured to process **up to 150 PDFs and all non-empty website `.txt` files**. Therefore, the number of PDFs downloaded by the crawler is not the same as the number indexed.

These counts describe the crawl output; they do not prove that every page on the website was discovered. Pages blocked from crawling, inaccessible links, dynamically generated content, or unlinked documents may not be included.

### Output structure

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

The exact files in the project may vary according to the scripts retained in the working directory.

## 5. Implementation

### Step 1 — Website crawling

The crawler visits public pages on the S.P.I.T. website, follows eligible internal links, discovers PDFs, and saves the extracted website content and downloaded PDFs locally.

Source metadata is saved separately in JSON files. The crawler is restricted to the institutional domain and should respect the site's crawling rules.

### Step 2 — Text extraction

**Website pages:** The crawler saves webpage text as `.txt` files.

**Digital PDFs:** PyMuPDF extracts embedded text page by page.

**Scanned PDFs:** If a PDF page has no embedded text, the pipeline renders the page as an image and uses Tesseract OCR to recognize text.

The OCR fallback is important for scanned notices and documents. OCR output can contain recognition errors, so scanned content should be checked when answer accuracy matters.

### Step 3 — Metadata preservation

Each document is stored with metadata such as:

- Source filename
- Source URL
- Source title
- Source type (`pdf` or `html`)
- PDF page number, where applicable

This metadata helps identify the origin of retrieved passages and supports source attribution in the chatbot response.

### Step 4 — Chunking

LlamaIndex's `SentenceSplitter` divides extracted text into chunks with the following configuration:

```python
SentenceSplitter(
    chunk_size=500,
    chunk_overlap=75
)
```

- **Chunk size:** 500
- **Chunk overlap:** 75

The overlap preserves some surrounding context between adjacent chunks. Chunking allows the retriever to select relevant portions of a long document instead of passing an entire document to the language model.

### Step 5 — Embedding generation

Each chunk is converted into a vector using:

```python
HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    normalize=True
)
```

Embeddings represent semantic information numerically, allowing the system to retrieve passages related to a question even when the wording is not identical.

### Step 6 — FAISS indexing

The vectors are stored in a FAISS `IndexFlatIP` index.

```python
faiss.IndexFlatIP(dimension)
```

The index uses inner-product similarity. With normalized vectors, inner product corresponds to cosine similarity. The vectors must actually be normalized consistently for this interpretation to hold.

LlamaIndex manages the document-to-chunk conversion and connects the FAISS vector store to the ingestion pipeline. The persisted index is saved in `vector_store/`.

### Step 7 — Retrieval

When a user submits a question, the system embeds the query and retrieves the top three candidate chunks in the current test configuration:

```python
retriever = index.as_retriever(similarity_top_k=3)
```

The retrieved chunks and their source metadata are then passed to the generation stage.

### Step 8 — Answer generation

The local model used is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model receives the user question and retrieved context. A strict system instruction is used to ensure that it:

- Answers only questions directly related to S.P.I.T.
- Uses only the retrieved official knowledge-base context.
- Does not guess or fill gaps using its pretrained knowledge.
- Refuses unrelated questions.
- Refuses S.P.I.T.-related questions when the retrieved context does not contain sufficient evidence.
- Identifies supporting sources when source metadata is available.

Configured refusal responses:

**Unrelated question:**

> I can only answer questions directly related to SPIT using its official knowledge base.

**Insufficient evidence:**

> I could not find sufficient relevant information in the SPIT knowledge base to answer this question.

A prompt alone cannot guarantee that every unsupported answer will be rejected. A retrieval relevance threshold and systematic testing should also be used to reduce hallucinations.

## 6. Running the Project

Activate the Python virtual environment:

```bash
source venv/bin/activate
```

### Rebuild the vector index

```bash
python ingest.py
```

This processes up to 150 PDFs and all non-empty website text files, then rebuilds the index in `vector_store/`.

### Test retrieval only

```bash
python test.py
```

This is useful for checking whether the retrieved passages are relevant before evaluating generated answers.

### Run the RAG chatbot

```bash
python rag.py
```

Enter an S.P.I.T.-related question at the prompt. Enter `exit` to stop the interactive session.

The first run may download the embedding model and language model. Ensure the machine has sufficient available memory and disk space.

## 7. Evaluation Methodology

RAG quality should be evaluated at two separate levels:

1. **Retrieval quality:** Did the system retrieve the correct supporting passages?
2. **Generation quality:** Did the language model produce an accurate answer supported by those passages?

A plausible answer is not necessarily correct, and a correct answer without supporting retrieved evidence does not demonstrate successful RAG grounding.

### 7.1 Recommended test set

Use a fixed set of 30 questions divided into five categories:

| Category | Number | Purpose |
|---|---:|---|
| Factual questions | 10 | Check facts such as departments, programmes, or official procedures |
| Multi-document questions | 5 | Check whether information from multiple sources is retrieved |
| Comparison questions | 5 | Check comparisons between programmes, departments, or policies when documented |
| Reasoning questions | 5 | Check whether conclusions are supported by the retrieved evidence |
| Unanswerable questions | 5 | Check whether the chatbot refuses unsupported questions |
| **Total** | **30** | |

Questions should be based on verified public institutional information. For each answerable question, record the expected answer and supporting source. For each unanswerable question, record why the knowledge base does not support an answer.

### 7.2 Retrieval evaluation

Compare the following values of `similarity_top_k`:

```python
similarity_top_k = 1
similarity_top_k = 3
similarity_top_k = 5
similarity_top_k = 10
```

Measure:

- **Hit Rate / Recall@k:** Whether at least one relevant source appears among the top-k retrieved chunks.
- **Precision@k:** The proportion of retrieved chunks judged relevant.
- **Source correctness:** Whether the retrieved source actually supports the expected answer.

Increasing `k` can provide more context but may also introduce irrelevant passages. The best value should be selected from measured results, not assumed in advance.

### 7.3 Generation evaluation

Assess each generated response for:

- **Correctness:** Does it match the verified answer?
- **Faithfulness:** Is every substantive claim supported by the retrieved context?
- **Relevance:** Does it answer the question asked?
- **Citation/source accuracy:** Do the cited sources support the claims?
- **Abstention quality:** Does it refuse unrelated or unsupported questions?
- **Completeness:** Does it include the important information required by the question?

Use a manually verified answer key. Record the results in a table rather than reporting unmeasured accuracy.

### 7.4 Required robustness tests

| Test | Expected behaviour |
|---|---|
| Answerable S.P.I.T. question | Answer from relevant retrieved context |
| Unrelated general-knowledge question | Return the configured unrelated-question refusal |
| S.P.I.T. question with no supporting evidence | Return the insufficient-evidence refusal |
| Irrelevant retrieved context | Avoid presenting it as evidence |
| Conflicting source passages | Identify the conflict or abstain instead of inventing a resolution |
| OCR-derived source | Answer correctly when OCR text is legible and sufficient |
| Source attribution | Point to the correct page or PDF source when available |

### 7.5 RAG versus in-context learning baseline

For a useful comparison, ask the same questions in two configurations:

- **RAG:** Question plus retrieved knowledge-base passages.
- **In-context baseline:** Question plus a fixed, manually selected context, without vector retrieval.

Use the same language model and evaluation criteria where possible. This comparison helps assess whether dynamic retrieval improves answer quality. Do not claim one method performs better without measured results.

## 8. Results and Observations

### Results established during implementation

- The crawler collected 236 website pages and downloaded 687 PDFs.
- The crawler visited 333 URLs and recorded 3 failed requests.
- A scanned PDF with no embedded text was identified.
- Tesseract OCR successfully extracted text from the scanned PDF during testing.
- A FAISS vector store was built using the BGE embedding model and LlamaIndex.
- Retrieval was tested through a separate script.
- The local Qwen model was used for RAG answer generation.
- An unrelated question initially produced an unsupported answer based on irrelevant retrieved context. This exposed the need for stricter grounding instructions and a retrieval relevance check.

### Results that must be measured

The following values should be filled in after running the complete evaluation:

| Metric | Result |
|---|---|
| Number of questions evaluated | Not yet recorded |
| Retrieval Hit Rate / Recall@k | Not yet measured |
| Precision@k | Not yet measured |
| Answer correctness | Not yet measured |
| Faithfulness / groundedness | Not yet measured |
| Correct refusal rate | Not yet measured |
| Incorrect answer rate on unanswerable questions | Not yet measured |
| Best `similarity_top_k` | Not yet determined |
| RAG versus in-context baseline | Not yet compared quantitatively |

Do not report a single “accuracy” value unless the evaluation set, scoring method, and observed results are documented.

## 9. Limitations

- The ingestion configuration indexes at most 150 PDFs, not all 687 downloaded PDFs.
- The crawler's collected pages do not guarantee complete coverage of the website.
- OCR may introduce spelling, formatting, and table-extraction errors.
- Retrieval can return semantically similar but irrelevant passages.
- The 0.5B-parameter language model may struggle with complex reasoning or combining information from several documents.
- A strict prompt cannot completely eliminate hallucinations.
- The current FAISS configuration uses exact inner-product search; relevance scores still need to be calibrated against a verified test set.
- Website updates are not automatically reflected until the crawl and ingestion pipeline are run again.

## 10. Conclusion

A Retrieval-Augmented Generation chatbot for S.P.I.T. was developed using publicly collected website content and institutional PDFs. The pipeline includes document collection, PDF text extraction, OCR fallback for scanned pages, metadata preservation, overlapping text chunking, BGE embeddings, FAISS vector search, and local answer generation with Qwen.

The implementation demonstrates how a language model can answer institutional questions using a searchable external knowledge base instead of relying solely on its pretrained knowledge. Initial testing also showed that irrelevant retrieval results can lead to unsupported answers, motivating stricter grounding instructions and relevance-based refusal.

The next step is to evaluate the system systematically using answerable, multi-document, comparison, reasoning, and unanswerable questions. Retrieval and generation metrics must be measured before making quantitative claims about accuracy or reliability.

## 11. References

1. Sardar Patel Institute of Technology official website: https://www.spit.ac.in/
2. LlamaIndex documentation: https://docs.llamaindex.ai/
3. FAISS documentation: https://faiss.ai/
4. Hugging Face model — BGE Small English v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5
5. Hugging Face model — Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
6. PyMuPDF documentation: https://pymupdf.readthedocs.io/
7. Tesseract OCR: https://github.com/tesseract-ocr/tesseract

---

**End of Experiment 8**
