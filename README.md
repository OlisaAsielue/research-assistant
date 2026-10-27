# Local Research Assistant (RAG)

Ask questions about your own collection of research papers. PDFs are cleaned, chunked, embedded into a persistent Chroma index, and answered by Mistral-7B running locally through Ollama. No paper text leaves the machine.

## Problem

Literature reviews mean repeatedly searching dozens of PDFs for methods, datasets and claims. Hosted chat tools either can't see your papers or require uploading them. The aim is a small, fully local pipeline that answers from your papers and shows which papers each answer came from.

## Approach

- **Ingestion** (`src/data_ingestion.py`). Text is extracted page by page with PyMuPDF. Numeric citation markers such as `[12]` are removed and whitespace is normalised, so chunks embed the content rather than reference noise.
- **Chunking and embedding** (`src/rag_pipeline.py`). LlamaIndex's sentence-aware splitter (512 tokens, 64 overlap) and `BAAI/bge-small-en-v1.5` embeddings, a small model that runs comfortably on a laptop CPU.
- **Vector store.** A persistent ChromaDB collection. The index is built once and reloaded on later runs; `--rebuild` re-embeds after adding papers.
- **Generation.** Top-4 retrieved chunks are passed to Mistral-7B via Ollama.
- **Interface.** A Gradio app that shows the answer along with the source file and similarity score of every chunk used, so each claim can be checked against the paper.

## Results

This is a working local tool rather than a benchmarked system. Retrieval and answer quality haven't been formally evaluated; the next step below describes how they would be.

## Tech stack

Python, LlamaIndex, ChromaDB, Hugging Face embeddings (bge-small), Ollama (Mistral-7B), PyMuPDF, Gradio, pytest.

## How to run

```bash
git clone https://github.com/OlisaAsielue/research-assistant.git
cd research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Local LLM (https://ollama.com)
ollama pull mistral

# Add your PDFs to data/raw_pdfs/, then:
python src/data_ingestion.py                      # PDFs -> cleaned text in data/processed/
python src/rag_pipeline.py --query "What datasets were used?"   # builds the index on first run
python src/app.py                                 # web UI at http://127.0.0.1:7860
pytest                                            # ingestion tests
```

## Limitations and next steps

- **Evaluation.** Build a small question set with known answer passages for a handful of papers, then measure retrieval hit rate at k and answer faithfulness (for example with RAGAS or LLM-as-judge with spot checks).
- **Chunking ignores layout.** Two-column PDFs, tables and equations come through as flattened text. A layout-aware parser such as GROBID would keep sections and references intact and allow section-level citations.
- **Plain dense retrieval.** Hybrid search (BM25 plus embeddings) and a cross-encoder re-ranker would help with exact terms such as dataset names.
- **Indexing is all-or-nothing.** New papers need a full rebuild; incremental upserts keyed on file hash would fix that.
