# Gen-AI PDF Semantic Search & Q&A

A lightweight, local semantic search and retrieval tool for PDF documents powered by LangChain, ChromaDB, and HuggingFace embeddings (`all-MiniLM-L6-v2`).

## Features

- **Automatic PDF Discovery**: Automatically targets `Individual_Contributor_License_Agreement.pdf` or scans the local directory for available PDFs.
- **Page-Aware Chunking**: Extracts text page-by-page and chunks content with sliding overlaps while retaining exact page numbers in metadata.
- **Local Vector Storage**: Embeds chunks using sentence-transformers and persists them locally inside a ChromaDB collection.
- **Interactive Query Engine**: Interactive terminal interface to execute semantic similarity searches against the indexed document.

## Project Structure

```
├── Individual_Contributor_License_Agreement.pdf  # Sample source document
├── main.py                                      # Core PDF extraction, embedding & query logic
├── pyproject.toml                               # uv project specification
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Ignore environments, cache, and DB binaries
└── README.md                                    # Documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package and project manager)

### Installation

1. Clone repository:
   ```bash
   git clone https://github.com/rishav-raj-genx/Gen-AI-Project-py.git
   cd Gen-AI-Project-py
   ```

2. Initialize virtual environment and install dependencies using `uv`:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

### Usage

Run the application:
```bash
uv run python main.py
```

Type your query in natural language (e.g., `grant of copyright license`), view the relevant passages with page references, and type `exit` to stop.
