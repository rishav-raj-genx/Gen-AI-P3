from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# 1. LOAD PDF
# ==========================================

pdf_path = Path(__file__).parent / "Individual_Contributor_License_Agreement.pdf"
if not pdf_path.exists():
    available_pdfs = list(Path(__file__).parent.glob("*.pdf"))
    if not available_pdfs:
        raise FileNotFoundError(f"No PDF file found in {Path(__file__).parent}")
    pdf_path = available_pdfs[0]

print(f"Loading PDF: {pdf_path.name}")
reader = PdfReader(str(pdf_path))
print("Total pages:", len(reader.pages))


# ==========================================
# 2. CHUNKING & EXTRACTION (WITH PAGE METADATA)
# ==========================================

text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

all_docs = []

for i, page in enumerate(reader.pages, start=1):
    raw_text = page.extract_text() or ""
    text = " ".join(raw_text.split())

    if not text:
        continue

    page_docs = text_splitter.create_documents(
        texts=[text],
        metadatas=[{"page": i}]
    )
    all_docs.extend(page_docs)
    print(f"Page {i} extracted successfully ({len(page_docs)} chunks)")

print(f"\nPDF processing completed! Total chunks created: {len(all_docs)}")


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ==========================================
# 4. STORE IN CHROMADB
# ==========================================

db_path = str(Path(__file__).parent / "chroma_db")

# Open or create persistent database
vector_store = Chroma(
    collection_name="pdf_knowledge_base",
    embedding_function=embeddings,
    persist_directory=db_path
)

# Add chunks only if the collection is empty
if vector_store._collection.count() == 0:
    vector_store.add_documents(
        documents=all_docs,
        ids=[f"chunk_{i}" for i in range(len(all_docs))]
    )
    print("Chunks stored in ChromaDB!")
else:
    print("Existing ChromaDB data loaded!")


# ==========================================
# 5. INTERACTIVE QUESTION RETRIEVAL
# ==========================================

print("\n================================")
print("       PDF CHATBOT READY")
print("================================")
print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")


while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nChatbot stopped.")
        break

    if query.lower() == "exit":
        print("Chatbot stopped.")
        break

    if not query:
        continue

    # Search relevant chunks
    results = vector_store.similarity_search(
        query,
        k=3
    )

    print("\nRelevant information:\n")

    if not results:
        print("No relevant information found.")
    else:
        for i, result in enumerate(results, start=1):
            print(f"Result {i}:")
            print(result.page_content)
            print("Page:", result.metadata.get("page", "Not available"))
            print("-" * 50)