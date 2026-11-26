from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def resolve_pdf_path() -> Path:
    default_pdf = Path(__file__).parent / "Individual_Contributor_License_Agreement.pdf"
    if default_pdf.exists():
        return default_pdf

    available_pdfs = list(Path(__file__).parent.glob("*.pdf"))
    if not available_pdfs:
        raise FileNotFoundError(f"No PDF file found in {Path(__file__).parent}")
    return available_pdfs[0]


def extract_and_chunk_pdf(pdf_path: Path) -> List[Document]:
    reader = PdfReader(str(pdf_path))
    print(f"Loading PDF: {pdf_path.name} ({len(reader.pages)} pages)")

    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    documents: List[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = " ".join(raw_text.split())

        if not cleaned_text:
            continue

        page_chunks = text_splitter.create_documents(
            texts=[cleaned_text],
            metadatas=[{"page": page_num, "source": pdf_path.name}]
        )
        documents.extend(page_chunks)
        print(f"Page {page_num} extracted ({len(page_chunks)} chunks)")

    print(f"PDF processing complete: {len(documents)} total chunks created.")
    return documents


def get_vector_store(documents: List[Document], persist_dir: Path) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="pdf_knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(persist_dir)
    )

    if vector_store._collection.count() == 0:
        vector_store.add_documents(
            documents=documents,
            ids=[f"chunk_{i}" for i in range(len(documents))]
        )
        print("Embeddings generated and indexed in ChromaDB.")
    else:
        print("Loaded existing ChromaDB index.")

    return vector_store


def start_chat_loop(vector_store: Chroma) -> None:
    print("\n" + "=" * 40)
    print("         PDF CHATBOT READY")
    print("=" * 40)
    print("Ask questions about the document.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if query.lower() == "exit":
            print("Exiting chatbot.")
            break

        if not query:
            continue

        results = vector_store.similarity_search(query, k=3)
        print("\nRelevant Passages:\n")

        if not results:
            print("No relevant results found.")
        else:
            for i, result in enumerate(results, start=1):
                page = result.metadata.get("page", "N/A")
                print(f"[{i}] Page {page}:")
                print(result.page_content)
                print("-" * 50)
            print()


def main() -> None:
    pdf_path = resolve_pdf_path()
    docs = extract_and_chunk_pdf(pdf_path)
    db_path = Path(__file__).parent / "chroma_db"
    vector_store = get_vector_store(docs, db_path)
    start_chat_loop(vector_store)


if __name__ == "__main__":
    main()
