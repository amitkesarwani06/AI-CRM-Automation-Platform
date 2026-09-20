import logging
import os
import warnings

# Suppress langchain-community sunset deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY and USER_AGENT from .env

# Ensure USER_AGENT is set for WebBaseLoader (suppresses its warning)
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "ai-crm-platform/1.0"))

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)



class DocumentIngestionService:
    """
    Single entry point for loading documents from any source into the CRM
    Knowledge Base. Returns a standardized list of Document objects
    regardless of source format.
    """

    def load_pdf(self, file_path: str) -> list[Document]:
        """Load a PDF — one Document per page, with page number in metadata."""
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Enrich metadata so we always know the source
        for doc in docs:
            doc.metadata["doc_type"] = "pdf"
            doc.metadata["file_name"] = Path(file_path).name
        logger.info(f"Loaded {len(docs)} pages from {file_path}")
        return docs

    def load_csv(self, file_path: str, content_columns: Optional[list[str]] = None) -> list[Document]:
        """
        Load a CSV — one Document per row.
        content_columns: which columns to include in page_content.
        If None, all columns are included.
        """
        logger.info(f"Loading CSV: {file_path}")
        loader = CSVLoader(
            file_path=file_path,
            content_columns=content_columns,
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = "csv"
            doc.metadata["file_name"] = Path(file_path).name
        logger.info(f"Loaded {len(docs)} rows from {file_path}")
        return docs

    def load_web(self, urls: list[str]) -> list[Document]:
        """Load one or more web pages — strips HTML, returns clean text."""
        logger.info(f"Loading {len(urls)} web page(s)")
        loader = WebBaseLoader(web_paths=urls)
        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = "web"
        logger.info(f"Loaded {len(docs)} web document(s)")
        return docs

    def load_text(self, file_path: str) -> list[Document]:
        """Load a plain text file into a Document."""
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = "text"
            doc.metadata["file_name"] = Path(file_path).name
        logger.info(f"Loaded {len(docs)} text document(s) from {file_path}")
        return docs

    def load_directory(self, dir_path: str, glob: str = "**/*.pdf") -> list[Document]:
        """
        Load all files matching a pattern from a directory.
        Default: all PDFs recursively.
        Change glob to '**/*.txt' for text files, etc.
        """
        logger.info(f"Loading directory: {dir_path} (pattern: {glob})")
        loader = DirectoryLoader(
            dir_path,
            glob=glob,
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = "pdf"
            doc.metadata["file_name"] = Path(doc.metadata.get("source", "")).name
        logger.info(f"Loaded {len(docs)} documents from directory")
        return docs

    def load_all(self, dir_path: str) -> list[Document]:
        """
        Convenience method: load ALL supported file types from a directory.
        Used for bulk ingestion at startup.
        """
        all_docs = []
        path = Path(dir_path)

        # Load PDFs
        for pdf_file in path.glob("**/*.pdf"):
            all_docs.extend(self.load_pdf(str(pdf_file)))

        # Load CSVs
        for csv_file in path.glob("**/*.csv"):
            all_docs.extend(self.load_csv(str(csv_file)))

        # Load TXT files
        for txt_file in path.glob("**/*.txt"):
            all_docs.extend(self.load_text(str(txt_file)))

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = DocumentIngestionService()

    base_raw = Path(__file__).parent.parent / "raw_docs"

    # 1. Test CSV loading
    csv_path = str(base_raw / "leads_sample.csv")
    csv_docs = service.load_csv(csv_path)
    print("\n--- CSV Loader Output (First 3 Rows) ---")
    for i, doc in enumerate(csv_docs[:3]):
        print(f"Row {i+1} Content:\n{doc.page_content}")
        print(f"Row {i+1} Metadata: {doc.metadata}\n")

    # 2. Test Text loading (Mini Assignment)
    txt_path = str(base_raw / "company_about.txt")
    txt_docs = service.load_text(txt_path)
    print("--- Text Loader Output ---")
    print(f"Content:\n{txt_docs[0].page_content}")
    print(f"Metadata: {txt_docs[0].metadata}\n")

    # 3. Test Web loading
    web_docs = service.load_web(["https://python.org"])
    print("--- Web Loader Output ---")
    print(f"Content (first 300 chars): {web_docs[0].page_content[:300]}...")
    print(f"Metadata: {web_docs[0].metadata}")
    print(f"Document type: {type(web_docs[0])}")
    print(f"page_content length: {len(web_docs[0].page_content)} chars\n")

    # 4. Test load_all convenience method
    all_raw_docs = service.load_all(str(base_raw))
    print(f"--- Bulk Load All ---")
    print(f"Total raw docs loaded: {len(all_raw_docs)}")
