import logging
from enum import Enum
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Different content types need different chunk sizes."""
    FAQ            = "faq"
    TECHNICAL      = "technical"
    CONVERSATIONAL = "conversational"
    GENERAL        = "general"


CHUNK_CONFIGS = {
    ContentType.FAQ:            {"chunk_size": 400,  "chunk_overlap": 50},
    ContentType.TECHNICAL:      {"chunk_size": 800,  "chunk_overlap": 100},
    ContentType.CONVERSATIONAL: {"chunk_size": 300,  "chunk_overlap": 30},
    ContentType.GENERAL:        {"chunk_size": 500,  "chunk_overlap": 50},
}


class DocumentChunkingService:
    """
    Takes Document objects from DocumentIngestionService (Day 8)
    and splits them into smaller, embedding-ready chunks.
    """

    def __init__(self, content_type: ContentType = ContentType.GENERAL):
        config = CHUNK_CONFIGS[content_type]
        self.content_type = content_type
        self.chunk_size = config["chunk_size"]       # stored so split() can log it
        self.chunk_overlap = config["chunk_overlap"]  # stored so split() can log it
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split Document objects into chunks. Metadata is preserved."""
        if not documents:
            logger.warning("No documents to split.")
            return []

        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["content_type"] = self.content_type
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            f"Split {len(documents)} document(s) into {len(chunks)} chunks "
            f"(type={self.content_type}, size={self.chunk_size}, "
            f"overlap={self.chunk_overlap})"
        )
        return chunks

    def split_text(self, text: str, source: str = "unknown") -> list[Document]:
        """Split a raw string directly into Document chunks."""
        raw_chunks = self.splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "content_type": self.content_type,
                }
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def inspect(self, chunks: list[Document]) -> None:
        """Print a readable summary of chunks — use during dev to tune sizes."""
        print(f"\n{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        sizes = [len(c.page_content) for c in chunks]
        print(f"Chunk sizes → min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")
        print(f"\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Size: {len(chunk.page_content)} chars")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content: {chunk.page_content[:150]}...")
        print('='*60)

    def smart_split(self, documents: list[Document]) -> list[Document]:
        """Auto-detect content type from metadata. Skip splitting CSV rows."""
        all_chunks = []
        for doc in documents:
            doc_type = doc.metadata.get("doc_type", "")
            file_name = doc.metadata.get("file_name", "")

            if doc_type == "csv":
                # CSV rows are already atomic — don't split
                all_chunks.append(doc)
            elif "faq" in file_name.lower():
                service = DocumentChunkingService(ContentType.FAQ)
                all_chunks.extend(service.split([doc]))
            else:
                all_chunks.extend(self.split([doc]))

        return all_chunks


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    Refund Policy
    We offer a 30-day money-back guarantee on all plans.
    To request a refund, contact our support team at support@crmplatform.com.
    Refunds are processed within 5-7 business days.

    Pricing Plans
    Starter Plan: Rs.999/month -- up to 3 users, 500 leads, basic analytics.
    Growth Plan: Rs.2,999/month -- up to 15 users, unlimited leads, AI features.
    Enterprise Plan: Custom pricing -- unlimited users, dedicated support.

    Getting Started
    After signing up, you will receive a welcome email with setup instructions.
    Your first 14 days are free -- no credit card required.
    Book an onboarding call at calendly.com/crmplatform/onboarding.

    Support Hours
    Our support team is available Monday to Saturday, 9am to 7pm IST.
    For urgent issues outside hours, use the in-app emergency chat.
    Average response time is under 2 hours during business hours.
    """

    # Test 1: GENERAL chunking
    print("\n>>> GENERAL chunking:")
    service = DocumentChunkingService(content_type=ContentType.GENERAL)
    chunks = service.split_text(sample_text, source="company_faq")
    service.inspect(chunks)

    # Test 2: FAQ chunking (smaller size = more chunks)
    print("\n>>> FAQ chunking:")
    faq_service = DocumentChunkingService(content_type=ContentType.FAQ)
    faq_chunks = faq_service.split_text(sample_text, source="company_faq")
    faq_service.inspect(faq_chunks)

    print(f"\nComparison: GENERAL={len(chunks)} chunks | FAQ={len(faq_chunks)} chunks")

    # Test 3: Integration test — Day 8 loader → Day 9 splitter
    print("\n>>> Integration test: Day 8 loader → Day 9 splitter")
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    from app.knowledge_base.loaders.document_loader import DocumentIngestionService

    loader = DocumentIngestionService()

    # Use company_about.txt (already exists from Day 8)
    raw_docs = loader.load_text("app/knowledge_base/raw_docs/company_about.txt")
    chunked = service.split(raw_docs)
    service.inspect(chunked)
