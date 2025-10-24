"""Ingest ESG PDF documents into Qdrant vector store."""
import sys
import argparse
import re
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.qdrant_store import QdrantStore
from pypdf import PdfReader


def is_table_row(text: str) -> bool:
    """Detect if text appears to be a table row with numerical data."""
    # Count digits and special table characters
    digit_count = sum(c.isdigit() for c in text)
    has_multiple_numbers = len(re.findall(r'\d+[.,]?\d*', text)) >= 2
    has_separators = text.count('|') >= 2 or text.count('\t') >= 2
    
    # Table rows typically have multiple numbers or clear separation
    return has_multiple_numbers or has_separators


def extract_context_keywords(text: str) -> str:
    """Extract key financial/ESG terms to add as context."""
    # Important keywords for ESG metrics
    keywords = []
    text_lower = text.lower()
    
    # Financial metrics
    metric_terms = ['emissions', 'carbon', 'ghg', 'scope', 'energy', 'water', 'waste', 
                    'revenue', 'turnover', 'spend', 'investment', 'cost', 'target',
                    'reduction', 'training', 'employees', 'stores', 'fines', 'taxation']
    
    # Units
    unit_terms = ['kwh', 'mwh', 'tons', 'tonnes', 'kilotons', 'co2', 'co₂', 
                  'million', 'billion', 'percent', '%', 'r million', 'r billion']
    
    for term in metric_terms + unit_terms:
        if term in text_lower:
            keywords.append(term)
    
    # Extract years
    years = re.findall(r'20\d{2}', text)
    keywords.extend(years)
    
    return ' '.join(set(keywords)) if keywords else ''


def process_pdf(pdf_path: Path) -> list:
    """
    Extract text from PDF and create chunks optimized for numerical/tabular data.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of (text, metadata) tuples
    """
    try:
        reader = PdfReader(str(pdf_path))
        chunks = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            if text.strip():
                # Enhanced chunking strategy
                chunk_size = 1500  # Increased for better context
                chunk_overlap = 300  # More overlap to preserve relationships
                
                # Split by lines first to detect tables
                lines = text.split('\n')
                
                # Group lines into semantic blocks
                current_chunk = ""
                table_buffer = []  # Buffer for table rows
                in_table = False
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detect table rows
                    if is_table_row(line):
                        in_table = True
                        table_buffer.append(line)
                        continue
                    
                    # If we were in a table and now aren't, flush the table
                    if in_table and not is_table_row(line):
                        # Add table context
                        if table_buffer:
                            table_text = "\n".join(table_buffer)
                            # Add preceding context (headers)
                            context_lines = lines[max(0, i-len(table_buffer)-3):i-len(table_buffer)]
                            context = " ".join([l.strip() for l in context_lines if l.strip()])
                            
                            full_table_chunk = f"{context}\n{table_text}"
                            
                            # Extract keywords for better retrieval
                            keywords = extract_context_keywords(full_table_chunk)
                            if keywords:
                                full_table_chunk = f"[Keywords: {keywords}] {full_table_chunk}"
                            
                            metadata = {
                                "source_file": pdf_path.name,
                                "page": page_num,
                                "doc_type": "esg_report",
                                "content_type": "table"
                            }
                            chunks.append((full_table_chunk.strip(), metadata))
                            table_buffer = []
                        in_table = False
                    
                    # Regular text processing
                    if not in_table:
                        # Check if adding this line exceeds chunk size
                        if len(current_chunk) + len(line) > chunk_size and current_chunk:
                            # Extract and add keywords
                            keywords = extract_context_keywords(current_chunk)
                            if keywords:
                                current_chunk = f"[Keywords: {keywords}] {current_chunk}"
                            
                            metadata = {
                                "source_file": pdf_path.name,
                                "page": page_num,
                                "doc_type": "esg_report",
                                "content_type": "text"
                            }
                            chunks.append((current_chunk.strip(), metadata))
                            
                            # Keep overlap - last 300 characters
                            if len(current_chunk) > chunk_overlap:
                                current_chunk = current_chunk[-chunk_overlap:]
                            current_chunk += " " + line
                        else:
                            current_chunk += " " + line
                
                # Flush any remaining table
                if table_buffer:
                    table_text = "\n".join(table_buffer)
                    keywords = extract_context_keywords(table_text)
                    if keywords:
                        table_text = f"[Keywords: {keywords}] {table_text}"
                    metadata = {
                        "source_file": pdf_path.name,
                        "page": page_num,
                        "doc_type": "esg_report",
                        "content_type": "table"
                    }
                    chunks.append((table_text.strip(), metadata))
                
                # Add remaining text chunk
                if current_chunk.strip():
                    keywords = extract_context_keywords(current_chunk)
                    if keywords:
                        current_chunk = f"[Keywords: {keywords}] {current_chunk}"
                    metadata = {
                        "source_file": pdf_path.name,
                        "page": page_num,
                        "doc_type": "esg_report",
                        "content_type": "text"
                    }
                    chunks.append((current_chunk.strip(), metadata))
        
        logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks from {len(reader.pages)} pages")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return []


def ingest_documents(data_dir: Path, clear_existing: bool = False):
    """
    Ingest all PDFs from data directory.
    
    Args:
        data_dir: Directory containing PDF files
        clear_existing: Whether to clear existing collection
    """
    logger.info(f"Starting document ingestion from: {data_dir}")
    
    # Initialize vector store
    store = QdrantStore()
    
    # Clear existing collection if requested
    if clear_existing:
        logger.warning("Clearing existing collection...")
        store.delete_collection()
        store._ensure_collection_exists()
    
    # Find all PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {data_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process all PDFs
    all_chunks = []
    all_metadata = []
    
    for pdf_file in pdf_files:
        chunks = process_pdf(pdf_file)
        for text, metadata in chunks:
            all_chunks.append(text)
            all_metadata.append(metadata)
    
    if not all_chunks:
        logger.error("No chunks extracted from PDFs")
        return
    
    # Add to vector store
    logger.info(f"Adding {len(all_chunks)} chunks to Qdrant...")
    success = store.add_documents(
        documents=all_chunks,
        metadata=all_metadata
    )
    
    if success:
        info = store.get_collection_info()
        logger.success(f"✓ Successfully ingested documents!")
        logger.info(f"Collection info: {info}")
    else:
        logger.error("Failed to ingest documents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest ESG PDF documents into Qdrant"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before ingesting"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    ingest_documents(args.data_dir, args.clear)


if __name__ == "__main__":
    main()
