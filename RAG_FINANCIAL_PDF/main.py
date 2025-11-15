#!/usr/bin/env python3
"""
Financial ESG RAG System
Main entry point for the RAG pipeline
"""

import argparse
from rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Financial ESG RAG System - Query ESG reports using RAG"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of vector store from documents"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode - provide a question"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b",
        help="Ollama model to use (default: gemma3:4b)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    rag.initialize(force_rebuild=args.rebuild)
    
    if args.query:
        # Single query mode
        result = rag.query(args.query, k=args.top_k, model=args.model)
        print(f"\n💡 Answer:\n{result['answer']}")
        
        if result.get('sources'):
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['file']} (Page {source['page']})")
    else:
        # Interactive mode
        rag.interactive_query()


if __name__ == "__main__":
    main()
