"""Quick diagnostic script to identify RAG system issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.qdrant_store import QdrantStore
from src.chatbot.esg_chatbot import ESGChatbot
from loguru import logger


def diagnose_vector_db():
    """Check vector database status."""
    print("\n" + "="*60)
    print("1. VECTOR DATABASE DIAGNOSIS")
    print("="*60)
    
    try:
        store = QdrantStore()
        info = store.get_collection_info()
        
        print(f"Collection Name: {info.get('name')}")
        print(f"Points Count: {info.get('points_count', 0)}")
        print(f"Vectors Count: {info.get('vectors_count', 0)}")
        
        if info.get('points_count', 0) == 0:
            print("\n❌ PROBLEM: Vector database is EMPTY!")
            print("   FIX: Run 'python scripts/ingest_esg_documents.py --clear'")
            return False
        else:
            print(f"\n✅ Database has {info.get('points_count')} documents")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def diagnose_retrieval():
    """Test retrieval with sample query."""
    print("\n" + "="*60)
    print("2. RETRIEVAL DIAGNOSIS")
    print("="*60)
    
    try:
        store = QdrantStore()
        test_query = "What are Absa's carbon emissions targets?"
        
        print(f"Test Query: {test_query}")
        results = store.search(test_query, limit=5, score_threshold=0.15)
        
        print(f"\nRetrieved: {len(results)} contexts")
        
        if len(results) == 0:
            print("\n❌ PROBLEM: No contexts retrieved!")
            print("   Possible causes:")
            print("   1. Score threshold too high")
            print("   2. Embedding model mismatch")
            print("   3. Query processing issue")
            return False
        
        print("\n✅ Retrieval working")
        print("\nTop 3 results:")
        for idx, result in enumerate(results[:3], 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            source = metadata.get('source_file', 'Unknown')
            text_preview = result.get('text', '')[:100]
            print(f"\n  [{idx}] Score: {score:.3f}")
            print(f"      Source: {source}")
            print(f"      Preview: {text_preview}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def diagnose_chatbot():
    """Test chatbot response."""
    print("\n" + "="*60)
    print("3. CHATBOT DIAGNOSIS")
    print("="*60)
    
    try:
        chatbot = ESGChatbot()
        test_query = "What are Absa's carbon emissions targets?"
        
        print(f"Test Query: {test_query}")
        response = chatbot.process_message(test_query, "diagnostic_session")
        
        print(f"\nResponse Length: {len(response)} characters")
        print(f"\nResponse Preview:\n{response[:300]}...")
        
        # Check if response contains source citations
        has_citation = any(keyword in response.lower() for keyword in 
                          ['source', 'absa', 'report', 'pdf', '['])
        
        if has_citation:
            print("\n✅ Response contains source references")
        else:
            print("\n⚠️  WARNING: Response may lack source citations")
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "i don't have", "not available", "cannot find",
            "no information", "no data", "i apologize"
        ]
        
        is_uncertain = any(phrase in response.lower() for phrase in hallucination_phrases)
        
        if is_uncertain:
            print("⚠️  Response indicates missing information")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def diagnose_test_data():
    """Check if test dataset expectations match reality."""
    print("\n" + "="*60)
    print("4. TEST DATASET DIAGNOSIS")
    print("="*60)
    
    from src.evaluation.test_dataset import ESGTestDataset
    
    test_cases = ESGTestDataset.get_test_cases()
    print(f"Total test cases: {len(test_cases)}")
    
    # Check if expected sources exist in vector DB
    store = QdrantStore()
    
    expected_sources = set()
    for case in test_cases:
        expected_sources.update(case.get('expected_sources', []))
    
    print(f"\nExpected source documents: {len(expected_sources)}")
    for source in sorted(expected_sources):
        print(f"  - {source}")
    
    print("\n⚠️  Verify these PDFs exist in your data/ folder")
    print("   and were ingested correctly.")
    
    return True


def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("RAG SYSTEM DIAGNOSTIC TOOL")
    print("="*60)
    
    results = {
        "vector_db": diagnose_vector_db(),
        "retrieval": diagnose_retrieval(),
        "chatbot": diagnose_chatbot(),
        "test_data": diagnose_test_data()
    }
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.replace('_', ' ').title()}: {'OK' if status else 'FAILED'}")
    
    print("\n" + "="*60)
    print("RECOMMENDED ACTIONS")
    print("="*60)
    
    if not results['vector_db']:
        print("\n🔧 URGENT: Re-ingest documents")
        print("   Command: python scripts/ingest_esg_documents.py --clear")
    
    if not results['retrieval']:
        print("\n🔧 Fix retrieval settings")
        print("   - Lower score_threshold in qdrant_store.py")
        print("   - Check embedding model compatibility")
    
    if all(results.values()):
        print("\n✅ All components working!")
        print("   Issue may be in test dataset expectations.")
        print("   Check EVALUATION_GUIDE.md for test data customization.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
