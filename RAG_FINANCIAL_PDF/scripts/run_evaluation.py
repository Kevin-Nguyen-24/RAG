"""Wrapper script that ensures vector DB is populated before running evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.qdrant_store import QdrantStore
import subprocess


def check_and_populate_db():
    """Check if vector DB has data, populate if empty."""
    print("\n" + "="*60)
    print("CHECKING VECTOR DATABASE STATUS")
    print("="*60)
    
    try:
        store = QdrantStore()
        info = store.get_collection_info()
        points_count = info.get('points_count', 0)
        
        print(f"\n📊 Current database status:")
        print(f"   Documents: {points_count}")
        
        if points_count == 0:
            print("\n⚠️  WARNING: Vector database is EMPTY!")
            print("   Re-ingesting documents...")
            
            # Close store before running subprocess
            store.close()
            del store
            
            # Run ingestion
            result = subprocess.run(
                [sys.executable, "scripts/ingest_esg_documents.py", "--clear"],
                capture_output=False,
                text=True
            )
            
            if result.returncode != 0:
                print("\n❌ ERROR: Failed to ingest documents!")
                return False
            
            # Verify ingestion worked
            store = QdrantStore()
            info = store.get_collection_info()
            new_count = info.get('points_count', 0)
            
            # Close store after checking
            store.close()
            del store
            
            if new_count > 0:
                print(f"\n✅ SUCCESS: Ingested {new_count} documents!")
                return True
            else:
                print("\n❌ ERROR: Ingestion failed - still 0 documents!")
                return False
        else:
            # Close store before running evaluation subprocess
            store.close()
            del store
            print(f"\n✅ Database OK: {points_count} documents ready")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def main():
    """Main entry point."""
    # Check and populate database
    if not check_and_populate_db():
        print("\n❌ Cannot run evaluation - database issue!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60 + "\n")
    
    # Run evaluation
    args = ["python", "scripts/evaluate_rag.py"] + sys.argv[1:]
    subprocess.run(args)


if __name__ == "__main__":
    main()
