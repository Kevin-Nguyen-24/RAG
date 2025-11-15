"""Test script to verify dynamic PDF detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.test_dataset import ESGTestDataset


def main():
    """Test dynamic PDF filename detection."""
    
    print("\n" + "="*60)
    print("TESTING DYNAMIC PDF FILENAME DETECTION")
    print("="*60)
    
    # Get detected PDFs
    pdfs = ESGTestDataset._get_company_pdfs()
    
    print("\n📁 Detected PDF files:")
    print("-"*60)
    for company, filename in pdfs.items():
        icon = "✅" if Path(f"./data/{filename}").exists() else "❌"
        print(f"{icon} {company:.<20} {filename}")
    
    # Get test cases
    test_cases = ESGTestDataset.get_test_cases()
    
    print(f"\n📝 Test cases: {len(test_cases)}")
    print("-"*60)
    
    # Show sample test case
    sample = test_cases[0]
    print(f"\nSample test case:")
    print(f"  Question: {sample['question']}")
    print(f"  Expected source: {sample['expected_sources'][0]}")
    print(f"  Company: {sample['company']}")
    
    # Verify all expected sources exist
    print(f"\n🔍 Verifying expected sources...")
    print("-"*60)
    
    all_sources = set()
    for case in test_cases:
        all_sources.update(case.get('expected_sources', []))
    
    missing = []
    for source in sorted(all_sources):
        if source:  # Skip empty strings
            exists = Path(f"./data/{source}").exists()
            icon = "✅" if exists else "❌"
            print(f"{icon} {source}")
            if not exists:
                missing.append(source)
    
    print("\n" + "="*60)
    if missing:
        print(f"⚠️  WARNING: {len(missing)} expected files not found:")
        for f in missing:
            print(f"  - {f}")
    else:
        print("✅ SUCCESS: All expected files exist!")
    print("="*60)
    
    print("\n💡 Benefits of dynamic detection:")
    print("  • No manual filename updates needed")
    print("  • Automatically adapts to your PDFs")
    print("  • Prevents filename mismatch errors")
    print("  • Works even if you rename files")


if __name__ == "__main__":
    main()
