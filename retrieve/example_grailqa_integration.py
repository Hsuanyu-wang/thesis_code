#!/usr/bin/env python3
"""
Example script demonstrating how to integrate GrailQA dataset
using the new adapter system while keeping existing pipeline unchanged.
"""

import os
import sys
from src.dataset.adapters import ensure_processed_data, get_adapter


def test_grailqa_adapter():
    """Test the GrailQA adapter with the provided JSON files."""
    print("🧪 Testing GrailQA Adapter...")
    
    # Configuration for GrailQA
    config = {
        'dataset': {
            'name': 'grailQA',
            'source': 'grailQA',
            'raw_data_path': 'data_files/grailQA',
            'splits': ['train', 'test']
        }
    }
    
    # Test adapter creation
    try:
        adapter = get_adapter('grailQA', config)
        print("✅ GrailQA adapter created successfully")
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return False
    
    # Test data loading and processing
    for split in ['train', 'test']:
        print(f"\n🔄 Testing {split} split...")
        
        try:
            # Ensure processed data exists
            processed_file = ensure_processed_data('grailQA', config, split)
            print(f"✅ Processed data saved to: {processed_file}")
            
            # Load and inspect processed data
            import pickle
            with open(processed_file, 'rb') as f:
                processed_data = pickle.load(f)
            
            print(f"📊 Loaded {len(processed_data)} samples")
            
            if processed_data:
                sample = processed_data[0]
                print(f"📝 Sample keys: {list(sample.keys())}")
                print(f"❓ Question: {sample['question'][:100]}...")
                print(f"🏷️  Answer entities: {sample['a_entity'][:3]}")
                print(f"🔗 Relations: {len(sample['relation_list'])}")
                print(f"📊 Triples: {len(sample['h_id_list'])}")
            
        except Exception as e:
            print(f"❌ Failed to process {split}: {e}")
            return False
    
    print("\n✅ GrailQA adapter test completed successfully!")
    return True


def demonstrate_pipeline_integration():
    """Demonstrate how the new system integrates with existing pipeline."""
    print("\n🔗 Demonstrating Pipeline Integration...")
    
    # This shows how the existing pipeline can now work with GrailQA
    print("""
    The new adapter system allows GrailQA to work with the existing pipeline:
    
    1. Data Processing:
       python emb_grailqa.py -d grailQA
    
    2. Training:
       python train_grailqa.py -d grailQA
    
    3. The existing RetrieverDataset classes work unchanged:
       - RetrieverDataset
       - OptimizedRetrieverDataset  
       - SmartBatchRetrieverDataset
    
    4. All existing features are preserved:
       - Embedding computation
       - Triple scoring
       - KGE reranking
       - PRA integration
    """)


def show_compatibility_with_existing():
    """Show how WebQSP/CWQ continue to work unchanged."""
    print("\n🔄 Existing Datasets (WebQSP/CWQ) Compatibility...")
    
    print("""
    The adapter system maintains full backward compatibility:
    
    ✅ WebQSP (unchanged):
       python emb.py -d webqsp
       python train.py -d webqsp
    
    ✅ CWQ (unchanged):
       python emb.py -d cwq  
       python train.py -d cwq
    
    ✅ All existing scripts work exactly as before
    ✅ No changes to existing data formats
    ✅ No changes to existing model architectures
    """)


def main():
    """Main demonstration function."""
    print("🚀 GrailQA Integration Demonstration")
    print("=" * 50)
    
    # Check if GrailQA JSON files exist
    train_file = "data_files/grailQA/graphquestions_v1_fb15_training_091420.json"
    test_file = "data_files/grailQA/graphquestions_v1_fb15_test_091420.json"
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        print("Please ensure the GrailQA JSON files are in the correct location.")
        return
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Please ensure the GrailQA JSON files are in the correct location.")
        return
    
    print("✅ GrailQA JSON files found")
    
    # Test the adapter
    if test_grailqa_adapter():
        demonstrate_pipeline_integration()
        show_compatibility_with_existing()
        
        print("\n🎉 Integration demonstration completed successfully!")
        print("\nNext steps:")
        print("1. Run: python emb_grailqa.py -d grailQA")
        print("2. Run: python train_grailqa.py -d grailQA")
        print("3. Use the existing evaluation scripts with GrailQA")
    else:
        print("\n❌ Integration test failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
