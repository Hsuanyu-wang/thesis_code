#!/usr/bin/env python3
"""
Test script for KGE integration in SubgraphRAG

This script tests the KGE models and integration functionality.
"""

import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

from model.kge_models import create_kge_model, KGELoss
from model.kge_utils import create_kge_config_from_model

def test_kge_models():
    """Test KGE model creation and forward pass"""
    print("🧪 Testing KGE Models...")
    
    # Test parameters
    num_entities = 100
    num_relations = 50
    embedding_dim = 64
    batch_size = 32
    
    # Test each KGE model
    models = ['transe', 'distmult', 'ptranse']
    
    for model_type in models:
        print(f"\nTesting {model_type.upper()}...")
        
        try:
            # Create model
            model = create_kge_model(
                model_type=model_type,
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=embedding_dim
            )
            
            # Create dummy data
            pos_h = torch.randint(0, num_entities, (batch_size,))
            pos_r = torch.randint(0, num_relations, (batch_size,))
            pos_t = torch.randint(0, num_entities, (batch_size,))
            neg_h = torch.randint(0, num_entities, (batch_size,))
            neg_r = torch.randint(0, num_relations, (batch_size,))
            neg_t = torch.randint(0, num_entities, (batch_size,))
            
            # Forward pass
            pos_score, neg_score = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            # Check output shapes
            assert pos_score.shape == (batch_size,), f"Wrong pos_score shape: {pos_score.shape}"
            assert neg_score.shape == (batch_size,), f"Wrong neg_score shape: {neg_score.shape}"
            
            # Test prediction
            pred_score = model.predict(pos_h, pos_r, pos_t)
            assert pred_score.shape == (batch_size,), f"Wrong pred_score shape: {pred_score.shape}"
            
            # Test embeddings
            entity_embs, relation_embs = model.get_embeddings()
            assert entity_embs.shape == (num_entities, embedding_dim)
            assert relation_embs.shape == (num_relations, embedding_dim)
            
            print(f"✅ {model_type.upper()} passed all tests!")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} failed: {e}")
            return False
    
    return True

def test_kge_loss():
    """Test KGE loss function"""
    print("\n🧪 Testing KGE Loss...")
    
    try:
        criterion = KGELoss(margin=1.0)
        
        # Create dummy scores
        pos_score = torch.randn(32)
        neg_score = torch.randn(32)
        
        # Compute loss
        loss = criterion(pos_score, neg_score)
        
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        print("✅ KGE Loss passed all tests!")
        return True
        
    except Exception as e:
        print(f"❌ KGE Loss failed: {e}")
        return False

def test_kge_config():
    """Test KGE configuration creation"""
    print("\n🧪 Testing KGE Configuration...")
    
    try:
        # Test with non-existent model (should return None)
        config = create_kge_config_from_model('webqsp', 'transe', 'train')
        if config is None:
            print("✅ KGE config correctly returns None for non-existent model")
        else:
            print("⚠️  KGE config found existing model")
        
        return True
        
    except Exception as e:
        print(f"❌ KGE Configuration failed: {e}")
        return False

def test_retriever_integration():
    """Test Retriever integration with KGE"""
    print("\n🧪 Testing Retriever Integration...")
    
    try:
        # Skip this test if torch_geometric is not available
        try:
            from model.retriever import Retriever
        except ImportError as e:
            if 'torch_geometric' in str(e):
                print("⚠️  Skipping Retriever test (torch_geometric not available)")
                return True
            else:
                raise e
        
        # Create dummy KGE config
        kge_config = {
            'model_type': 'transe',
            'num_entities': 100,
            'num_relations': 50,
            'embedding_dim': 64,
            'margin': 1.0,
            'norm': 1
        }
        
        # Create retriever with KGE
        emb_size = 1024  # GTE embedding size
        retriever = Retriever(
            emb_size=emb_size,
            topic_pe=True,
            DDE_kwargs={'num_rounds': 2, 'num_reverse_rounds': 2},
            kge_config=kge_config
        )
        
        # Test that KGE is enabled
        assert retriever.use_kge == True, "KGE should be enabled"
        assert hasattr(retriever, 'kge_model'), "Retriever should have KGE model"
        assert hasattr(retriever, 'kge_projection'), "Retriever should have KGE projection"
        
        print("✅ Retriever Integration passed all tests!")
        return True
        
    except Exception as e:
        print(f"❌ Retriever Integration failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end KGE integration"""
    print("\n🧪 Testing End-to-End Integration...")
    
    try:
        # Create a simple KGE model
        kge_model = create_kge_model('transe', 10, 5, 32)
        
        # Create dummy triple data
        h_ids = torch.tensor([0, 1, 2])
        r_ids = torch.tensor([0, 1, 0])
        t_ids = torch.tensor([1, 2, 3])
        
        # Get embeddings
        entity_embs, relation_embs = kge_model.get_embeddings()
        
        # Test that we can get embeddings for specific entities
        h_embs = entity_embs[h_ids]
        t_embs = entity_embs[t_ids]
        r_embs = relation_embs[r_ids]
        
        assert h_embs.shape == (3, 32), f"Wrong head embeddings shape: {h_embs.shape}"
        assert t_embs.shape == (3, 32), f"Wrong tail embeddings shape: {t_embs.shape}"
        assert r_embs.shape == (3, 32), f"Wrong relation embeddings shape: {r_embs.shape}"
        
        print("✅ End-to-End Integration passed all tests!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-End Integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting KGE Integration Tests")
    print("=" * 50)
    
    tests = [
        test_kge_models,
        test_kge_loss,
        test_kge_config,
        test_retriever_integration,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KGE integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 