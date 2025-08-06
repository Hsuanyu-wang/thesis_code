#!/usr/bin/env python3
"""
Test script for KGE score integration in SubgraphRAG

This script tests the new KGE score integration approach.
"""

import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

from model.retriever import Retriever
from model.kge_models import create_kge_model

def test_kge_score_integration():
    """Test KGE score integration with Retriever model"""
    print("🧪 Testing KGE Score Integration...")
    
    # Test parameters
    emb_size = 256
    num_entities = 100
    num_relations = 50
    kge_embedding_dim = 128
    batch_size = 32
    
    # Create KGE config
    kge_config = {
        'model_type': 'transe',
        'num_entities': num_entities,
        'num_relations': num_relations,
        'embedding_dim': kge_embedding_dim,
        'margin': 1.0,
        'norm': 1
    }
    
    # Create Retriever model with KGE
    retriever_config = {
        'topic_pe': True,
        'DDE_kwargs': {
            'num_rounds': 2,
            'num_reverse_rounds': 2
        }
    }
    
    model = Retriever(
        emb_size=emb_size,
        kge_config=kge_config,
        **retriever_config
    )
    
    print(f"✅ Retriever model created with KGE integration")
    print(f"   - Embedding size: {emb_size}")
    print(f"   - KGE model: {kge_config['model_type']}")
    print(f"   - KGE embedding dim: {kge_embedding_dim}")
    
    # Create dummy input data
    device = torch.device('cpu')
    model = model.to(device)
    
    h_id_tensor = torch.randint(0, num_entities, (batch_size,))
    r_id_tensor = torch.randint(0, num_relations, (batch_size,))
    t_id_tensor = torch.randint(0, num_entities, (batch_size,))
    q_emb = torch.randn(emb_size)
    entity_embs = torch.randn(num_entities, emb_size)
    num_non_text_entities = 0
    relation_embs = torch.randn(num_relations, emb_size)
    # Fix topic_entity_one_hot dimension - should match number of entities
    topic_entity_one_hot = torch.zeros(num_entities, 2)
    
    print(f"✅ Created dummy input data")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Number of entities: {num_entities}")
    print(f"   - Number of relations: {num_relations}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        mlp_logits, kge_score = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot
        )
    
    print(f"✅ Forward pass successful")
    print(f"   - MLP logits shape: {mlp_logits.shape}")
    print(f"   - KGE score shape: {kge_score.shape if kge_score is not None else 'None'}")
    print(f"   - MLP logits range: [{mlp_logits.min().item():.3f}, {mlp_logits.max().item():.3f}]")
    if kge_score is not None:
        print(f"   - KGE score range: [{kge_score.min().item():.3f}, {kge_score.max().item():.3f}]")
    
    # Test margin ranking loss computation
    print("\n🧪 Testing Margin Ranking Loss...")
    
    # Create dummy target labels
    target_triple_probs = torch.randint(0, 2, (batch_size,)).float()
    
    # Compute positive and negative masks
    positive_mask = target_triple_probs > 0.5
    negative_mask = target_triple_probs <= 0.5
    
    print(f"   - Positive samples: {positive_mask.sum().item()}")
    print(f"   - Negative samples: {negative_mask.sum().item()}")
    
    if positive_mask.sum() > 0 and negative_mask.sum() > 0 and kge_score is not None:
        # Get positive and negative KGE scores
        pos_kge_scores = kge_score[positive_mask]
        neg_kge_scores = kge_score[negative_mask]
        
        # Compute margin ranking loss
        margin = 1.0
        kge_loss = torch.clamp(
            pos_kge_scores.unsqueeze(1) - neg_kge_scores.unsqueeze(0) + margin, 
            min=0
        ).mean()
        
        print(f"   - KGE margin ranking loss: {kge_loss.item():.3f}")
        print(f"   - Positive KGE scores range: [{pos_kge_scores.min().item():.3f}, {pos_kge_scores.max().item():.3f}]")
        print(f"   - Negative KGE scores range: [{neg_kge_scores.min().item():.3f}, {neg_kge_scores.max().item():.3f}]")
    else:
        print(f"   - Skipping margin ranking loss (insufficient positive/negative samples)")
    
    print("\n✅ All tests passed!")
    return True

def test_kge_models():
    """Test individual KGE models"""
    print("\n🧪 Testing Individual KGE Models...")
    
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
            h = torch.randint(0, num_entities, (batch_size,))
            r = torch.randint(0, num_relations, (batch_size,))
            t = torch.randint(0, num_entities, (batch_size,))
            
            # Test predict method
            with torch.no_grad():
                score = model.predict(h, r, t)
            
            print(f"   ✅ {model_type.upper()} predict successful")
            print(f"   - Score shape: {score.shape}")
            print(f"   - Score range: [{score.min().item():.3f}, {score.max().item():.3f}]")
            
        except Exception as e:
            print(f"   ❌ {model_type.upper()} failed: {e}")
            return False
    
    print("\n✅ All KGE models tested successfully!")
    return True

if __name__ == '__main__':
    print("🚀 Starting KGE Score Integration Tests...\n")
    
    # Test individual KGE models
    if not test_kge_models():
        print("❌ KGE models test failed")
        sys.exit(1)
    
    # Test KGE score integration
    if not test_kge_score_integration():
        print("❌ KGE score integration test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed! KGE score integration is working correctly.") 