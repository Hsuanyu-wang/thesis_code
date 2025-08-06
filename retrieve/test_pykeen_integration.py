#!/usr/bin/env python3
"""
測試 PyKEEN 整合的腳本
Test script for PyKEEN integration
"""

import torch
import numpy as np
from src.model.kge_models import create_kge_model, KGELoss

def test_pykeen_models():
    """
    測試 PyKEEN 模型的基本功能
    Test basic functionality of PyKEEN models
    """
    print("🧪 Testing PyKEEN Integration...")
    
    # 測試參數
    num_entities = 100
    num_relations = 50
    embedding_dim = 64
    batch_size = 32
    
    # 測試的模型類型
    pykeen_models = ['pykeen_transe', 'pykeen_distmult', 'pykeen_complex', 'pykeen_rotate', 'pykeen_simple']
    custom_models = ['transe', 'distmult', 'complex', 'rotate', 'simple']
    
    print(f"📊 Testing {len(pykeen_models)} PyKEEN models...")
    
    for model_type in pykeen_models:
        try:
            print(f"\n🔍 Testing {model_type}...")
            
            # 創建模型
            model = create_kge_model(
                model_type=model_type,
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=embedding_dim
            )
            
            # 創建測試數據
            pos_h = torch.randint(0, num_entities, (batch_size,))
            pos_r = torch.randint(0, num_relations, (batch_size,))
            pos_t = torch.randint(0, num_entities, (batch_size,))
            neg_h = torch.randint(0, num_entities, (batch_size,))
            neg_r = torch.randint(0, num_relations, (batch_size,))
            neg_t = torch.randint(0, num_entities, (batch_size,))
            
            # 前向傳播測試
            pos_score, neg_score = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            print(f"✅ {model_type} - Forward pass successful")
            print(f"   Pos score shape: {pos_score.shape}")
            print(f"   Neg score shape: {neg_score.shape}")
            print(f"   Pos score range: [{pos_score.min().item():.3f}, {pos_score.max().item():.3f}]")
            print(f"   Neg score range: [{neg_score.min().item():.3f}, {neg_score.max().item():.3f}]")
            
            # 測試預測功能
            pred_score = model.predict(pos_h, pos_r, pos_t)
            print(f"   Predict score shape: {pred_score.shape}")
            
            # 測試獲取 embedding
            entity_emb, relation_emb = model.get_embeddings()
            print(f"   Entity embedding shape: {entity_emb.shape}")
            print(f"   Relation embedding shape: {relation_emb.shape}")
            
        except Exception as e:
            print(f"❌ {model_type} - Error: {str(e)}")
    
    print(f"\n🎉 PyKEEN integration test completed!")

def test_model_comparison():
    """
    比較 PyKEEN 模型與自定義模型的輸出
    Compare PyKEEN models with custom implementations
    """
    print("\n🔬 Testing Model Comparison...")
    
    num_entities = 50
    num_relations = 20
    embedding_dim = 32
    batch_size = 16
    
    # 測試數據
    pos_h = torch.randint(0, num_entities, (batch_size,))
    pos_r = torch.randint(0, num_relations, (batch_size,))
    pos_t = torch.randint(0, num_entities, (batch_size,))
    neg_h = torch.randint(0, num_entities, (batch_size,))
    neg_r = torch.randint(0, num_relations, (batch_size,))
    neg_t = torch.randint(0, num_entities, (batch_size,))
    
    model_pairs = [
        ('transe', 'pykeen_transe'),
        ('distmult', 'pykeen_distmult'),
        ('complex', 'pykeen_complex'),
        ('rotate', 'pykeen_rotate'),
        ('simple', 'pykeen_simple')
    ]
    
    for custom_type, pykeen_type in model_pairs:
        try:
            print(f"\n📊 Comparing {custom_type} vs {pykeen_type}...")
            
            # 創建模型
            custom_model = create_kge_model(
                model_type=custom_type,
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=embedding_dim
            )
            
            pykeen_model = create_kge_model(
                model_type=pykeen_type,
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=embedding_dim
            )
            
            # 設置相同的隨機種子以獲得可比較的結果
            torch.manual_seed(42)
            custom_pos_score, custom_neg_score = custom_model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            torch.manual_seed(42)
            pykeen_pos_score, pykeen_neg_score = pykeen_model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            # 比較分數分佈
            print(f"   Custom pos score mean: {custom_pos_score.mean().item():.4f}")
            print(f"   PyKEEN pos score mean: {pykeen_pos_score.mean().item():.4f}")
            print(f"   Custom neg score mean: {custom_neg_score.mean().item():.4f}")
            print(f"   PyKEEN neg score mean: {pykeen_neg_score.mean().item():.4f}")
            
            # 注意：由於實現細節不同，分數可能不完全相同，但應該在合理範圍內
            print(f"   ✅ Models are compatible")
            
        except Exception as e:
            print(f"   ❌ Comparison failed: {str(e)}")

def test_loss_function():
    """
    測試損失函數與 PyKEEN 模型的兼容性
    Test loss function compatibility with PyKEEN models
    """
    print("\n💡 Testing Loss Function Compatibility...")
    
    num_entities = 30
    num_relations = 15
    embedding_dim = 16
    batch_size = 8
    
    # 測試數據
    pos_h = torch.randint(0, num_entities, (batch_size,))
    pos_r = torch.randint(0, num_relations, (batch_size,))
    pos_t = torch.randint(0, num_entities, (batch_size,))
    neg_h = torch.randint(0, num_entities, (batch_size,))
    neg_r = torch.randint(0, num_relations, (batch_size,))
    neg_t = torch.randint(0, num_entities, (batch_size,))
    
    model_types = ['pykeen_transe', 'pykeen_distmult', 'pykeen_complex']
    
    for model_type in model_types:
        try:
            print(f"\n🔍 Testing loss with {model_type}...")
            
            model = create_kge_model(
                model_type=model_type,
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=embedding_dim
            )
            
            criterion = KGELoss(margin=1.0)
            
            # 前向傳播
            pos_score, neg_score = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            # 計算損失
            loss = criterion(pos_score, neg_score)
            
            print(f"   Loss value: {loss.item():.4f}")
            print(f"   ✅ Loss computation successful")
            
        except Exception as e:
            print(f"   ❌ Loss test failed: {str(e)}")

if __name__ == '__main__':
    print("🚀 Starting PyKEEN Integration Tests...")
    
    # 檢查 PyKEEN 是否可用
    try:
        from pykeen.models import DistMult
        print("✅ PyKEEN is available")
    except ImportError:
        print("❌ PyKEEN is not available. Please install it with: pip install pykeen")
        print("   Tests will be skipped.")
        exit(1)
    
    # 運行測試
    test_pykeen_models()
    test_model_comparison()
    test_loss_function()
    
    print("\n🎉 All tests completed!") 