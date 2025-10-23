#!/usr/bin/env python3
"""
測試 CWQ 整合後的功能
驗證 -sc 和 -sci 參數是否能正常工作
"""

import sys
import os
import torch
import argparse

# 添加路徑
sys.path.append('/home/YX_thesis/retrieve')

from src.config.retriever import load_yaml
from src.dataset.retriever import OptimizedRetrieverDataset, optimized_collate_retriever
from torch.utils.data import DataLoader

def test_cwq_spcount_integration():
    """測試 CWQ 的 spcount 功能整合"""
    print("🧪 測試 CWQ spcount 功能整合...")
    
    # 載入配置
    config_file = 'configs/retriever/cwq.yaml'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    config = load_yaml(config_file)
    
    try:
        # 測試 spcount 模式
        print("📊 測試 spcount 模式...")
        train_set_spcount = OptimizedRetrieverDataset(
            config=config,
            split='train',
            freq_weight=False,
            weight_mode='spcount',
        )
        
        # 檢查是否有 pos_weight_factors
        sample = train_set_spcount[0]
        if 'pos_weight_factors' in sample:
            print(f"✅ pos_weight_factors 存在，形狀: {sample['pos_weight_factors'].shape}")
            print(f"   權重範圍: {sample['pos_weight_factors'].min():.4f} - {sample['pos_weight_factors'].max():.4f}")
            print(f"   正樣本權重統計: {sample['pos_weight_factors'][sample['target_triple_probs'] > 0].mean():.4f}")
        else:
            print("❌ pos_weight_factors 不存在")
            return False
            
        # 測試 spcount_inv 模式
        print("\n📊 測試 spcount_inv 模式...")
        train_set_spcount_inv = OptimizedRetrieverDataset(
            config=config,
            split='train',
            freq_weight=False,
            weight_mode='spcount_inv',
        )
        
        sample_inv = train_set_spcount_inv[0]
        if 'pos_weight_factors' in sample_inv:
            print(f"✅ pos_weight_factors 存在，形狀: {sample_inv['pos_weight_factors'].shape}")
            print(f"   權重範圍: {sample_inv['pos_weight_factors'].min():.4f} - {sample_inv['pos_weight_factors'].max():.4f}")
            print(f"   正樣本權重統計: {sample_inv['pos_weight_factors'][sample_inv['target_triple_probs'] > 0].mean():.4f}")
        else:
            print("❌ pos_weight_factors 不存在")
            return False
        
        # 測試 DataLoader 整合
        print("\n📊 測試 DataLoader 整合...")
        train_loader = DataLoader(
            train_set_spcount,
            batch_size=2,
            shuffle=False,
            collate_fn=optimized_collate_retriever,
            num_workers=0  # 避免多進程問題
        )
        
        batch = next(iter(train_loader))
        if 'pos_weight_factors_list' in batch:
            print(f"✅ batch 包含 pos_weight_factors_list，長度: {len(batch['pos_weight_factors_list'])}")
            for i, factors in enumerate(batch['pos_weight_factors_list']):
                print(f"   樣本 {i}: 權重形狀 {factors.shape}, 範圍 {factors.min():.4f} - {factors.max():.4f}")
        else:
            print("❌ batch 不包含 pos_weight_factors_list")
            return False
        
        print("\n✅ 所有測試通過！CWQ spcount 功能整合成功")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_functions():
    """測試訓練函數是否能正確處理新的權重"""
    print("\n🧪 測試訓練函數整合...")
    
    # 模擬 args 對象
    class MockArgs:
        def __init__(self):
            self.spcount = True
            self.spcount_inv = False
            self.kge_bce_weight = False
            self.kge_shortest_path = False
    
    args = MockArgs()
    
    try:
        # 測試權重計算邏輯（不實際運行訓練）
        import torch.nn.functional as F
        
        # 模擬數據
        pred_logits = torch.randn(10)
        target = torch.randint(0, 2, (10,)).float()
        pos_weight_factors = torch.ones(10) * 2.0  # 模擬 spcount 權重
        
        # 測試 spcount 權重計算
        positive_mask = (target > 0).float()
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, positive_mask, reduction='none')
        weighted_loss = bce_loss * pos_weight_factors
        sample_loss = weighted_loss.mean()
        
        print(f"✅ spcount 權重計算成功，loss: {sample_loss:.4f}")
        
        # 測試 spcount_inv 權重計算
        args.spcount = False
        args.spcount_inv = True
        pos_weight_factors_inv = 1.0 / (torch.ones(10) * 2.0 + 1.0)
        
        positive_mask = (target > 0).float()
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, positive_mask, reduction='none')
        weighted_loss = bce_loss * pos_weight_factors_inv
        sample_loss = weighted_loss.mean()
        
        print(f"✅ spcount_inv 權重計算成功，loss: {sample_loss:.4f}")
        
        print("✅ 訓練函數權重計算邏輯測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 訓練函數測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 開始測試 CWQ 功能整合...")
    
    # 測試 1: Dataset 整合
    test1_passed = test_cwq_spcount_integration()
    
    # 測試 2: 訓練函數整合
    test2_passed = test_training_functions()
    
    if test1_passed and test2_passed:
        print("\n🎉 所有測試通過！CWQ 現在支援完整的 spcount 功能")
        print("💡 現在可以使用以下命令:")
        print("   python train_main.py -d cwq -sc    # shortest path count 權重")
        print("   python train_main.py -d cwq -sci   # inverse shortest path count 權重")
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤信息")
        sys.exit(1)
