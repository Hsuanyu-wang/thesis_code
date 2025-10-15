#!/usr/bin/env python3
"""
測試KGE整合功能
"""
import torch
import os
import sys
sys.path.append('/home/YX_thesis/retrieve')

from src.model.kge_integration import KGEIntegration, KGEEnhancedRetriever
from src.model.retriever import Retriever

def test_kge_integration():
    """測試KGE整合功能"""
    print("🧪 開始測試KGE整合功能...")
    
    # 檢查可用的KGE模型
    kge_checkpoints = [
        '/home/pykeen_checkpoints/webqsp_transe.pt',
        '/home/pykeen_checkpoints/webqsp_DistMult.pt'
    ]
    
    available_models = []
    for checkpoint in kge_checkpoints:
        if os.path.exists(checkpoint):
            available_models.append(checkpoint)
            print(f"✅ 找到KGE模型: {checkpoint}")
        else:
            print(f"❌ 未找到KGE模型: {checkpoint}")
    
    if not available_models:
        print("❌ 沒有可用的KGE模型，跳過測試")
        return False
    
    # 使用第一個可用的模型進行測試
    kge_model_path = available_models[0]
    kge_model_type = 'TransE' if 'transe' in kge_model_path.lower() else 'DistMult'
    
    print(f"🔗 使用KGE模型: {kge_model_path} (類型: {kge_model_type})")
    
    try:
        # 創建KGE整合
        kge_integration = KGEIntegration(
            kge_model_path=kge_model_path,
            kge_model_type=kge_model_type,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            freeze_kge=True,
            kge_weight=0.1
        )
        print("✅ KGE整合創建成功")
        
        # 創建基礎Retriever模型
        emb_size = 768
        base_model = Retriever(emb_size, topic_pe=True, DDE_kwargs={'num_rounds': 2, 'num_reverse_rounds': 2})
        print("✅ 基礎Retriever模型創建成功")
        
        # 創建KGE增強模型
        enhanced_model = KGEEnhancedRetriever(
            base_retriever=base_model,
            kge_integration=kge_integration,
            kge_weight=0.1
        )
        print("✅ KGE增強模型創建成功")
        
        # 創建測試數據
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_data = {
            'q_emb': torch.randn(2, emb_size).to(device),
            'entity_embs_list': [
                torch.randn(5, emb_size).to(device),
                torch.randn(6, emb_size).to(device)
            ],
            'relation_embs_list': [
                torch.randn(3, emb_size).to(device),
                torch.randn(4, emb_size).to(device)
            ],
            'topic_entity_one_hot_list': [
                torch.randn(5, 2).to(device),
                torch.randn(6, 2).to(device)
            ],
            'h_id_tensors': [
                torch.randint(0, 5, (8,)).to(device),
                torch.randint(0, 6, (10,)).to(device)
            ],
            'r_id_tensors': [
                torch.randint(0, 3, (8,)).to(device),
                torch.randint(0, 4, (10,)).to(device)
            ],
            't_id_tensors': [
                torch.randint(0, 5, (8,)).to(device),
                torch.randint(0, 6, (10,)).to(device)
            ],
            'target_triple_probs_list': [
                torch.randint(0, 2, (8,)).float().to(device),
                torch.randint(0, 2, (10,)).float().to(device)
            ],
            'num_non_text_entities': [2, 3]
        }
        
        print("✅ 測試數據創建成功")
        
        # 測試前向傳播
        enhanced_model.eval()
        with torch.no_grad():
            base_predictions, kge_outputs = enhanced_model(batch_data)
            print(f"✅ 前向傳播成功")
            print(f"   基礎預測數量: {len(base_predictions)}")
            print(f"   KGE輸出鍵: {list(kge_outputs.keys())}")
            
            if 'kge_scores_list' in kge_outputs:
                kge_scores = kge_outputs['kge_scores_list']
                print(f"   KGE分數數量: {len(kge_scores)}")
                for i, score in enumerate(kge_scores):
                    print(f"   樣本{i} KGE分數形狀: {score.shape}")
        
        # 測試損失計算
        loss = enhanced_model.compute_combined_loss(
            base_predictions,
            kge_outputs.get('kge_scores_list', []),
            batch_data['target_triple_probs_list']
        )
        print(f"✅ 組合損失計算成功: {loss.item():.4f}")
        
        print("🎉 KGE整合測試全部通過！")
        return True
        
    except Exception as e:
        print(f"❌ KGE整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_kge_integration()
    if success:
        print("\n✅ 所有測試通過，KGE整合功能正常")
    else:
        print("\n❌ 測試失敗，請檢查錯誤信息")
