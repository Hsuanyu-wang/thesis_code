#!/usr/bin/env python3
"""
KGE整合訓練示例
展示如何使用PyKEEN預訓練的KGE模型來增強Retriever訓練
"""
import subprocess
import os

def run_kge_training_example():
    """運行KGE整合訓練示例"""
    
    print("🚀 KGE整合訓練示例")
    print("=" * 50)
    
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
        print("❌ 沒有可用的KGE模型，請先訓練PyKEEN模型")
        return
    
    # 選擇一個模型進行示例
    kge_model_path = available_models[0]
    kge_model_type = 'TransE' if 'transe' in kge_model_path.lower() else 'DistMult'
    
    print(f"\n🔗 使用KGE模型: {kge_model_path}")
    print(f"📊 模型類型: {kge_model_type}")
    
    # 構建訓練命令
    cmd = [
        'python', '/home/YX_thesis/retrieve/train_with_kge.py',
        '-d', 'webqsp',  # 數據集
        '-m', 'default',  # 方法
        '--batch_size', '2',  # 小批次大小用於測試
        '--num_workers', '4',
        '--grad_accum_steps', '2',
        '--samples_per_epoch', '100',  # 限制樣本數量用於快速測試
        '--kge_model_path', kge_model_path,
        '--kge_model_type', kge_model_type,
        '--kge_weight', '0.1',
        '--freeze_kge',  # 凍結KGE參數
        '-id_sup', 'kge_test'  # 實驗標識
    ]
    
    print(f"\n📝 訓練命令:")
    print(' '.join(cmd))
    
    print(f"\n🎯 訓練配置:")
    print(f"  - 數據集: webqsp")
    print(f"  - KGE模型: {kge_model_type}")
    print(f"  - KGE權重: 0.1")
    print(f"  - 凍結KGE: True")
    print(f"  - 批次大小: 2")
    print(f"  - 梯度累積: 2")
    print(f"  - 每epoch樣本數: 100 (測試用)")
    
    # 詢問是否運行
    response = input("\n❓ 是否運行此訓練示例? (y/n): ")
    if response.lower() != 'y':
        print("❌ 取消訓練")
        return
    
    print(f"\n🚀 開始訓練...")
    try:
        # 運行訓練
        result = subprocess.run(cmd, cwd='/home/YX_thesis/retrieve', 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ 訓練完成!")
            print("📊 訓練輸出:")
            print(result.stdout)
        else:
            print("❌ 訓練失敗!")
            print("📊 錯誤輸出:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ 訓練超時 (1小時)")
    except Exception as e:
        print(f"❌ 運行錯誤: {e}")

def show_usage_examples():
    """顯示使用示例"""
    print("\n📚 使用示例:")
    print("=" * 50)
    
    print("\n1. 基礎訓練 (無KGE):")
    print("python train_with_kge.py -d webqsp --batch_size 4")
    
    print("\n2. 使用TransE KGE模型:")
    print("python train_with_kge.py -d webqsp \\")
    print("  --kge_model_path /home/pykeen_checkpoints/webqsp_transe.pt \\")
    print("  --kge_model_type TransE \\")
    print("  --kge_weight 0.1 \\")
    print("  --freeze_kge")
    
    print("\n3. 使用DistMult KGE模型 (微調):")
    print("python train_with_kge.py -d webqsp \\")
    print("  --kge_model_path /home/pykeen_checkpoints/webqsp_DistMult.pt \\")
    print("  --kge_model_type DistMult \\")
    print("  --kge_weight 0.2 \\")
    print("  --no_freeze_kge")
    
    print("\n4. 快速測試 (小樣本):")
    print("python train_with_kge.py -d webqsp \\")
    print("  --kge_model_path /home/pykeen_checkpoints/webqsp_transe.pt \\")
    print("  --samples_per_epoch 50 \\")
    print("  --batch_size 1 \\")
    print("  -id_sup quick_test")
    
    print("\n5. 完整訓練 (CWQ數據集):")
    print("python train_with_kge.py -d cwq \\")
    print("  --kge_model_path /home/pykeen_checkpoints/webqsp_transe.pt \\")
    print("  --kge_model_type TransE \\")
    print("  --kge_weight 0.15 \\")
    print("  --batch_size 2 \\")
    print("  --grad_accum_steps 4")

if __name__ == '__main__':
    print("🎯 KGE整合訓練系統")
    print("=" * 50)
    
    while True:
        print("\n選擇操作:")
        print("1. 運行KGE訓練示例")
        print("2. 顯示使用示例")
        print("3. 退出")
        
        choice = input("\n請選擇 (1-3): ")
        
        if choice == '1':
            run_kge_training_example()
        elif choice == '2':
            show_usage_examples()
        elif choice == '3':
            print("👋 再見!")
            break
        else:
            print("❌ 無效選擇，請重新輸入")
