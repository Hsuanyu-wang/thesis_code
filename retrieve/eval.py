import numpy as np
import pandas as pd
import torch
import os
import json
import time
from datetime import datetime, timezone, timedelta

# 檢索評估腳本 (Evaluation script)
def main(args):
    # 1. 載入預測結果與標準答案 (Load prediction and ground truth)
    # pred_dict = torch.load(args.path)
    # gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth')
    pred_dict = torch.load(args.path, weights_only=True)
    gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth', weights_only=True)
    # 2. 計算各種 recall 指標 (Compute recall metrics)
    k_list = [int(k) for k in args.k_list.split(',')]
    
    metric_dict = dict()
    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
    
    for sample_id in pred_dict:
        if len(pred_dict[sample_id]['scored_triples']) == 0:
            continue
        
        h_list, r_list, t_list, _ = zip(*pred_dict[sample_id]['scored_triples'])
        
        a_entity_in_graph = set(pred_dict[sample_id]['a_entity_in_graph'])
        if len(a_entity_in_graph) > 0:
            for k in k_list:
                entities_k = set(h_list[:k] + t_list[:k])
                metric_dict[f'ans_recall@{k}'].append(
                    len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                )
        
        triples = list(zip(h_list, r_list, t_list))
        shortest_path_triples = set(pred_dict[sample_id]['target_relevant_triples'])
        if len(shortest_path_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k) / len(shortest_path_triples)
                )
        
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))
        if len(gpt_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k) / len(gpt_triples)
                )

    for metric, val in metric_dict.items():
        metric_dict[metric] = np.mean(val)
    
    # 3. 輸出 recall 結果表格 (Print recall table)
    # 建立 table_dict，包含所有 K 的 recall 結果，並加入 use_kge_method 與 subgraph_method 欄位
    table_dict = {
        'K': k_list,
        'ans_recall': [
            round(metric_dict.get(f'ans_recall@{k}', 0), 3) for k in k_list
        ],
        'shortest_path_triple_recall': [
            round(metric_dict.get(f'shortest_path_triple_recall@{k}', 0), 3) for k in k_list
        ],
        'gpt_triple_recall': [
            round(metric_dict.get(f'gpt_triple_recall@{k}', 0), 3) for k in k_list
        ],
        # 新增紀錄 use_kge_method 與 subgraph method
        'use_kge': [args.use_kge for _ in k_list],
        'subgraph_method': [args.subgraph_method for _ in k_list]
    }
    df = pd.DataFrame(table_dict)
    print(df.to_string(index=False))
    
    # 4. 儲存詳細與摘要結果 (Save detailed and summary results)
    save_results(args, metric_dict, table_dict, df)

def save_results(args, metric_dict, table_dict, df):
    """
    儲存評估結果到 retrieve_result 資料夾，並將 use_kge 與 subgraph_method 一併寫入
    """
    # 建立 retrieve_result 資料夾
    result_dir = "retrieve_result"
    os.makedirs(result_dir, exist_ok=True)
    
    # 產生時間戳記
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dt = datetime.now(timezone(timedelta(hours=8)))
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    
    # 從路徑中提取實驗名稱
    exp_name = os.path.basename(os.path.dirname(args.path))
    if not exp_name or exp_name == ".":
        exp_name = "unknown_exp"
    
    # 建立檔案名稱
    base_filename = f"{args.dataset}_{exp_name}_{timestamp}"
    
    # 1. 儲存 CSV 表格，包含 use_kge 與 subgraph_method 欄位
    dataset_dir = os.path.join(result_dir, f"{args.dataset}")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV 結果已儲存至: {csv_path}")
    
    # 2. 儲存詳細的 JSON 結果，包含 use_kge_method 與 subgraph_method
    json_result = {
        "dataset": args.dataset,
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "prediction_path": args.path,
        "k_list": [int(k) for k in args.k_list.split(',')],
        "metrics": metric_dict,
        "table_data": table_dict,
        "use_kge": args.use_kge,  # 紀錄 KGE 方法或 False
        "subgraph_method": args.subgraph_method  # 新增 subgraph_method
    }
    
    json_path = os.path.join(result_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"JSON 詳細結果已儲存至: {json_path}")
    
    # 3. 儲存簡化的摘要檔案，動態根據 k_list 生成所有 K 的 recall
    k_list = [int(k) for k in args.k_list.split(',')]
    key_metrics = {}
    
    for k in k_list:
        key_metrics[f"ans_recall@{k}"] = round(metric_dict.get(f'ans_recall@{k}', 0), 3)
        key_metrics[f"shortest_path_triple_recall@{k}"] = round(metric_dict.get(f'shortest_path_triple_recall@{k}', 0), 3)
        key_metrics[f"gpt_triple_recall@{k}"] = round(metric_dict.get(f'gpt_triple_recall@{k}', 0), 3)
    
    summary = {
        "dataset": args.dataset,
        "experiment": exp_name,
        "timestamp": timestamp,
        "use_kge": args.use_kge,  # 紀錄 KGE 方法或 False
        "subgraph_method": args.subgraph_method,  # 新增 subgraph_method
        "key_metrics": key_metrics
    }
    
    summary_path = os.path.join(result_dir, f"{base_filename}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要結果已儲存至: {summary_path}")
    
    # 4. 更新或建立比較表格，傳遞 use_kge 與 subgraph_method
    update_comparison_table(result_dir, args.dataset, summary, k_list)

def update_comparison_table(result_dir, dataset, summary, k_list):
    """
    更新或建立實驗比較表格，並將 use_kge 與 subgraph_method 一併寫入
    所有比較表格都使用 append 模式，不覆蓋舊紀錄
    """
    comparison_file = os.path.join(result_dir, f'{dataset}', f"{dataset}_comparison.csv")
    
    # 準備要新增的資料行，動態根據 k_list 生成所有 K 的 recall
    new_row = {
        "experiment": summary["experiment"],
        "timestamp": summary["timestamp"],
        "use_kge": summary["use_kge"],  # 紀錄 KGE 方法或 False
        "subgraph_method": summary["subgraph_method"],  # 新增 subgraph_method
    }
    
    # 動態添加所有 k 值的 recall 指標
    for k in k_list:
        new_row[f"ans_recall@{k}"] = summary["key_metrics"][f"ans_recall@{k}"]
        new_row[f"shortest_path_triple_recall@{k}"] = summary["key_metrics"][f"shortest_path_triple_recall@{k}"]
        new_row[f"gpt_triple_recall@{k}"] = summary["key_metrics"][f"gpt_triple_recall@{k}"]
    
    # 如果比較檔案存在，使用 append 模式
    if os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
        
        # 檢查新資料的列是否與現有表格的列相同
        new_columns = set(new_row.keys())
        existing_columns = set(comparison_df.columns)
        
        if new_columns == existing_columns:
            # 列相同，直接 append 新記錄
            comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # 列不同，合併欄位並 append
            # 找出所有欄位的聯集
            all_columns = list(existing_columns.union(new_columns))
            
            # 為現有資料添加缺失的欄位（設為 NaN）
            for col in new_columns - existing_columns:
                comparison_df[col] = np.nan
            
            # 為新資料添加缺失的欄位（設為 NaN）
            for col in existing_columns - new_columns:
                new_row[col] = np.nan
            
            # 確保欄位順序一致
            new_row_ordered = {col: new_row.get(col, np.nan) for col in all_columns}
            comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row_ordered])], ignore_index=True)
            
        print(f"比較表格已更新 (append 模式): {comparison_file}")
    else:
        # 建立新的比較表格
        comparison_df = pd.DataFrame([new_row])
        print(f"新的比較表格已建立: {comparison_file}")
    
    # 儲存比較表格
    comparison_df.to_csv(comparison_file, index=False)
    
    # 顯示最新的比較結果
    print("\n=== 實驗比較表格 ===")
    print(comparison_df.to_string(index=False))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'kgqagen'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result, e.g., training result/webqsp_xxx/retrieval_result.pth')
    parser.add_argument('--k_list', type=str, default='5,10,20,50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    # 新增 use_kge_method 與 subgraph_method 參數
    parser.add_argument('--use_kge', type=str, required=True, help='Whether KGE is used (False or specify method, e.g., transe, distmult, etc.)')
    parser.add_argument('--subgraph_method', type=str, required=True, help='Optimal subgraph method (e.g., shortestpath)')
    args = parser.parse_args()
    
    main(args)
