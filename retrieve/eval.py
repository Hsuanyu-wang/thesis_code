import numpy as np
import pandas as pd
import torch
import os
import time


def evaluate_single(path: str, dataset: str, k_list_str: str, gpt_triples_path: str = None) -> dict:
    pred_dict = torch.load(path)
    gpt_path = gpt_triples_path if gpt_triples_path else f'data_files/{dataset}/gpt_triples.pth'
    gpt_triple_dict = torch.load(gpt_path)
    k_list = [int(k) for k in k_list_str.split(',')]
    
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
        metric_dict[metric] = np.mean(val) if len(val) > 0 else 0.0
    
    exp_id = os.path.basename(os.path.dirname(path))
    
    # 嘗試讀取 training_info.txt 來獲取 KGE 模型信息
    kge_model = 'none'
    exp_dir = os.path.dirname(path)
    training_info_path = os.path.join(exp_dir, 'training_info.txt')
    
    if os.path.exists(training_info_path):
        try:
            with open(training_info_path, 'r') as f:
                content = f.read()
                
            # 查找 KGE model 信息
            for line in content.split('\n'):
                if line.strip().startswith('KGE model:'):
                    kge_model = line.split(':', 1)[1].strip()
                    break
                elif line.strip().startswith('KGE model path:'):
                    # 如果找到了模型路徑，從路徑中提取模型名稱
                    model_path = line.split(':', 1)[1].strip()
                    if 'transe' in model_path.lower():
                        kge_model = 'transe'
                    elif 'distmult' in model_path.lower():
                        kge_model = 'distmult'
                    elif 'complex' in model_path.lower():
                        kge_model = 'complex'
                    break
        except Exception as e:
            print(f"Warning: Failed to read training_info.txt for {exp_id}: {e}")
    
    save_data = {
        'exp_id': exp_id,
        'dataset': dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600)),
        'kge_model': kge_model
    }
    
    metrics_to_save = ['ans_recall', 'shortest_path_triple_recall', 'gpt_triple_recall']
    for metric_name in metrics_to_save:
        for k in k_list:
            save_data[f'{metric_name}@{k}'] = round(metric_dict[f'{metric_name}@{k}'], 4)
    return save_data


def save_table_format_csv(df, csv_path, k_list):
    """儲存表格格式的 CSV 檔案"""
    # 創建表格格式的標題行
    header_row1 = ['exp_id', 'dataset', 'timestamp', 'kge_model']
    header_row2 = ['', '', '', '']
    
    # 添加 metrics 標題
    metrics = ['ans', 'sp_triple', 'gpt_triple']
    for metric in metrics:
        for k in k_list:
            header_row1.append(metric)
            header_row2.append(f'@{k}')
    
    # 創建表格格式的 CSV
    with open(csv_path, 'w') as f:
        # 寫入第一行（主要類別）
        f.write(','.join(header_row1) + '\n')
        # 寫入第二行（@k 值）
        f.write(','.join(header_row2) + '\n')
        
        # 寫入數據行
        for _, row in df.iterrows():
            data_row = [
                str(row['exp_id']),
                str(row['dataset']),
                str(row['timestamp']),
                str(row['kge_model'])
            ]
            
            # 添加 ans 數據
            for k in k_list:
                data_row.append(str(row.get(f'ans_recall@{k}', 0.0)))
            
            # 添加 sp_triple 數據
            for k in k_list:
                data_row.append(str(row.get(f'shortest_path_triple_recall@{k}', 0.0)))
            
            # 添加 gpt_triple 數據
            for k in k_list:
                data_row.append(str(row.get(f'gpt_triple_recall@{k}', 0.0)))
            
            f.write(','.join(data_row) + '\n')


def main(args):
    # 確保儲存目錄與 CSV 存在
    save_dir = os.path.join('/home/YX_thesis/retrieve/results/evaluation', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'evaluation_results.csv')
    table_csv_path = os.path.join(save_dir, 'evaluation_results_table_format.csv')

    if getattr(args, 'batch_dir', None):
        root = args.batch_dir
        if not os.path.isdir(root):
            print(f"Batch directory not found: {root}")
            return
        # 讀取既有 exp_id，避免重複
        existing_ids = set()
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if 'exp_id' in existing_df.columns:
                    existing_ids = set(existing_df['exp_id'].astype(str).tolist())
            except Exception:
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()
        rows = []
        # 遍歷子資料夾，尋找 retrieval_result.pth
        for sub in sorted(os.listdir(root)):
            folder = os.path.join(root, sub)
            if not os.path.isdir(folder):
                continue
            exp_id = os.path.basename(folder)
            if exp_id in existing_ids:
                print(f"Skip evaluated exp: {exp_id}")
                continue
            cand = os.path.join(folder, 'retrieval_result.pth')
            if not os.path.exists(cand):
                print(f"No retrieval_result.pth in {folder}, skip")
                continue
            try:
                save_data = evaluate_single(cand, args.dataset, args.k_list, args.gpt_triples_path)
                rows.append(save_data)
                print(f"Evaluated: {exp_id}")
            except Exception as e:
                print(f"Failed {exp_id}: {e}")
                continue
        # append 結果
        if rows:
            new_df = pd.DataFrame(rows)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
            combined_df.to_csv(csv_path, index=False)
            
            # 生成表格格式的 CSV
            k_list = [int(k) for k in args.k_list.split(',')]
            save_table_format_csv(combined_df, table_csv_path, k_list)
            
            print(f"Appended {len(rows)} rows to: {csv_path}")
            print(f"Table format saved to: {table_csv_path}")
        else:
            print("No new experiments to evaluate.")
        return

    # 單檔模式
    save_data = evaluate_single(args.path, args.dataset, args.k_list, args.gpt_triples_path)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if 'exp_id' in existing_df.columns and save_data['exp_id'] in set(existing_df['exp_id'].astype(str)):
            print(f"Experiment {save_data['exp_id']} already exists. Skipped.")
            return
        combined_df = pd.concat([existing_df, pd.DataFrame([save_data])], ignore_index=True)
    else:
        combined_df = pd.DataFrame([save_data])
    
    combined_df.to_csv(csv_path, index=False)
    
    # 生成表格格式的 CSV
    k_list = [int(k) for k in args.k_list.split(',')]
    save_table_format_csv(combined_df, table_csv_path, k_list)
    
    print(f"Results saved to: {csv_path}")
    print(f"Table format saved to: {table_csv_path}")

    # 顯示當前實驗結果
    metric_mapping = {
        'ans_recall': 'ans',
        'shortest_path_triple_recall': 'sp_triple',
        'gpt_triple_recall': 'gpt_triple'
    }
    columns = pd.MultiIndex.from_product([list(metric_mapping.values()), [f'@{k}' for k in k_list]])
    data = []
    for metric_name in metric_mapping.keys():
        for k in k_list:
            data.append(save_data.get(f'{metric_name}@{k}', 0.0))
    df = pd.DataFrame([data], columns=columns)
    print(f"\nExperiment ID: {save_data['exp_id']}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=False,
                        help='Path to retrieval result')
    parser.add_argument('--k_list', type=str, default='5,10,30,50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    parser.add_argument('--batch_dir', type=str, default=None,
                        help='If set, iterate folders under this directory and append results for new experiments.')
    parser.add_argument('--gpt_triples_path', type=str, default=None,
                        help='Override path to GPT triples; if set, use this file instead of data_files/{dataset}/gpt_triples.pth')
    args = parser.parse_args()
    
    if not args.path and not args.batch_dir:
        # default batch root
        args.batch_dir = '/home/YX_thesis/retrieve/results/training'
    
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start evaluation retriever {ts_start} ==========")
    
    main(args)

    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End evaluation retriever {ts_end} ==========")
    print(f"Retriever evaluation time: {end_time - start_time:.2f} seconds")
