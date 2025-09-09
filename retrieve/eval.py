import numpy as np
import pandas as pd
import torch
import os
import time


def evaluate_single(path: str, dataset: str, k_list_str: str) -> dict:
    pred_dict = torch.load(path)
    gpt_triple_dict = torch.load(f'data_files/{dataset}/gpt_triples.pth')
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
    
    save_data = {
        'exp_id': exp_id,
        'dataset': dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600)),
        'kge_model': 'none'
    }
    
    metrics_to_save = ['ans_recall', 'shortest_path_triple_recall', 'gpt_triple_recall']
    for metric_name in metrics_to_save:
        for k in k_list:
            save_data[f'{metric_name}@{k}'] = round(metric_dict[f'{metric_name}@{k}'], 4)
    return save_data


def main(args):
    # 確保儲存目錄與 CSV 存在
    save_dir = os.path.join('/home/SubgraphRAG/retrieve/results/evaluation', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'evaluation_results.csv')

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
                save_data = evaluate_single(cand, args.dataset, args.k_list)
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
            print(f"Appended {len(rows)} rows to: {csv_path}")
        else:
            print("No new experiments to evaluate.")
        return

    # 單檔模式
    save_data = evaluate_single(args.path, args.dataset, args.k_list)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if 'exp_id' in existing_df.columns and save_data['exp_id'] in set(existing_df['exp_id'].astype(str)):
            print(f"Experiment {save_data['exp_id']} already exists. Skipped.")
            return
        combined_df = pd.concat([existing_df, pd.DataFrame([save_data])], ignore_index=True)
    else:
        combined_df = pd.DataFrame([save_data])
    combined_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # 顯示當前實驗結果
    k_list = [int(k) for k in args.k_list.split(',')]
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
    args = parser.parse_args()
    
    if not args.path and not args.batch_dir:
        # default batch root
        args.batch_dir = '/home/SubgraphRAG/retrieve/results/training'
    
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start evaluation retriever {ts_start} ==========")
    
    main(args)

    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End evaluation retriever {ts_end} ==========")
    print(f"Retriever evaluation time: {end_time - start_time:.2f} seconds")