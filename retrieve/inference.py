import os
import torch
import time
import random

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from src.config.retriever import load_yaml

@torch.no_grad()
def _run_no_train_inference(dataset: str, ranking_method: str, max_K: int = 500):
    """運行 no-train 模式的推理，直接使用 shortest path triplets 進行 ranking"""
    device = torch.device(f'cuda:0')
    
    # Load config
    config_file = f'configs/retriever/{dataset}.yaml'
    config = load_yaml(config_file)
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
        
        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        
        # Get all triplets from shortest paths
        all_triplet_ids = target_triple_probs.nonzero().squeeze(-1).tolist()
        
        if len(all_triplet_ids) == 0:
            # No triplets available
            top_K_triples = []
        else:
            # Apply different ranking methods
            if ranking_method == 'random':
                # Method 1: Random selection of all triplets (up to max_K)
                random.shuffle(all_triplet_ids)
                selected_triplet_ids = all_triplet_ids[:max_K]
                scores = [1.0] * len(selected_triplet_ids)  # Equal scores
                
            elif ranking_method == 'sp_count':
                # Method 2: Rank by SP count (higher count = higher rank)
                triplet_counts = []
                for tid in all_triplet_ids:
                    count = target_triple_probs[tid].item()
                    triplet_counts.append((tid, count))
                # Sort by count (descending)
                triplet_counts.sort(key=lambda x: x[1], reverse=True)
                selected_triplet_ids = [tid for tid, _ in triplet_counts[:max_K]]
                scores = [count for _, count in triplet_counts[:max_K]]
                
            elif ranking_method == 'sp_count_inv':
                # Method 3: Rank by inverse SP count (lower count = higher rank)
                triplet_counts = []
                for tid in all_triplet_ids:
                    count = target_triple_probs[tid].item()
                    # Use inverse count (1/count) for ranking
                    inv_count = 1.0 / (count + 1.0)  # Add 1 to avoid division by zero
                    triplet_counts.append((tid, inv_count))
                # Sort by inverse count (descending)
                triplet_counts.sort(key=lambda x: x[1], reverse=True)
                selected_triplet_ids = [tid for tid, _ in triplet_counts[:max_K]]
                scores = [inv_count for _, inv_count in triplet_counts[:max_K]]
                
            elif ranking_method == 'freq_weight':
                # Method 4: Similar to freq_weight (use frequency-based weighting)
                triplet_counts = []
                for tid in all_triplet_ids:
                    count = target_triple_probs[tid].item()
                    # Use frequency weight (count + 1)
                    freq_weight = count + 1.0
                    triplet_counts.append((tid, freq_weight))
                # Sort by frequency weight (descending)
                triplet_counts.sort(key=lambda x: x[1], reverse=True)
                selected_triplet_ids = [tid for tid, _ in triplet_counts[:max_K]]
                scores = [freq_weight for _, freq_weight in triplet_counts[:max_K]]
                
            elif ranking_method == 'freq_weight_inv':
                # Method 5: Similar to freq_weight_inv (use inverse frequency-based weighting)
                triplet_counts = []
                for tid in all_triplet_ids:
                    count = target_triple_probs[tid].item()
                    # Use inverse frequency weight
                    inv_freq_weight = 1.0 / (count + 1.0)
                    triplet_counts.append((tid, inv_freq_weight))
                # Sort by inverse frequency weight (descending)
                triplet_counts.sort(key=lambda x: x[1], reverse=True)
                selected_triplet_ids = [tid for tid, _ in triplet_counts[:max_K]]
                scores = [inv_freq_weight for _, inv_freq_weight in triplet_counts[:max_K]]
            
            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")
            
            # Convert triplet IDs to actual triplets
            top_K_triples = []
            for j, triple_id in enumerate(selected_triplet_ids):
                # Get entity and relation IDs from the tensor
                h_id = h_id_tensor[triple_id].item()
                r_id = r_id_tensor[triple_id].item()
                t_id = t_id_tensor[triple_id].item()
                
                top_K_triples.append((
                    entity_list[h_id],
                    relation_list[r_id],
                    entity_list[t_id],
                    scores[j]
                ))
        
        # Get target relevant triples
        target_relevant_triple_ids = target_triple_probs.nonzero().reshape(-1).tolist()
        target_relevant_triples = []
        for triple_id in target_relevant_triple_ids:
            h_id = h_id_tensor[triple_id].item()
            r_id = r_id_tensor[triple_id].item()
            t_id = t_id_tensor[triple_id].item()
            target_relevant_triples.append((
                entity_list[h_id],
                relation_list[r_id],
                entity_list[t_id],
            ))
        
        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }
        
        pred_dict[raw_sample['id']] = sample_dict
    
    return pred_dict

@torch.no_grad()
def _run_inference(path: str, max_K: int = 500):
    device = torch.device(f'cuda:0')
    
    # cpt = torch.load(path, map_location='cpu')
    cpt = torch.load(path, map_location='cpu', weights_only=False)
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model((
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot
            ))
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }
        
        pred_dict[raw_sample['id']] = sample_dict

    root_path = os.path.dirname(path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))

@torch.no_grad()
def main(args):
    # Handle no-train mode
    if args.no_train:
        if args.dataset is None:
            raise ValueError('Please provide --dataset for no-train mode')
        
        print(f"🚀 Running no-train inference...")
        print(f"   Dataset: {args.dataset}")
        print(f"   Ranking method: {args.ranking_method}")
        print(f"   Max-K triplets: {args.max_K}")
        
        pred_dict = _run_no_train_inference(
            dataset=args.dataset,
            ranking_method=args.ranking_method,
            max_K=args.max_K
        )
        
        # Save results to training directory structure
        ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
        exp_name = f'no_train_{args.ranking_method}_{ts}'
        save_dir = os.path.join('/home/YX_thesis/retrieve/results/training', args.dataset, exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        result_path = os.path.join(save_dir, 'retrieval_result.pth')
        torch.save(pred_dict, result_path)
        
        # Save training_info.txt to match the expected format for eval.py
        info_path = os.path.join(save_dir, 'training_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))}\n")
            f.write(f"No-train mode: True\n")
            f.write(f"Ranking method: {args.ranking_method}\n")
            f.write(f"Max-K triplets: {args.max_K}\n")
            f.write(f"Total samples processed: {len(pred_dict)}\n")
            f.write(f"Results saved to: {result_path}\n")
            f.write("\n")
            
            f.write("Configuration:\n")
            f.write(f"  No-train mode: True\n")
            f.write(f"  Ranking method: {args.ranking_method}\n")
            f.write(f"  Max-K triplets: {args.max_K}\n")
            f.write(f"  KGE model: none\n")
        
        print(f"✅ No-train inference completed!")
        print(f"🎯 Results saved to: {result_path}")
        return
    
    # Batch mode: iterate over subfolders and run inference per checkpoint
    if args.batch_dir is not None:
        root = args.batch_dir
        if not os.path.isdir(root):
            print(f"Batch directory not found: {root}")
            return
        subfolders = sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        succeeded = []
        skipped = []
        failed = []
        for folder in subfolders:
            cpt_path = os.path.join(folder, 'cpt.pth')
            out_path = os.path.join(folder, 'retrieval_result.pth')
            if not os.path.exists(cpt_path):
                failed.append((folder, 'missing cpt.pth'))
                continue
            if os.path.exists(out_path) and not args.overwrite:
                skipped.append(folder)
                continue
            try:
                _run_inference(cpt_path, args.max_K)
                succeeded.append(folder)
            except Exception as e:
                failed.append((folder, str(e)))
                continue
        print("=== Batch inference summary ===")
        print(f"Root: {root}")
        print(f"Succeeded: {len(succeeded)}")
        print(f"Skipped (exists): {len(skipped)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("Failures:")
            for f, reason in failed:
                print(f"  - {f}: {reason}")
        return

    # Single-file mode (original behavior)
    if args.path is None:
        raise ValueError('Please provide -p/--path or --batch_dir')
    _run_inference(args.path, args.max_K)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=False,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    parser.add_argument('--batch_dir', type=str, default=None,
                        help='Run inference for all experiment folders under this directory (expects cpt.pth in each).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing retrieval_result.pth if present in a folder during batch mode.')
    
    # No-train mode arguments
    parser.add_argument('--no_train', action='store_true', help='Skip model loading and directly use shortest path triplets for ranking.')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], help='Dataset name for no-train mode.')
    parser.add_argument('--ranking_method', type=str, default='random', 
                       choices=['random', 'sp_count', 'sp_count_inv', 'freq_weight', 'freq_weight_inv'], 
                       help='Ranking method for no-train mode.')
    
    args = parser.parse_args()
    
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start inference retriever {ts_start} ==========")
    
    main(args)
    
    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End inference retriever {ts_end} ==========")
    print(f"Retriever inference time: {end_time - start_time:.2f} seconds")