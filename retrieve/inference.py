import os
import torch
import numpy as np

# from tqdm import tqdm  # 移除 tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from src.model.kge_utils import create_kge_config_from_model

# 推論腳本 (Inference script)
@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0')
    
    # 1. 載入資料集 (Load dataset)
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    # KGE integration: load KGE config if enabled
    kge_config = None
    if config.get('kge', {}) and config['kge'].get('enabled', False):
        dataset_name = config['dataset']['name']
        model_type = config['kge']['model_type']
        kge_config = create_kge_config_from_model(dataset_name, model_type, 'train')
        if kge_config is None:
            print("Warning: KGE is enabled but no trained model found. Inference will proceed without KGE.")
        else:
            print(f"KGE configuration loaded: {kge_config['model_type']} with {kge_config['embedding_dim']} dimensions")
    # 2. 載入訓練好的模型 (Load trained model)
    model = Retriever(emb_size, **config['retriever'], kge_config=kge_config).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    for i in range(len(infer_set)):
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
            # 3. 執行推論 (Run inference)
            pred_triple_logits, kge_score = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
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

    root_path = os.path.dirname(args.path)
    # 若路徑包含 webqsp_xxx 這類，則改為 training result/xxx
    if os.path.basename(os.path.dirname(args.path)).startswith('webqsp') or os.path.basename(os.path.dirname(args.path)).startswith('cwq'):
        exp_name = os.path.basename(os.path.dirname(args.path))
        result_root = os.path.join('training result', exp_name)
    else:
        result_root = os.path.dirname(args.path)
    os.makedirs(result_root, exist_ok=True)
    # 4. 儲存推論結果 (Save inference results)
    torch.save(pred_dict, os.path.join(result_root, 'retrieval_result.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()
    
    main(args)
