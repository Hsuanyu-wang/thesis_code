import torch
import os
from .kge_models import create_kge_model

def load_kge_model(dataset_name, model_type, split='train'):
    """
    載入已訓練的KGE模型 (Load trained KGE model)
    """
    model_path = f'data_files/{dataset_name}/kge/{model_type}/{split}_model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"KGE model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    kge_model = create_kge_model(
        model_type=checkpoint['model_type'],
        num_entities=checkpoint['num_entities'],
        num_relations=checkpoint['num_relations'],
        embedding_dim=checkpoint['embedding_dim']
    )
    
    # Load state dict
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    
    return kge_model, checkpoint

def create_kge_config_from_model(dataset_name, model_type, split='train'):
    """
    Create KGE configuration from trained model
    """
    try:
        _, checkpoint = load_kge_model(dataset_name, model_type, split)
        
        kge_config = {
            'model_type': checkpoint['model_type'],
            'num_entities': checkpoint['num_entities'],
            'num_relations': checkpoint['num_relations'],
            'embedding_dim': checkpoint['embedding_dim'],
            'margin': 1.0,  # Default margin
            'norm': 1       # Default norm for TransE/PTransE
        }
        
        return kge_config
    except FileNotFoundError:
        print(f"Warning: KGE model not found for {dataset_name}/{model_type}/{split}")
        return None

def get_kge_embeddings(kge_model, entity_ids, relation_ids=None):
    """
    根據模型類型取得entity/relation embedding
    For RotatE/ComplEx: 回傳 entity_embeddings, relation_embeddings
    For SimplE: 回傳 entity_head, entity_tail, relation_head, relation_tail
    """
    kge_model.eval()
    with torch.no_grad():
        # RotatE, ComplEx: entity_embeddings, relation_embeddings
        # SimplE: dict of entity_head, entity_tail, relation_head, relation_tail
        embs = kge_model.get_embeddings()
        if isinstance(embs, dict):
            # SimplE
            entity_head_embs = embs['entity_head'][entity_ids]
            entity_tail_embs = embs['entity_tail'][entity_ids]
            if relation_ids is not None:
                relation_head_embs = embs['relation_head'][relation_ids]
                relation_tail_embs = embs['relation_tail'][relation_ids]
                return entity_head_embs, entity_tail_embs, relation_head_embs, relation_tail_embs
            return entity_head_embs, entity_tail_embs
        else:
            entity_embs, relation_embs = embs
            entity_embeddings = entity_embs[entity_ids]
            if relation_ids is not None:
                relation_embeddings = relation_embs[relation_ids]
                return entity_embeddings, relation_embeddings
            return entity_embeddings

def map_entity_ids_to_kge_ids(entity_ids, entity_to_id_mapping):
    """
    Map original entity IDs to KGE internal IDs
    """
    kge_entity_ids = []
    for entity_id in entity_ids:
        if entity_id in entity_to_id_mapping:
            kge_entity_ids.append(entity_to_id_mapping[entity_id])
        else:
            # Use a default ID for unknown entities
            kge_entity_ids.append(0)
    
    return torch.LongTensor(kge_entity_ids)

def map_relation_ids_to_kge_ids(relation_ids, relation_to_id_mapping):
    """
    Map original relation IDs to KGE internal IDs
    """
    kge_relation_ids = []
    for relation_id in relation_ids:
        if relation_id in relation_to_id_mapping:
            kge_relation_ids.append(relation_to_id_mapping[relation_id])
        else:
            # Use a default ID for unknown relations
            kge_relation_ids.append(0)
    
    return torch.LongTensor(kge_relation_ids) 