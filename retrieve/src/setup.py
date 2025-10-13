import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sample(device, sample):
    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = sample

    non_blocking = (device.type == 'cuda')
    h_id_tensor = h_id_tensor.to(device, non_blocking=non_blocking)
    r_id_tensor = r_id_tensor.to(device, non_blocking=non_blocking)
    t_id_tensor = t_id_tensor.to(device, non_blocking=non_blocking)
    q_emb = q_emb.to(device, non_blocking=non_blocking)
    entity_embs = entity_embs.to(device, non_blocking=non_blocking)
    relation_embs = relation_embs.to(device, non_blocking=non_blocking)
    topic_entity_one_hot = topic_entity_one_hot.to(device, non_blocking=non_blocking)
    
    return h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list
