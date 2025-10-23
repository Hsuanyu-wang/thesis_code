"""
Dataset Adapter System for Multi-format Support

This module provides adapters for different dataset formats while keeping
the existing WebQSP/CWQ processing pipeline unchanged.
"""

import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datasets import load_dataset


class DatasetAdapter(ABC):
    """Base class for dataset adapters."""
    
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.config = config
    
    @abstractmethod
    def load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load raw data for the given split."""
        pass
    
    @abstractmethod
    def convert_to_standard_format(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert raw data to standard format expected by the pipeline."""
        pass
    
    def process_and_save(self, split: str, save_dir: str) -> str:
        """Process raw data and save in standard format."""
        # Load raw data
        raw_data = self.load_raw_data(split)
        
        # Convert to standard format
        processed_data = self.convert_to_standard_format(raw_data)
        
        # Save processed data
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✅ Processed {len(processed_data)} samples for {self.dataset_name}/{split}")
        return save_path


class WebQSPAdapter(DatasetAdapter):
    """Adapter for WebQSP dataset (uses existing HuggingFace format)."""
    
    def load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load WebQSP data from HuggingFace."""
        dataset = load_dataset('ml1996/webqsp', split=split)
        return [sample for sample in dataset]
    
    def convert_to_standard_format(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """WebQSP data is already in the expected format."""
        return raw_data


class CWQAdapter(DatasetAdapter):
    """Adapter for CWQ dataset (uses existing HuggingFace format)."""
    
    def load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load CWQ data from HuggingFace."""
        dataset = load_dataset('rmanluo/RoG-cwq', split=split)
        return [sample for sample in dataset]
    
    def convert_to_standard_format(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CWQ data is already in the expected format."""
        return raw_data


class GrailQAAdapter(DatasetAdapter):
    """Adapter for GrailQA dataset (converts from JSON format)."""
    
    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        super().__init__(dataset_name, config)
        self.raw_data_path = config.get('raw_data_path', f'data_files/{dataset_name}')
    
    def load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load GrailQA data from JSON files."""
        if split == 'train':
            file_path = os.path.join(self.raw_data_path, 'graphquestions_v1_fb15_training_091420.json')
        elif split == 'test':
            file_path = os.path.join(self.raw_data_path, 'graphquestions_v1_fb15_test_091420.json')
        else:
            raise ValueError(f"Split {split} not supported for GrailQA")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def convert_to_standard_format(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert GrailQA JSON format to standard format."""
        processed_samples = []
        
        for sample in raw_data:
            # Extract basic info
            sample_id = f"grailqa_{sample['qid']}"
            question = sample['question']
            
            # Extract answer entities
            a_entity = []
            for answer in sample.get('answer', []):
                if answer.get('entity_name'):
                    a_entity.append(answer['entity_name'])
            
            # Extract entities and relations from graph_query
            graph_query = sample.get('graph_query', {})
            nodes = graph_query.get('nodes', [])
            edges = graph_query.get('edges', [])
            
            # Build entity lists
            text_entity_list = []
            non_text_entity_list = []
            entity_to_id = {}
            
            for node in nodes:
                node_id = node.get('id', '')
                node_type = node.get('node_type', '')
                friendly_name = node.get('friendly_name', '')
                
                if node_type == 'entity' and node_id:
                    # Use Freebase ID as entity identifier
                    if node_id.startswith('m.'):
                        non_text_entity_list.append(node_id)
                        entity_to_id[node_id] = len(text_entity_list) + len(non_text_entity_list) - 1
                    else:
                        text_entity_list.append(friendly_name or node_id)
                        entity_to_id[node_id] = len(text_entity_list) + len(non_text_entity_list) - 1
            
            # Build relation list
            relation_list = []
            rel_to_id = {}
            for edge in edges:
                relation = edge.get('relation', '')
                if relation and relation not in rel_to_id:
                    relation_list.append(relation)
                    rel_to_id[relation] = len(relation_list) - 1
            
            # Build triples
            h_id_list = []
            r_id_list = []
            t_id_list = []
            
            for edge in edges:
                start_idx = edge.get('start', -1)
                end_idx = edge.get('end', -1)
                relation = edge.get('relation', '')
                
                if start_idx >= 0 and end_idx >= 0 and relation in rel_to_id:
                    # Find corresponding node IDs
                    start_node = nodes[start_idx] if start_idx < len(nodes) else None
                    end_node = nodes[end_idx] if end_idx < len(nodes) else None
                    
                    if start_node and end_node:
                        start_id = start_node.get('id', '')
                        end_id = end_node.get('id', '')
                        
                        if start_id in entity_to_id and end_id in entity_to_id:
                            h_id_list.append(entity_to_id[start_id])
                            r_id_list.append(rel_to_id[relation])
                            t_id_list.append(entity_to_id[end_id])
            
            # Extract question entities (entities with question_node=1)
            q_entity = []
            q_entity_id_list = []
            for node in nodes:
                if node.get('question_node', 0) == 1:
                    node_id = node.get('id', '')
                    friendly_name = node.get('friendly_name', '')
                    if node_id in entity_to_id:
                        q_entity.append(friendly_name or node_id)
                        q_entity_id_list.append(entity_to_id[node_id])
            
            # Map answer entities to entity IDs
            a_entity_id_list = []
            for answer in sample.get('answer', []):
                answer_arg = answer.get('answer_argument', '')
                if answer_arg in entity_to_id:
                    a_entity_id_list.append(entity_to_id[answer_arg])
            
            # Create processed sample
            processed_sample = {
                'id': sample_id,
                'question': question,
                'q_entity': q_entity,
                'q_entity_id_list': q_entity_id_list,
                'text_entity_list': text_entity_list,
                'non_text_entity_list': non_text_entity_list,
                'relation_list': relation_list,
                'h_id_list': h_id_list,
                'r_id_list': r_id_list,
                't_id_list': t_id_list,
                'a_entity': a_entity,
                'a_entity_id_list': a_entity_id_list
            }
            
            processed_samples.append(processed_sample)
        
        return processed_samples


# Registry for dataset adapters
DATASET_ADAPTERS = {
    'webqsp': WebQSPAdapter,
    'cwq': CWQAdapter,
    'grailQA': GrailQAAdapter,
}


def get_adapter(dataset_name: str, config: Dict[str, Any]) -> DatasetAdapter:
    """Get the appropriate adapter for the dataset."""
    if dataset_name not in DATASET_ADAPTERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_ADAPTERS.keys())}")
    
    adapter_class = DATASET_ADAPTERS[dataset_name]
    return adapter_class(dataset_name, config)


def ensure_processed_data(dataset_name: str, config: Dict[str, Any], split: str) -> str:
    """Ensure processed data exists for the given dataset and split."""
    processed_dir = f'data_files/{dataset_name}/processed'
    processed_file = os.path.join(processed_dir, f'{split}.pkl')
    
    if os.path.exists(processed_file):
        print(f"✅ Processed data already exists: {processed_file}")
        return processed_file
    
    print(f"🔄 Processing {dataset_name}/{split} data...")
    adapter = get_adapter(dataset_name, config)
    return adapter.process_and_save(split, processed_dir)
