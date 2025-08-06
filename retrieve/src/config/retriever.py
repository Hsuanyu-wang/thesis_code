import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    topic_pe: bool
    DDE_kwargs: DDEYaml

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class KGEYaml(pydantic.BaseModel):
    enabled: bool = False
    model_type: str = 'transe'
    embedding_dim: int = 256
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.001
    margin: float = 1.0
    num_negatives: int = 1
    norm: int = 1

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    kge: KGEYaml = None
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    # Handle optional KGE configuration
    if 'kge' not in yaml_data:
        yaml_data['kge'] = None
    
    config = RetrieverTrainYaml(**yaml_data).dict()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config
