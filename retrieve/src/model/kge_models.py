import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add PyKEEN imports
try:
    from pykeen.models import DistMult, TransE, ComplEx, RotatE, SimplE
    from pykeen.triples import TriplesFactory
    from pykeen.losses import MarginRankingLoss
    PYKEEEN_AVAILABLE = True
except ImportError:
    PYKEEEN_AVAILABLE = False
    print("Warning: PyKEEN not available. Custom implementations will be used.")

class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    Paper: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, norm=1):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Normalize relation embeddings
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        Forward pass for training
        Args:
            pos_h, pos_r, pos_t: positive triples (head, relation, tail)
            neg_h, neg_r, neg_t: negative triples (head, relation, tail)
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        """
        Compute TransE score: ||h + r - t||
        """
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        score = h_emb + r_emb - t_emb
        score = torch.norm(score, p=self.norm, dim=1)
        
        return score
    
    def predict(self, h, r, t):
        """
        Predict score for a single triple or batch of triples
        """
        return self._score(h, r, t)
    
    def get_embeddings(self):
        """
        Get entity and relation embeddings
        """
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

class DistMult(nn.Module):
    """
    DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    Paper: https://arxiv.org/abs/1412.6575
    """
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        Forward pass for training
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        """
        Compute DistMult score: <h, r, t>
        """
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        score = torch.sum(h_emb * r_emb * t_emb, dim=1)
        
        return score
    
    def predict(self, h, r, t):
        """
        Predict score for a single triple or batch of triples
        """
        return self._score(h, r, t)
    
    def get_embeddings(self):
        """
        Get entity and relation embeddings
        """
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

class PTransE(nn.Module):
    """
    PTransE: Learning Entity and Relation Embeddings for Knowledge Graph Completion
    Paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, norm=1):
        super(PTransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Normalize relation embeddings
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        Forward pass for training
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        """
        Compute PTransE score: ||h + r - t||
        Similar to TransE but with different training strategy
        """
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        score = h_emb + r_emb - t_emb
        score = torch.norm(score, p=self.norm, dim=1)
        
        return score
    
    def predict(self, h, r, t):
        """
        Predict score for a single triple or batch of triples
        """
        return self._score(h, r, t)
    
    def get_embeddings(self):
        """
        Get entity and relation embeddings
        """
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

class KGELoss(nn.Module):
    """
    Loss function for KGE models
    """
    def __init__(self, margin=1.0):
        super(KGELoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        """
        Compute margin ranking loss
        """
        loss = torch.clamp(pos_score - neg_score + self.margin, min=0)
        return loss.mean()

class RotatE(nn.Module):
    """
    RotatE: Reasoning over Entities, Relations, and Time in Knowledge Graphs
    Paper: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=6.0):
        super(RotatE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = 2.0
        # 計算embedding範圍 (embedding range for initialization)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / embedding_dim]), requires_grad=False
        )
        # entity用複數表示，維度*2 (real+imag)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)  # complex: real+imag
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        # 均勻初始化 (Uniform initialization)
        nn.init.uniform_(self.entity_embeddings.weight.data, -self.embedding_range.item(), self.embedding_range.item())
        nn.init.uniform_(self.relation_embeddings.weight.data, -self.embedding_range.item(), self.embedding_range.item())

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        前向傳播，計算正負樣本分數 (Forward pass for training)
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score

    def _score(self, h, r, t):
        """
        RotatE打分公式：將關係視為複數平面上的旋轉 (RotatE scoring function)
        """
        h_emb = self.entity_embeddings(h).view(-1, self.embedding_dim, 2)  # (batch, dim, 2)
        t_emb = self.entity_embeddings(t).view(-1, self.embedding_dim, 2)
        r_emb = self.relation_embeddings(r)
        pi = 3.14159265358979323846
        # phase_r: 關係的旋轉角度 (relation rotation phase)
        phase_r = r_emb / (self.embedding_range.item() / pi)
        re_h, im_h = h_emb[..., 0], h_emb[..., 1]
        re_t, im_t = t_emb[..., 0], t_emb[..., 1]
        re_r, im_r = torch.cos(phase_r), torch.sin(phase_r)
        # 複數乘法 (complex multiplication)
        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r
        # 與tail的距離 (distance to tail)
        re_diff = re_score - re_t
        im_diff = im_score - im_t
        score = torch.stack([re_diff, im_diff], dim=0)
        score = score.norm(dim=0).sum(dim=1)  # L2 norm
        return score

    def predict(self, h, r, t):
        """
        預測單一三元組分數或批次三元組分數 (Predict score for a single triple or batch of triples)
        """
        return self._score(h, r, t)

    def get_embeddings(self):
        """
        取得entity與relation的embedding (Get entity and relation embeddings)
        """
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction
    Paper: https://arxiv.org/abs/1606.06357
    """
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        # entity/relation皆用複數 (real+imag)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)  # real+imag
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim * 2)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        前向傳播，計算正負樣本分數 (Forward pass for training)
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score

    def _score(self, h, r, t):
        """
        ComplEx打分公式 (ComplEx scoring function)
        """
        h_emb = self.entity_embeddings(h).view(-1, self.embedding_dim, 2)
        r_emb = self.relation_embeddings(r).view(-1, self.embedding_dim, 2)
        t_emb = self.entity_embeddings(t).view(-1, self.embedding_dim, 2)
        re_h, im_h = h_emb[..., 0], h_emb[..., 1]
        re_r, im_r = r_emb[..., 0], r_emb[..., 1]
        re_t, im_t = t_emb[..., 0], t_emb[..., 1]
        # 複數內積 (complex bilinear product)
        score = (re_h * re_r * re_t + re_h * im_r * im_t + im_h * re_r * im_t - im_h * im_r * re_t).sum(dim=1)
        return score

    def predict(self, h, r, t):
        """
        預測單一三元組分數或批次三元組分數 (Predict score for a single triple or batch of triples)
        """
        return self._score(h, r, t)

    def get_embeddings(self):
        """
        取得entity與relation的embedding (Get entity and relation embeddings)
        """
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

class SimplE(nn.Module):
    """
    SimplE: A Simple Embedding Model for Link Prediction
    Paper: https://arxiv.org/abs/1802.04868
    """
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(SimplE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        # head/tail entity, head/tail relation embedding
        self.entity_head = nn.Embedding(num_entities, embedding_dim)
        self.entity_tail = nn.Embedding(num_entities, embedding_dim)
        self.relation_head = nn.Embedding(num_relations, embedding_dim)
        self.relation_tail = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_head.weight.data)
        nn.init.xavier_uniform_(self.entity_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_head.weight.data)
        nn.init.xavier_uniform_(self.relation_tail.weight.data)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        前向傳播，計算正負樣本分數 (Forward pass for training)
        """
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score

    def _score(self, h, r, t):
        """
        SimplE打分公式 (SimplE scoring function)
        """
        h_head = self.entity_head(h)
        t_tail = self.entity_tail(t)
        r_head = self.relation_head(r)
        h_tail = self.entity_tail(h)
        t_head = self.entity_head(t)
        r_tail = self.relation_tail(r)
        # head模式與tail模式平均 (average of head and tail mode)
        score1 = (h_head * r_head * t_tail).sum(dim=1)
        score2 = (t_head * r_tail * h_tail).sum(dim=1)
        score = 0.5 * (score1 + score2)
        return score

    def predict(self, h, r, t):
        """
        預測單一三元組分數或批次三元組分數 (Predict score for a single triple or batch of triples)
        """
        return self._score(h, r, t)

    def get_embeddings(self):
        """
        取得所有embedding (Get all embeddings as a dict)
        """
        return {
            'entity_head': self.entity_head.weight.data,
            'entity_tail': self.entity_tail.weight.data,
            'relation_head': self.relation_head.weight.data,
            'relation_tail': self.relation_tail.weight.data
        }

class InterHT(nn.Module):
    """
    InterHT: Knowledge Graph Embeddings by Interaction between Head and Tail Entities
    Paper: https://arxiv.org/abs/2202.04897
    Official code: https://github.com/destwang/InterHT
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, u=1.0):
        super(InterHT, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.u = u
        # InterHT uses double entity embedding and triple relation embedding
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim * 3)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score

    def _score(self, h, r, t):
        # h, r, t: (batch,)
        h_emb = self.entity_embeddings(h)  # (batch, 2*dim)
        r_emb = self.relation_embeddings(r)  # (batch, 3*dim)
        t_emb = self.entity_embeddings(t)  # (batch, 2*dim)
        # Split embeddings
        a_head, b_head = torch.chunk(h_emb, 2, dim=1)  # (batch, dim), (batch, dim)
        re_head, re_mid, re_tail = torch.chunk(r_emb, 3, dim=1)  # (batch, dim) x3
        a_tail, b_tail = torch.chunk(t_emb, 2, dim=1)
        # Normalize and add bias
        e_h = torch.ones_like(b_head)
        e_t = torch.ones_like(b_tail)
        a_head = F.normalize(a_head, 2, -1)
        a_tail = F.normalize(a_tail, 2, -1)
        b_head = F.normalize(b_head, 2, -1) + self.u * e_h
        b_tail = F.normalize(b_tail, 2, -1) + self.u * e_t
        # InterHT score
        score = a_head * b_tail - a_tail * b_head + re_mid
        score = torch.norm(score, p=1, dim=1)
        return score

    def predict(self, h, r, t):
        return self._score(h, r, t)

    def get_embeddings(self):
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data

def create_kge_model(model_type, num_entities, num_relations, embedding_dim, **kwargs):
    """
    KGE模型工廠函數 (Factory function to create KGE models)
    根據model_type建立對應的KGE模型
    Supported: transe, distmult, ptranse, rotate, complex, simple, interht
    """
    model_type_lower = model_type.lower()
    
    # Handle PyKEEN models with 'pykeen_' prefix
    if model_type_lower.startswith('pykeen_'):
        pykeen_type = model_type_lower.replace('pykeen_', '')
        if PYKEEEN_AVAILABLE and pykeen_type in ['transe', 'distmult', 'complex', 'rotate', 'simple']:
            return create_pykeen_model(pykeen_type, num_entities, num_relations, embedding_dim, **kwargs)
        else:
            raise ValueError(f"PyKEEN model {pykeen_type} not supported or PyKEEN not available")
    
    # Check if PyKEEN is available and use it for supported models
    if PYKEEEN_AVAILABLE and model_type_lower in ['transe', 'distmult', 'complex', 'rotate', 'simple']:
        return create_pykeen_model(model_type_lower, num_entities, num_relations, embedding_dim, **kwargs)
    
    # Fall back to custom implementations
    if model_type_lower == 'transe':
        # TransE supports margin and norm
        margin = kwargs.get('margin', 1.0)
        norm = kwargs.get('norm', 1)
        return TransE(num_entities, num_relations, embedding_dim, margin=margin, norm=norm)
    
    elif model_type_lower == 'distmult':
        # DistMult only supports basic parameters
        return DistMult(num_entities, num_relations, embedding_dim)
    
    elif model_type_lower == 'ptranse':
        # PTransE supports margin and norm
        margin = kwargs.get('margin', 1.0)
        norm = kwargs.get('norm', 1)
        return PTransE(num_entities, num_relations, embedding_dim, margin=margin, norm=norm)
    
    elif model_type_lower == 'rotate':
        # RotatE only supports margin
        margin = kwargs.get('margin', 6.0)
        return RotatE(num_entities, num_relations, embedding_dim, margin=margin)
    
    elif model_type_lower == 'complex':
        # ComplEx only supports basic parameters
        return ComplEx(num_entities, num_relations, embedding_dim)
    
    elif model_type_lower == 'simple':
        # SimplE only supports basic parameters
        return SimplE(num_entities, num_relations, embedding_dim)
    
    elif model_type_lower == 'interht':
        # InterHT supports margin and u
        margin = kwargs.get('margin', 1.0)
        u = kwargs.get('u', 1.0)
        return InterHT(num_entities, num_relations, embedding_dim, margin=margin, u=u)
    
    ####################################
    elif model_type_lower == 'cmkge':
        margin = kwargs.get('margin', 1.0)
        norm = kwargs.get('norm', 1)
        return CMKGE(num_entities, num_relations, embedding_dim, margin=margin, norm=norm)

    elif model_type_lower == 'cake':
        commonsense_dim = kwargs.get('commonsense_dim', 100)
        return CAKE(num_entities, num_relations, embedding_dim, commonsense_dim=commonsense_dim)

    elif model_type_lower == 'kgeprisma':
        margin = kwargs.get('margin', 1.0)
        return KGEPrisma(num_entities, num_relations, embedding_dim, margin=margin)

    elif model_type_lower == 'rdf2vec':
        # walk_sequences 應以關鍵字參數傳入
        walk_sequences = kwargs['walk_sequences']  # 必須提供
        return RDF2Vec(num_entities, embedding_dim, walk_sequences=walk_sequences)
    ####################################
    else:
        raise ValueError(f"Unknown KGE model type: {model_type}") 

def create_pykeen_model(model_type, num_entities, num_relations, embedding_dim, **kwargs):
    """
    使用 PyKEEN 套件建立 KGE 模型
    Create KGE models using PyKEEN library
    """
    if not PYKEEEN_AVAILABLE:
        raise ImportError("PyKEEN is not available. Please install it with: pip install pykeen")
    
    # Create a dummy triples factory for initialization
    # This is needed for PyKEEN models but we'll override the forward method
    dummy_triples = torch.zeros((1, 3), dtype=torch.long)
    triples_factory = TriplesFactory.from_labeled_triples(dummy_triples)
    
    if model_type == 'transe':
        margin = kwargs.get('margin', 1.0)
        return PyKEENTransE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            margin=margin
        )
    
    elif model_type == 'distmult':
        return PyKEENDistMult(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
    
    elif model_type == 'complex':
        return PyKEENComplEx(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
    
    elif model_type == 'rotate':
        margin = kwargs.get('margin', 6.0)
        return PyKEENRotatE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            margin=margin
        )
    
    elif model_type == 'simple':
        return PyKEENSimplE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
    
    else:
        raise ValueError(f"PyKEEN model type {model_type} not supported")

# PyKEEN-based model wrappers
class PyKEENTransE(nn.Module):
    """
    PyKEEN TransE wrapper
    """
    def __init__(self, triples_factory, embedding_dim, margin=1.0):
        super().__init__()
        self.pykeen_model = TransE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            margin=margin
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        # Use PyKEEN's scoring function
        return self.pykeen_model.score_hrt(h, r, t)
    
    def predict(self, h, r, t):
        return self._score(h, r, t)
    
    def get_embeddings(self):
        return self.pykeen_model.entity_representations[0].weight.data, self.pykeen_model.relation_representations[0].weight.data

class PyKEENDistMult(nn.Module):
    """
    PyKEEN DistMult wrapper
    """
    def __init__(self, triples_factory, embedding_dim):
        super().__init__()
        self.pykeen_model = DistMult(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.embedding_dim = embedding_dim
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        # Use PyKEEN's scoring function
        return self.pykeen_model.score_hrt(h, r, t)
    
    def predict(self, h, r, t):
        return self._score(h, r, t)
    
    def get_embeddings(self):
        return self.pykeen_model.entity_representations[0].weight.data, self.pykeen_model.relation_representations[0].weight.data

class PyKEENComplEx(nn.Module):
    """
    PyKEEN ComplEx wrapper
    """
    def __init__(self, triples_factory, embedding_dim):
        super().__init__()
        self.pykeen_model = ComplEx(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.embedding_dim = embedding_dim
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        # Use PyKEEN's scoring function
        return self.pykeen_model.score_hrt(h, r, t)
    
    def predict(self, h, r, t):
        return self._score(h, r, t)
    
    def get_embeddings(self):
        return self.pykeen_model.entity_representations[0].weight.data, self.pykeen_model.relation_representations[0].weight.data

class PyKEENRotatE(nn.Module):
    """
    PyKEEN RotatE wrapper
    """
    def __init__(self, triples_factory, embedding_dim, margin=6.0):
        super().__init__()
        self.pykeen_model = RotatE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            margin=margin
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        # Use PyKEEN's scoring function
        return self.pykeen_model.score_hrt(h, r, t)
    
    def predict(self, h, r, t):
        return self._score(h, r, t)
    
    def get_embeddings(self):
        return self.pykeen_model.entity_representations[0].weight.data, self.pykeen_model.relation_representations[0].weight.data

class PyKEENSimplE(nn.Module):
    """
    PyKEEN SimplE wrapper
    """
    def __init__(self, triples_factory, embedding_dim):
        super().__init__()
        self.pykeen_model = SimplE(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
        self.num_entities = triples_factory.num_entities
        self.num_relations = triples_factory.num_relations
        self.embedding_dim = embedding_dim
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        # Use PyKEEN's scoring function
        return self.pykeen_model.score_hrt(h, r, t)
    
    def predict(self, h, r, t):
        return self._score(h, r, t)
    
    def get_embeddings(self):
        return self.pykeen_model.entity_representations[0].weight.data, self.pykeen_model.relation_representations[0].weight.data

###########################

class CMKGE(nn.Module):
    """
    CMKGE: Continual Mask Knowledge Graph Embedding
    核心: 用兩組mask控制參數plasticity (可塑性) 與 stability (穩定性)
    參考框架：動態更新實體及關係embedding，減少遺忘
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, norm=1):
        super(CMKGE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # 雙重mask參數，用以分辨"穩定"與"新知識"通路
        self.entity_stability_mask = nn.Parameter(torch.ones(num_entities, embedding_dim))
        self.entity_plasticity_mask = nn.Parameter(torch.ones(num_entities, embedding_dim))
        self.relation_stability_mask = nn.Parameter(torch.ones(num_relations, embedding_dim))
        self.relation_plasticity_mask = nn.Parameter(torch.ones(num_relations, embedding_dim))

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # 應用mask於embedding
        def masked_embedding(e_idx, embedding, stability_mask, plasticity_mask):
            e_emb = embedding(e_idx)
            e_stability = e_emb * stability_mask[e_idx]
            e_plasticity = e_emb * plasticity_mask[e_idx]
            return e_stability + e_plasticity

        pos_h_emb = masked_embedding(pos_h, self.entity_embeddings, self.entity_stability_mask, self.entity_plasticity_mask)
        pos_r_emb = masked_embedding(pos_r, self.relation_embeddings, self.relation_stability_mask, self.relation_plasticity_mask)
        pos_t_emb = masked_embedding(pos_t, self.entity_embeddings, self.entity_stability_mask, self.entity_plasticity_mask)

        neg_h_emb = masked_embedding(neg_h, self.entity_embeddings, self.entity_stability_mask, self.entity_plasticity_mask)
        neg_r_emb = masked_embedding(neg_r, self.relation_embeddings, self.relation_stability_mask, self.relation_plasticity_mask)
        neg_t_emb = masked_embedding(neg_t, self.entity_embeddings, self.entity_stability_mask, self.entity_plasticity_mask)

        pos_score = torch.norm(pos_h_emb + pos_r_emb - pos_t_emb, p=self.norm, dim=1)
        neg_score = torch.norm(neg_h_emb + neg_r_emb - neg_t_emb, p=self.norm, dim=1)

        return pos_score, neg_score

    def predict(self, h, r, t):
        h_emb = self.entity_embeddings(h) * (self.entity_stability_mask[h] + self.entity_plasticity_mask[h])
        r_emb = self.relation_embeddings(r) * (self.relation_stability_mask[r] + self.relation_plasticity_mask[r])
        t_emb = self.entity_embeddings(t) * (self.entity_stability_mask[t] + self.entity_plasticity_mask[t])
        score = torch.norm(h_emb + r_emb - t_emb, p=self.norm, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data


class CAKE(nn.Module):
    """
    CAKE: Commonsense-augmented Knowledge Embedding
    核心: 除了傳統embedding，加入外部常識庫embedding增強融合
    此範例簡化用額外latent commonsense embedding矩陣模擬
    """
    def __init__(self, num_entities, num_relations, embedding_dim, commonsense_dim=100):
        super(CAKE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.commonsense_dim = commonsense_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.commonsense_embeddings = nn.Embedding(num_entities, commonsense_dim)  # commonsense latent vector

        self.fc_fuse = nn.Linear(embedding_dim + commonsense_dim, embedding_dim)  # 融合層

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.commonsense_embeddings.weight.data)
        nn.init.xavier_uniform_(self.fc_fuse.weight.data)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        def fused_embedding(e_idx):
            emb = self.entity_embeddings(e_idx)
            cs_emb = self.commonsense_embeddings(e_idx)
            fused = torch.cat([emb, cs_emb], dim=1)
            fused = F.relu(self.fc_fuse(fused))
            return fused

        pos_h_emb = fused_embedding(pos_h)
        pos_r_emb = self.relation_embeddings(pos_r)
        pos_t_emb = fused_embedding(pos_t)

        neg_h_emb = fused_embedding(neg_h)
        neg_r_emb = self.relation_embeddings(neg_r)
        neg_t_emb = fused_embedding(neg_t)

        pos_score = torch.norm(pos_h_emb + pos_r_emb - pos_t_emb, p=1, dim=1)
        neg_score = torch.norm(neg_h_emb + neg_r_emb - neg_t_emb, p=1, dim=1)

        return pos_score, neg_score

    def predict(self, h, r, t):
        h_emb = torch.cat([self.entity_embeddings(h), self.commonsense_embeddings(h)], dim=1)
        h_emb = F.relu(self.fc_fuse(h_emb))
        r_emb = self.relation_embeddings(r)
        t_emb = torch.cat([self.entity_embeddings(t), self.commonsense_embeddings(t)], dim=1)
        t_emb = F.relu(self.fc_fuse(t_emb))

        score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data, self.commonsense_embeddings.weight.data, self.relation_embeddings.weight.data


class KGEPrisma(nn.Module):
    """
    KGEPrisma: 可解釋知識圖嵌入
    核心: 增加監督信號促使embedding空間符合符號規則
    此版本以簡化版約束loss示範
    """
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(KGEPrisma, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._score(pos_h, pos_r, pos_t)
        neg_score = self._score(neg_h, neg_r, neg_t)
        return pos_score, neg_score
    
    def _score(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        # 基本TransE風格分數
        score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)

        # 添加符號性約束，例如可加正則化或規則約束loss
        # 這裡示意性地回傳分數，實際約束可以在訓練時加入額外loss
        return score

    def interpretability_loss(self):
        """
        示範加入可解釋性約束
        如embedding稀疏性、正交性或符號邏輯近似等
        簡單示範embedding稀疏正則化
        """
        return torch.mean(torch.abs(self.entity_embeddings.weight)) + torch.mean(torch.abs(self.relation_embeddings.weight))

    def predict(self, h, r, t):
        return self._score(h, r, t)

    def get_embeddings(self):
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data


class RDF2Vec(nn.Module):
    """
    RDF2Vec: 基於序列化隨機漫遊的表示學習
    核心：將隨機漫遊序列視為word2vec結構學習entity embedding
    此為簡化版模擬，需預先提供entity序列數據以學習embedding
    """
    def __init__(self, num_entities, embedding_dim, walk_sequences):
        """
        :param walk_sequences: List[List[int]]，每個元素是隨機漫遊的實體ID序列
        """
        super(RDF2Vec, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)

        # 用word2vec Skip-gram架構學習embedding的簡化版
        # 這裡示範用負采樣法訓練，需外部調用train_step實現

        self.walk_sequences = walk_sequences  # 用於外部采樣上下文

    def forward(self, center_entities, context_entities, negative_entities):
        """
        Skip-gram正負樣本訓練
        center_entities: 中心詞entity batch
        context_entities: 正確上下文entity batch
        negative_entities: 負樣本entity batch
        """
        center_emb = self.entity_embeddings(center_entities)  # (batch, dim)
        context_emb = self.entity_embeddings(context_entities)  # (batch, dim)
        neg_emb = self.entity_embeddings(negative_entities)  # (batch * neg_sample, dim)

        pos_score = torch.sum(center_emb * context_emb, dim=1)
        neg_score = torch.bmm(neg_emb.view(center_emb.size(0), -1, self.embedding_dim),
                              center_emb.unsqueeze(2)).squeeze()

        loss = - torch.mean(torch.log(torch.sigmoid(pos_score)) + torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1))
        return loss

    def get_embeddings(self):
        return self.entity_embeddings.weight.data
    
###########################