import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class GTELargeEN:
    def __init__(self,
                 device,
                 normalize=True):
        self.device = device
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True).to(device)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list, sub_batch_size=64):
        if len(text_list) == 0:
            return torch.zeros(0, 1024)
        
        all_embs = []
        
        # 分批處理以避免記憶體過載
        for i in range(0, len(text_list), sub_batch_size):
            sub_batch = text_list[i:i + sub_batch_size]
        
            batch_dict = self.tokenizer(
                    sub_batch, max_length=8196, padding=True,
                truncation=True, return_tensors='pt').to(self.device)
            
            outputs = self.model(**batch_dict).last_hidden_state
            emb = outputs[:, 0]
            
            if self.normalize:
                emb = F.normalize(emb, p=2, dim=1)
            
                # 立即移到 CPU 並清理 GPU 記憶體
                emb = emb.cpu()
                all_embs.append(emb)
                
                # 清理中間變數
                del batch_dict, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合併所有結果
        return torch.cat(all_embs, dim=0)

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        
        return q_emb, entity_embs, relation_embs
