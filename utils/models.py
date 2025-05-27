import torch
import torch.nn as nn

from utils.losses import TwhinLoss, InBatchSampledSoftmax, CalibratedPairwiseLogistic


class TwhinModel(nn.Module):
    def __init__(self, embedding_dim, item_cardinality=25834, user_cardinality=3315, relation_cardinality=3, reg_weight=0.):
        super().__init__()
        self.item_embeddings = nn.Embedding(item_cardinality, embedding_dim)
        self.user_embeddings = nn.Embedding(user_cardinality, embedding_dim) 
        self.relation_embeddings = nn.Embedding(relation_cardinality, embedding_dim) 
        self.criterion = TwhinLoss(reg_weight=reg_weight)

    def forward(self, inputs):
        item_embeddings = self.item_embeddings(inputs['items'])
        user_embeddings = self.user_embeddings(inputs['users'])
        relation_embeddings = self.relation_embeddings(inputs['relations'])
        return self.criterion(
            torch.nn.functional.normalize(user_embeddings + relation_embeddings),
            torch.nn.functional.normalize(item_embeddings)
        )


class ResNet(nn.Module):
    def __init__(self, embedding_dim, dropout=0.):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.layernorm(self.layer(x) + x)


class PickFirst(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 0, :]


class ModelBackbone(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 num_heads=2,
                 hidden_dim=64,
                 item_embeddings=None, 
                 item_cardinality=25834,
                 max_seq_len=256,
                 dropout_rate=0.2,
                 num_transformer_layers=2,
                 num_tokens=13982):
        super().__init__()
        if item_embeddings is None:
            self.item_embeddings = nn.Embedding(item_cardinality, embedding_dim)
        else:
            self.item_embeddings = item_embeddings 
        self.token_embeddings = nn.EmbeddingBag(num_tokens, embedding_dim)
        self.action_embeddings = nn.Embedding(4, embedding_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.cls_embedding = nn.Embedding(1, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.user_encoder = PickFirst()
        self.item_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            *[ResNet(embedding_dim=embedding_dim * 4) for _ in range(num_transformer_layers)],
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
    
    def forward(self, inputs):
        batch_size, seq_len = inputs['product_id'].shape

        offsets = torch.cumsum(inputs['product_names']['lengths'], dim=0) - inputs['product_names']['lengths']

        products_embeddings = self.item_embeddings(inputs['product_id'])
        products_embeddings[inputs['product_id'] != 0] = products_embeddings[inputs['product_id'] != 0] + self.token_embeddings(inputs['product_names']['ids'], offsets)
        action_embeddings = self.action_embeddings(inputs['action_type'])

        input_embeddings = products_embeddings + action_embeddings + \
                           self.position_embeddings(torch.arange(seq_len, device=inputs['product_id'].device)).unsqueeze(0)
        cls_embedding = self.cls_embedding(torch.zeros(batch_size, 1, device=inputs['product_id'].device, dtype=torch.long))
        input_embeddings_with_cls = torch.cat([cls_embedding, input_embeddings], dim=1)

        padding_mask = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.bool, device=inputs['product_id'].device),
            (inputs['product_id'] == 0)
        ], dim=1)
        user_embeddings = self.user_encoder(self.transformer_encoder(src=input_embeddings_with_cls, src_key_padding_mask=padding_mask))
        offsets = torch.cumsum(inputs['candidate_names']['lengths'], dim=0) - inputs['candidate_names']['lengths'] 
        candidates_embeddings = self.item_encoder(self.item_embeddings(inputs['candidate']) + self.token_embeddings(inputs['candidate_names']['ids'], offsets)) 

        return {
            'user_embeddings': nn.functional.normalize(user_embeddings),
            'candidates_embeddings': nn.functional.normalize(candidates_embeddings)
        }

        
class PretrainModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion = InBatchSampledSoftmax()
        
    def forward(self, inputs):
        backbone_output = self.backbone(inputs) 
        user_embeddings = backbone_output['user_embeddings']
        candidates_embeddings = backbone_output['candidates_embeddings']
        return self.criterion(user_embeddings, candidates_embeddings.squeeze(1))
   

class FinetuneModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.scale_pointwise = nn.Parameter(torch.zeros(1))
        self.bias_pointwise = nn.Parameter(torch.zeros(1))
        self.scale_pairwise = nn.Parameter(torch.zeros(1))
        self.bias_pairwise = nn.Parameter(torch.zeros(1))
        self.criterion_pairwise = CalibratedPairwiseLogistic() 

    def forward(self, inputs):
        backbone_output = self.backbone(inputs) 
        user_embeddings = backbone_output['user_embeddings']
        candidates_embeddings = backbone_output['candidates_embeddings']

        candidates_lengths = inputs['candidate_lengths']
        candidates_mask = inputs['candidate_mask']

        scores = torch.einsum(
            'te,te->t',
            torch.repeat_interleave(user_embeddings, candidates_lengths, dim=0),
            candidates_embeddings
        )
        logits_pointwise = torch.exp(self.scale_pointwise) * scores + self.bias_pointwise
        logits_pairwise = torch.exp(self.scale_pairwise) * scores + self.bias_pairwise

        return nn.functional.binary_cross_entropy_with_logits(logits_pointwise, candidates_mask.to(torch.float), reduction='mean') * 0.1 \
               + self.criterion_pairwise(logits_pairwise, candidates_mask, candidates_lengths)