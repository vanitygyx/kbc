import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import numpy as np

class KBGAT_conv(MessagePassing):
    """
    基于KRACL的KBGAT_conv实现，用于知识图谱的图注意力网络
    """
    def __init__(self, in_channel, out_channel, rel_dim, dropout=0, final_layer=False):
        super(KBGAT_conv, self).__init__(aggr="add", node_dim=0)
        self.ent_input_dim = in_channel
        self.out_dim = out_channel
        self.rel_dim = rel_dim
        self.w_1 = nn.Linear(2 * self.ent_input_dim + rel_dim, self.out_dim, bias=True)
        self.w_2 = nn.Linear(self.out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.final_layer = final_layer
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.w_1.weight.data)
        torch.nn.init.xavier_uniform_(self.w_2.weight.data)

    def forward(self, x, relation_embedding, edge_index, edge_type, edge_weight=None):
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, 
                                edge_weight=edge_weight, relation_embedding=relation_embedding)
        
        if self.final_layer:
            node_emb = self.elu(node_emb.mean(dim=1))
        else:
            node_emb = self.leaky_relu(node_emb)
        
        return node_emb

    def message(self, x_i, x_j, index, ptr, size_i, edge_type, relation_embedding, edge_weight):
        edge_emb = torch.index_select(relation_embedding, 0, edge_type)
        triple_emb = torch.cat((x_i, x_j, edge_emb), dim=1)
        c = self.w_1(triple_emb)

        b = self.leaky_relu(self.w_2(c))
        alpha = softmax(b, index, ptr, size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = c * alpha.view(-1, 1)

        if edge_weight is None:
            return out
        else:
            out = out * edge_weight.view(-1, 1)
            return out
        
    def update(self, aggr_out):
        return aggr_out


class MultimodalKBGAT(nn.Module):
    """
    模态分离的多KBGAT架构
    为每个模态分别构建图注意力编码器，然后进行后融合
    """
    def __init__(self, sizes, rank, visual_dim, textual_dim, rel_dim, dropout=0):
        super(MultimodalKBGAT, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.rel_dim = rel_dim
        
        # 三个独立的KBGAT编码器
        self.kbgat_structural = KBGAT_conv(
            in_channel=rank * 2, 
            out_channel=rank * 2, 
            rel_dim=rel_dim, 
            dropout=dropout
        )
        self.kbgat_visual = KBGAT_conv(
            in_channel=rank * 2, 
            out_channel=rank * 2, 
            rel_dim=rel_dim, 
            dropout=dropout
        )
        self.kbgat_textual = KBGAT_conv(
            in_channel=rank * 2, 
            out_channel=rank * 2, 
            rel_dim=rel_dim, 
            dropout=dropout
        )
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 视觉权重
        self.gamma = nn.Parameter(torch.tensor(0.8))  # 文本权重
        
        # 模态特征投影层
        self.visual_proj = nn.Linear(visual_dim, rank * 2)
        self.textual_proj = nn.Linear(textual_dim, rank * 2)
        
        # 结构模态的可学习嵌入
        self.structural_embeddings = nn.Embedding(sizes[0], rank * 2, sparse=True)
        
        # 关系嵌入（共享）
        self.relation_embeddings = nn.Embedding(sizes[1], rel_dim, sparse=True)
        
        # 初始化
        self.structural_embeddings.weight.data *= 1e-3
        self.relation_embeddings.weight.data *= 1e-3
        
    def forward(self, x, edge_index, edge_type, multimodal_data):
        """
        前向传播
        Args:
            x: 三元组 [batch_size, 3] (head, relation, tail)
            edge_index: 图的边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            multimodal_data: 多模态数据字典 {'visual': tensor, 'textual': tensor}
        """
        device = x.device
        
        # 获取关系嵌入
        relation_emb = self.relation_embeddings.weight
        
        # 1. 分别对每个模态进行KBGAT编码
        
        # 结构模态：使用可学习的实体嵌入
        structural_features = self.structural_embeddings.weight
        h_structural = self.kbgat_structural(
            x=structural_features, 
            relation_embedding=relation_emb,
            edge_index=edge_index, 
            edge_type=edge_type
        )
        
        # 视觉模态：投影预训练的图像特征
        visual_features = self.visual_proj(multimodal_data['visual'].to(device))
        h_visual = self.kbgat_visual(
            x=visual_features,
            relation_embedding=relation_emb,
            edge_index=edge_index,
            edge_type=edge_type
        )
        
        # 文本模态：投影预训练的文本特征
        textual_features = self.textual_proj(multimodal_data['textual'].to(device))
        h_textual = self.kbgat_textual(
            x=textual_features,
            relation_embedding=relation_emb,
            edge_index=edge_index,
            edge_type=edge_type
        )
        
        # 2. 后融合策略
        h_final = (1 - self.alpha - self.gamma) * h_structural + \
                  self.alpha * h_visual + self.gamma * h_textual
                  
        return h_final
    
    def get_embeddings(self):
        """获取当前的实体嵌入（用于ComplEx解码）"""
        return self.structural_embeddings.weight
    
    def get_relation_embeddings(self):
        """获取关系嵌入"""
        return self.relation_embeddings.weight


class AdaptiveFusionKBGAT(MultimodalKBGAT):
    """
    带有自适应注意力融合机制的多模态KBGAT
    """
    def __init__(self, sizes, rank, visual_dim, textual_dim, rel_dim, dropout=0):
        super(AdaptiveFusionKBGAT, self).__init__(sizes, rank, visual_dim, textual_dim, rel_dim, dropout)
        
        # 注意力融合网络
        self.fusion_attention = nn.Sequential(
            nn.Linear(rank * 2, rank),
            nn.ReLU(),
            nn.Linear(rank, 3),  # 3个模态
            nn.Softmax(dim=-1)
        )
        
    def adaptive_fusion(self, h_struct, h_visual, h_textual):
        """
        自适应注意力融合
        """
        # 计算每个实体的融合权重
        # 使用结构嵌入作为查询来计算注意力权重
        attention_weights = self.fusion_attention(h_struct)  # [num_entities, 3]
        
        # 加权融合
        h_fused = attention_weights[:, 0:1] * h_struct + \
                  attention_weights[:, 1:2] * h_visual + \
                  attention_weights[:, 2:3] * h_textual
        
        return h_fused
    
    def forward(self, x, edge_index, edge_type, multimodal_data, use_adaptive=True):
        """
        前向传播，支持自适应融合
        """
        device = x.device
        relation_emb = self.relation_embeddings.weight
        
        # 分别编码各模态
        structural_features = self.structural_embeddings.weight
        h_structural = self.kbgat_structural(
            x=structural_features, 
            relation_embedding=relation_emb,
            edge_index=edge_index, 
            edge_type=edge_type
        )
        
        visual_features = self.visual_proj(multimodal_data['visual'].to(device))
        h_visual = self.kbgat_visual(
            x=visual_features,
            relation_embedding=relation_emb,
            edge_index=edge_index,
            edge_type=edge_type
        )
        
        textual_features = self.textual_proj(multimodal_data['textual'].to(device))
        h_textual = self.kbgat_textual(
            x=textual_features,
            relation_embedding=relation_emb,
            edge_index=edge_index,
            edge_type=edge_type
        )
        
        # 选择融合策略
        if use_adaptive:
            h_final = self.adaptive_fusion(h_structural, h_visual, h_textual)
        else:
            # 固定权重融合
            h_final = (1 - self.alpha - self.gamma) * h_structural + \
                      self.alpha * h_visual + self.gamma * h_textual
                      
        return h_final