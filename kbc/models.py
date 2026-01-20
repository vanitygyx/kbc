from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .graph_utils import build_graph_from_triples


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    来源: https://arxiv.org/pdf/2004.11362.pdf
    支持SimCLR无监督对比学习
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        计算对比学习损失
        Args:
            features: 特征向量 [bsz, n_views, ...]
            labels: 真实标签 [bsz]
            mask: 对比掩码 [bsz, bsz]
        Returns:
            损失标量
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supervised contrastive loss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本的平均log-likelihood
        pos_per_sample = mask.sum(1)
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample

        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class RelationContrastLoss(nn.Module):
    """
    关系对比学习损失
    """
    def __init__(self, temperature=0.07, num_neg=10):
        super(RelationContrastLoss, self).__init__()
        self.temperature = temperature
        self.num_neg = num_neg

    def forward(self, pos_scores, neg_scores):
        """
        计算关系对比损失
        Args:
            pos_scores: 正样本得分 [batch_size, 1]
            neg_scores: 负样本得分 [batch_size, num_neg]
        """
        neg_scores = neg_scores.view(-1, self.num_neg, 1)
        pos = torch.exp(torch.div(pos_scores, self.temperature))
        neg = torch.exp(torch.div(neg_scores, self.temperature)).sum(dim=1)
        loss = -torch.log(torch.div(pos, neg)).mean()
        return loss

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            multimodal_data: Dict[str, torch.Tensor] = None,
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        统一的get_ranking方法，支持多模态数据
        """
        ranks = torch.ones(len(queries))
        
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    
                    # 根据模型类型调用不同的forward方法
                    if multimodal_data is not None:
                        scores, _ = self.forward(these_queries, multimodal_data)
                    else:
                        scores, _ = self.forward(these_queries)
                    
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks


class KBGAT_conv(nn.Module):
    """
    基于KRACL的KBGAT_conv实现，用于知识图谱的图注意力网络
    使用纯PyTorch实现，不依赖torch_geometric
    """
    def __init__(self, in_channel, out_channel, rel_dim, dropout=0, final_layer=False):
        super(KBGAT_conv, self).__init__()
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
        """
        前向传播
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            relation_embedding: 关系嵌入 [num_relations, rel_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
        """
        # 手动实现消息传递
        src, dst = edge_index[0], edge_index[1]
        
        # 获取源节点和目标节点特征
        x_src = x[src]  # [num_edges, feature_dim]
        x_dst = x[dst]  # [num_edges, feature_dim]
        
        # 获取边的关系嵌入
        edge_emb = relation_embedding[edge_type]  # [num_edges, rel_dim]
        
        # 构建三元组嵌入
        triple_emb = torch.cat([x_src, x_dst, edge_emb], dim=1)  # [num_edges, 2*feature_dim + rel_dim]
        
        # 计算注意力
        c = self.w_1(triple_emb)  # [num_edges, out_dim]
        b = self.leaky_relu(self.w_2(c))  # [num_edges, 1]
        
        # 手动实现softmax注意力
        alpha = torch.zeros_like(b)
        for node_id in range(x.size(0)):
            mask = (dst == node_id)
            if mask.sum() > 0:
                node_edges = b[mask]
                alpha[mask] = F.softmax(node_edges, dim=0)
        
        # 应用dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 计算消息
        messages = c * alpha  # [num_edges, out_dim]
        
        # 聚合消息
        out = torch.zeros(x.size(0), self.out_dim, device=x.device)
        out.index_add_(0, dst, messages)
        
        if self.final_layer:
            out = self.elu(out)
        else:
            out = self.leaky_relu(out)
        
        return out


class MultimodalKBGAT(nn.Module):
    """
    模态分离的多KBGAT架构
    为每个模态分别构建图注意力编码器，然后进行后融合
    """
    def __init__(self, sizes, rank, visual_dim, textual_dim, rel_dim, dropout=0.0):
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


class ComplEx(KBCModel):
    """
    基础ComplEx模型实现
    """
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )


class MultimodalComplEx(KBCModel):
    """
    多模态ComplEx模型，集成模态分离的KBGAT编码器和对比学习
    """
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, 
            visual_dim: int, textual_dim: int, rel_dim: int,
            init_size: float = 1e-3, dropout: float = 0.0,
            temperature: float = 0.07, use_contrastive: bool = True
    ):
        super(MultimodalComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.rel_dim = rel_dim
        self.use_contrastive = use_contrastive

        # 多模态KBGAT编码器
        self.encoder = MultimodalKBGAT(
            sizes=sizes,
            rank=rank,
            visual_dim=visual_dim,
            textual_dim=textual_dim,
            rel_dim=rel_dim,
            dropout=dropout
        )
        
        # 关系嵌入（用于ComplEx解码）
        self.relation_embeddings = nn.Embedding(sizes[1], 2 * rank, sparse=True)
        self.relation_embeddings.weight.data *= init_size
        
        # 对比学习模块
        if self.use_contrastive:
            self.contrastive_loss = SupConLoss(
                temperature=temperature,
                contrast_mode='all',
                base_temperature=temperature
            )
            self.relation_contrast_loss = RelationContrastLoss(
                temperature=temperature,
                num_neg=10
            )
        
        # 图结构（将在训练时构建）
        self.edge_index = None
        self.edge_type = None

    def build_graph(self, triples):
        """
        从训练三元组构建图结构
        """
        self.edge_index, self.edge_type = build_graph_from_triples(
            triples, self.sizes[0], self.sizes[1]
        )
        
    def forward(self, x, multimodal_data):
        """
        前向传播
        Args:
            x: 三元组 [batch_size, 3]
            multimodal_data: 多模态数据 {'visual': tensor, 'textual': tensor}
        """
        device = x.device
        
        # 确保图结构在正确的设备上
        if self.edge_index is not None:
            if self.edge_index.device != device:
                self.edge_index = self.edge_index.to(device)
            if self.edge_type.device != device:
                self.edge_type = self.edge_type.to(device)
        
        # 1. 多模态图注意力编码
        if self.edge_index is not None:
            encoded_embeddings = self.encoder(x, self.edge_index, self.edge_type, multimodal_data)
        else:
            # 如果没有图结构，使用结构嵌入
            encoded_embeddings = self.encoder.get_embeddings()
        
        # 2. 使用编码后的嵌入进行ComplEx评分
        lhs = encoded_embeddings[x[:, 0]]
        rhs = encoded_embeddings[x[:, 2]]
        rel = self.relation_embeddings(x[:, 1])
        
        # 3. ComplEx双线性评分
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        
        to_score = encoded_embeddings
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
    
    def score(self, x, multimodal_data):
        """
        计算三元组得分
        """
        device = x.device
        
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
            self.edge_type = self.edge_type.to(device)
            encoded_embeddings = self.encoder(x, self.edge_index, self.edge_type, multimodal_data)
        else:
            encoded_embeddings = self.encoder.get_embeddings()
        
        lhs = encoded_embeddings[x[:, 0]]
        rel = self.relation_embeddings(x[:, 1])
        rhs = encoded_embeddings[x[:, 2]]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )