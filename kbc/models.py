from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from .encoders import MultimodalKBGAT
from .graph_utils import build_graph_from_triples

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        fb_ling_f=r'../../pre_train/matrix_fb_ling.npy'
        fb_visual_f=r'../../pre_train/matrix_fb_visual.npy'
        wn_ling_f=r"../../pre_train/matrix_wn_ling.npy"
        wn_visual_f=r"../../pre_train/matrix_wn_visual.npy"
        fb_ling,fb_visual,wn_ling,wn_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f)),torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        multimodal_embeddings=[wn_ling,wn_visual]
        multimodal_embeddings1=[fb_ling,fb_visual]
        
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries,multimodal_embeddings)
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

class OTKGE_wn(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(OTKGE_wn, self).__init__()
        self.sizes = sizes
        self.rank = rank
        alpha=0.1#select the parameter
        gamma=0.8#select the parameter
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        
        wn_ling_f=r"../../pre_train/matrix_wn_ling.npy"
        wn_visual_f=r"../../pre_train/matrix_wn_visual.npy"
        wn_ling,wn_visual=torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        self.img_vec=wn_visual.to(torch.float32)
        self.img_dimension=wn_visual.shape[-1]
        self.ling_vec=wn_ling.to(torch.float32)
        self.ling_dimension=wn_ling.shape[-1]
        
        # 简单的线性投影层，替代最优传输
        self.img_proj = nn.Linear(self.img_dimension, 2 * rank)
        self.ling_proj = nn.Linear(self.ling_dimension, 2 * rank)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings1[0].weight.data *= init_size
        self.embeddings1[1].weight.data *= init_size      
           
    def forward(self, x, multi_modal):
        device = x.device
        
        # 简单线性投影替代最优传输
        img_embeddings = self.img_proj(self.img_vec.to(device))
        ling_embeddings = self.ling_proj(self.ling_vec.to(device))
        
        # 简单线性融合
        embedding = (1 - self.alpha - self.gamma) * self.embeddings[0].weight + \
                   self.alpha * img_embeddings + self.gamma * ling_embeddings
                   
        lhs = embedding[(x[:, 0])]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[(x[:, 2])] 
        rel1 = self.embeddings1[1](x[:,1])     
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank], rel1[:, self.rank:]
        
        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )

class OTKGE_fb(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(OTKGE_fb, self).__init__()
        self.sizes = sizes
        self.rank = rank
        alpha=0.1#select the parameter
        gamma=0.7#select the parameter
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        
        fb_ling_f=r'../../pre_train/matrix_fb_ling.npy'
        fb_visual_f=r'../../pre_train/matrix_fb_visual.npy'
        fb_ling,fb_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f))        
        self.img_vec=fb_visual.to(torch.float32)
        self.img_dimension=fb_visual.shape[-1]
        self.ling_vec=fb_ling.to(torch.float32)
        self.ling_dimension=fb_ling.shape[-1]
        
        # 简单的线性投影层，替代最优传输
        self.img_proj = nn.Linear(self.img_dimension, 2 * rank)
        self.ling_proj = nn.Linear(self.ling_dimension, 2 * rank)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
           
    def forward(self, x, multi_modal):
        device = x.device
        
        # 简单线性投影替代最优传输
        img_embeddings = self.img_proj(self.img_vec.to(device))
        ling_embeddings = self.ling_proj(self.ling_vec.to(device))
        
        # 简单线性融合
        embedding = (1 - self.alpha - self.gamma) * self.embeddings[0].weight + \
                   self.alpha * img_embeddings + self.gamma * ling_embeddings
                   
        lhs = embedding[(x[:, 0])]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[(x[:, 2])] 
        rel1 = self.embeddings1[1](x[:,1])     
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank], rel1[:, self.rank:]
        
        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )
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

    def forward(self, x, multi_modal=None):
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
    多模态ComplEx模型，集成模态分离的KBGAT编码器
    """
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, 
            visual_dim: int, textual_dim: int, rel_dim: int,
            init_size: float = 1e-3, dropout: float = 0.0
    ):
        super(MultimodalComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.rel_dim = rel_dim

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
            self.edge_index = self.edge_index.to(device)
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
    
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            multimodal_data: Dict[str, torch.Tensor],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        重写get_ranking方法以支持多模态数据
        """
        ranks = torch.ones(len(queries))
        
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries, multimodal_data)
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