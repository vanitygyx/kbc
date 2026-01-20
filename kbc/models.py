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
