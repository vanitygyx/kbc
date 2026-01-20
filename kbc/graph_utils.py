import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List

def build_graph_from_triples(triples, num_entities, num_relations, add_inverse=True):
    """
    从三元组构建图结构
    
    Args:
        triples: 三元组列表 [[h, r, t], ...]
        num_entities: 实体数量
        num_relations: 关系数量  
        add_inverse: 是否添加逆关系
        
    Returns:
        edge_index: [2, num_edges] 边索引
        edge_type: [num_edges] 边类型
    """
    edges = []
    edge_types = []
    
    for triple in triples:
        h, r, t = triple
        # 正向边
        edges.append([h, t])
        edge_types.append(r)
        
        if add_inverse:
            # 逆向边
            edges.append([t, h])
            edge_types.append(r + num_relations)  # 逆关系ID
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    return edge_index, edge_type

def create_adjacency_dict(triples, num_entities):
    """
    创建邻接字典，用于快速查找邻居
    
    Args:
        triples: 三元组列表
        num_entities: 实体数量
        
    Returns:
        adj_dict: {entity_id: [(neighbor_id, relation_id), ...]}
    """
    adj_dict = defaultdict(list)
    
    for h, r, t in triples:
        adj_dict[h].append((t, r))
        adj_dict[t].append((h, r))  # 无向图
        
    return dict(adj_dict)

def sample_subgraph(edge_index, edge_type, center_nodes, num_hops=2, max_neighbors=50):
    """
    采样子图，用于大规模图的高效训练
    
    Args:
        edge_index: 完整图的边索引
        edge_type: 完整图的边类型
        center_nodes: 中心节点列表
        num_hops: 采样跳数
        max_neighbors: 每个节点最大邻居数
        
    Returns:
        subgraph_edge_index: 子图边索引
        subgraph_edge_type: 子图边类型
        node_mapping: 原节点ID到子图节点ID的映射
    """
    device = edge_index.device
    
    # 构建邻接列表
    adj_list = defaultdict(list)
    for i, (src, dst) in enumerate(edge_index.t()):
        adj_list[src.item()].append((dst.item(), edge_type[i].item(), i))
    
    # BFS采样
    visited = set()
    current_nodes = set(center_nodes)
    all_nodes = set(center_nodes)
    
    for hop in range(num_hops):
        next_nodes = set()
        for node in current_nodes:
            if node in visited:
                continue
            visited.add(node)
            
            # 采样邻居
            neighbors = adj_list.get(node, [])
            if len(neighbors) > max_neighbors:
                neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
            
            for neighbor, _, _ in neighbors:
                next_nodes.add(neighbor)
                all_nodes.add(neighbor)
        
        current_nodes = next_nodes
    
    # 创建节点映射
    node_list = sorted(all_nodes)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}
    
    # 构建子图边
    subgraph_edges = []
    subgraph_edge_types = []
    
    for i, (src, dst) in enumerate(edge_index.t()):
        src_id, dst_id = src.item(), dst.item()
        if src_id in node_mapping and dst_id in node_mapping:
            subgraph_edges.append([node_mapping[src_id], node_mapping[dst_id]])
            subgraph_edge_types.append(edge_type[i].item())
    
    subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long, device=device).t()
    subgraph_edge_type = torch.tensor(subgraph_edge_types, dtype=torch.long, device=device)
    
    return subgraph_edge_index, subgraph_edge_type, node_mapping

def normalize_adjacency(edge_index, num_nodes, add_self_loops=True):
    """
    对邻接矩阵进行归一化
    
    Args:
        edge_index: 边索引
        num_nodes: 节点数量
        add_self_loops: 是否添加自环
        
    Returns:
        normalized_edge_index: 归一化后的边索引
        edge_weight: 边权重
    """
    from torch_geometric.utils import add_self_loops as add_loops, degree
    
    if add_self_loops:
        edge_index, _ = add_loops(edge_index, num_nodes=num_nodes)
    
    # 计算度
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # 对称归一化
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return edge_index, edge_weight

def create_bidirectional_graph(edge_index, edge_type, num_relations):
    """
    创建双向图（添加逆关系）
    
    Args:
        edge_index: 原始边索引
        edge_type: 原始边类型
        num_relations: 关系数量
        
    Returns:
        bidirectional_edge_index: 双向边索引
        bidirectional_edge_type: 双向边类型
    """
    # 原始边
    forward_edges = edge_index
    forward_types = edge_type
    
    # 逆向边
    backward_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
    backward_types = edge_type + num_relations
    
    # 合并
    bidirectional_edge_index = torch.cat([forward_edges, backward_edges], dim=1)
    bidirectional_edge_type = torch.cat([forward_types, backward_types], dim=0)
    
    return bidirectional_edge_index, bidirectional_edge_type

def filter_graph_by_entities(edge_index, edge_type, entity_mask):
    """
    根据实体掩码过滤图
    
    Args:
        edge_index: 边索引
        edge_type: 边类型
        entity_mask: 实体掩码 [num_entities]
        
    Returns:
        filtered_edge_index: 过滤后的边索引
        filtered_edge_type: 过滤后的边类型
    """
    # 找到有效的边（两个端点都在掩码中）
    src_valid = entity_mask[edge_index[0]]
    dst_valid = entity_mask[edge_index[1]]
    edge_mask = src_valid & dst_valid
    
    filtered_edge_index = edge_index[:, edge_mask]
    filtered_edge_type = edge_type[edge_mask]
    
    return filtered_edge_index, filtered_edge_type

def compute_graph_statistics(edge_index, num_entities, num_relations):
    """
    计算图的统计信息
    
    Args:
        edge_index: 边索引
        num_entities: 实体数量
        num_relations: 关系数量
        
    Returns:
        stats: 统计信息字典
    """
    num_edges = edge_index.size(1)
    
    # 计算度分布
    from torch_geometric.utils import degree
    in_degree = degree(edge_index[1], num_entities)
    out_degree = degree(edge_index[0], num_entities)
    
    stats = {
        'num_entities': num_entities,
        'num_relations': num_relations,
        'num_edges': num_edges,
        'avg_degree': (in_degree + out_degree).float().mean().item(),
        'max_in_degree': in_degree.max().item(),
        'max_out_degree': out_degree.max().item(),
        'density': num_edges / (num_entities * num_entities)
    }
    
    return stats

class GraphBatch:
    """
    图批处理工具类
    """
    def __init__(self, edge_indices, edge_types, batch_sizes):
        self.edge_indices = edge_indices
        self.edge_types = edge_types
        self.batch_sizes = batch_sizes
        self.num_graphs = len(edge_indices)
    
    def to(self, device):
        """移动到指定设备"""
        self.edge_indices = [ei.to(device) for ei in self.edge_indices]
        self.edge_types = [et.to(device) for et in self.edge_types]
        return self
    
    def get_batch_graph(self, batch_idx):
        """获取批次中的特定图"""
        return self.edge_indices[batch_idx], self.edge_types[batch_idx]

def create_graph_batch(triples_list, num_entities_list, num_relations):
    """
    创建图批次
    
    Args:
        triples_list: 三元组列表的列表
        num_entities_list: 每个图的实体数量列表
        num_relations: 关系数量
        
    Returns:
        GraphBatch: 图批次对象
    """
    edge_indices = []
    edge_types = []
    batch_sizes = []
    
    for triples, num_entities in zip(triples_list, num_entities_list):
        edge_index, edge_type = build_graph_from_triples(
            triples, num_entities, num_relations
        )
        edge_indices.append(edge_index)
        edge_types.append(edge_type)
        batch_sizes.append(num_entities)
    
    return GraphBatch(edge_indices, edge_types, batch_sizes)