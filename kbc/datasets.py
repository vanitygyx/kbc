import pickle
from typing import Dict, Tuple, List
import os

import numpy as np
import torch
from models import KBCModel
from graph_utils import build_graph_from_triples

class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)
        self.name = name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        
        # 构建图结构
        self.edge_index = None
        self.edge_type = None
        self._build_graph()
        
        # 加载多模态数据
        self.multimodal_data = None
        self._load_multimodal_data()

    def _build_graph(self):
        """
        从训练数据构建图结构
        """
        train_triples = self.data['train']
        self.edge_index, self.edge_type = build_graph_from_triples(
            train_triples, self.n_entities, self.n_predicates // 2
        )
        
    def _load_multimodal_data(self):
        """
        加载多模态特征数据
        """
        try:
            if self.name in ['WN9IMG', 'FBIMG',"MKG-W","MKG-Y","DB15K"]:
                # 加载预训练的多模态特征
                if self.name == 'WN9IMG':
                    visual_path = '../pre_train/matrix_wn_visual.npy'
                    textual_path = '../pre_train/matrix_wn_ling.npy'
                elif self.name == 'FBIMG':  
                    visual_path = '../pre_train/matrix_fb_visual.npy'
                    textual_path = '../pre_train/matrix_fb_ling.npy'
                elif self.name == 'MKG-W':  
                    visual_path = '../pre_train/MKG-W-visual.pth'
                    textual_path = '../pre_train/MKG-W-textual.pth'
                elif self.name == 'MKG-Y': 
                    visual_path = '../pre_train/MKG-Y-visual.pth'
                    textual_path = '../pre_train/MKG-Y-textual.pth'
                else:
                    visual_path = '../pre_train/DB15K-visual.pth'
                    textual_path = '../pre_train/DB15K-textual.pth'
                if self.name in ['MKG-W','MKG-Y','DB15K']:
                    visual_features = torch.load(visual_path)
                    textual_features = torch.load(textual_path)
                else:
                    visual_features = torch.tensor(np.load(visual_path), dtype=torch.float32)
                    textual_features = torch.tensor(np.load(textual_path), dtype=torch.float32)
                
                self.multimodal_data = {
                    'visual': visual_features,
                    'textual': textual_features
                }
                
                print(f"Loaded multimodal data: visual {visual_features.shape}, textual {textual_features.shape}")
                
        except Exception as e:
            print(f"Warning: Could not load multimodal data: {e}")
            # 创建虚拟的多模态数据
            self.multimodal_data = {
                'visual': torch.randn(self.n_entities, 768),
                'textual': torch.randn(self.n_entities, 768)
            }

    def get_graph_structure(self):
        """
        获取图结构
        """
        return self.edge_index, self.edge_type
    
    def get_multimodal_data(self):
        """
        获取多模态数据
        """
        return self.multimodal_data

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        model.eval()
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            # 检查是否为MultimodalComplEx模型
            if hasattr(model, 'encoder'):
                # MultimodalComplEx模型需要多模态数据
                if self.multimodal_data is not None:
                    ranks = model.get_ranking(q, self.to_skip[m], self.multimodal_data, batch_size=500)
                else:
                    ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            else:
                # 原有模型
                ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)

            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
