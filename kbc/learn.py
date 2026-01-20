import os
import json
import argparse
import numpy as np

import torch
from torch import optim

from .datasets import Dataset
from .models import *
from .regularizers import *
from .optimizers import KBCOptimizer

datasets = ['WN9IMG','FBIMG']

parser = argparse.ArgumentParser(
    description="Multi-modal Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='OTKGE_wn'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model = None
regularizer = None

# 特殊处理MultimodalComplEx模型
if args.model == 'MultimodalComplEx':
    # 获取多模态数据维度
    multimodal_data = dataset.get_multimodal_data()
    if multimodal_data is not None:
        visual_dim = multimodal_data['visual'].shape[1]
        textual_dim = multimodal_data['textual'].shape[1]
    else:
        # 默认维度
        visual_dim = 512
        textual_dim = 768
    rel_dim = args.rank  # 关系维度设为rank
    
    model = MultimodalComplEx(
        sizes=dataset.get_shape(),
        rank=args.rank,
        visual_dim=visual_dim,
        textual_dim=textual_dim,
        rel_dim=rel_dim,
        init_size=args.init
    )
    
    # 构建图结构
    train_triples = dataset.get_train()
    if train_triples is not None:
        model.build_graph(train_triples)
    
else:
    # 原有模型创建方式
    exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init)')

exec('regularizer = '+args.regularizer+'(args.reg)')
regularizer = [regularizer, N3(args.reg)]

device = 'cuda'
model.to(device)
for reg in regularizer:
    reg.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

# 为MultimodalComplEx模型传递多模态数据
multimodal_data = dataset.get_multimodal_data() if args.model == 'MultimodalComplEx' else None
optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, multimodal_data=multimodal_data)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    #m =  (mrrs['lhs'] + mrrs['rhs'])/2
    #h = (hits['lhs'] + hits['rhs'])/2
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0

if args.checkpoint != '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))

if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e+1))

            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)

            if (e + 1) % args.valid == 0:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]

                print("\t TRAIN: ", train)
                print("\t VALID: ", valid)

                log_file.write("Epoch: {}\n".format(e+1))
                log_file.write("\t TRAIN: {}\n".format(train))
                log_file.write("\t VALID: {}\n".format(valid))

                log_file.flush()

        test = avg_both(*dataset.eval(model, 'test', 50000))
        print("\t TEST : ", test)

if args.do_save:
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
    elif len_emb == 3:
        np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')

