# Knowledge Base Completion (kbc)
This code reproduces results in [Canonical Tensor Decomposition for Knowledge Base Completion](https://arxiv.org/abs/1806.07297) (ICML 2018).

kbc为baseline 构建multimodal graph embedding model with graph attention and contrastive learning

## Running the code
Reproduce the results below with the following command :
```
python kbc/learn.py --dataset FB15K --model ComplEx --rank 500 --optimizer
Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 1e-2
 --max_epochs 100 --valid 5
```


## License
kbc is CC-BY-NC licensed, as found in the LICENSE file.
