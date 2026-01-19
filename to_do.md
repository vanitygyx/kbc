# MKBC改造方案 - 多模态知识图谱补全框架

## 项目概述

基于kbc项目架构，融合OTKGE的多模态融合策略和KRACL的图注意力编码器，构建统一的多模态知识图谱补全框架。核心思路是直接使用OTKGE现成的多模态数据和融合方案，将多模态信息集成到KRACL的KBGAT_conv编码器中，最终通过ComplEx进行链接预测。

## 改造步骤

### 阶段1：多模态数据集成（最小化修改）

#### 1.1 OTKGE多模态数据直接使用
**技术细节**：
- 直接使用OTKGE现成的多模态数据（WN9IMG、FBIMG）
- 复用OTKGE的预训练特征文件（matrix_wn_ling.npy、matrix_wn_visual.npy等）
- 保持OTKGE的数据格式和加载方式

**涉及文件**：
- `kbc/datasets.py` (修改，添加多模态数据加载)
- `kbc/data/` (新建目录，存放OTKGE数据)

### 阶段2：多模态图注意力编码器创新

#### 2.1 多模态KBGAT_conv设计（核心创新）
**技术细节**：
- 基于KRACL的KBGAT_conv，集成OTKGE的多模态融合策略
- **创新点1**：多模态实体嵌入初始化
  - 使用OTKGE的加权融合方案：`embedding = (1-α-γ)*structural + α*visual + γ*textual`
  - 在KBGAT_conv的forward中动态计算多模态融合嵌入
- **创新点2**：多模态注意力机制
  - 在attention计算中同时考虑结构、视觉、文本信息
  - 修改`message`函数，将多模态特征融入三元组表示：`(x_i_multimodal, x_j_multimodal, edge_emb)`
- **创新点3**：模态感知的关系变换
  - 为不同模态设计专门的关系变换操作
  - 实现模态特定的注意力权重学习

**涉及文件**：
- `kbc/models.py` (修改，添加MultimodalKBGAT类)
- `kbc/encoders.py` (新建，移植并改进KBGAT_conv)

#### 2.2 图结构构建（最小化修改）
**技术细节**：
- 复用kbc现有的三元组处理逻辑
- 添加双向边构建功能
- 集成到现有Dataset类中

**涉及文件**：
- `kbc/datasets.py` (修改，添加图结构构建)
- `kbc/graph_utils.py` (新建，图处理工具函数)

### 阶段3：ComplEx解码器适配

#### 3.1 多模态ComplEx模型（最小化修改）
**技术细节**：
- 基于现有ComplEx类，添加多模态编码器支持
- 修改forward函数，集成KBGAT编码器输出
- 保持ComplEx双线性评分函数不变

**核心修改**：
```python
def forward(self, x, multimodal_data=None):
    # 1. 多模态图注意力编码
    if hasattr(self, 'encoder'):
        encoded_emb = self.encoder(x, multimodal_data)  # KBGAT输出
        # 使用编码后的嵌入替换原始嵌入
        lhs = encoded_emb[x[:, 0]]
        rhs = encoded_emb[x[:, 2]]
    else:
        # 原始ComplEx逻辑
        lhs = self.embeddings[0](x[:, 0])
        rhs = self.embeddings[0](x[:, 2])
    
    rel = self.embeddings[1](x[:, 1])  # 关系嵌入保持不变
    
    # 后续ComplEx评分逻辑保持不变
    lhs = lhs[:, :self.rank], lhs[:, self.rank:]
    rel = rel[:, :self.rank], rel[:, self.rank:]
    rhs = rhs[:, :self.rank], rhs[:, self.rank:]
    
    to_score = encoded_emb if hasattr(self, 'encoder') else self.embeddings[0].weight
    to_score = to_score[:, :self.rank], to_score[:, self.rank:]
    
    return (
        (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
        (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
    ), (
        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
    )
```

**涉及文件**：
- `kbc/models.py` (修改ComplEx类，添加MultimodalComplEx类)

### 阶段4：训练框架集成（最小化修改）

#### 4.1 学习脚本适配
**技术细节**：
- 修改现有learn.py，支持多模态模型训练
- 集成OTKGE的训练逻辑和参数设置
- 保持kbc原有的优化器和正则化器

**涉及文件**：
- `kbc/learn.py` (修改，添加多模态支持)

#### 4.2 评估框架扩展
**技术细节**：
- 扩展现有Dataset.eval方法
- 添加多模态数据的评估支持
- 保持标准KGC评估指标（MRR、Hits@K）

**涉及文件**：
- `kbc/datasets.py` (修改eval方法)

## 核心技术创新

### 多模态KBGAT_conv架构设计

#### 创新点1：多模态实体表示融合
```python
class MultimodalKBGAT(KBGAT_conv):
    def __init__(self, ...):
        super().__init__(...)
        # OTKGE风格的多模态参数
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 视觉权重
        self.gamma = nn.Parameter(torch.tensor(0.8))  # 文本权重
        
    def get_multimodal_embeddings(self, structural_emb, visual_emb, textual_emb):
        # OTKGE融合策略
        return (1 - self.alpha - self.gamma) * structural_emb + \
               self.alpha * visual_emb + self.gamma * textual_emb
```

#### 创新点2：多模态注意力机制
```python
def message(self, x_i, x_j, edge_type, relation_embedding, ...):
    # 获取多模态融合后的实体表示
    x_i_multimodal = self.get_multimodal_embeddings(x_i, visual_i, textual_i)
    x_j_multimodal = self.get_multimodal_embeddings(x_j, visual_j, textual_j)
    
    # 原KBGAT_conv注意力计算，但使用多模态特征
    edge_emb = torch.index_select(relation_embedding, 0, edge_type)
    triple_emb = torch.cat((x_i_multimodal, x_j_multimodal, edge_emb), dim=1)
    
    # 后续注意力计算保持不变
    c = self.w_1(triple_emb)
    b = self.leaky_relu(self.w_2(c))
    alpha = softmax(b, index, ptr, size_i)
    
    return c * alpha.view(-1, 1)
```

#### 创新点3：端到端多模态ComplEx
```python
class MultimodalComplEx(ComplEx):
    def __init__(self, sizes, rank, multimodal_data, init_size=1e-3):
        super().__init__(sizes, rank, init_size)
        self.encoder = MultimodalKBGAT(...)
        self.multimodal_data = multimodal_data
        
    def forward(self, x):
        # 多模态图注意力编码
        encoded_embeddings = self.encoder(x, self.multimodal_data)
        
        # 使用编码后的嵌入进行ComplEx评分
        lhs = encoded_embeddings[x[:, 0]]
        rhs = encoded_embeddings[x[:, 2]]
        rel = self.embeddings[1](x[:, 1])
        
        # ComplEx双线性评分（保持不变）
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
```

## 实施计划（最小化修改）

### 文件修改清单

#### 必须修改的文件（3个）
1. **`kbc/models.py`**：
   - 添加MultimodalKBGAT类（基于KRACL的KBGAT_conv）
   - 添加MultimodalComplEx类（扩展现有ComplEx）

2. **`kbc/datasets.py`**：
   - 添加多模态数据加载功能（复用OTKGE数据格式）
   - 扩展eval方法支持多模态评估

3. **`kbc/learn.py`**：
   - 添加多模态模型的训练支持
   - 集成OTKGE的训练参数和逻辑

#### 新建文件（2个）
1. **`kbc/encoders.py`**：移植KRACL的KBGAT_conv和相关工具函数
2. **`kbc/graph_utils.py`**：图结构构建和处理工具

### 技术优势

#### 1. 最小化修改
- 复用OTKGE现成数据和融合策略
- 基于kbc现有架构，最小化代码变动
- 保持ComplEx解码器的理论优势

#### 2. 核心创新
- **多模态图注意力**：首次将多模态信息集成到图注意力机制中
- **端到端训练**：编码器-解码器联合优化
- **模态感知注意力**：不同模态信息的差异化处理

#### 3. 实验优势
- 直接对比OTKGE（相同数据，不同编码器）
- 直接对比KRACL（相同编码器，增加多模态）
- 直接对比kbc baseline（相同解码器，增加编码器）

### 预期效果

#### 性能提升
- **相比OTKGE**：图注意力编码器带来的结构建模优势
- **相比KRACL**：多模态信息带来的表示学习优势  
- **相比kbc**：编码器+多模态的双重提升

#### 技术贡献
1. **首个多模态图注意力KGC框架**
2. **OTKGE+KRACL+ComplEx的有效融合**
3. **最小化修改的工程实践方案**

通过这个精简的改造方案，我们将以最小的代码修改量实现多模态图注意力知识图谱补全的技术创新，为多模态KGC研究提供新的技术路径。