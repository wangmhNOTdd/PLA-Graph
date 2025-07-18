# GET项目 - 分子图可视化工具

本工具集用于可视化GET项目中PDBbind identity30数据集的分子复合物图结构。

## 功能特性

- **3D分子结构可视化**：显示分子的三维空间结构
- **2D图结构可视化**：显示块级别的图连接关系和统计信息
- **连接矩阵热图**：显示块之间连接关系的矩阵表示
- **交互式探索**：通过Jupyter notebook进行交互式数据探索
- **批量处理**：支持批量可视化多个样本

## 文件说明

- `visualize_graph.py` - 主要的可视化工具类和命令行接口
- `visualize_pdbbind_example.py` - PDBbind数据集的示例脚本
- `molecular_graph_visualization.ipynb` - 交互式Jupyter notebook
- `install_visualization_deps.py` - 依赖包安装脚本

## 安装依赖

首先运行安装脚本来确保所有依赖包都已安装：

```bash
python install_visualization_deps.py
```

## 使用方法

### 1. 命令行使用

#### 基本可视化
```bash
python visualize_graph.py --dataset ./datasets/PDBBind/processed/identity30/test.pkl --index 0
```

#### 指定输出目录
```bash
python visualize_graph.py --dataset ./datasets/PDBBind/processed/identity30/test.pkl --index 0 --output_dir ./output
```

#### 调整k邻居数
```bash
python visualize_graph.py --dataset ./datasets/PDBBind/processed/identity30/test.pkl --index 0 --k_neighbors 15
```

#### 完整参数列表
```bash
python visualize_graph.py --help
```

### 2. PDBbind示例脚本

#### 快速演示（不保存文件）
```bash
python visualize_pdbbind_example.py --demo
```

#### 完整可视化（保存到文件）
```bash
python visualize_pdbbind_example.py
```

### 3. Jupyter Notebook交互式使用

启动Jupyter notebook：
```bash
jupyter notebook molecular_graph_visualization.ipynb
```

在notebook中可以：
- 交互式选择数据集、样本索引和可视化参数
- 实时查看可视化结果
- 批量处理多个样本
- 分析数据集统计信息

## 数据集路径配置

默认的数据集路径配置如下：
```python
dataset_paths = {
    'train': './datasets/PDBBind/processed/identity30/train.pkl',
    'valid': './datasets/PDBBind/processed/identity30/valid.pkl',
    'test': './datasets/PDBBind/processed/identity30/test.pkl'
}
```

如果您的数据集路径不同，请相应修改路径。

## 可视化类型说明

### 3D Structure (3D分子结构)
- 显示分子的三维空间结构
- 不同颜色表示不同的段(segment)
- 原子用小球表示，块中心用方块表示
- 坐标轴单位为埃(Å)

### 2D Graph (2D图结构)
- 左侧：显示块级别的图连接关系
- 右侧：显示详细的统计信息
- 包含块数量、原子数量、边数量等信息
- 显示段分布和原子类型分布

### Connectivity Matrix (连接矩阵热图)
- 显示块之间连接关系的矩阵表示
- 热图形式，1表示连接，0表示未连接
- 有助于理解图的连接模式

## 输出文件

可视化结果将保存在指定的输出目录中，包含：
- `*_3d_structure.png` - 3D分子结构图
- `*_2d_graph.png` - 2D图结构和统计信息
- `*_connectivity_matrix.png` - 块连接矩阵热图

## 参数说明

- `--dataset` - 数据集文件路径（.pkl格式）
- `--index` - 要可视化的样本索引（默认：0）
- `--k_neighbors` - k近邻图构建的邻居数量（默认：9）
- `--output_dir` - 输出目录路径（可选）
- `--figsize` - 图形尺寸（宽度 高度，默认：15 10）

## 示例输出

运行可视化工具后，您将看到：

1. **3D分子结构图**：显示分子的三维空间排列
2. **2D图结构图**：显示块级别的连接关系和统计信息
3. **连接矩阵热图**：显示块之间的连接模式

## 故障排除

### 常见问题

1. **数据集文件未找到**
   - 检查数据集路径是否正确
   - 确认已正确处理PDBbind数据集

2. **依赖包未安装**
   - 运行 `python install_visualization_deps.py`
   - 手动安装缺失的包

3. **内存不足**
   - 对于大型分子，减少可视化的样本数量
   - 降低图形分辨率

4. **图形显示问题**
   - 在服务器环境中，可能需要设置 `DISPLAY` 环境变量
   - 使用 `--output_dir` 参数保存图形到文件

### 性能优化

- 对于大型数据集，建议使用批量处理而非逐个可视化
- 可以通过调整 `k_neighbors` 参数来平衡图的复杂度和计算时间
- 使用较小的图形尺寸可以加快渲染速度

## 技术细节

### 图结构构建

1. **原子节点**：每个原子作为一个节点
2. **块节点**：每个残基/分子片段作为一个块节点
3. **连接关系**：
   - 原子与所属块的连接
   - 块之间的k近邻连接（基于空间距离）

### 数据格式

输入数据应包含以下字段：
- `X`: 原子坐标 `[n_atoms, 3]`
- `B`: 块类型 `[n_blocks]`
- `A`: 原子类型 `[n_atoms]`
- `block_lengths`: 每个块的原子数量 `[n_blocks]`
- `segment_ids`: 每个块属于哪个段 `[n_blocks]`
- `id`: 分子ID
- `affinity`: 结合亲和力

## 扩展功能

### 自定义可视化

您可以通过修改 `GraphVisualizer` 类来添加自定义的可视化功能：

```python
from visualize_graph import GraphVisualizer

# 创建自定义可视化器
visualizer = GraphVisualizer(figsize=(12, 8))

# 加载数据
sample = visualizer.load_data(dataset_path, index=0)

# 自定义可视化
# ... 您的自定义代码 ...
```

### 集成到训练流程

可以将可视化工具集成到训练流程中，用于：
- 训练数据的质量检查
- 模型预测结果的可视化
- 异常样本的识别

## 贡献和反馈

如果您在使用过程中遇到问题或有改进建议，请：
1. 检查本README的故障排除部分
2. 查看代码注释中的详细说明
3. 提交Issue或Pull Request

## 许可证

本工具集遵循GET项目的许可证。
