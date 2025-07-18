# GET项目训练集前三个复合物可视化指南

## 概述
这个可视化工具专门用于查看GET项目中PDBbind identity30训练集的前三个复合物的图结构。

## 使用步骤

### 1. 安装依赖
```bash
python install_visualization_deps.py
```

### 2. 测试环境
```bash
python test_train_visualization.py
```

### 3. 运行可视化

#### 快速演示（不保存图片）
```bash
python visualize_pdbbind_example.py --demo
```
这将显示训练集第一个复合物的可视化，但不会保存图片文件。

#### 完整可视化（保存图片）
```bash
python visualize_pdbbind_example.py
```
这将可视化训练集的前三个复合物，并将图片保存到 `./visualization_output/train_first_3/` 目录中。

## 输出文件说明

每个复合物会生成三种类型的图片：

1. **`*_3d_structure.png`** - 3D分子结构图
   - 显示蛋白质和配体的3D空间结构
   - 不同颜色表示不同的原子类型
   - 包含坐标轴和距离信息

2. **`*_2d_graph.png`** - 2D图结构和统计信息
   - 显示图的拓扑结构
   - 包含节点和边的连接关系
   - 显示图的统计信息（节点数、边数等）

3. **`*_connectivity_matrix.png`** - 块连接矩阵热图
   - 显示不同分子块之间的连接模式
   - 热图颜色表示连接强度
   - 有助于理解分子间的相互作用

## 输出目录结构
```
./visualization_output/train_first_3/
├── sample_0_3d_structure.png
├── sample_0_2d_graph.png
├── sample_0_connectivity_matrix.png
├── sample_1_3d_structure.png
├── sample_1_2d_graph.png
├── sample_1_connectivity_matrix.png
├── sample_2_3d_structure.png
├── sample_2_2d_graph.png
└── sample_2_connectivity_matrix.png
```

## 数据集要求

确保您已经处理过PDBbind identity30数据集，并且以下文件存在：
- `./datasets/PDBBind/processed/identity30/train.pkl`

如果文件不存在，请先运行数据处理脚本。

## 故障排除

### 问题1：找不到训练集文件
```
❌ 训练集文件不存在: ./datasets/PDBBind/processed/identity30/train.pkl
```
**解决方案**：请确保已经运行过PDBbind数据处理脚本。

### 问题2：缺少依赖包
```
❌ 缺少依赖包: No module named 'matplotlib'
```
**解决方案**：运行 `python install_visualization_deps.py`

### 问题3：可视化过程中出错
**解决方案**：
1. 检查数据集文件是否完整
2. 确保所有依赖包都已正确安装
3. 查看终端输出的详细错误信息

## 自定义选项

如果您想修改可视化参数，可以编辑 `visualize_pdbbind_example.py` 文件中的以下参数：

- `k_neighbors`: KNN图的邻居数量（默认为9）
- `figsize`: 图片大小（默认为(15, 10)）
- `max_samples`: 可视化的样本数量（目前固定为3）

## 联系方式

如果遇到问题，请检查：
1. 数据集文件是否存在且完整
2. 依赖包是否正确安装
3. Python环境是否配置正确
