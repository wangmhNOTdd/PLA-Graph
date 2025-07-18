# GET项目可视化修改总结

## 修改内容

### 1. 主要修改 - visualize_pdbbind_example.py
- **简化功能**：专注于训练集前三个复合物的可视化
- **移除多数据集支持**：不再处理验证集和测试集
- **固定样本数量**：仅可视化训练集的前3个样本
- **优化输出目录**：结果保存到 `./visualization_output/train_first_3/`
- **改进中文提示**：所有输出信息更改为中文

### 2. 新增文件

#### test_train_visualization.py
- 环境检查脚本
- 验证训练集文件是否存在
- 检查依赖包是否可用
- 提供使用指南

#### run_train_visualization.py
- 一键运行脚本
- 交互式选择运行模式
- 自动检查环境和运行可视化

#### VISUALIZATION_GUIDE.md
- 详细的使用指南
- 故障排除说明
- 输出文件说明

### 3. 更新现有文件

#### install_visualization_deps.py
- 更新使用说明
- 添加新的测试脚本引用

## 使用方法

### 方法1：一键运行（推荐）
```bash
python run_train_visualization.py
```

### 方法2：直接运行
```bash
# 快速演示
python visualize_pdbbind_example.py --demo

# 完整可视化
python visualize_pdbbind_example.py
```

### 方法3：测试环境
```bash
python test_train_visualization.py
```

## 输出结果

### 文件结构
```
./visualization_output/train_first_3/
├── sample_0_3d_structure.png        # 第1个复合物的3D结构
├── sample_0_2d_graph.png           # 第1个复合物的2D图结构
├── sample_0_connectivity_matrix.png # 第1个复合物的连接矩阵
├── sample_1_3d_structure.png        # 第2个复合物的3D结构
├── sample_1_2d_graph.png           # 第2个复合物的2D图结构
├── sample_1_connectivity_matrix.png # 第2个复合物的连接矩阵
├── sample_2_3d_structure.png        # 第3个复合物的3D结构
├── sample_2_2d_graph.png           # 第3个复合物的2D图结构
└── sample_2_connectivity_matrix.png # 第3个复合物的连接矩阵
```

### 图片内容
1. **3D结构图** - 显示蛋白质和配体的三维空间结构
2. **2D图结构** - 显示图的拓扑结构和统计信息
3. **连接矩阵** - 显示分子块之间的连接模式

## 技术细节

### 数据来源
- 训练集文件：`./datasets/PDBBind/processed/identity30/train.pkl`
- 仅处理前3个复合物样本

### 可视化参数
- KNN邻居数：k=9
- 图片大小：(15, 10)
- 输出格式：PNG

### 依赖包
- matplotlib（绘图）
- numpy（数值计算）
- networkx（图结构）
- pickle（数据加载）

## 优势

1. **专注性**：只关注训练集前三个样本，避免信息过载
2. **简单性**：一键运行，无需复杂配置
3. **完整性**：提供多种可视化视角
4. **可靠性**：包含环境检查和错误处理
5. **中文化**：所有提示信息为中文

## 注意事项

1. 确保PDBbind identity30数据集已处理
2. 运行前先安装依赖包
3. 可视化过程可能需要几分钟时间
4. 生成的图片文件较大，注意磁盘空间
