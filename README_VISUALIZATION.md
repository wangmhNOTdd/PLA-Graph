# 🎯 GET项目训练集前三个复合物可视化

## 📋 快速开始

### 1️⃣ 一键运行（推荐）
```bash
python run_train_visualization.py
```
按提示选择运行模式即可。

### 2️⃣ 直接运行
```bash
# 快速演示（不保存图片）
python visualize_pdbbind_example.py --demo

# 完整可视化（保存图片）
python visualize_pdbbind_example.py
```

## 📊 输出结果

您将获得训练集前3个复合物的以下可视化：

### 🔬 每个复合物包含3种图片：
1. **3D结构图** - 三维分子结构
2. **2D图结构** - 拓扑连接关系
3. **连接矩阵** - 分子块相互作用

### 📁 文件保存位置：
```
./visualization_output/train_first_3/
├── sample_0_* (第1个复合物)
├── sample_1_* (第2个复合物)
└── sample_2_* (第3个复合物)
```

## 🛠️ 故障排除

### 问题：找不到训练集文件
```bash
python test_train_visualization.py  # 检查环境
```

### 问题：缺少依赖包
```bash
python install_visualization_deps.py  # 安装依赖
```

## 📖 详细文档

- `VISUALIZATION_GUIDE.md` - 完整使用指南
- `MODIFICATIONS_SUMMARY.md` - 技术细节说明

---
🎉 **准备就绪！现在您可以可视化训练集的前三个复合物了！**
