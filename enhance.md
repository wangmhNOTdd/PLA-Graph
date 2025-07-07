下面给出 **Hyper‑Edge GET (HE‑GET)** 的完整实现指南——从输入嵌入到输出读数，精确到每一层的张量形状、参数矩阵大小与数学公式。所有符号与坐标约定沿用 GET 原论文【turn1file4】【turn1file15】。

---

## 0 符号约定与总体超参

| 记号             | 默认值          | 说明                          |
| -------------- | ------------ | --------------------------- |
| $d_\text{h}$   | **128**      | 原子特征向量通道数（Hi 列数）            |
| $d_\text{r}$   | **64**       | 注意力中间维度（Query/Key/Value 行数） |
| $d_\text{H}$   | **32**       | 块‑级双曲空间维度                   |
| $d_\text{rbf}$ | **16**       | RBF 距离嵌入维度                  |
| $d_\text{e}$   | **32**       | 边特征维度（更新后）                  |
| $L$            | **6**        | HE‑GET 层数                   |
| 头数 $h$         | **8**        | 多头 = $d_\text{r}/h = 8$     |
| 曲率 $K_\ell$    | **可学习，初值=1** | 每层独立                        |

---

## 1 输入嵌入层

### 1.1 原子嵌入

$$
H_i^{(0)}[p]=E_{\text{atom}}(a_i[p]) \;+\;E_{\text{block}}(b_i)\;+\;E_{\text{pos}}(p_i[p])\quad\in\mathbb R^{d_\text{h}}
$$

### 1.2 坐标

$$
\tilde X_i^{(0)}\in\mathbb R^{n_i\times3},\qquad
\text{不做缩放，直接输入后续等变模块。}
$$

### 1.3 边特征初始化

$$
E_{pq}^{(0)}=[\,\text{RBF}(\|\tilde x_p-\tilde x_q\|_2)\,] \in\mathbb R^{d_\text{rbf}}
$$

---

## 2 HE‑GET 基本层（第 $\ell=1\!\sim\!L$ 层）

<table>
<thead><tr><th>模块</th><th>张量/参数尺寸</th><th>公式</th></tr></thead>
<tbody>

<tr><td colspan=3 align=center><b>2.1 块‑级 Hyperbolic Attention</b></td></tr>

<tr><td>指数映射</td><td>\(c_i=\tfrac1{n_i}\sum_p\tilde X_i[p]\)<br>\(\mathbf z_i^{(\ell)}=\exp_{\mathbf o}^{K_\ell}\!\bigl((0,c_i)\bigr)\in\mathbb H_{K_\ell}^{d_\text{H}}\)</td><td></td></tr>

<tr><td>距离</td><td>标量</td><td>\(d_{K_\ell}(\mathbf z_i,\mathbf z_j)=\sqrt{K_\ell}\,\operatorname{arcosh}\!\bigl(-\langle \mathbf z_i,\mathbf z_j\rangle_L/K_\ell\bigr)\)</td></tr>

<tr><td>块注意力</td><td>\(\beta_{ij}^{(\ell)}\in\mathbb R\)</td><td>\(\displaystyle
\beta_{ij}^{(\ell)}=\frac{\exp(-d_{K_\ell}(\mathbf z_i,\mathbf z_j))}{\sum_{k\in\mathcal N(i)}\exp(-d_{K_\ell}(\mathbf z_i,\mathbf z_k))}\)</td></tr>

<tr><td colspan=3 align=center><b>2.2 原子级 Edge‑Enhanced Cross‑Attention</b></td></tr>

<tr><td>边更新</td><td>\(E_{pq}^{(\ell)} = \text{MLP}_\text{edge}^{(\ell)}\!\bigl([H_i^{(\ell-1)}[p]\!\Vert\! H_j^{(\ell-1)}[q]]\bigr)+E_{pq}^{(\ell-1)}\in\mathbb R^{d_\text{e}}\)</td><td></td></tr>

<tr><td>Q/K/V</td><td>\(W_Q^{(\ell)},W_K^{(\ell)},W_V^{(\ell)}\in\mathbb R^{d_\text{h}\times d_\text{r}}\)</td><td>\(Q_i=H_iW_Q,\;K_j=H_jW_K,\;V_j=H_jW_V\)</td></tr>

<tr><td>关系张量</td><td>\(R_{ij}\in\mathbb R^{n_i\times n_j\times d_\text{r}}\)</td><td>\(R_{ij}[p,q]=\varphi_A(Q_i[p],K_j[q],\text{RBF}(D_{ij}[p,q]),E_{pq}^{(\ell)})\)【turn1file4】</td></tr>

<tr><td>原子注意力</td><td>\(\alpha_{ij}\in\mathbb R^{n_i\times n_j}\)</td><td>\(\alpha_{ij}=\operatorname{Softmax}(R_{ij}W_A^{(\ell)})\), \(W_A^{(\ell)}\!\in\!\mathbb R^{d_\text{r}\times1}\)</td></tr>

<tr><td>信息聚合<br>(与块权重结合)</td><td>\(H_i^{\dagger},\;\tilde X_i^{\dagger}\)</td><td>
\(\displaystyle
m_{ij,p}=\alpha_{ij}[p]\cdot\varphi_v(V_j\!\Vert\! \text{RBF}(D_{ij}[p]))\)<br>
\(H_i^{\dagger}[p]=\sum_{j}\beta_{ij}^{(\ell)}\,\varphi_m(m_{ij,p})\)<br>
\(\tilde X_i^{\dagger}[p]=\sum_{j}\beta_{ij}^{(\ell)}\bigl(\sigma_m(m_{ij,p})\odot (\alpha_{ij}[p]\tilde X_{ij}[p])\bigr)\)
</td></tr>

<tr><td colspan=3 align=center><b>2.3 残差与等变 FFN</b></td></tr>

<tr><td>更新</td><td>详见【turn1file15】</td><td>
\(\displaystyle
H_i^{\star}[p]=H_i^{(\ell-1)}[p]+H_i^{\dagger}[p]\)<br>
\(\tilde X_i^{\star}[p]=\tilde X_i^{(\ell-1)}[p]+\tilde X_i^{\dagger}[p]\)
</td></tr>

<tr><td>FFN</td><td>\(W_{1}^{(\ell)}\in\mathbb R^{d_\text{h}\times4d_\text{h}}\)<br>\(W_{2}^{(\ell)}\in\mathbb R^{4d_\text{h}\times d_\text{h}}\)</td><td>
\[
\begin{aligned}
h' &= H_i^{\star}[p]+W_2^{(\ell)}\sigma(W_1^{(\ell)}H_i^{\star}[p])\\
x' &= \tilde X_i^{\star}[p]+(\tilde X_i^{\star}[p]-\bar x_i)\odot \sigma_x(h')
\end{aligned}
\]
</td></tr>

<tr><td colspan=3 align=center><b>2.4 等变 LayerNorm</b></td></tr>

<tr><td>特征</td><td>\(\gamma,\beta\in\mathbb R^{d_\text{h}}\)</td><td>\(H_i^{(\ell)}=\operatorname{LN}(h';\gamma,\beta)\)【turn1file15】</td></tr>

<tr><td>坐标</td><td>\(\sigma\in\mathbb R^{1}\)</td><td>\(\tilde X_i^{(\ell)}=\operatorname{ELN}(x';\sigma)\)【turn1file15】</td></tr>

</tbody></table>

> **维度校验**
>
> * 所有矩阵‑向量乘结果行数都保持 $n_i$。
> * 注意力内对 $n_i\!\times\! n_j$ 行/列做 softmax，不改变形状。
> * 双曲空间操作只针对 **块心坐标**，与原子数无关，因此 O($B^2$) 而非 O($\sum n_i^2$)。

---

## 3 全局 APPNP 增强

在 **每两个 HE‑GET 层之间** 追加一次 APPNP 传播以捕获长程拓扑信息（来自 EIGN）：

$$
\mathbf H^{(\ell)}\leftarrow(1-\alpha)\,\hat A \mathbf H^{(\ell)}+\alpha\,\mathbf H^{(0)},\quad\hat A=D^{-1/2}AD^{-1/2},\;\alpha=0.1
$$

张量尺寸保持 $(\sum_i n_i)\times d_\text{h}$。

---

## 4 输出读取

* **图级读数**：对所有块取平均
  $\displaystyle h_\text{graph}=\frac1{\sum n_i}\sum_{i,p} H_i^{(L)}[p]$

* **任务头**

  * 分类：$ \text{MLP}_{\text{cls}}: \mathbb R^{d_\text{h}}\!\to\!\mathbb R^{C}$
  * 回归：$ \text{MLP}_{\text{reg}}: \mathbb R^{d_\text{h}}\!\to\!\mathbb R^{1}$

---

## 5 实现要点

1. **双曲运算**

   * 使用 *torch‑hype* 或 *geoopt* 库；对数/指数映射按公式 (6)(7) 实现【HAT 公式 (7)】。
   * 曲率 $K_\ell$ 作为 nn.Parameter 并约束 $K_\ell>10^{-4}$（softplus）。

2. **边增强**

   * Edge‑MLP：两层 $d_\text{h}\!\to\! d_\text{e}\!\to\! d_\text{e}$，SiLU 激活，残差加回原 $E^{(\ell-1)}$。

3. **多头实现**

   * 将 $d_\text{r}$ 按头分割；α、β 在每头独立计算，再 concat。

4. **批处理**

   * 以 **块** 为批单元；双曲坐标使用 `manifold.batch_tangent2exp`。

5. **复杂度**

   * 时间：O$\bigl(B^2 d_\text{H} + \sum_{i,j} n_i n_j d_\text{r}\bigr)$。实践中 $B\ll\sum n_i$，瓶颈仍在原子注意力。

---

## 6 训练建议

| 项             | 设置                             |
| ------------- | ------------------------------ |
| Optimizer     | AdamW, lr $2\!\times\!10^{-4}$ |
| Warm‑up       | 2 000 steps，余 Cosine decay     |
| Dropout       | 0.1 (attention / FFN)          |
| Batch size    | 8 \~ 16 个复合体（依显存）              |
| Gradient clip | 1.0                            |
| 正则            | 曲率 L2（0.01），边置乱比例 15 %         |

---

## 7 最小 PyTorch 伪代码（核心）

```python
class HEGETLayer(nn.Module):
    def __init__(self, d_h=128, d_r=64, d_H=32, d_e=32, n_head=8):
        super().__init__()
        self.W_Q = nn.Linear(d_h, d_r, bias=False)
        self.W_K = nn.Linear(d_h, d_r, bias=False)
        self.W_V = nn.Linear(d_h, d_r, bias=False)
        self.W_A = nn.Linear(d_r, 1,  bias=False)
        self.W_B = nn.Linear(d_r, 1,  bias=False)
        self.edge_mlp = MLP(2*d_h, d_e)
        ...
    def forward(self, H, X, blocks, E):
        # 1) hyperbolic block attention → β
        z = exp_map_zero(block_centroids(X), self.K)
        d = arcosh_dist(z)                       # (B,B)
        beta = softmax(-d, dim=-1)
        # 2) edge update
        E = self.edge_mlp(cat(Hi,Hj))+E
        # 3) atom‑level attention → α
        ...
        return H_new, X_new, E
```

---

## 8 总结

HE‑GET 在 **块层面引入可学习曲率的双曲注意力**，充分容纳层次/指数增长结构；在 **原子层面融合边更新 + APPNP**，显式强化局部‑全局通路。按照以上尺寸与公式逐层堆叠，即可复现本文报告的性能，并保留 GET 的 E(3) 等变与块内置换不变性质【turn1file5】【turn1file15】。若有进一步的工程化需求（混合精度、分布式训练等），只需对前述张量形状保持一致即可安全替换。
