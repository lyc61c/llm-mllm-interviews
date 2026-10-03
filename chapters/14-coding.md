# 手撕代码与算法

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [模型算子与数值实现](#topic-1)
  - [COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？](#cod-001)
  - [COD-002 · 手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？](#cod-002)
  - [COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。](#cod-004)
  - [COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。](#cod-007)
  - [COD-018 · 用 PyTorch 实现两层 MLP，图像输入应该怎样组织？](#cod-018)
- [损失函数与训练代码](#topic-2)
  - [COD-005 · 手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？](#cod-005)
  - [COD-008 · 实现 DPO loss，怎样避免符号和序列概率错误？](#cod-008)
  - [COD-009 · 实现 GRPO 组内优势，标准差为零时怎么办？](#cod-009)
- [采样与缓存实现](#topic-3)
  - [COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？](#cod-003)
  - [COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？](#cod-006)
  - [COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。](#cod-014)
- [通用算法与数据结构](#topic-4)
  - [COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？](#cod-010)
  - [COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。](#cod-011)
  - [COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。](#cod-012)
  - [COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？](#cod-013)
  - [COD-015 · 手撕代码时怎样设计能揭露错误的测试？](#cod-015)
  - [COD-016 · 手写最长公共子序列：怎样定义状态、推导转移并压缩空间？](#cod-016)
  - [COD-017 · 手写两数之和：怎样用单遍哈希表返回两个不同元素的下标？](#cod-017)

<a id="topic-1"></a>
## 模型算子与数值实现

<a id="cod-001"></a>
### COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？

**L1**

#### 答案

Softmax 对所有 logits 加同一常数不变，因为分子与分母中的公因子会抵消。令 $`m=\max_i z_i`$，先减最大值再计算指数，可以避免大正数导致溢出。

交叉熵直接计算 $`\mathrm{logsumexp}(z)-z_y`$，比先求概率再取对数更稳定；对 logits 的梯度为预测概率减去目标的 one-hot 向量。实现需要测试大正负 logits、平移不变性和部分 `-inf` mask。全为 `-inf` 的行没有有效概率分布，必须显式处理。

减去任意共同常数的等价性来自 $`e^{z-c}=e^ze^{-c}`$，归一化时公共因子抵消。它不能自动推广到任意函数：例如先 sigmoid 再归一化，一般有 $`\sigma(z_i-c)/\sum_j\sigma(z_j-c)\ne\sigma(z_i)/\sum_j\sigma(z_j)`$；ReLU 后归一化还可能出现全零分母。替换函数会改变权重、梯度甚至概率语义，数值稳定化必须按新公式推导，而不是照搬 softmax 的技巧。

```math
\begin{aligned}m&=\max_j z_j,\qquad p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}\\\mathcal L_{\mathrm{CE}}&=m+\log\sum_j e^{z_j-m}-z_y\end{aligned}
```

#### 易错点

- 只写 exp(z)/sum(exp(z))；或者用 log(softmax) 在极端值下得到 log(0)。

#### 追问

- 如何实现 label smoothing 和 ignore_index？

<a id="cod-002"></a>
### COD-002 · 手撕 Multi-Head Attention：形状、缩放与 mask 怎么写？

**L1** · 字节跳动 / 腾讯

#### 答案

输入形状为 `[B,T,D]`，Q/K/V 投影后拆成 `[B,H,T,d]`，其中 $`D=Hd`$。可以使用一个 `D→3D` 线性层一次生成三组投影，但它们仍有各自的参数。

计算缩放点积后，分数形状为 `[B,H,Tq,Tk]`。先加因果或 padding mask，再沿 `Tk` 轴做 softmax，与 V 相乘；各头输出拼接回 `[B,T,D]`，最后经过输出投影。用输出形状和“未来 token 改变不影响较早位置”的性质检查实现。教学代码可显式生成二次方大小的分数矩阵，生产 SDPA 通常用融合内核；训练与推理的 dropout 设置也要区分。

若追问原始Transformer还包括什么，应根据层级说明：注意力子层内部有缩放、mask、softmax/dropout、多头拼接和输出投影；完整block还包括残差、LayerNorm与FFN，输入序列还需位置表达。题目未指名“关键一步”时先澄清是在问注意力算子还是整个block，不能把位置编码硬塞进softmax，或把后续模型的RoPE当作原始Transformer的实现。

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d}+M\right)V
```

#### 易错点

- 把布尔 mask 语义弄反；PyTorch 不同 API 的 True 可能表示允许或屏蔽。

#### 追问

- 加入 GQA 后 Q 头数与 KV 头数如何对应？

<a id="cod-004"></a>
### COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。

**L2**

#### 答案

RoPE 将偶数维向量按二维对旋转，每对使用不同频率。位置 $`m`$ 对应的角度为 $`\phi_i=m\,\mathrm{base}^{-2i/d}`$，一对分量 $`(a,b)`$ 变为 $`(a\cos\phi_i-b\sin\phi_i,\ a\sin\phi_i+b\cos\phi_i)`$。

旋转矩阵正交，因此保持向量范数；又因为 $`R_m^\top R_n=R_{n-m}`$，旋转后的 Q/K 内积通过旋转项依赖位置差。相邻维配对与 split-half 配对可以通过维度排列对应，但已有权重不能直接混用两种布局。实现应匹配模型的频率、配对方式和位置编号，并测试范数及相对位置内积。

```math
\begin{aligned}\phi_i&=m\,\mathrm{base}^{-2i/d},\qquad\lVert R_mx\rVert_2=\lVert x\rVert_2\\(R_mq)^\top(R_nk)&=q^\top R_{n-m}k\end{aligned}
```

#### 易错点

- 把 RoPE 当作加到 embedding 的位置向量，或对 V 也默认旋转。

#### 追问

- 频率缩放改变了哪个项？为什么公式可外推不代表效果必然可外推？

<a id="cod-007"></a>
### COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。

**L2**

#### 答案

冻结基座权重 $`W`$，增加低秩更新 $`\Delta W=(\alpha/r)BA`$。采用行 batch 输入时，前向为 $`xW^\top+(\alpha/r)(xA^\top)B^\top`$。矩阵形状分别为 `W[out,in]`、`A[r,in]`、`B[out,r]`，新增参数量为 $`r(\mathrm{in}+\mathrm{out})`$。

常见初始化是 A 随机、B 为零，使初始输出与基座一致；首步 A 的梯度为零，B 仍可学习。当 dropout 关闭且权重精度一致时，把 $`\Delta W`$ 合并到 W 应与未合并前向在容差内一致。测试还应确认冻结参数没有梯度；量化基座的合并需要额外处理。

#### 易错点

- 把低秩更新写成元素乘法，或 merge 后仍重复加 adapter。

#### 追问

- B 随机、A 零是否也能让初始更新为零？

<a id="cod-018"></a>
### COD-018 · 用 PyTorch 实现两层 MLP，图像输入应该怎样组织？

**L2** · 字节跳动

#### 答案

两层 MLP 通常指两个可学习的线性层，中间加入非线性：Linear(d,h) → GELU → Linear(h,c)。继承 nn.Module，在 __init__ 中定义子层并调用 super().__init__，在 forward 中组合计算；这样权重和偏置会被自动注册，优化器才能找到它们。若两个线性层之间没有非线性，整体仍可合并成一次仿射变换。

PyTorch 的 Linear 只变换最后一个维度，权重形状是 [out_features,in_features]，其余前导维保持不变。普通向量批次 [B,d] 会得到 [B,c]；对视觉 patch/token 特征 [B,N,d] 使用同一 MLP，会得到 [B,N,c]，每个 token 共用参数，但这一步不会让不同 patch 之间交换信息。

若任务是最简单的固定尺寸图像分类，可以将 [B,C,H,W] 用 flatten(start_dim=1) 变成 [B,CHW]，再送入 MLP，输出 [B,num_classes] 的 logits。例如 ImageMLP((3,32,32),128,10) 接收32×32的RGB图像并输出10类分数；训练时可把原始 logits 直接传给 CrossEntropyLoss。不能把 batch 维一起摊平，也不能把 NCHW 图像直接交给 Linear 后误以为它会自动处理整个图像——它实际只会映射最后的 W 维。

整图 flatten 的第一层参数量随 CHW 增长，且固定输入尺寸；它保留了像素在向量中的顺序，但没有 CNN 的局部连接等空间归纳偏置。实际视觉模型常先提取 patch 或编码器特征，再用逐 token MLP 做投影，若需要跨 patch 交互，还要加入注意力、卷积或其他空间混合模块。

参考类 `MLP` 实现逐向量或逐 token 映射，`ImageMLP` 封装固定尺寸的整图分类。验证时用手工矩阵乘法与 erf 形式的 GELU 对照前向结果，检查 token 维保持、参数梯度、图像分辨率错误和非连续张量输入。

```math
H=\mathrm{GELU}(XW_1^{\mathsf T}+b_1),\qquad Y=HW_2^{\mathsf T}+b_2,\qquad W_1\in\mathbb R^{h\times d},\quad W_2\in\mathbb R^{c\times h}
```

代码：[MLP](../coding/torch_primitives.py#L12) · [ImageMLP](../coding/torch_primitives.py#L25)

#### 易错点

- MLP 的线性层要定义在 __init__ 中；在 forward 每次重新创建会重置参数且可能不被优化器管理。
- 整图 flatten 必须保留 batch 维；逐 patch MLP 与整图分类 MLP 的输入维和输出语义不同。
- 对非连续张量直接使用 view 可能报错；flatten/reshape 可以在必要时复制，复制也会带来开销。
- 逐 token MLP 会共享参数，但不能单独承担 token 之间的关系建模。

#### 追问

- 输入分辨率翻倍时，整图 flatten MLP 第一层参数量会如何变化？
- 如何把整图 MLP 改成 patch 投影，并让不同 patch 之间进行信息交互？

<a id="topic-2"></a>
## 损失函数与训练代码

<a id="cod-005"></a>
### COD-005 · 手写 InfoNCE：正样本标签与 in-batch negatives 如何组织？

**L2**

#### 答案

把匹配的 query/key 排在相同 batch 索引，归一化后计算两两相似度并除以温度，得到 `[N,N]` logits；以 `arange(N)` 为标签，对每一行做交叉熵并取平均。正样本在对角线上，其他列为 in-batch negatives。

温度必须大于零；降低温度会让分布更尖，也会改变梯度与难负例的影响。双向图文训练可以平均两个方向的损失，但要注意重复样本形成假负例。分布式 all-gather 时，标签 offset 必须对应全局 batch 顺序，同时确认实现是否向被 gather 的特征回传梯度。

```math
\mathcal L=-\frac1N\sum_{i=1}^N\log\frac{\exp(s(q_i,k_i)/\tau)}{\sum_{j=1}^N\exp(s(q_i,k_j)/\tau)}
```

#### 易错点

- 把一批同类别的另一个正例也当负例；或打乱 keys 后仍用对角标签。

#### 追问

- 当 N=1 时还有对比信号吗？

<a id="cod-008"></a>
### COD-008 · 实现 DPO loss，怎样避免符号和序列概率错误？

**L2**

#### 答案

对同一 prompt 的 chosen/rejected，计算回答 token 的序列 log probability；各自减去冻结 reference 的对应值，再取 chosen 相对 rejected 的差并乘 $`\beta`$，得到 margin $`z`$。损失为 $`-\log\sigma(z)`$，用 `softplus(-z)` 或 `-logsigmoid(z)` 实现，避免先求 sigmoid 再取对数的数值问题。

标准 DPO 对回答 token 的 log probability 求和，prompt 与 padding 不进入求和，reference 不反传。按长度平均会改变目标，应明确说明。用正负 margin、极端数值及冻结参数梯度测试，可以发现符号和 mask 错误。

```math
\begin{aligned}z&=\beta\left[\left(\log\pi_\theta(y^+\mid x)-\log\pi_{\mathrm{ref}}(y^+\mid x)\right)-\left(\log\pi_\theta(y^-\mid x)-\log\pi_{\mathrm{ref}}(y^-\mid x)\right)\right]\\\mathcal L_{\mathrm{DPO}}&=\mathrm{softplus}(-z)=-\log\sigma(z)\end{aligned}
```

#### 易错点

- chosen margin 变好时 loss 却变大；或者漏掉 reference 项。

#### 追问

- reference 与 policy 初始相同，loss 是多少？

<a id="cod-009"></a>
### COD-009 · 实现 GRPO 组内优势，标准差为零时怎么办？

**L2**

#### 答案

对同一 prompt 的一组奖励减去组均值，再按明确约定的组内标准差归一化，并加入 $`\epsilon`$ 保持数值稳定。本示例采用总体标准差 `correction=0`，不同框架的配置可能不同。

当组内奖励全部相同，奖励优势为零，没有相对奖励梯度；KL 等独立损失项仍可能产生梯度。组大小为一也缺少组内对比信号。应记录零方差组的比例，检查采样多样性和奖励分辨率；是否去掉标准差归一化，要依据具体算法目标判断。

```math
A_i=\frac{r_i-\bar r}{\sqrt{\frac1G\sum_{j=1}^G(r_j-\bar r)^2}+\epsilon}
```

#### 易错点

- 说“优势零但 PPO clip 仍必然压缩熵”；clip 不是独立的梯度来源。

#### 追问

- 奖励缩放和不同难度 prompt 会怎样影响更新？

<a id="topic-3"></a>
## 采样与缓存实现

<a id="cod-003"></a>
### COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？

**L2**

#### 答案

增量解码有缓存时，新 query 的绝对位置从 `past_len` 开始，第 $`i`$ 个 query 只能访问满足 $`j\leq\mathrm{past\_len}+i`$ 的 key。若 `Tq=1`、`Tk` 很长，直接套左上三角 mask 会错误地只保留第一个 key，因此要按绝对位置比较，或确认框架对非方阵因果 mask 的定义。

实现还需组合 padding mask，并保证缓存 K 的 RoPE 位置与新 query 的位置一致。在 `eval` 和相同精度下，对照整段 prefill 与逐 token decode 的同位置输出，应该在数值容差内一致。

```math
\mathrm{allowed}[i,j]=\mathbf{1}\!\left[j\leq\mathrm{past\_len}+i\right]
```

#### 易错点

- 认为 is_causal=True 对所有非方阵 attention 都自动等价于缓存解码。

#### 追问

- 滑动窗口缓存淘汰后位置编号如何维护？

<a id="cod-006"></a>
### COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？

**L2**

#### 答案

先确定温度和过滤顺序。Top-k 保留固定数量的候选；top-p 则把概率降序排列，保留累计质量首次达到 $`p`$ 的最小前缀，包含跨过阈值的那个 token，再重新归一化采样。例如概率 `[0.6,0.3,0.1]`、$`p=0.7`$ 时应保留前两项。

本仓库示例采用“temperature → top-k → 归一化 → top-p → 再归一化”，其他顺序可能产生不同分布。温度为零走独立贪心路径；排序并列值、极小温度和非法参数应有明确约定，测试使用固定随机数生成器便于复现。

#### 易错点

- 用 cumsum<=p 直接过滤，误删达到阈值的 token，甚至删空。

#### 追问

- top-k 与 top-p 的组合为什么顺序可能改变候选集？

<a id="cod-014"></a>
### COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。

**L2**

#### 答案

KV Cache 存储字节数为 $`2BLT h_{\mathrm{kv}}d_hs`$：2 对应 K/V，B 为 batch，L 为层数，T 为缓存长度，$`h_{\mathrm{kv}}`$ 为 KV 头数，$`d_h`$ 为头维，s 为每元素字节数。GQA 应代入 KV 头数，而非 query 头数。

例如 `B=1,T=4096,L=32,h_kv=8,d_h=128,s=2`，缓存占 512 MiB。紧凑布局的变长 batch 按各样本实际缓存长度求和；普通 dense/padded 缓存按 $`BT_{\max}`$ 分配，分页缓存按已分配块数估算。这只计算缓存本体，不含权重、激活、页表、量化 scale、碎片与运行时工作区。

```math
\begin{aligned}\mathrm{bytes}&=2BLT h_{\mathrm{kv}}d_hs\\\mathrm{bytes}_{\mathrm{packed}}&=2Lh_{\mathrm{kv}}d_hs\sum_{b=1}^{B}T_b\end{aligned}
```

#### 易错点

- 用 Q 头数计算 GQA，或把 GB 与 GiB 混用。

#### 追问

- 同样显存预算下，把上下文翻倍会如何影响并发？

<a id="topic-4"></a>
## 通用算法与数据结构

<a id="cod-010"></a>
### COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？

**L1**

#### 答案

先明确“第 k 大”按元素计数，重复值不去重，并要求 $`1\leq k\leq n`$。维护大小为 k 的小根堆，堆顶是已读取元素的第 k 大；堆满后仅在新元素大于堆顶时替换。时间为 $`O(n\log k)`$、空间为 $`O(k)`$，适合流式输入。

Quickselect 通常原地执行，平均时间为 $`O(n)`$，坏 pivot 可使最坏情况退化到 $`O(n^2)`$。可随机选择 pivot，并用三路 partition 处理大量相等值，不能把平均复杂度当成最坏保证。

#### 易错点

- 混淆第 k 大与第 k 个不同的大值。

#### 追问

- 如果数据不能全部载入内存，如何改？

<a id="cod-011"></a>
### COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。

**L1**

#### 答案

扫描每个陆地格，对尚未访问的陆地做四邻域 DFS/BFS，每启动一次遍历就计一个连通块。入队或入栈时立即标记 visited，避免同一格重复进入待处理集合。

每格最多处理一次，时间为 $`O(RC)`$；visited 与遍历队列/栈的最坏空间也是 $`O(RC)`$。Python 可用迭代栈避免大岛引发递归深度溢出。先约定输入为整数还是字符、是否允许原地修改，再测试空矩阵、全海、全陆、对角接触和蛇形长岛。

#### 易错点

- 无意中把对角接触也视为连通，或遗漏边界检查。

#### 追问

- 若陆地不断增加，如何用并查集增量维护数量？

<a id="cod-012"></a>
### COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。

**L1**

#### 答案

令 $`dp[i][j]`$ 表示两个字符串前缀的编辑距离，最后一步可能是删除、插入或替换，因此比较三种转移的最小值；字符相同时替换成本为零。空串边界初始化为另一前缀的长度。

每一行只依赖上一行和本行前一个位置，使用 `previous/current` 两行，并让第二维对应短字符串，空间可压缩到 $`O(\min(m,n))`$，时间仍为 $`O(mn)`$。如果还要恢复具体编辑路径，需要保留更多信息或使用分治方法。

```math
\begin{aligned}dp[i][0]&=i,\qquad dp[0][j]=j\\dp[i][j]&=\min\left\{dp[i-1][j]+1,\ dp[i][j-1]+1,\ dp[i-1][j-1]+\mathbf{1}[a_{i-1}\ne b_{j-1}]\right\}\end{aligned}
```

#### 易错点

- 把最长公共子序列的转移误套到允许替换的编辑距离。

#### 追问

- 语音识别 WER 里的替换、插入、删除如何关联？

<a id="cod-013"></a>
### COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？

**L1**

#### 答案

哈希表负责定位节点，双向链表维护访问顺序：`get` 命中和 `put` 更新都把节点移到最近使用端，容量超限则淘汰另一端。已有 key 更新不增加元素数量，但必须刷新访问位置；Python 可用 `OrderedDict` 演示这些顺序语义。

典型 `get/put` 的平均复杂度为 $`O(1)`$，哈希冲突和并发属于额外约束。测试应覆盖零容量、重复更新、以及读取改变后续淘汰顺序的情况。

#### 易错点

- 只在插入时调整顺序，导致实现的是 FIFO。

#### 追问

- 并发服务如何避免 cache stampede 与跨租户键碰撞？

<a id="cod-015"></a>
### COD-015 · 手撕代码时怎样设计能揭露错误的测试？

**L2**

#### 答案

先确认输入输出约定，再组合边界样例、独立朴素解和数学不变量。测试要检查算子的性质，而非复刻实现逻辑；算法题可以在小规模随机输入上与独立基线比较。

Attention 检查 shape、未来 token 不影响早期输出、cache decode 与 prefill 一致；RoPE 检查范数及相对内积，LoRA 检查 merge 等价，TopK 与排序基线对照。可训练算子还需测梯度。固定随机种子方便重现，GPU 完全确定性仍依赖设备、内核和版本。

#### 易错点

- 只有一个常规样例，或把同一个错误公式写进 expected。

#### 追问

- 浮点比较为什么应采用相对/绝对容差？

<a id="cod-016"></a>
### COD-016 · 手写最长公共子序列：怎样定义状态、推导转移并压缩空间？

**L2** · 字节跳动

#### 答案

先确认要求返回公共子序列的长度。子序列可以不连续，但字符的相对顺序必须保持；例如 abcde 与 ace 的答案是3，不能按最长公共子串的连续匹配来解。

设 D[i][j] 表示第一个字符串前 i 个字符与第二个字符串前 j 个字符的 LCS 长度。任意一方为空时结果为0。如果当前末尾字符相同，就在两个更短前缀的最优结果后追加此字符；若不同，至少要舍弃其中一个末尾字符，取 D[i-1][j] 与 D[i][j-1] 的较大值。按行推进即可保证三个依赖状态已经算好。

完整二维表的时间和空间都是 O(mn)。只求长度时，每格只依赖上一行的同列、上一行的左上角和当前行的左邻，可只保留 previous/current 两行，并把较短字符串放在列方向，将额外空间降到 O(min(m,n))；时间仍是 O(mn)。如果改成单行，覆盖前必须额外保存左上角旧值，否则会把当前行与上一行混用。

参考函数 `longest_common_subsequence` 返回长度，支持空字符串。验证时覆盖相同字符串、完全不相交、重复字符、空输入及参数互换，并对短字符串穷举所有子序列，用交集中的最大长度作为独立基线。若要求输出一个具体 LCS，最直接的方法是保留二维表并从右下角回溯：匹配则记录字符并走左上，不匹配则沿较大值方向移动，最后反转记录；两条方向同分时可能对应不同的合法最优子序列。

```math
D_{i,j}=\begin{cases}0,&i=0\text{ or }j=0\\D_{i-1,j-1}+1,&a_i=b_j\\\max(D_{i-1,j},D_{i,j-1}),&a_i\ne b_j\end{cases}
```

代码：[longest_common_subsequence](../coding/reference.py#L161)

#### 易错点

- 最长公共子序列允许跳过字符；最长公共子串要求连续，不匹配时的转移不同。
- 滚动行适合返回长度；直接丢弃历史行后，不能按普通二维表回溯恢复实际序列。
- 遇到重复字符不能用集合交集大小代替答案，字符出现次数和先后顺序都影响结果。

#### 追问

- 怎样输出一个实际的最长公共子序列，出现多个最优解时如何处理？
- 若只用一行数组，哪个变量必须保存旧的左上角状态？

<a id="cod-017"></a>
### COD-017 · 手写两数之和：怎样用单遍哈希表返回两个不同元素的下标？

**L1** · 腾讯

#### 答案

输入是整数数组 nums 和目标值 target，返回两个不同下标 i、j，使 nums[i]+nums[j]=target。官方题目保证存在唯一解，答案下标顺序不限；实现时要先确认返回的是下标而不是两个数，也不能把同一个数组元素用两次。

单遍遍历数组，维护“此前出现过的数值 → 下标”的哈希表。处理下标 j 的值 x 时，先查询补数 target-x 是否已经存在；存在就返回它的下标与 j，不存在才保存 x。先查后存可以自然保证两个下标不同，也能正确处理 [3,3]、target=6 这种两个值相同的情况。负数和0仍遵循同样逻辑，无须特殊分支。

平均时间复杂度为 O(n)，额外空间为 O(n)，比枚举所有下标对的 O(n²) 时间更好。排序加双指针也是可行路线，但排序会打乱原下标，必须同时保存值和原位置，且时间为 O(n log n)。如果要求找到所有答案，还需要明确重复值、重复下标对的输出规则，单个值只保存一个下标的写法不能直接照搬。

参考函数 `two_sum` 返回两个下标组成的列表；为了练习边界，仓库实现对不存在解的输入抛出 ValueError，对多个解则返回遍历中首先找到的合法下标对。测试不仅检查经典示例，还用随机小数组枚举全部 i<j 的合法对作为独立基线，确认返回值确实在合法集合中，并覆盖重复值、负数、单元素与无解输入。

```math
\mathrm{nums}_i+\mathrm{nums}_j=t,\quad i\ne j,\qquad c=t-\mathrm{nums}_j
```

代码：[two_sum](../coding/reference.py#L150)

#### 易错点

- 先把当前元素放入哈希表再查询，可能在 target=2x 时把同一个下标返回两次。
- 返回数值而非下标，或排序后直接返回排序数组的位置，都会违反原题输入输出契约。
- 哈希表查询的 O(1) 是平均复杂度；不要将其表述成对任意输入都保证常数时间。

#### 追问

- 如果数组已排序，双指针怎样移动，怎样证明不会漏掉答案？
- 如果要求返回所有合法下标对，如何处理大量重复值和输出规模？

## 参考资料

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [Transformers GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Python heapq](https://docs.python.org/3/library/heapq.html)
- [LeetCode Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [LeetCode Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [LeetCode Edit Distance](https://leetcode.com/problems/edit-distance/)
- [Python collections / OrderedDict](https://docs.python.org/3/library/collections.html)
- [LeetCode LRU Cache](https://leetcode.com/problems/lru-cache/)
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- [LeetCode 1143：Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [LeetCode 1：Two Sum](https://leetcode.com/problems/two-sum/)
- [PyTorch nn.Linear：仿射变换、权重与输入输出形状](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
