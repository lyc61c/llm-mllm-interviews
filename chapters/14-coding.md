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
  - [COD-019 · 手写 VAE 训练 loss：ELBO、重参数化、KL 闭式和 reduction 怎样对应？](#cod-019)
- [采样与缓存实现](#topic-3)
  - [COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？](#cod-003)
  - [COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？](#cod-006)
  - [COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。](#cod-014)
  - [COD-020 · 手写 BucketBatchSampler：怎样减少 padding 并保证 epoch 无遗漏、无重复？](#cod-020)
- [通用算法与数据结构](#topic-4)
  - [COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？](#cod-010)
  - [COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。](#cod-011)
  - [COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。](#cod-012)
  - [COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？](#cod-013)
  - [COD-015 · 手撕代码时怎样设计能揭露错误的测试？](#cod-015)
  - [COD-016 · 手写最长公共子序列：怎样定义状态、推导转移并压缩空间？](#cod-016)
  - [COD-017 · 手写两数之和：怎样用单遍哈希表返回两个不同元素的下标？](#cod-017)
  - [COD-021 · 手写整数平方根：怎样用二分避免浮点误差与乘法溢出？](#cod-021)
  - [COD-022 · 手写最长回文子串：区间 DP 和中心扩展怎样取舍？](#cod-022)
  - [COD-023 · 手写全排列：回溯如何恢复现场，重复元素怎样去重？](#cod-023)
  - [COD-024 · 反转单链表怎样原地改指针，如何证明不丢节点也不引入环？](#cod-024)
  - [COD-025 · 手写股票最大利润：交易次数、手续费和冷冻期不同，解法怎样变化？](#cod-025)

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

<a id="cod-019"></a>
### COD-019 · 手写 VAE 训练 loss：ELBO、重参数化、KL 闭式和 reduction 怎样对应？

**L2** · 腾讯

#### 答案

VAE定义生成模型 p(z)pθ(x|z)，用编码器 qφ(z|x) 近似通常不可解析的后验。对数似然等于 ELBO 加 qφ(z|x) 到真实后验的KL，因此最大化ELBO给出可训练的似然下界。训练时最小化负ELBO，即重建负对数似然与后验到先验的正KL之和；不能把最大化下界中的负KL直接作为要最小化的loss。

常见编码器输出均值 μ 和 logvar=log σ²，后验为对角高斯、先验为 N(0,I)。重参数化用 z=μ+exp(logvar/2)⊙ε、ε∼N(0,I)，把随机性放在与参数无关的噪声里，让重建梯度经 z 回到编码器；直接 detach z 会切断这条路径。此设置下KL有闭式，无须对KL再做蒙特卡洛估计。

重建项取决于观测似然：二值观测的Bernoulli解码器可输出 logits，使用 binary_cross_entropy_with_logits；若灰度值作为软标签，需说明采用了该重建代理目标。实值观测可用固定方差高斯，负对数似然是带方差系数和常数的平方误差；可学习方差时还要保留对数方差项，不能一律称普通MSE就是完整似然。参考 vae_loss 支持Bernoulli logits和单位方差Gaussian均值，Gaussian分支保留常数。

归约要把每个样本的像素/观测维求和，也把每个样本的潜变量维求和，然后分别对batch取平均；这样两项具有相同的逐样本尺度。若重建项对所有像素取均值而KL仍按潜维求和，分辨率改变就会改变两项的相对权重。参考函数返回 total、reconstruction_nll、posterior_kl；默认 β=1 对应标准负ELBO，β≠1为加权变体。训练循环是编码、重参数化采样、解码、计算loss、zero_grad、backward、step，并分别监控重建与KL以识别后验坍塌。

图像生成中的压缩自编码器可以加入感知损失和对抗训练，提高细节重建；这些属于特定视觉模型的扩展，不能说标准VAE必然包含LPIPS和GAN。混合精度时可将指数和归约放在FP32，仍须检查logvar异常导致的溢出；单纯转换精度并不保证任意logvar数值稳定。

```math
\begin{aligned}\mathrm{ELBO}(x)&=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))\\\log p_\theta(x)&=\mathrm{ELBO}(x)+D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\\K_i&=\tfrac12\sum_j(\mu_{ij}^{2}+e^{\ell_{ij}}-1-\ell_{ij}),\quad \ell=\log\sigma^2\\\mathcal L&=\tfrac1B\sum_i[-\log p_\theta(x_i\mid z_i)+\beta K_i]\end{aligned}
```

代码：[vae_reparameterize](../coding/torch_primitives.py#L130) · [vae_loss](../coding/torch_primitives.py#L145)

#### 易错点

- logvar表示log σ²，标准差应取exp(logvar/2)，不是exp(logvar)。
- 标准ELBO的KL方向是近似后验到先验；重建项和KL使用不同归约会隐式改变权重。
- 参考实现只给基础VAE目标，不包含图像生成系统的感知/GAN训练流程。

#### 追问

- 怎样通过KL和重建曲线判断posterior collapse，KL warmup会改变什么？
- 若解码器预测可学习方差，高斯重建NLL怎样修改？

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

<a id="cod-020"></a>
### COD-020 · 手写 BucketBatchSampler：怎样减少 padding 并保证 epoch 无遗漏、无重复？

**L2** · 腾讯

#### 答案

先确认练习契约：输入每条样本的长度、batch_size、局部桶大小、seed和drop_last，输出一批批样本索引；采样器不读取样本，不负责padding。批内长度相近时，补到最长序列所浪费的token通常更少。图文/视频数据可把长度替换为估计token数或计算成本，但单个长度值未必准确代表动态分辨率和时序计算开销。

参考 BucketBatchSampler 先按长度稳定排序，把相邻索引划入大小为 batch_size×bucket_multiplier 的局部桶；每个epoch用局部随机数生成器按seed+epoch在桶内洗牌，再切成批，并洗牌批次顺序。桶太大接近随机组批，padding收益减弱；桶太小则每轮批次成员变化较少，要在计算效率与训练随机性之间选择。它是独立参考实现，PyTorch并没有要求所有bucket sampler都采用这一算法。

桶大小设为batch_size的整数倍，只有最后一个桶可能产生非满批，避免每个桶都丢一个尾巴。drop_last=False时，原始索引只经历排序、无放回洗牌和不重叠切片，因此每个epoch恰好覆盖N条样本；drop_last=True时只删除最后非满批，总保留floor(N/B)×B条且不重复。启用局部洗牌后，每轮被丢的样本可能不同；不洗牌时可能一直丢长度排序尾部，应检查是否造成偏置。

实现 __iter__ 返回索引列表、__len__ 返回批次数，set_epoch改变随机种子；同seed和epoch重复迭代得到相同批次。DataLoader接入用 batch_sampler=实例，collate_fn再读取样本并做padding、mask和标签归约，不同时设置batch_size、shuffle、sampler、drop_last。初始化排序O(N log N)、每轮组批O(N)，参考实现保留全部索引与批次，辅助空间O(N)。

验证以集合不变量为主：索引合法、不重复、样本覆盖与drop_last计数正确，空数据和N<B不出错，同epoch可复现、不同epoch改变排列；再用短长序列混合数据检查padding预算。参考实现是单进程版本，不能直接让所有DDP rank各跑一份，否则会重复训练全量数据；分布式需要额外设计rank分片、批次数一致和尾批策略。

```math
\mathrm{padding}(\mathcal B)=|\mathcal B|\max_{i\in\mathcal B}L_i-\sum_{i\in\mathcal B}L_i,\qquad N_{\mathrm{batch}}=\begin{cases}\lfloor N/B\rfloor,&\mathrm{drop\_last}\\\lceil N/B\rceil,&\text{otherwise}\end{cases}
```

代码：[BucketBatchSampler](../coding/reference.py#L239)

#### 易错点

- 返回一批索引的对象应接到batch_sampler；传给sampler会把索引列表当作单条样本key。
- 桶尺寸不对齐batch_size又逐桶drop_last，会丢掉多个桶的尾部样本。
- set_epoch只控制本采样器的随机次序，不自动管理worker的数据增强随机性或分布式分片。

#### 追问

- DDP怎样保证各rank样本不重叠且更新步数一致？
- 如果按总token预算而非固定样本数组批，__len__和梯度归约如何定义？

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

**L1** · 字节跳动

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

<a id="cod-021"></a>
### COD-021 · 手写整数平方根：怎样用二分避免浮点误差与乘法溢出？

**L1** · 小红书

#### 答案

明确返回非负整数n的平方根向下取整，即最大满足r²≤n的整数，不是浮点近似值。0和1直接返回；其他值可在[1,n//2+1]上二分，因为正整数平方随r单调增长。中点满足条件时继续向右找更大合法值，否则向左缩小，循环结束返回high，保证r²≤n<(r+1)²。

判断可以写mid≤n//mid，以整数除法避免固定宽度语言中mid*mid溢出；mid始终至少为1，不会除零。中点在固定宽度语言用low+(high-low)//2避免low+high溢出。Python整数可自动扩展，但仍应解释跨语言的溢出风险。不要先sqrt再int：浮点数对大整数会丢失精度，在完全平方数邻近尤其容易错一个。

参考 integer_sqrt 不调用math.isqrt或浮点幂，二分迭代次数O(log n)，使用常数个整数变量；若考虑Python任意精度大整数，除法和变量存储有额外位复杂度，不应把每步大整数运算都说成严格O(1)。输入负数或非整数时明确报错。测试用math.isqrt作独立oracle，覆盖0/1、完全平方数、平方数±1以及远超过64位的整数，并同时检查答案的平方区间不变量。

```math
r=\max\{k\in\mathbb Z_{\ge0}:k^2\le n\},\qquad r^2\le n\lt (r+1)^2,\qquad m^2\le n\iff m\le\lfloor n/m\rfloor\ (m\gt 0)
```

代码：[integer_sqrt](../coding/reference.py#L290)

#### 易错点

- 返回round(sqrt(n))会四舍五入，题目要求向下取整。
- 二分更新边界不排除mid，会在相邻整数边界无限循环。
- math.isqrt可用于测试oracle，不能冒充手写二分实现。

#### 追问

- 如果改用整数Newton迭代，怎样选初值并处理停止条件？
- 固定宽度语言为什么还要避免中点的low+high溢出？

<a id="cod-022"></a>
### COD-022 · 手写最长回文子串：区间 DP 和中心扩展怎样取舍？

**L2** · 小红书

#### 答案

回文子串必须连续，不能跳过中间字符；最长回文子序列是另一题。先确认返回长度还是实际子串、同长答案的选择规则。参考函数返回实际子串，多个最长结果时取最靠左的出现位置，并把空字符串作为额外练习边界。

区间DP定义P[i][j]表示s[i..j]是否为回文。两端字符相等，且区间长度≤2或内部P[i+1][j-1]为真时，整个区间为回文。按区间长度从小到大枚举，或i从大到小、j从i向右遍历，保证内部状态已计算；每个真状态更新最大长度和起点。时间O(n²)、布尔表空间O(n²)，初始化单字符为真；长度2时不能访问不存在的内部状态。

只需要一个最长子串时可用中心扩展：对每个位置分别以(i,i)检查奇数长度、以(i,i+1)检查偶数长度，只要左右字符相同就向外扩张，记录最佳边界。参考 longest_palindromic_substring 用该方法，最坏时间O(n²)、额外工作空间O(1)，最终切片返回字符串需要O(L)空间。省掉DP表的同时，仍覆盖所有回文，因为每个回文都有唯一的单字符或字符间隙中心。

测试不能只比较babad的某一个答案，除非实现已约定tie规则。参考测试穷举短字符串的全部连续子串，以正反相同判定回文、按长度和最早起点得到独立oracle，覆盖奇数、偶数、重复字符、空串和无长回文。若长度很大且需要最坏线性时间，可追问Manacher；需讲清奇偶统一处理、镜像半径和右边界，不能只报算法名称。

```math
P_{i,j}=(s_i=s_j)\land\big((j-i\le1)\lor P_{i+1,j-1}\big),\qquad 0\le i\le j\lt n
```

代码：[longest_palindromic_substring](../coding/reference.py#L307)

#### 易错点

- 最长回文子串需要连续；用最长回文子序列的max转移会解错题。
- 只遍历单字符中心会漏掉abba等偶数长度回文。
- 区间DP按错误顺序遍历，会读到尚未计算的内部状态。

#### 追问

- 为什么中心扩展可覆盖所有回文，最坏O(n²)输入是什么？
- Manacher中当前点在最右回文区间内时，镜像半径怎样初始化？

<a id="cod-023"></a>
### COD-023 · 手写全排列：回溯如何恢复现场，重复元素怎样去重？

**L2** · 字节跳动

#### 答案

先确认输入是否含重复值，要求按元素下标区分排列还是只返回不同数值序列。若所有元素互异，维护当前path和used数组，每一层从尚未使用的元素中选择一个；path长度达到n时得到完整排列。选择后标记used、追加到path，递归结束必须pop并清除标记，恢复当前层的状态；保存答案时要复制path，否则所有结果可能指向同一个可变列表。

若有重复值且要求唯一排列，可先对输入副本排序，再用同层去重：遍历下标i时，若该元素已用过则跳过；若它与前一个元素相同且前一个元素当前未使用，也跳过。后一个条件表示同一递归层的相同值只开启一个分支。若前一个相同元素已经在path中，则允许选择当前元素，从而正常生成[1,1,2]等包含多个1的排列；不能无条件跳过所有相邻相等元素。

参考 unique_permutations 接受元素间可以排序和比较相等的输入，返回不同排列组成的列表，保留原输入，输出次序按排序后的遍历确定。空输入定义为只有一个排列——空排列，因此返回[[]]；这使回溯终止条件和0!=1一致。混合不可排序类型会在排序时报告TypeError，不承诺任意Python对象都能使用。

无重复值时共有n!个长度n的结果，复制并输出它们本身就需要O(n×n!)时间，参考回溯的最坏时间也为O(n×n!)，排序另需O(n log n)。含重复值时，唯一输出数U=n!/∏c_j!，保存答案需要O(nU)空间；递归栈、used、path和排序副本的额外工作空间是O(n)，要把输出存储与工作空间分别说明。

测试用itertools.permutations枚举短输入，再把元组结果去重作为独立oracle；验证结果集合一致、没有重复序列、每个结果的元素多重集与输入相同、输入未被修改，各答案列表互不别名。覆盖空输入、全相同、部分重复和互异元素。若面试要求原地交换版本，也须说明每层交换前后均要恢复，以及重复值在当前层如何去重。

```math
U=\frac{n!}{\prod_j c_j!},\qquad \mathrm{skip}(i)=\mathrm{used}_i\lor(i\gt 0\land a_i=a_{i-1}\land\neg\mathrm{used}_{i-1}),\qquad 0!=1
```

代码：[unique_permutations](../coding/reference.py#L321)

#### 易错点

- 递归返回不pop或不清除used，会污染后续分支，遗漏或产生错误结果。
- 直接把path对象追加到答案列表，会让结果共享同一个可变对象。
- 同层去重判断依赖排序和前一个同值元素是否已使用，不能无条件跳过重复值。

#### 追问

- 如何用原地交换代替used数组，并对重复元素进行同层去重？
- 如果只需逐个消费排列，如何改为生成器以减少答案存储？

<a id="cod-024"></a>
### COD-024 · 反转单链表怎样原地改指针，如何证明不丢节点也不引入环？

**L1** · 腾讯

#### 答案

输入为单链表头节点，输出反转后的新头；要求修改原节点的next关系，保持每个节点的身份和值。空链表返回None，单节点返回自身。先明确输入是否保证无环，环形链表没有以None结束的普通线性反转语义；参考 reverse_linked_list 在修改任何指针前用快慢指针检查环，发现环就报ValueError，且不修改原结构。

迭代反转维护previous和current：开始previous=None、current=head。每轮必须先保存following=current.next，再令current.next=previous，随后把previous移到当前节点、current移到following。current为空时，previous就是原尾节点，也是反转后的新头。先保存following是关键，否则改写next后会失去剩余未处理链表的入口。

循环不变量是previous指向已处理前缀的反向链，current指向尚未处理的原顺序后缀，两个部分的节点集合不重叠且并集始终是全部原节点。每轮把后缀第一个节点移到反向前缀，不创建或丢弃节点；前缀末尾为原头且next为None，因此没有新增环。结合输入无环或预检通过，current沿保存的原后继有限前进，最终终止。

三指针反转时间O(n)、辅助空间O(1)，快慢指针预检也为O(n)时间和O(1)空间，因此参考实现的总复杂度不变。递归版本也能反转，但递归栈占O(n)，Python长链表可能触及递归深度限制，不能把递归写成常数空间。

测试除了值序列反向，还必须检查节点对象按原身份的逆序出现，原头成为尾且next为空，遍历不重复遇到节点；再次反转应恢复原身份顺序。值重复的输入可揭露只反转值或重建新节点的伪实现，带环和自环输入则验证预检拒绝且原next关系保持。

```math
(v_0\to v_1\to\cdots\to v_{n-1}\to\varnothing)\longmapsto(v_{n-1}\to\cdots\to v_1\to v_0\to\varnothing),\qquad T(n)=O(n),\ S(n)=O(1)
```

代码：[ListNode](../coding/reference.py#L348) · [reverse_linked_list](../coding/reference.py#L353)

#### 易错点

- 没有先保存旧next就改指针，会失去尚未处理后缀。
- 仅反转值或创建新节点不能满足原地修改节点连接的契约。
- 返回旧head或忘记让旧head.next为None，分别会返回尾节点或产生错误连接。

#### 追问

- 如何只反转指定区间或每k个节点反转一次？
- 快慢指针为什么可判断有环，递归版为什么占O(n)额外空间？

<a id="cod-025"></a>
### COD-025 · 手写股票最大利润：交易次数、手续费和冷冻期不同，解法怎样变化？

**L2** · 腾讯

#### 答案

先问清最多交易几次、是否只持有一股、能否不交易、是否有手续费或冷冻期，再定义状态。只说“股票最大利润”不能确定具体版本。本题参考 max_stock_profit 明确采用至多一次买入、在更晚一天卖出、允许不交易；这只是标准练习契约，不把面经未说明的交易次数补成事实。

单次交易遍历每天价格，维护此前最低买入价lowest和已知最佳利润best。对当天作为卖出日，先比较price-lowest，再更新lowest，这样候选买入日始终在卖出日之前。任意最优交易若在当天卖出，最好的买入价必是此前最低价；遍历所有卖出日就不会漏最优解。best从0开始，因此空输入、单元素、价格持平或不断下跌都返回0。时间O(n)、辅助空间O(1)，支持流式遍历且不改变输入。

不限次数、没有费用或冷冻期、至多持有一股时，可以累加每对相邻日价格的正增量；每段上涨区间等价于在最低点买、最高点卖。但只做一次交易不能累加多段上涨：例如[1,3,1,3]单次利润2、多次利润4。若至多k次交易，可用“第j次买入后的持有收益”和“第j次卖出后的空仓收益”状态，第j次买入依赖第j-1次卖出，第j次卖出依赖第j次买入；按天推进，时间O(nk)、滚动空间O(k)，初始化不可达状态为负无穷并允许零次交易。

手续费版本通常在卖出时只扣一次fee，用hold和cash两状态，从前一天状态计算新状态；也可等价改为买入时扣费，但不能两边各扣一次。一个完整冷冻日意味着昨天卖出后今天不能买入，此时可保留rest/hold/sold三状态，或让当天买入引用前两天可空仓收益，不能沿用没有冷冻期的cash前一天转移。状态更新必须明确是否允许同日操作，并保存需要的旧值，避免新旧状态混用。

单次版本测试用所有买卖日i<j的价格差和0作为穷举oracle，覆盖空、单元素、上涨、下跌、重复低点与多个上涨波段；这样可揭露使用全局max-min却忽略时间先后，或者把多次交易利润误当单次交易答案的问题。若要求输出交易日，维护最低价下标和最佳下标对，再约定同利润的选择规则。

```math
P=\max\big(\{0\}\cup\{p_j-p_i:0\le i\lt j\lt n\}\big),\qquad m_t=\min_{0\le i\le t}p_i,\qquad P_t=\max(P_{t-1},p_t-m_{t-1})\ (t\ge1)
```

代码：[max_stock_profit](../coding/reference.py#L368)

#### 易错点

- 全局最大价减最小价可能先卖后买，违反时间顺序。
- 累加所有正增量只适用于相应的不限次数无费用/冷冻期版本。
- 手续费、冷冻期和交易次数限制都会改变状态，不能只换一个初始化值就套用同一代码。

#### 追问

- 最多两次交易怎样写四个状态，并控制更新顺序？
- 手续费和一个冷冻日同时存在时，怎样定义可买入的空仓状态？

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
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [PyTorch examples：VAE MNIST训练示例](https://github.com/pytorch/examples/blob/main/vae/main.py)
- [Latent Diffusion官方感知与对抗自编码器loss](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/ldm/modules/losses/contperceptual.py)
- [PyTorch torch.utils.data 文档](https://docs.pytorch.org/docs/2.14/data.html)
- [PyTorch官方Sampler与BatchSampler实现](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py)
- [LeetCode 69：Sqrt(x)](https://leetcode.com/problems/sqrtx/)
- [LeetCode 5：Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [LeetCode 46：Permutations](https://leetcode.com/problems/permutations/)
- [LeetCode 47：Permutations II](https://leetcode.com/problems/permutations-ii/)
- [Algorithms 4th edition：Bags, Queues, Stacks与单链表节点](https://algs4.cs.princeton.edu/13stacks/)
- [WPI CS2223：Stocks动态规划课程解答](https://web.cs.wpi.edu/~cs2223/b05/HW/HW6/SolutionsHW6/)
- [University of Washington CSE421：Dynamic Programming交易手续费题](https://courses.cs.washington.edu/courses/cse421/25wi/files/homework/homework5_problems.pdf)
