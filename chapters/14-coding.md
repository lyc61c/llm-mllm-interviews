# 手撕代码与算法

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [模型算子与数值实现](#topic-1)
  - [COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？](#cod-001)
  - [COD-002 · 手撕 MHA/GQA：形状、分组头映射、缩放与 mask 怎么写？](#cod-002)
  - [COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。](#cod-004)
  - [COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。](#cod-007)
  - [COD-018 · 用 PyTorch 实现两层 MLP，图像输入应该怎样组织？](#cod-018)
  - [COD-026 · 如何实现基于跨层变化的渐进视觉 token 裁剪？](#cod-026)
  - [COD-029 · 手写 MoE Top-k 路由：专家索引、门权重与溢出策略怎么定义？](#cod-029)
  - [COD-030 · 手写 LayerNorm：归一化轴、biased variance、epsilon 与 gamma/beta 怎样实现？](#cod-030)
- [损失函数与训练代码](#topic-2)
  - [COD-005 · 手写 InfoNCE：正样本标签、归一化、温度与排除自身如何定义？](#cod-005)
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
  - [COD-025 · 手写股票最大利润：单笔、至多两笔、手续费与冷冻期怎样区分？](#cod-025)
  - [COD-027 · 不调用 sqrt 求非负实数平方根，如何控制误差并处理极大、极小值？](#cod-027)
  - [COD-028 · 矩阵中的最长递增路径怎么求，如何避免递归深度溢出？](#cod-028)

<a id="topic-1"></a>
## 模型算子与数值实现

<a id="cod-001"></a>
### COD-001 · 手写稳定 Softmax 与交叉熵，为什么要减最大值？

**L1**

#### 答案

稳定 Softmax 先把每一行 logits 减去该行最大值，再做指数和归一化；稳定交叉熵则直接计算 logsumexp 减正确类别的 logit。这样既保留数学结果，又避免大数取指数溢出，或概率下溢成零后再取对数。

公式中的 z 是原分数，m 是最大分数，y 是正确类别。所有分数减同一个常数，只会给分子分母同时乘一个公共因子，因此概率不变。例如 [1000,1001] 可以先变成 [−1,0]，最大指数为 1，结果仍一样。交叉熵写成 m+log∑exp(z−m)−z_y，直接从分数计算；对 logits 的梯度是预测概率减 one-hot 目标。

我会沿类别轴保留维度计算最大值，核对 batch 广播、标签和有效位置，再测试极大正负分数、整体平移和部分负无穷 mask。全部负无穷的行没有可用分布，需要明确报错或特殊策略。时间与输出空间随分数元素数线性增长。这个减常数技巧依赖指数的性质，不能直接套到“sigmoid 后归一化”或“ReLU 后归一化”；后者甚至可能全零，替换函数后必须重新推导目标与稳定算法。

```math
\begin{aligned}m&=\max_j z_j,\qquad p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}\\\mathcal L_{\mathrm{CE}}&=m+\log\sum_j e^{z_j-m}-z_y\end{aligned}
```

#### 易错点

- 直接 exp 原 logits 可能溢出，先算 softmax 再 log 又可能遇到 log(0)。

#### 追问

- 加入标签平滑和 ignore_index 后，稳定损失与有效样本归约如何实现？

<a id="cod-002"></a>
### COD-002 · 手撕 MHA/GQA：形状、分组头映射、缩放与 mask 怎么写？

**L1** · 字节跳动 / 腾讯 / 阶跃星辰

#### 答案

手写 MHA/GQA 的关键是先把张量形状和查询头到 KV 头的对应关系定清楚，再计算匹配、屏蔽、归一化与汇总。输入 B×T×D 投影后，MHA 常把 Q/K/V 拆成 B×H×T×d，D=Hd；匹配分数为 B×H×T_q×T_k，沿最后的 key 轴 softmax，与 V 相乘后拼接各头，再做输出投影。一次大 QKV 投影并不表示三者共用相同参数。

GQA 保留 H_q 个查询头，只用 H_kv 个 KV 头。本题约定连续等大小分组，g=H_q/H_kv 必须为整数，第 h 个查询头用 floor(h/g) 的 KV 头。例如 8 个查询头、2 个 KV 头，前 4 个用第 0 组，后 4 个用第 1 组。repeat_interleave 可以得到这个映射，普通 repeat 却会产生交错映射；不能先平均查询头。H_kv=H_q 是 MHA，H_kv=1 是 MQA。输出和分数的头轴仍是 H_q。

布尔 mask 在这里 True 表示允许，禁止位置先设成负无穷，每个查询至少要有一个允许 key，否则参考实现报错。Torch 版允许广播到完整分数形状，标准库版只接受共享二维或完整四维布尔 mask。缓存因果关系要保留 query_offset，不能机械套矩形三角阵；cross-attention 也要测试两种长度不同的情况。

教学实现直接计算已投影的 Q/K/V，不包含完整 block、位置旋转、dropout 或缓存分配。物理展开 K/V 有临时内存，生产融合内核可以保持紧凑缓存；显式注意力空间仍随 H_qT_qT_k 增长。验证要按查询头独立选 KV 做基线，覆盖两个端点和掩码扰动，而不是只检查输出尺寸。

```math
\begin{aligned}g&=H_q/H_{\mathrm{kv}},\quad j(h)=\lfloor h/g\rfloor,\quad H_q\bmod H_{\mathrm{kv}}=0\\O_h&=\mathrm{softmax}\!\left(Q_hK_{j(h)}^\top/\sqrt d+M_h\right)V_{j(h)}\\\mathrm{shape}(O)&=[B,H_q,T_q,d],\quad\mathrm{ConcatHeads}(O)=[B,T_q,H_qd]\end{aligned}
```

代码：[grouped_query_attention](../coding/reference.py#L394) · [grouped_query_attention](../coding/torch_primitives.py#L175)

#### 易错点

- 连续分组用 repeat_interleave，普通 repeat 得到交错映射，不能混用。
- GQA 减少 KV 头，但查询、分数和输出的头轴仍是 H_q。
- 布尔 mask 的语义要看 API，全屏蔽行与缓存 query_offset 也要明确处理。

#### 追问

- 不物理展开 KV 怎样按组计算，省下的缓存与注意力矩阵有什么区别？
- 张量并行分片后，怎样保持查询头到连续 KV 组的映射？

<a id="cod-004"></a>
### COD-004 · 手写 RoPE，并证明旋转保持范数与相对位置内积。

**L2**

#### 答案

手写 RoPE 就是按位置把 Q/K 的通道两两旋转，证明时利用旋转矩阵的正交性和角度相减性质。旋转维度 d 必须为偶数，第 i 对的角度是位置 m 乘 base^(−2i/d)；代码先生成频率，再按位置计算 cos、sin，并广播到 batch 与头维。

一对分量 [a,b] 变成 [a cosφ−b sinφ, a sinφ+b cosφ]。比如旋转 90 度，[1,0] 变成 [0,1]，长度仍为 1。因为 RᵀR=I，所以任意向量的平方范数不变；查询用 R_m、键用 R_n，R_mᵀR_n 合成 R_(n−m)，因此点积里的位置项只依赖两者距离，而内容向量仍参与计算。

实现要匹配 checkpoint 的相邻配对或前后半配对布局、base、旋转比例与真实位置。它通常不旋转 V，也不是加到 embedding 的位置向量。每个旋转元素只做常数次操作，时间线性于元素数；cos/sin 表可以复用。测试要同时比较范数和相对内积，并检查缓存 K 是否只旋转一次。位置频率可在数学上计算到更远位置，但模型是否可靠使用训练外长度，仍需长上下文训练与评测验证。

```math
\begin{aligned}\phi_i&=m\,\mathrm{base}^{-2i/d},\qquad\lVert R_mx\rVert_2=\lVert x\rVert_2\\(R_mq)^\top(R_nk)&=q^\top R_{n-m}k\end{aligned}
```

#### 易错点

- RoPE 通常旋转 Q/K，不能写成 embedding 加法，也不应默认旋转 V。

#### 追问

- 频率缩放改变哪个旋转角度项，为什么能计算长位置不保证效果？

<a id="cod-007"></a>
### COD-007 · 实现 LoRA Linear 并证明 merge 前后输出一致。

**L2**

#### 答案

LoRA Linear 在冻结原权重的同时，增加一个可训练的低秩修正；merge 是把这个修正直接加回原矩阵，使推理少走一个分支。按 PyTorch 权重布局，W 是 out×in，A 是 r×in，B 是 out×r，增量 ΔW=(α/r)BA，新增参数约 r(in+out)。

行批次输入 x 的输出是 xWᵀ+(α/r)(xAᵀ)Bᵀ，与 x(W+ΔW)ᵀ 在精确算术下相等，因为矩阵乘法可结合。比如输入宽度 1000、输出宽度 1000、r=8，修正只需约 16000 参数，远少于完整百万参数。常见初始化 A 随机、B 为零，让初始 ΔW 为零；首步 A 梯度因 B 为零而为零，B 仍可以开始学习，交换零初始化也要相应理解梯度。

实现要冻结 W，正确注册 A/B，统一尺度与 bias，并避免 merge 后仍重复加 adapter。有 dropout 时，应在关闭随机扰动的推理条件下比较；浮点舍入需要容差，量化基座的合并还涉及反量化或重新量化。未合并额外计算约与 r(in+out) 成正比，合并后恢复普通 Linear。测试应检查输出等价、冻结权重无梯度、adapter 有梯度，并覆盖重复 merge 与状态切换。

#### 易错点

- 低秩更新用矩阵乘法 BA；merge 后不能再重复计算同一 adapter 增量。

#### 追问

- 把 A 初始化为零、B 随机也能保持初始输出吗，首步梯度怎样变化？

<a id="cod-018"></a>
### COD-018 · 用 PyTorch 实现两层 MLP，图像输入应该怎样组织？

**L2** · 字节跳动

#### 答案

两层 MLP 用两个线性层加中间非线性把输入特征映射到目标维度，图像怎么输入取决于做整图分类还是逐 patch 变换。PyTorch 中应继承 nn.Module，在初始化里注册 Linear(d,h)、GELU、Linear(h,c)，forward 负责调用；每次 forward 重新创建层会重置参数，优化器也可能找不到它们。

公式采用 PyTorch 权重 out×in 的布局：第一层权重 h×d，第二层 c×h。Linear 只变换最后一个轴，因此 B×d 得到 B×c，B×N×d 的 token 特征得到 B×N×c，N 个位置使用同一套参数，但不会互相交流。如果两层之间没有非线性，它们可以合成一次仿射变换，表达能力不会因层数自动增加。

最简单的固定尺寸图像分类可把 B×C×H×W 用 flatten(start_dim=1) 变成 B×CHW，再输出类别 logits。例如 3×32×32 图像每样本输入 3072 维，不能把 batch 一起展开；直接把 NCHW 输入 Linear 只会映射最后的 W 轴。整图展开第一层参数约 CHW×h，固定尺寸且缺少卷积的局部连接偏置。逐 patch 方案则要另外用注意力或卷积做空间交互。计算随输入向量数与两层矩阵规模增长，训练把原始 logits 交给交叉熵；验证形状、梯度、分辨率错误及非连续张量，使用 reshape/flatten 时也留意必要复制。

```math
H=\mathrm{GELU}(XW_1^{\mathsf T}+b_1),\qquad Y=HW_2^{\mathsf T}+b_2,\qquad W_1\in\mathbb R^{h\times d},\quad W_2\in\mathbb R^{c\times h}
```

代码：[MLP](../coding/torch_primitives.py#L12) · [ImageMLP](../coding/torch_primitives.py#L25)

#### 易错点

- 线性层应在初始化里注册，forward 反复新建会重置参数。
- 整图展开保留 batch，逐 patch 变换与整图分类的输出语义不同。
- 非连续张量 view 可能失败，reshape 或 flatten 的必要复制也有成本。
- 逐 token MLP 共享参数，但不负责跨 token 交换信息。

#### 追问

- 图像高和宽都翻倍时，整图 MLP 首层参数量如何变化？
- 怎样改成 patch 投影，并加入跨 patch 的信息交互？

<a id="cod-026"></a>
### COD-026 · 如何实现基于跨层变化的渐进视觉 token 裁剪？

**L3** · 快手

#### 答案

基于跨层变化的视觉 token 裁剪，可以比较同一个视觉位置在相邻语言模型层前后的表示变化，逐阶段保留变化大的 token。V2Drop 默认把 L2 差异作为重要性代理，它不是相邻视频帧差，也不要求显式注意力权重；小变化只是启发式信号，不证明删除后所有任务都无损。

最小接口输入两份 N×D hidden、N 维视觉 mask 与当前预算 k。只对视觉位置计算分数 s_i=‖h_i^ℓ−h_i^(ℓ−1)‖，选前 k 个，再与全部文本和特殊 token 合并，按原索引排序返回。保留原顺序很重要，不能把高分排到前面改变序列。Torch 版用 FP32 评分，标准库用 Python float，极接近分数可能排名不同；并列值优先早位置可以复现。评分约 O(ND)，全排序选择约 O(N log N)，输出索引至多 N。

模型集成时，要保证两份 hidden 对应同一逻辑 token，剪后同步收集 hidden、视觉与有效 mask 及原 position IDs；下一阶段也比较相同保留集合。Qwen 的时空位置不能重新编成密集编号。二维 mask 删对应序列轴，方阵 mask 需要处理 query/key 两轴，带历史的矩形 mask 则按各自逻辑位置重建。不同样本预算不同，还需变长布局或再 padding。

通常在 prefill 阶段裁剪，各层 KV 长度可能不同，需要各层逻辑位置到缓存槽的映射，不能用深层索引统一删浅层缓存；只加 mask 也不会自动减少矩阵算术。硬 top-k 对选择边界不可直接求导，训练需另定义未选项梯度与标签对应。现有函数只覆盖单样本、单阶段评分索引，完整位置、缓存与模型执行还要集成。测试文本全保留、预算 0/全保留、空视觉、并列、非有限分数和多阶段索引一致性，层位与预算通过质量和速度消融决定。

```math
\begin{aligned}s_i^{(\ell)}&=\lVert h_i^{(\ell)}-h_i^{(\ell-1)}\rVert_2\\ M_0&\ge M_1\ge\cdots\ge M_K\ge0\end{aligned}
```

代码：[variation_keep_indices](../coding/reference.py#L456) · [variation_keep_indices](../coding/torch_primitives.py#L223)

#### 易错点

- 表征变化小只是剪枝信号，不证明删除对所有任务无影响。
- 比较前后 hidden 必须对齐同一逻辑 token，错位会算成不同 token 距离。
- 各层缓存长度、位置与矩形 mask 要分别维护，评分函数不包含完整集成。
- 某实现的固定视觉跨度不能当成所有 LLaVA 或 Qwen 的通用规则。

#### 追问

- 并列分数怎样约定稳定保留顺序并固定评分精度？
- 裁剪后为什么仍要保留原位置，而不是重新密集编号？
- 最终 token 数相同，提前裁剪与多阶段裁剪为何成本不同？

<a id="cod-029"></a>
### COD-029 · 手写 MoE Top-k 路由：专家索引、门权重与溢出策略怎么定义？

**L2** · 阿里巴巴

#### 答案

手写 MoE Top-k 路由要明确选哪些专家、门权重是否重归一，以及函数是否包含容量和专家执行。本题核心接收 N×E 的有限 router logits，输出 N×k 的专家 ID 和权重；hidden 可以先从 B×T×D 展平再投影，但核心函数本身不包含这个线性层或完整 MoE。

默认只对选中 logits 做 softmax，等价于完整分布取 top-k 后重归一；renormalize=False 保留原概率，所选权重和可能小于 1。比如 logits=log([1,2,4,8])，选 ID [3,2]，两种权重分别是 [2/3,1/3] 与 [8/15,4/15]。索引离散不可微，门权重仍可反传；特别是 k=1 重归一后权重恒为 1，主损失经门权重传到 router 的梯度为零，不能直接视为 Switch 的训练等价实现。敏感归一化常用 FP32，参考 float64 输入保留双精度；并列按较小 ID 优先是此接口约定。

真正执行专家时，按专家收集 token，算 FFN，再用对应权重 scatter_add 回原 token，同 token 的 k 个分支必须相加。先算全部专家再 mask 并没有稀疏计算收益。教学稳定全排序约 O(NE log E)，概率和返回空间约 O(NE)、O(Nk)，生产可用部分选择与融合分发，这不等于整层全部成本。

容量 C 常取平均 Nk/E 乘 capacity factor 后向上取整。现有核心没有容量筛选；扩展时要规定接收顺序、溢出丢分支或回退、剩余权重是否重归一。若全部分支落选，要定义零专家输出加残差或 fallback，避免除以零；这不等于从序列删除 token。Dropless 也会有负载不均和小 batch 矩阵效率问题。验证上述闭式例子、分数平移、并列、非法 k、非有限值和门梯度，并将路由、专家分发、均衡损失与通信的责任分别讲清楚。

```math
\begin{aligned}z&=XW_r^\top,\quad S_t=\mathrm{TopK}(z_t,k)\\\tilde p_{t,e}&=\frac{e^{z_{t,e}}}{\sum_{j\in S_t}e^{z_{t,j}}}\quad(e\in S_t)\\y_t&=\sum_{e\in S_t}\tilde p_{t,e}\,E_e(x_t)\\C&=\left\lceil\mathrm{capacity\_factor}\cdot\frac{Nk}{E}\right\rceil\end{aligned}
```

代码：[moe_top_k_router](../coding/reference.py#L574) · [moe_top_k_router](../coding/torch_primitives.py#L252)

#### 易错点

- 选中 gate 是否重归一影响尺度与梯度，Top-1 归一到 1 尤其不能忽略。
- 返回专家 ID 与 gate 只是核心选择，不包含分发、容量、专家并行或完整训练。
- token dropping 常丢溢出分支，不一定删除序列里的 token。

#### 追问

- Top-1 原 softmax gate 与恒为 1 的 gate，主损失怎样传到 router？
- 同 token 两个专家分支怎样累加回原位置，并处理容量溢出？

<a id="cod-030"></a>
### COD-030 · 手写 LayerNorm：归一化轴、biased variance、epsilon 与 gamma/beta 怎样实现？

**L2** · 百度

#### 答案

手写 LayerNorm 先确定统计轴：本题对任意 [...,D] 输入，只沿最后的 D 个特征独立计算均值与总体方差，再做缩放偏移。Transformer 的 B×T×D 就是每个 token 单独归一，不是跨 batch 或整个时间轴统计；完整 PyTorch LayerNorm 可归一最后多个轴，参考核心不覆盖全部通用接口。

计算 μ=mean(x)、v=mean((x−μ)²)，保留统计维方便广播，再用 (x−μ)/√(v+ε)。方差分母是 D，对应 correction=0，直接用默认无偏修正可能不同；ε 在根号里面。例如 x=[1,3]、ε=1，结果为 [−1/√2,1/√2]，能同时检验方差与 ε 的位置。常量或 D=1 时归一部分是零，仿射后为 β，避免 0/0。

γ、β 是 D 维逐特征参数，常以 1、0 初始化，并广播到所有领先位置。仿射后不保证均值仍为零、方差仍为一。函数接收参数不等于自动注册可训练模块，训练时需用 nn.Parameter 或已有模块管理；train、eval 都使用当前输入统计，没有 BatchNorm 的运行均值。

Torch 参考保留 float64，否则在 FP32 统计与归一后转回输入类型；转换仍可传 γ、β 梯度，但极大输入也可能让 FP32 统计溢出。标准库支持非空矩形嵌套列表并返回新列表，均要校验 ε、形状与有限统计。N 个 D 维向量计算 O(ND)，输出 O(ND)、统计 O(N)。测试闭式例子、常量、领先维、仿射与非法输入，并与 F.layer_norm 比较前向和梯度，不能只看随机输出大致像零均值。

```math
\begin{aligned}\mu_i&=\frac1D\sum_{j=1}^{D}x_{ij},\qquad v_i=\frac1D\sum_{j=1}^{D}(x_{ij}-\mu_i)^2\\y_{ij}&=\gamma_j\frac{x_{ij}-\mu_i}{\sqrt{v_i+\epsilon}}+\beta_j,\qquad\epsilon\gt 0\end{aligned}
```

代码：[layer_norm_last_dim](../coding/reference.py#L602) · [layer_norm_last_dim](../coding/torch_primitives.py#L278)

#### 易错点

- 最后 D 轴总体方差除 D，不能改成 D−1 或跨 batch 统计。
- ε 放在平方根内，后续 γ、β 可以改变均值与方差。
- 函数接收 γ、β 不代表自动注册参数，训练模块需显式管理它们。

#### 追问

- D=1 时 LN 为什么得到 β，RMSNorm 的结果为何不同？
- 归一化最后 H、W 两轴时，统计和仿射参数形状如何改变？

<a id="topic-2"></a>
## 损失函数与训练代码

<a id="cod-005"></a>
### COD-005 · 手写 InfoNCE：正样本标签、归一化、温度与排除自身如何定义？

**L2** · 阿里巴巴

#### 答案

InfoNCE 把匹配样本作为正确候选、其他样本作为竞争候选，用分类损失拉开相似度；手写时最重要的是先确定候选矩阵与正样本标签。本库的成对版本输入两组 N×D 特征，按匹配关系对齐，同一行列索引是正例，得到 N×N 相似度分数并除以正温度 τ，标签为 arange(N)。双向版本再平均 query→key 和 key→query 的交叉熵，计算约 O(N²D)，完整分数空间 O(N²)。

特征 L2 归一化与概率 softmax 是两件事。前者把点积变成余弦相似度，减少向量长度直接放大分数的影响，是相似度设计；交叉熵本身不要求特征单位长度，接收任意实数 logits。不能先 softmax 再传 cross_entropy；需要对数概率的 NLLLoss 才要配 log_softmax。标准库版本允许原始点积或余弦，Torch 参考固定归一化，并用 eps 处理近零范数；标准库余弦版本拒绝零向量，接口行为不能混称完全相同。

公式中的每行损失是稳定 logsumexp 减正例分数，温度越低，分布越尖锐，对相似度的梯度也多一个 1/τ。低温会突出困难负例，也会放大错误配对和假负例；FP32 归约能改善稳定性，但极小温度可能在除法时已让 logits 溢出。N=1 没有竞争候选，成对损失与表示梯度均为零。

“排除自身”要按矩阵定义判断。成对 N×N 对角线是正例，必须保留；SimCLR 将两个视图合成 2N 个样本，同视图自身的对角线才要屏蔽，另一视图正例索引为 (i+N) mod 2N。参考函数没有实现完整 SimCLR。测试可用两个正交匹配向量得到 log(1+exp(−1/τ))，并覆盖错位标签、整体缩放和极端分数。分布式汇集还要核对全局标签偏移与梯度；多个候选都正确时，要明确多正例目标，避免把等价答案强行当负例。

```math
\begin{aligned}z_{ij}&=s(q_i,k_j)/\tau,\quad p_{ij}=\mathrm{softmax}_j(z_i)\\\mathcal L_i&=\log\sum_j e^{z_{ij}}-z_{ii},\quad \mathcal L=\tfrac1N\sum_i\mathcal L_i\\\frac{\partial\mathcal L_i}{\partial s_{ij}}&=\frac{p_{ij}-\mathbf1[j=i]}{\tau}\\s_{\mathrm{cos}}(q,k)&=\frac{q^\top k}{\lVert q\rVert_2\lVert k\rVert_2}\end{aligned}
```

代码：[info_nce](../coding/reference.py#L73) · [info_nce](../coding/torch_primitives.py#L100)

#### 易错点

- CE 不要求特征先做 L2 归一；特征范数与 softmax 概率归一是不同操作。
- 成对矩阵对角线是正例，SimCLR 合并两视图后的对角线才是同一视图自身。
- key 重排要更新正例标签；重复或语义等价样本也可能是假负例。
- 极小温度使除法先产生无穷时，后续 FP32 或 logsumexp 无法补救。

#### 追问

- 同一 query 有多个正确文档时，怎样设计多正例目标？
- 分布式 all-gather 后怎样定位正例，梯度是否传到其他 rank 特征？
- SimCLR 为什么屏蔽自身，却仍要把另一视图正例保留在分母？

<a id="cod-008"></a>
### COD-008 · 实现 DPO loss，怎样避免符号和序列概率错误？

**L2**

#### 答案

DPO loss 比较同一问题下好回答与差回答的相对概率，并用冻结参考模型作为基准；手写时先算回答部分的序列对数概率，再构造偏好 margin。公式中 x 是 prompt，y⁺ 是 chosen，y⁻ 是 rejected，π_θ 是正在训练的策略，π_ref 是参考，β 控制相对差异尺度。

先分别计算策略相对参考的 log probability 增量，再做 chosen 减 rejected，得到 z。loss 用 softplus(−z) 或 −logsigmoid(z)，避免 sigmoid 后再 log 的下溢。chosen 相对概率改善时 z 变大、loss 下降；若初始策略与参考相同，z=0，loss=ln 2，这能快速检查符号。reference 应停止梯度。

序列 log probability 要对正确右移目标 gather，并只把回答 token 求和，prompt 与 padding 不进入求和；标准目标是求和，改成长度平均会改变优化目标。log-softmax 词表计算约 O(BTV)，汇总约 O(BT)，B、T、V 分别是 batch、长度、词表。验证可检查正负 margin、极端值、回答边界、变长样本与参考梯度。长度、数据质量和 β 会影响训练行为，因此不能把一个正确公式当作完整偏好训练配方。

```math
\begin{aligned}z&=\beta\left[\left(\log\pi_\theta(y^+\mid x)-\log\pi_{\mathrm{ref}}(y^+\mid x)\right)-\left(\log\pi_\theta(y^-\mid x)-\log\pi_{\mathrm{ref}}(y^-\mid x)\right)\right]\\\mathcal L_{\mathrm{DPO}}&=\mathrm{softplus}(-z)=-\log\sigma(z)\end{aligned}
```

#### 易错点

- chosen 相对改善应使 loss 下降，符号写反或漏参考项会改变目标。

#### 追问

- 策略与参考初始完全相同，margin 与 DPO loss 分别是多少？

<a id="cod-009"></a>
### COD-009 · 实现 GRPO 组内优势，标准差为零时怎么办？

**L2**

#### 答案

GRPO 的组内优势表示一条回答比同题其他回答好多少，常用奖励减组均值，再除以组标准差。公式里 G 是同一 prompt 的回答数，r_i 是奖励，ε 防止分母过小；本题参考采用除以 G 的总体标准差，即 correction=0，不能无意换成除以 G−1。

例如奖励 [0,1,1]，均值为 2/3，失败回答优势为负，成功回答为正；这给策略一个相对方向。如果全部奖励相同，分子都为零，奖励优势就是零，即使加 ε 也不会创造学习信号。G=1 同样没有组内对比。PPO/GRPO 的 clip 是限制更新幅度的机制，不是独立梯度来源，不能说零优势时它还必然压低熵；KL 等其他损失则仍可能产生梯度。

实现应按 prompt 分组，避免把不同问题混成一组，明确奖励是否停止梯度和是否采用标准差归一。计算时间、输出空间均随组样本数线性增长。测试要覆盖恒定奖励、单样本、平移和缩放，并记录零方差组比例。若比例很高，可检查采样多样性、题目难度与奖励分辨率；是否去掉标准差归一需要结合算法目标，因为它会改变不同问题与奖励尺度的更新权重。

```math
A_i=\frac{r_i-\bar r}{\sqrt{\frac1G\sum_{j=1}^G(r_j-\bar r)^2}+\epsilon}
```

#### 易错点

- 零优势没有奖励更新信号，clip 本身不是独立梯度，不能保证继续压低熵。

#### 追问

- 奖励缩放与 prompt 难度变化，标准差归一如何改变各组更新权重？

<a id="cod-019"></a>
### COD-019 · 手写 VAE 训练 loss：ELBO、重参数化、KL 闭式和 reduction 怎样对应？

**L2** · 腾讯 / 百度

#### 答案

VAE 训练最小化负 ELBO，也就是重建负对数似然加近似后验到先验的 KL，先把损失符号和概率模型讲清楚。VAE 用 p(z)p_θ(x|z) 生成数据，用编码器 q_φ(z|x) 近似难算的真实后验；对数似然等于 ELBO 加一个非负的后验误差 KL，所以 ELBO 是可优化下界。最大化下界里的负 KL，转换为最小化 loss 时要变成正 KL。

常见做法让近似后验 q_φ(z|x) 为 N(μ,diag(σ²)) 的对角高斯，编码器输出均值 μ 和 ℓ=log σ²；先验 p(z) 取 N(0,I)。采样写成 z=μ+exp(ℓ/2)⊙ε，ε 来自标准正态；随机性由独立噪声提供，重建梯度仍可通过 z 回到 μ、ℓ，这叫重参数化。不能把标准差写成 exp(ℓ)，也不能 detach z。公式里的 K_i 是第 i 个样本的闭式 KL，对潜变量维求和，不用额外采样估计。

重建项取决于观测模型。二值数据用 Bernoulli，解码器输出 logits，配 binary_cross_entropy_with_logits；灰度软标签时应明确它是采用的重建代理。实值数据可用固定方差高斯，对应带尺度和常数的平方误差；若方差可学习，还必须保留对数方差项。本题函数支持 Bernoulli logits 与单位方差高斯均值，后者保留常数。

每个样本先对观测维求重建和、对潜维求 KL 和，再分别对 batch 平均，返回总损失与两项分量。默认 β=1 是标准负 ELBO，其他 β 是加权变体；若重建按像素平均而 KL 求和，分辨率变化会隐式改变权重。训练顺序是编码、采样、解码、计算损失、清梯度、反传和更新，并监控重建与 KL 识别后验坍塌。运算随观测和潜变量元素数线性增长；FP32 指数归约仍不能保证异常 ℓ 不溢出。感知损失与对抗训练是视觉自编码器的额外设计，基础目标并不自动包含它们。

```math
\begin{aligned}\mathrm{ELBO}(x)&=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))\\\log p_\theta(x)&=\mathrm{ELBO}(x)+D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\\K_i&=\tfrac12\sum_j(\mu_{ij}^{2}+e^{\ell_{ij}}-1-\ell_{ij}),\quad \ell=\log\sigma^2\\\mathcal L&=\tfrac1B\sum_i[-\log p_\theta(x_i\mid z_i)+\beta K_i]\end{aligned}
```

代码：[vae_reparameterize](../coding/torch_primitives.py#L130) · [vae_loss](../coding/torch_primitives.py#L145)

#### 易错点

- logvar 是 log σ²，采样标准差用 exp(logvar/2)。
- KL 方向是近似后验到先验；不同归约会隐式改变重建与 KL 权重。
- 基础 VAE 目标不自动包含视觉感知损失或 GAN 训练。

#### 追问

- 怎样根据 KL 与重建曲线识别后验坍塌，KL warmup 改变什么？
- 解码器同时预测方差时，高斯重建负对数似然要加哪些项？

<a id="topic-3"></a>
## 采样与缓存实现

<a id="cod-003"></a>
### COD-003 · KV Cache 增量解码的因果 mask 为什么容易写错？

**L2**

#### 答案

KV Cache 下的因果掩码要按真实位置比较，新查询已经位于历史之后，因此矩形矩阵的左上角三角掩码常会写错。设缓存有 past_len 个 token，第 i 个新查询允许读取 j≤past_len+i 的 key；i、j 在各自张量中从零编号。

例如缓存有 5 个历史 token，一次只输入 1 个新 token，Q 长度为 1、K 长度为 6，这一个查询应该读取全部 6 个有效位置。若对 1×6 分数直接取左上角下三角，只会留下第一个 key，历史几乎全被屏蔽。若一次输入多个新 token，后面的新查询还可以读前面的新 token，但不能读尚未到达的后续位置。

实现时把位置比较与 padding、静态缓存未填槽位等限制组合，核对框架对非方阵 is_causal 的具体对齐方式。RoPE 的逻辑位置也必须与缓存中的 K 一致，滑动窗口淘汰旧条目不代表位置归零。验证可关闭 dropout，在相同权重、位置与精度下比较整段 prefill 和逐 token decode 的同位置 logits，使用合理数值容差。构造 mask 的空间为 T_q×T_k，单步注意力随有效历史长度增长。

```math
\mathrm{allowed}[i,j]=\mathbf{1}\!\left[j\leq\mathrm{past\_len}+i\right]
```

#### 易错点

- 非方阵 is_causal 的对齐方式由接口决定，不能默认等于缓存中的真实位置比较。

#### 追问

- 滑动窗口删除旧缓存后，逻辑位置与缓存槽位应怎样分别维护？

<a id="cod-006"></a>
### COD-006 · 实现 top-k / top-p 采样，截断边界怎么处理？

**L2**

#### 答案

Top-k 保留固定数量的高分候选，Top-p 保留累计概率达到阈值的最小高概率前缀，再在保留集合内重新归一化采样。Top-p 必须包括让累计概率第一次跨过 p 的那个 token，否则可能漏掉应保留项，甚至把候选删空。

例如概率 [0.6,0.3,0.1]、p=0.7，要保留前两项而不是只保留 0.6。实现时先排序，找累计概率首次大于等于 p 的位置，保留至该项，再恢复原索引；相等时也停止。例如 [0.5,0.25,0.25]、p=0.5 只保留第一项。p=1 的目标是保留全部概率质量，还需检查浮点累加边界。本题参考顺序是温度缩放、Top-k、归一化、Top-p、再归一化；Top-p 使用的是经过 Top-k 后的分布，改变顺序可能得到不同集合。温度为零则走独立贪心分支，不能做除零。

我会明确 k、p、温度的有效范围、并列值规则及全无效分数处理，用固定随机数生成器检查采样复现。完整排序通常为 O(V log V)、工作空间 O(V)，Top-k 可用部分选择优化；V 是词表大小。测试至少覆盖 p 很小或为 1、k=1、并列候选和极小温度。过滤改变概率分布，因此质量、随机性与输出成本要一起考虑。

#### 易错点

- 只保留累计概率≤p 会漏掉首次跨阈值的 token，甚至使候选为空。

#### 追问

- Top-k 后再算 Top-p，与先算 Top-p 为什么可能留下不同候选？

<a id="cod-014"></a>
### COD-014 · 手算并编码 MHA/GQA 的 KV Cache 显存。

**L2**

#### 答案

MHA/GQA 的 KV Cache 本体可按“两个键值张量 × 批大小 × 层数 × 缓存长度 × KV 头数 × 每头宽度 × 字节数”计算。公式中的 2 对应 K、V，B、L、T 是批大小、层数和长度，h_kv 是 KV 头数，d_h 是每头维度，s 是每元素字节数；GQA 不能把查询头数代进去。

例如 B=1、T=4096、L=32、h_kv=8、d_h=128、FP16 每元素 2 字节，得到 536870912 字节，也就是 512 MiB。若查询头更多，但仍共享这 8 组 K/V，紧凑缓存大小不会随查询头一起增加。同等条件下长度翻倍，缓存本体也翻倍，可能降低可容纳并发。

变长序列的紧凑缓存可把各样本实际长度相加；普通 padded 缓存通常按 B×T_max 分配，分页实现还要按已分配块数和尾块浪费估算。计算器应校验维度和 dtype 字节数，明确 GB 与 GiB 单位，时间为常数或对变长列表线性。这个数字不包含权重、激活、页表、碎片、量化 scale 与工作区，因此只是部署显存预算的一部分，需要再与实际运行峰值核对。

```math
\begin{aligned}\mathrm{bytes}&=2BLT h_{\mathrm{kv}}d_hs\\\mathrm{bytes}_{\mathrm{packed}}&=2Lh_{\mathrm{kv}}d_hs\sum_{b=1}^{B}T_b\end{aligned}
```

#### 易错点

- GQA 用 KV 头数估算缓存，字节转换时也要区分 GB 与 GiB。

#### 追问

- 显存预算固定且其他成本相同时，上下文翻倍怎样影响可用并发？

<a id="cod-020"></a>
### COD-020 · 手写 BucketBatchSampler：怎样减少 padding 并保证 epoch 无遗漏、无重复？

**L2** · 腾讯

#### 答案

BucketBatchSampler 把长度相近的样本放在同一批，减少补到最长序列时浪费的 padding，同时保持可复现的随机性和完整索引覆盖。它接收长度列表并输出一批批索引，不读取数据、不做 padding；DataLoader 的 collate_fn 才负责读取、填充、mask 和标签。公式中的 L_i 是长度，一批的浪费是批大小乘最大长度减实际长度总和。

本题算法先稳定按长度排序，把相邻索引划进大小为 batch_size×bucket_multiplier 的桶；每个 epoch 用 seed+epoch 的局部随机数生成器桶内洗牌、切批，再打乱批顺序。桶大时更接近随机组批，长度优势变弱；桶小时批成员变化少，需要平衡效率与训练随机性。图文或视频也可用估计 token 数或成本分桶，但一个标量不一定准确代表动态视觉计算。

桶大小是 batch_size 的整数倍，因此只有最后一桶可能有尾批。drop_last=False 时，排序、无放回洗牌和不重叠切片让每条索引恰好出现一次；True 时只丢最后非满批，保留 floor(N/B)×B 条。这里公式中的 B 是批大小，N 是样本数。不洗牌可能反复丢排序尾部，洗牌后应检查这种偏置。__len__ 返回批数，set_epoch 改变次序；接入 batch_sampler 后不要再同时设置 batch_size、shuffle、sampler、drop_last。

初始化排序 O(N log N)，每轮 O(N)，索引与批次存储 O(N)。验证合法索引、无重复、覆盖与尾批计数、空数据、N<B 和同 epoch 复现，再测 padding 预算。参考为单进程版本，DDP 每个 rank 各跑全量会重复训练；分布式必须另设分片、相同步数和尾批策略，set_epoch 也不自动管理 worker 增强随机性。

```math
\mathrm{padding}(\mathcal B)=|\mathcal B|\max_{i\in\mathcal B}L_i-\sum_{i\in\mathcal B}L_i,\qquad N_{\mathrm{batch}}=\begin{cases}\lfloor N/B\rfloor,&\mathrm{drop\_last}\\\lceil N/B\rceil,&\text{otherwise}\end{cases}
```

代码：[BucketBatchSampler](../coding/reference.py#L239)

#### 易错点

- 批索引列表接 batch_sampler；接 sampler 会被当作单个样本 key。
- 桶大小不对齐批大小又逐桶 drop_last，可能丢掉多个尾批。
- set_epoch 只控制本采样器次序，不自动管理增强随机性或 DDP 分片。

#### 追问

- DDP 怎样兼顾各 rank 样本不重叠与更新步数一致？
- 按总 token 预算组批时，批次数和梯度归约如何定义？

<a id="topic-4"></a>
## 通用算法与数据结构

<a id="cod-010"></a>
### COD-010 · 数组第 k 大：堆与 Quickselect 怎样取舍？

**L1**

#### 答案

数组第 k 大可以用大小为 k 的小根堆，也可以用 Quickselect；堆适合流式和稳定内存，Quickselect 适合已有完整数组并追求平均线性时间。先确认重复值照常计数，例如 [5,5,3] 的第 2 大是 5，而不是第 2 个不同值 3，并校验 1≤k≤n。

小根堆始终保存目前最大的 k 个元素，堆顶就是其中最小的，也就是当前第 k 大。未满时插入，满后只在新值大于堆顶时替换；小值不影响当前答案。扫描完返回堆顶，时间 O(n log k)、空间 O(k)，不需要一次加载全数据。

Quickselect 选择 pivot 并分区，只继续寻找包含目标排名的一侧，平均 O(n)，糟糕 pivot 最坏 O(n²)。随机 pivot 可以减轻退化，三路分区把小于、等于、大于分开，适合大量重复值；常见版本会修改原数组。测试用排序结果作为独立基线，覆盖 k=1、k=n、全相同、负值和非法 k。选择应看是否允许改输入、能否全部装内存及最坏延迟要求，不能把平均线性当成最坏保证。

#### 易错点

- 重复值通常照常计排名，第 k 大与第 k 个不同的大值是两种问题。

#### 追问

- 数组不能全部载入内存时，怎样流式维护第 k 大？

<a id="cod-011"></a>
### COD-011 · 岛屿问题：DFS/BFS 的时间、空间与边界。

**L1** · 字节跳动

#### 答案

岛屿数量就是陆地在指定邻接规则下的连通块数，可以扫描网格，每发现一块尚未访问的陆地，就用 DFS 或 BFS 把整个岛标记完，计数加一。本题通常只算上下左右四邻域，对角相碰不会连成同一个岛。

例如两个“1”只在对角线相邻，四邻域规则下是两个岛。遍历从当前陆地出发，把有效且未访问的邻居加入栈或队列，并在加入时立即标记，避免同一格被多个邻居重复加入。每格最多处理一次，时间 O(RC)，R、C 是行列数；visited 和最坏队列或栈可达 O(RC)。若允许改输入，可以把访问过的陆地改成水，省单独 visited，但遍历队列仍可能很大。

实现前要确定陆地是字符还是整数、是否允许原地修改，并检查空输入及行列边界。Python 大岛可能让递归 DFS 超出递归深度，迭代栈或 BFS 更稳。测试全水、全陆、狭长蛇形岛和对角接触。如果陆地持续新增，重新扫描成本高，可以用并查集维护连通块：新增一块先加一，再与不同邻接集合合并并减少计数。

#### 易错点

- 四邻域不含对角线，访问邻居时也必须检查行列边界。

#### 追问

- 陆地逐次增加时，并查集怎样增量维护岛屿数量？

<a id="cod-012"></a>
### COD-012 · 手写编辑距离，并压缩到 O(min(m,n)) 空间。

**L1**

#### 答案

编辑距离用动态规划计算把一个字符串变成另一个字符串的最少插入、删除和替换次数；只求数值时可以把空间压到较短字符串长度。dp[i][j] 表示两个长度为 i、j 的前缀的距离，空串变成长度 j 的串需要 j 次插入，反向需要 i 次删除。

比较末尾字符时，最后一步可以删除 a 的末尾，成本来自 dp[i−1][j]+1；插入 b 的末尾，来自 dp[i][j−1]+1；或者匹配、替换两端，来自 dp[i−1][j−1] 加字符不同的成本。例如“cat”变“cut”只需把 a 替成 u，距离为 1。取三者最小值，按行推进，依赖状态都已经算好。这里允许替换，所以不能直接套最长公共子序列公式。

每格只用上一行和本行左侧，可以保留 previous/current 两行，把短串放在列方向，时间 O(mn)、额外空间 O(min(m,n))。单行版本还要保存被覆盖前的左上角。若要恢复编辑路径，需要保留更多状态或用分治方法，不能只剩滚动行后普通回溯。测试空串、相同串、全不同和重复字符，语音识别的 WER 则对词级序列统计这些操作，再除以参考词数，评价单位要说明清楚。

```math
\begin{aligned}dp[i][0]&=i,\qquad dp[0][j]=j\\dp[i][j]&=\min\left\{dp[i-1][j]+1,\ dp[i][j-1]+1,\ dp[i-1][j-1]+\mathbf{1}[a_{i-1}\ne b_{j-1}]\right\}\end{aligned}
```

#### 易错点

- 编辑距离允许替换，不能直接套最长公共子序列的转移。

#### 追问

- 把编辑距离用于词级 WER 时，插入、删除、替换怎样计数与归一？

<a id="cod-013"></a>
### COD-013 · 实现 O(1) 的 LRU Cache，更新已有 key 怎么处理？

**L1**

#### 答案

LRU Cache 用哈希表快速找到元素，用双向链表保存最近访问顺序，命中读取和更新都会把元素移到最近使用端。容量超限时，淘汰另一端最久没访问的元素，这样典型 get、put 的平均时间都是 O(1)，存储随容量线性增长。

例如容量为 2，先放 A、B，再读取 A，随后放 C，应淘汰 B，因为 A 刚被使用。只在插入时更新顺序会变成先进先出，解错题。更新已有 key 时只修改值并移动同一节点，元素数不增加；新 key 才创建节点，超限后同时从链表和哈希表删除旧节点。哨兵头尾能简化空链和单节点边界，Python 也可用 OrderedDict 的移动与弹出操作演示。

我会测试零容量、重复更新、未命中和读操作改变淘汰顺序；检查移除节点的前后指针是否同步。O(1) 是常见哈希假设下的平均值，不是任何冲突输入的严格保证。真实并发服务还要保护共享结构、设计租户隔离的 key，并处理大量请求同时重建同一缓存项的问题；这些是工程扩展，不应让它们掩盖基本的数据结构不变量。

#### 易错点

- 命中读取和更新也要刷新顺序，只按插入时间淘汰会变成 FIFO。

#### 追问

- 并发服务怎样避免同一缺失项被同时重建，以及不同租户的 key 冲突？

<a id="cod-015"></a>
### COD-015 · 手撕代码时怎样设计能揭露错误的测试？

**L2**

#### 答案

能揭露错误的测试要结合输入输出约定、独立基线和数学不变量，不能只检查一个常见例子或把实现公式原样复制成期望值。先明确空输入、重复值、非法参数、是否修改输入及输出顺序，再选小样例和随机边界。

算法题可以用慢但明显正确的小规模穷举作基线：第 k 大对照排序，两数之和枚举所有下标对，动态规划对照短串穷举。神经算子则检查结构性质：因果注意力中改未来 token 不应影响早期输出；缓存解码应与整段计算一致；RoPE 保持范数；LoRA 合并前后输出一致。比如输出尺寸正确却读到了未来，只做 shape 测试根本发现不了。可训练算子还要检查梯度方向、冻结参数和必要的小张量梯度核对。

浮点比较使用绝对与相对容差：接近零时靠绝对容差，量级很大时靠相对容差，不能要求所有内核逐位一致。固定随机种子便于复现，但 GPU 是否完全确定还依赖设备、内核和版本。验证规模以足以覆盖边界为准，慢基线只跑小输入，避免测试成本超过任务价值；新增失败或实现变化再有针对性地扩展。

#### 易错点

- 单个普通样例覆盖不了边界，期望值照抄同一错误公式也发现不了问题。

#### 追问

- 近零与大尺度浮点值，为什么需要分别考虑绝对和相对容差？

<a id="cod-016"></a>
### COD-016 · 手写最长公共子序列：怎样定义状态、推导转移并压缩空间？

**L2** · 字节跳动

#### 答案

最长公共子序列用动态规划比较两个字符串的前缀，子序列允许跳过字符，但必须保持相对顺序。例如“abcde”和“ace”的公共子序列是“ace”，长度为 3；它不是要求连续的公共子串。先确认题目要长度还是实际序列，本题参考函数返回长度。

设 D[i][j] 是两个长度为 i、j 的前缀的最优长度，任一前缀为空时为 0。如果末尾字符相同，可以在更短前缀的最优结果后加这个字符，得到 D[i−1][j−1]+1；若不同，就分别考虑舍弃一方末尾，取 D[i−1][j] 与 D[i][j−1] 的最大值。公式用字符位置从 1 起的约定，代码通常访问 a[i−1]、b[j−1]，要避免下标混用。

二维表时间和空间都是 O(mn)。只求长度时保留上一行和当前行，把短串放列方向，额外空间变成 O(min(m,n))；单行版必须先保存旧左上角，再覆盖当前格。若要输出序列，可保留全表从右下回溯，匹配走左上、不匹配走较大方向，记录后反转；同分可能得到多个正确答案。测试空串、重复字符、全相同、全不相交，并用短串穷举作独立基线，不能用字符集合交集代替顺序和次数约束。

```math
D_{i,j}=\begin{cases}0,&i=0\text{ or }j=0\\D_{i-1,j-1}+1,&a_i=b_j\\\max(D_{i-1,j},D_{i,j-1}),&a_i\ne b_j\end{cases}
```

代码：[longest_common_subsequence](../coding/reference.py#L161)

#### 易错点

- 子序列可跳字符、子串必须连续，两者不匹配时转移不同。
- 滚动行只保留长度计算依赖，不能直接按完整表方法回溯序列。
- 集合交集会丢掉重复次数和顺序，不能代替 LCS。

#### 追问

- 怎样回溯一个实际 LCS，遇到多个同长答案时怎么选？
- 单行更新中，怎样保存被覆盖前的左上角状态？

<a id="cod-017"></a>
### COD-017 · 手写两数之和：怎样用单遍哈希表返回两个不同元素的下标？

**L1** · 腾讯

#### 答案

两数之和可以单遍遍历，用哈希表记住已经见过的数和下标，当前数只需要查找它的补数。目标是返回两个不同的原数组下标，使两数之和等于 target，而不是返回两个数本身。

处理位置 j 的 x 时，先查 target−x 是否在表里；若存在，返回此前下标与 j，否则再把 x 存入。例如 [3,3]、target=6，第一次存下标 0，第二次能找到它，返回 [0,1]。先查后存保证不会把同一个下标用两次；负数和零也遵循相同逻辑。平均时间 O(n)、额外空间 O(n)，哈希查询常数时间是平均假设，不是任意冲突下的严格保证。

排序加双指针也可行，但要携带原下标，且排序成本 O(n log n)。官方题常保证唯一解；本题函数对无解抛 ValueError，多解时返回遍历中首先遇到的合法对，因此测试要检查是否合法，而不强求某个未约定的顺序。若要求全部下标对，需要保存重复值的多个下标，并考虑答案本身可能二次增长。测试可枚举小数组所有 i<j 对，覆盖重复值、单元素、负数和无解，验证返回确实满足公式及不同下标约束。

```math
\mathrm{nums}_i+\mathrm{nums}_j=t,\quad i\ne j,\qquad c=t-\mathrm{nums}_j
```

代码：[two_sum](../coding/reference.py#L150)

#### 易错点

- 先存当前值再查补数，可能把同一元素下标使用两次。
- 需要原下标；返回数值或排序后位置都可能违反题目要求。
- 哈希常数查询是平均复杂度，不是任意冲突输入的保证。

#### 追问

- 已排序数组的双指针为何能根据当前和移动而不漏解？
- 返回全部下标对时，重复值与答案数量怎样影响存储和复杂度？

<a id="cod-021"></a>
### COD-021 · 手写整数平方根：怎样用二分避免浮点误差与乘法溢出？

**L1** · 小红书

#### 答案

整数平方根返回最大的整数 r，使 r²≤n，而不是返回近似浮点根或四舍五入结果。可以利用平方随非负整数单调增加的性质做二分；0、1 直接返回，其他数在 [1,n//2+1] 内搜索。

中点 m 合法时继续向右找更大答案，不合法时缩小右边界，每次排除已经判断的中点，最后返回 high。例如 n=8，结果是 2，因为 4≤8<9；这条平方区间不变量比只比较某个样例更能验证边界。判断 m²≤n 可改写成 m≤n//m，m>0 时等价，避免固定宽度语言的乘法溢出；中点也可用 low+(high−low)//2 避免加法溢出。

二分约 O(log n) 轮，使用常数个整数变量。Python 大整数会扩展，存储和除法成本还随位数增长，不能把每步任意大整数操作都视为严格常数。输入负数或非整数要明确拒绝，不应先浮点 sqrt 再转 int，因为大整数可能丢精度。测试可用 math.isqrt 作独立基线，覆盖完全平方、相邻值和超 64 位整数；它可以验证手写算法，但不能代替题目要求的实现。

```math
r=\max\{k\in\mathbb Z_{\ge0}:k^2\le n\},\qquad r^2\le n\lt (r+1)^2,\qquad m^2\le n\iff m\le\lfloor n/m\rfloor\ (m\gt 0)
```

代码：[integer_sqrt](../coding/reference.py#L290)

#### 易错点

- 题目要向下取整，round(sqrt(n)) 是不同输出。
- 更新边界要排除已判断中点，否则相邻边界可能无限循环。
- math.isqrt 可以做独立测试基线，不能代替手写二分。

#### 追问

- 整数牛顿迭代怎样设置初值并确定何时停止？
- 固定宽度整数中，low+high 的中点计算为什么也可能溢出？

<a id="cod-022"></a>
### COD-022 · 手写最长回文子串：区间 DP 和中心扩展怎样取舍？

**L2** · 小红书 / 阿里巴巴

#### 答案

最长回文子串可以用中心扩展，以常数工作空间检查所有奇偶中心；区间动态规划则用更多存储显式记录每个区间是否回文。子串必须连续，先确认与允许跳过字符的回文子序列不同，并约定返回长度还是实际文本。本题函数返回实际子串，同长取最靠左位置，空串返回空串。

区间状态 P[i][j] 表示 s[i..j] 是否回文，只有两端字符相等，并且“长度不超过 2”或“内部 P[i+1][j−1] 为真”至少一项成立，整个区间才是回文。按长度从短到长，或 i 从大到小推进，保证先算内部；时间和表空间 O(n²)。例如“abba”要在知道“bb”回文之后判断完整区间，长度 2 的分支又不能访问不存在的内部状态。

中心扩展对每个位置检查 (i,i) 的奇数中心，以及 (i,i+1) 的偶数中心，字符相同就向外走。每个回文都有这样的中心，因此不会漏；全相同字符串可导致最坏 O(n²)，额外工作空间 O(1)，返回切片还需结果长度空间。只检查单字符中心会漏掉“abba”。

测试可枚举短串全部连续区间，按正反相同与最早起点选答案。若输入很长，可进一步讨论线性时间 Manacher 的镜像半径和右边界。回文子序列则是另一套区间长度转移：两端相等加 2，不等取删左或删右的较大值；“bbbab”的子序列长度为 4，但最长子串为“bbb”，中心扩展不能解决前者。

```math
P_{i,j}=(s_i=s_j)\land\big((j-i\le1)\lor P_{i+1,j-1}\big),\qquad 0\le i\le j\lt n
```

代码：[longest_palindromic_substring](../coding/reference.py#L307)

#### 易错点

- 子串连续，子序列可跳字符，不能用后者的 max 转移解前者。
- 偶数回文的中心在字符间隙，只查单字符中心会漏解。
- 区间 DP 必须先算内部状态，不能按任意次序访问。

#### 追问

- 每个回文为什么都有扩展中心，什么输入会达到二次时间？
- Manacher 在当前点落入最右回文时，怎样利用镜像初始化半径？

<a id="cod-023"></a>
### COD-023 · 手写全排列：回溯如何恢复现场，重复元素怎样去重？

**L2** · 字节跳动

#### 答案

全排列用回溯逐个选择尚未使用的元素，完成一个长度为 n 的路径后保存副本，再恢复现场尝试其他选择。要先明确重复值是否按下标视为不同，还是只输出不同数值序列；本题函数输出唯一数值排列，保留原输入。

维护 path 和 used，每层选一个未用元素，标记、追加、递归，返回时 pop 并清除标记。保存必须复制 path，否则所有答案会引用同一个持续变化的列表。输入有重复时先排序，同一层若当前值等于前一个值、且前一个尚未使用，就跳过当前分支；如果前一个已经在 path 中，则可继续使用当前同值元素。例如 [1,1,2] 需要允许两个 1 同时出现在排列里，却不能用它们开启两个相同首元素分支。

公式中 c_j 是某个值的重复次数，唯一输出数 U=n!/∏c_j!。无重复时输出本身就需 O(n·n!) 时间，含重复时存答案需 O(nU)，栈、used、路径和排序副本工作空间 O(n)。空输入返回 [[]]，对应 0!=1；元素必须可排序比较，混合不兼容类型不在此接口保证内。测试用短输入的 itertools.permutations 去重作基线，检查答案多重集、无别名和输入未变。只逐个消费时可改生成器减少输出常驻存储；原地交换版本也要每层换回来并同层去重。

```math
U=\frac{n!}{\prod_j c_j!},\qquad \mathrm{skip}(i)=\mathrm{used}_i\lor(i\gt 0\land a_i=a_{i-1}\land\neg\mathrm{used}_{i-1}),\qquad 0!=1
```

代码：[unique_permutations](../coding/reference.py#L321)

#### 易错点

- 递归返回必须恢复 path 与 used，否则污染其他选择分支。
- 保存答案要复制 path，不能让所有结果指向同一个列表。
- 排序后的去重依据同层选择，不能无条件跳过所有同值元素。

#### 追问

- 原地交换回溯怎样恢复现场，并防止同层重复分支？
- 逐个输出排列时，生成器怎样减少答案常驻存储？

<a id="cod-024"></a>
### COD-024 · 反转单链表怎样原地改指针，如何证明不丢节点也不引入环？

**L1** · 腾讯

#### 答案

原地反转单链表要改变原节点的 next 连接，不能只把值倒过来或创建一套新节点。用三个引用保存已反转前缀、当前节点和未处理后继，就能在 O(n) 时间、O(1) 额外空间完成；空链表返回 None，单节点仍返回自身。

开始 previous=None、current=head，每轮先保存 following=current.next，再把 current.next 指向 previous，最后移动 previous=current、current=following。比如 A→B→C，处理 A 后得到 A→None，仍由 following 找到 B；若先改 next 再取后继，后半条链就丢了。结束时 previous 是原尾，也是新头，原头成为尾且 next 为 None。

循环始终把节点分成两部分：previous 指向已处理部分的反向链，current 指向未处理后缀，两部分互不重叠且覆盖全部原节点。每轮只移动一个节点，所以不创建、不遗失；在输入无环条件下有限前进，也不引入新环。参考函数先用快慢指针检查环，发现自环或其他环就报错，并在改指针前拒绝，原结构保持。预检不改变整体复杂度。递归版占 O(n) 栈，长链可能超过 Python 限制。测试除了值倒序，还要核对节点身份、尾指针和再次反转恢复原结构，重复值尤其能揭露伪实现。

```math
(v_0\to v_1\to\cdots\to v_{n-1}\to\varnothing)\longmapsto(v_{n-1}\to\cdots\to v_1\to v_0\to\varnothing),\qquad T(n)=O(n),\ S(n)=O(1)
```

代码：[ListNode](../coding/reference.py#L348) · [reverse_linked_list](../coding/reference.py#L353)

#### 易错点

- 先保存旧 next，再反向改指针，否则会失去未处理后缀。
- 倒序存值或创建新节点，不满足原地反转原节点连接的要求。
- 结束返回新头，原头 next 应为空，否则可能返回尾或形成错误连接。

#### 追问

- 只反转某段或每 k 个节点时，怎样连接各段边界？
- 快慢指针为什么能检出环，递归反转为什么占线性栈空间？

<a id="cod-025"></a>
### COD-025 · 手写股票最大利润：单笔、至多两笔、手续费与冷冻期怎样区分？

**L2** · 腾讯 / 百度

#### 答案

股票最大利润必须先确定交易次数、持仓、手续费和冷冻期，因为单笔、至多两笔与无限笔是不同问题。本题分别实现单笔与至多两笔，不重叠持仓、允许不交易；不是强迫完成两笔。

单笔遍历卖出日，保存此前最低买价与最佳利润，用当天价格减最低价更新答案，再纳入当天价格。初始利润为 0，因此空输入、单日或一直下跌都返回 0，时间 O(n)、空间 O(1)。全局最高减最低可能把卖出放到买入之前，所以不成立。

至多两笔用四个状态：h₁ 是第一轮买入后的持仓余额，c₁ 是至多完成一笔后的空仓余额，h₂、c₂ 对应第二轮。买入减价格 p，卖出加价格，第二次买入基于前一轮空仓。公式所有转移都用上一日状态，初始持仓负无穷、空仓零，返回 c₂，仍可包含零笔或一笔方案。例如 [3,3,5,0,0,3,1,4] 单笔为 4、至多两笔为 6。四状态也只需 O(n) 时间、O(1) 空间；参考接受有限有符号数，业务价格非负应按题意另校验。

无限笔且无费用冷冻期可累加相邻正增量；最多 k 笔扩展买卖状态为 O(nk) 时间、O(k) 空间。手续费统一在买或卖时扣一次，冷冻期则让买入依赖满足等待的旧空仓，不能直接套当天更新值。测试单笔枚举买卖日，两笔枚举有序、不重叠交易并包括单笔与不交易；需要交易日时再保存路径和同收益选择规则。

```math
\begin{aligned}h_{1,t}&=\max(h_{1,t-1},-p_t)\\c_{1,t}&=\max(c_{1,t-1},h_{1,t-1}+p_t)\\h_{2,t}&=\max(h_{2,t-1},c_{1,t-1}-p_t)\\c_{2,t}&=\max(c_{2,t-1},h_{2,t-1}+p_t)\\h_{1,-1}&=h_{2,-1}=-\infty,\quad c_{1,-1}=c_{2,-1}=0\end{aligned}
```

代码：[max_stock_profit](../coding/reference.py#L368) · [max_stock_profit_two_transactions](../coding/reference.py#L436)

#### 易错点

- 四状态允许零笔、一笔或两笔，空输入也可返回零。
- 当天新状态参与后续更新会隐含同日动作，加费用或冷冻期时必须重审。
- 股票 III 是至多两笔，单笔函数不能当完整解法。

#### 追问

- 最多 k 笔怎样扩展买卖状态，何时可退化成无限次贪心？
- 两笔再加手续费与一天冷冻期时，哪些依赖状态要改变？

<a id="cod-027"></a>
### COD-027 · 不调用 sqrt 求非负实数平方根，如何控制误差并处理极大、极小值？

**L2** · 深势科技

#### 答案

非负实数平方根需要返回受误差控制的近似值，稳妥做法是先把数值缩放到安全范围，再二分，最后恢复尺度。它与返回 floor 根的整数题不同；本题接受能转为有限 float 的非负数，拒绝负值、NaN、无穷，零直接返回。

普通二分可用 [0,max(1,x)]，但在原尺度算 mid² 可能溢出，极小 x 又可能被固定绝对误差吞掉。参考用 frexp 写成 x=m·2^e，若 e 为奇数，就把 m 乘 2、e 减 1，得到 x=m·2^(2k)，m 落在 [1/2,2)。只在 [0.5,2] 二分 √m，最后用 ldexp 乘回 2^k，因此平方不受原输入量级影响；这两个函数分解和恢复二进制尺度，并没有直接求根。

公式中的 a、b 是缩放后的夹逼区间，中点恢复为估计 r̂，半宽恢复为误差上界 E。当 E≤abs_tol+rel_tol|r̂| 时停止，默认绝对容差为 0、相对容差 10⁻¹²，保护很小的正根。中点若因浮点舍入不能再推进，就到达可表示精度；迭代上限前仍未满足也未停滞时，应报告未收敛，不能伪装成功。

二分轮数随所需有效精度对数增长，工作空间 O(1)，指数缩放减少量级影响。牛顿法也可用，但要处理初值、零和中间溢出。测试可用 math.sqrt 作独立基线，覆盖最大有限数、最小正数、0<x<1 和非法输入；任何浮点实现都不能保证任意数学精度。x<1 时根比 x 大，因此区间不能简单设成 [0,x]。

```math
\begin{aligned}x&=m\,2^{2k},\quad m\in[\tfrac12,2),\qquad\sqrt x=\sqrt m\,2^k\\\hat r&=2^k\frac{a+b}{2},\qquad E=2^k\frac{b-a}{2}\\E&\le\mathrm{abs\_tol}+\mathrm{rel\_tol}\,|\hat r|\end{aligned}
```

代码：[float_sqrt](../coding/reference.py#L489)

#### 易错点

- 实数近似根与整数 floor 根不同，输入输出契约不能混用。
- 固定绝对误差会吞掉极小数，原尺度平方也可能溢出。
- 浮点停止条件只保证可实现的近似，不能声称任意精度精确结果。

#### 追问

- 不用指数缩放时，怎样用 x/mid 比较来避免平方溢出？
- x 在 0 到 1 之间时，为什么 [0,x] 不能包住它的根？

<a id="cod-028"></a>
### COD-028 · 矩阵中的最长递增路径怎么求，如何避免递归深度溢出？

**L2** · 字节跳动

#### 答案

矩阵最长递增路径可以把格子看成有向无环图，再用拓扑分层 BFS 求最长长度，避免深递归。先明确只允许上下左右四邻移动、每步严格增大、返回经过的格子数；单格长度为 1，空矩阵为 0，非矩形输入应拒绝。

从小值格指向相邻大值格，沿边值不断增加，因此不可能绕回起点形成环，相等格不连边。记忆化 DFS 可用 dp(v)=1+最大前驱长度，但一条很长蛇形路径会占很多递归栈。拓扑方法先为每格数较小邻居作为入度，把入度为零的局部极小格入队。每轮处理当前全部格，再把更大邻居入度减一，降零者进入下一轮。

一个格必须等所有小前驱处理完才进入队列，其轮次等于最长前驱链加一，所以总轮数就是最长路径。例如 [1,2,3] 依次进三层，长度为 3。每格最多四条边，计算入度和遍历均 O(mn)，入度、队列空间 O(mn)，不改输入。需要实际路径时可同时维护最优前驱和长度，再回溯。

严格增加是无环证明的核心；改成不下降后相等邻居会构成环，不能直接套用；任意位置跳转也改变了题目。测试可用小矩阵穷举分支作基线，覆盖全相等、单行、边界和长递增行，后者可验证实现不依赖 Python 递归深度。长度按节点而非边计数，要避免差一。

```math
\begin{aligned}u\to v&\iff u,v\ \mathrm{are\ four\ neighbors}\ \land\ M_v\gt M_u\\\mathrm{dp}(v)&=1+\max\big(\{0\}\cup\{\mathrm{dp}(u):u\to v\}\big)\\L&=\max_v\mathrm{dp}(v)\end{aligned}
```

代码：[longest_increasing_matrix_path](../coding/reference.py#L531)

#### 易错点

- 路径长度计格子数，不是边数，因此单格是 1。
- 严格增加才保证无环，相等相邻格不能连递增边。
- 四邻移动、严格性与任意跳转是不同规则，必须先确定。

#### 追问

- 放宽成不下降后，相等节点为什么使拓扑方法失效？
- 怎样在求长度时记录前驱，并恢复一条最优路径？

## 参考资料

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/html/2305.13245v3)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [PyTorch functional.cross_entropy：直接接收未归一化logits](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch functional.normalize：Lp特征范数归一化](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html)
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
- [Best Time to Buy and Sell Stock III: official problem definition](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/)
- [Variation-aware Vision Token Dropping for Faster Large Vision-Language Models（v2）](https://arxiv.org/html/2509.01552v2)
- [V2Drop CVPR 2026 官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models_CVPR_2026_paper.html)
- [V2Drop 官方 LLaVA 实现](https://github.com/xuyang-liu16/V2Drop/blob/main/llava/model/language_model/V2Drop.py)
- [PyTorch 2.14 官方 torch.argsort 文档](https://docs.pytorch.org/docs/2.14/generated/torch.argsort.html)
- [Python math: frexp, ldexp, isfinite and floating-point tolerance](https://docs.python.org/3/library/math.html)
- [Longest Increasing Path in a Matrix: official four-neighbor problem definition](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/)
- [Mixtral of Experts](https://arxiv.org/html/2401.04088v1)
- [Transformers MixtralTopKRouter reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [torch.nn.LayerNorm — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)
