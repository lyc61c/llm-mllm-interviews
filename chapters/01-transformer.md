# Transformer 与数学基础

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 题目

- [TFM-001 · 缩放点积注意力如何计算，为什么除以 √d_k？](#tfm-001)
- [TFM-002 · 多头注意力与单头注意力有什么区别？](#tfm-002)
- [TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？](#tfm-003)
- [TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？](#tfm-004)
- [TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？](#tfm-005)
- [TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？](#tfm-006)
- [TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？](#tfm-007)
- [TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？](#tfm-008)
- [TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？](#tfm-009)
- [TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？](#tfm-010)
- [TFM-011 · Transformer 一层的时间、空间复杂度如何估算？](#tfm-011)
- [TFM-012 · 残差连接为什么能帮助深层模型训练？](#tfm-012)
- [TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？](#tfm-013)
- [TFM-014 · 输入 embedding 与输出 LM head 权重共享有什么利弊？](#tfm-014)
- [TFM-015 · MoE 与 Dense 的参数量和计算量应怎样比较？](#tfm-015)
- [TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？](#tfm-016)
- [TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？](#tfm-017)
- [TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？](#tfm-018)
- [TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？](#tfm-019)
- [TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？](#tfm-020)
- [TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？](#tfm-021)
- [TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？](#tfm-022)
- [TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？](#tfm-023)

<a id="tfm-001"></a>
## TFM-001 · 缩放点积注意力如何计算，为什么除以 √d_k？

**L1**

### 答案

缩放点积注意力先用 $Q$ 与 $K$ 的匹配分数，在 key 轴做 softmax，再加权汇聚 $V$。设 $Q\in\mathbb R^{L_q\times d_k}$、$K\in\mathbb R^{L_k\times d_k}$、$V\in\mathbb R^{L_k\times d_v}$，则 $S=QK^\top/\sqrt{d_k}+M$、$A=\operatorname{softmax}(S)$，输出 $AV\in\mathbb R^{L_q\times d_v}$。Self-attention 常取 $Q=XW_Q$、$K=XW_K$、$V=XW_V$，cross-attention 的 $Q$ 与 $K/V$ 可来自不同序列。

缩放的尺度分析假设各分量零均值、单位方差，$q_i$ 与 $k_i$ 独立，且不同维的乘积互不相关：此时 $\operatorname{Var}(q_i k_i)=1$，$\operatorname{Var}(q\cdot k)=d_k$，除以 $\sqrt{d_k}$ 后方差为 1。实际训练后的相关性和方差未必满足这些假设。分数差过大会使分布过于尖锐，softmax 的 Jacobian $\partial A_i/\partial S_j=A_i(\delta_{ij}-A_j)$ 部分元素变小；这不表示所有后续梯度必然消失，softmax 与交叉熵合并时 logits 梯度仍为预测概率减目标概率。

允许的连接加 0，禁止连接加 $-\infty$，再做 softmax。实现常先减去每行最大有限分数以避免指数溢出；全屏蔽行须单独处理，并检查 dtype、广播和内核的 mask 约定。有限且未屏蔽的 logits 在精确计算下得到严格为正、和为 1 的权重，所以 softmax 本身没有硬稀疏；数值下溢和 mask 产生的零是另外的机制。权重适合描述加权汇聚，但不能直接当作因果重要性。

$$
\begin{aligned}\operatorname{Attention}(Q,K,V)&=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V\\ \operatorname{Var}(q\cdot k)&=d_k\\ \frac{\partial p_i}{\partial s_j}&=p_i(\delta_{ij}-p_j)\end{aligned}
$$

### 易错点

- 把缩放分母写成 $d_{\rm model}$ 或 $d_k$，或省略独立/方差假设后把 $\operatorname{Var}(q\cdot k)=d_k$ 当普遍定律。
- 把 mask 加到已归一化权重后不重归一化，或把 softmax 说成自动只选少量 token。

### 追问

- 如果 Q/K 做 L2 归一化，点积分布与温度参数该怎样重新分析？
- cross-attention 的 $L_q\ne L_k$ 时，各中间矩阵与输出是什么形状？

<a id="tfm-002"></a>
## TFM-002 · 多头注意力与单头注意力有什么区别？

**L1**

### 答案

多头注意力用多组分别学习的 $Q/K/V$ 投影，在不同子空间并行计算注意力，拼接各头输出，再经 $W_O$ 混合回模型维度。以输入 `X:[B,T,D]` 为例，第 $i$ 头的 $W_Q^i,W_K^i\in\mathbb R^{D\times d_k}$、$W_V^i\in\mathbb R^{D\times d_v}$，输出 `head_i:[B,T,d_v]`。原始 base 模型取 $D=512$、$h=8$、$d_k=d_v=64$。

工程上把各头堆为 `Q/K:[B,h,T,d_k]`、`V:[B,h,T,d_v]`，分数为 `[B,h,T,T]`。输出从 `[B,h,T,d_v]` 转置并拼接为 `[B,T,h*d_v]`，乘 `W_O:[h*d_v,D]` 得到 `[B,T,D]`；拼接增加的是特征维度。Cross-attention 的权重形状为 `[B,h,L_q,L_k]`，输出长度仍为 $L_q$。一次大线性层后 reshape 与逐头投影等价，并不表示各头共用相同权重。

当 $hd_k=hd_v=D$ 时，忽略 bias，标准 MHA 的 Q/K/V/O 总权重约为 $4D^2$；固定 $D$ 增加头数只会缩小每头维度，参数量不随头数成倍增长。不过注意力矩阵元素数随 $hT^2$ 变化，显存与内核效率仍需比较。各头可以学习不同匹配模式，也可能冗余，不能预设每头必定负责语法或实体等固定任务。MQA/GQA 改变了 K/V 共享方式，其参数与 KV cache 要另外估算。

$$
\begin{aligned}\operatorname{head}_i&=\operatorname{softmax}\!\left(\frac{(XW_Q^i)(XW_K^i)^\top}{\sqrt{d_k}}+M\right)XW_V^i\\ \operatorname{MHA}(X)&=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O\end{aligned}
$$

### 易错点

- 将单头维度 $d_k$ 与总隐藏维度 D 混用，漏写转置/拼接或 $W_O$ 导致 shape 不闭合。
- 把实现里的 fused QKV 层误认为不同 head 使用完全相同的投影参数。

### 追问

- 固定 D 和 h 时，GQA 的参数量与 KV cache 怎样变化？
- 为什么 concatenation 后还需要 $W_O$，而不是直接相加各头输出？

<a id="tfm-003"></a>
## TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？

**L1**

### 答案

Attention mask 在 softmax 前排除不可见的 key：允许连接加 0，禁止连接加 $-\infty$。将禁止位置的分数乘 0 无法屏蔽，因为 $\exp(0)$ 仍会得到概率；全屏蔽行在朴素 softmax 中可能产生 NaN，须核对后端处理。

因果掩码从 0 编号时允许 $j\le i$，包括自身。位置 $i$ 的 logits 预测 $x_{i+1}$，所以看到 $x_i$ 不会泄漏下一词；只加 loss mask 而没有 causal mask，仍会泄漏未来信息。Padding mask 通常排除无效 key，pad query 的输出在后续计算或损失中忽略。Loss mask 只决定哪些目标计损：prompt 的 `labels=-100` 时，合法 prompt 仍应保持可见，否则回答无法读取问题。独立 packed 样本还要要求 query/key 属于同一样本。

已有 $P$ 个历史 token，一次处理 $L$ 个新 token 时，query 长度为 $L$、key 长度为 $P+L$，第 $i$ 个新 query 可看 $j\le P+i$。不能直接套用左上对齐的矩形下三角；须检查内核的因果偏置、真实缓存位置、左 padding 和静态缓存中未填的槽位。

框架的 bool mask 与加性 mask 含义不同：PyTorch 2.14 SDPA 的 `attn_mask=True` 表示允许关注，`MultiheadAttention.key_padding_mask=True` 表示屏蔽。广播应覆盖 `[B,H,L_q,L_k]`，可用小输入验证可见性。普通 batch 可按本批最长序列 padding，长度分桶能减少浪费；位置编号须与真实 token 和模型方案一致。Padding mask 通常不自动省去 pad 的算术计算，变长内核或 packing 还需显式段边界。

$$
\begin{aligned}A&=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)\\ M_{ij}&=\begin{cases}0,&\operatorname{allowed}(i,j)\\-\infty,&\text{otherwise}\end{cases}\\ \operatorname{allowed}_{\rm cached}(i,j)&\iff j\le P+i\end{aligned}
$$

### 易错点

- 把 prompt loss mask、EOS 边界或重置 `position_ids` 当成 attention 隔离。
- 不检查矩形 causal 对齐和 bool 含义，导致缓存只能关注最前面的 key 或泄漏未来。

### 追问

- 如何用扰动另一个 packed 样本的测试证明没有跨样本信息泄漏？
- 只有一个新 query 且所有 key 均为有效历史时，还需要显式三角 mask 吗？

<a id="tfm-004"></a>
## TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？

**L1**

### 答案

Encoder-Decoder 先编码源序列，再由 decoder 自回归生成目标，两边长度可以不同。原始 Transformer 的 encoder 和 decoder 各有 6 层：encoder 每层含双向 self-attention 和 FFN，decoder 另含因果 self-attention 与 cross-attention。源输入被编码为 $H_{\rm src}$，目标第 $t$ 步依赖源输入与目标历史。

Cross-attention 用 decoder hidden 作 $Q$、encoder memory 作 $K/V$，可访问完整已知源句并屏蔽源 padding；decoder 自身仍只看目标历史。源 memory 可一次编码，各层的 cross-attention K/V 投影可复用。条件概率分解为 $p(y\mid x)=\prod_t p(y_t\mid y_{<t},x)$，训练时用右移目标和 teacher forcing 并行计算各位置损失。原模型使用 PostNorm，并在两边注入位置编码；后续架构不必固定 6 层或同一种归一化。

BERT 等 encoder-only 常用双向表示和 MLM，适合理解及表示任务；GPT 类 decoder-only 用因果目标训练，可把输入与输出串到同一上下文。Prefix LM 的分段可见性不等于具有独立 encoder 和 cross-attention。选型要结合任务形态、源/目标长度、memory 复用、缓存、吞吐与训练数据：encoder-only 可经额外设计用于生成，decoder-only 也能分类，结构标签本身不能保证任务效果。

$$
\begin{aligned}H_{\rm src}&=\operatorname{Encoder}(x)\\p_\theta(y\mid x)&=\prod_t p_\theta(y_t\mid y_{<t},H_{\rm src})\\Q&=H_{\rm dec}W_Q,\quad K=H_{\rm src}W_K,\quad V=H_{\rm src}W_V\end{aligned}
$$

### 易错点

- 把 source 与 target 强制要求等长，或把 cross-attention 也无条件设置成目标三角 mask。
- 把 encoder 的双向 attention 直接移到生成中的未知目标位置而造成未来信息泄漏。

### 追问

- 改变 source 长度时，cross-attention 权重矩阵的哪一轴变化？
- 条件 memory 的缓存与 decoder 历史 KV cache 的增长方式有何不同？

<a id="tfm-005"></a>
## TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？

**L1**

### 答案

LayerNorm 对每个 token 的 hidden 维计算均值与方差，再用 $\epsilon$ 和可学习的 $\gamma,\beta$ 做标准化与仿射变换。对 `X:[B,T,D]`、`normalized_shape=D`，均值和方差形状为 `[B,T,1]`，$\gamma,\beta$ 为 `[D]`。方差使用分母 $D$ 的总体形式；$\epsilon$ 加在方差内，避免分母为零并影响小方差时的尺度。

归一化只覆盖指定的特征轴；若 `normalized_shape` 包含多个轴，统计范围也随之扩大。仿射或 bias 可由具体架构关闭。同一层各 token 共享这组参数，但不同层通常分别学习。LN 在训练与推理时都依据当前 token 重算统计，不依赖 batch size 或运行均值，适合变长、在线和自回归解码。

BatchNorm 通常按通道聚合 batch 与其他指定轴，训练时用批统计，常规推理用估计的总体或运行统计。小 batch、分布变化、序列位置差异和 padding 混入会影响估计；变长序列也能使用 BN，但须明确统计轴和有效位置处理。LN 有助于稳定激活与优化，训练是否稳定还依赖残差、Pre/PostNorm、初始化和学习率，不能只由 LN 推出梯度永不爆炸或消失。

$$
\begin{aligned}\mu_{bt}&=\frac1D\sum_{d=1}^D x_{btd}\\\sigma^2_{bt}&=\frac1D\sum_{d=1}^D(x_{btd}-\mu_{bt})^2\\\operatorname{LN}(x)_{btd}&=\gamma_d\frac{x_{btd}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}+\beta_d\end{aligned}
$$

### 易错点

- 对 `[B,T,D]` 在 T 轴归一化，或把每 token 均值误写成跨 batch 均值。
- 省略 $\epsilon/\gamma/\beta$、混用 D 与 $D-1$，或宣称 BN 一定不能处理任何变长输入。

### 追问

- `LN(D)` 与 `LN((T,D))` 的输出相同吗，为什么？
- RMSNorm 去掉了哪些计算，是否仍把每个 token 的均值归零？

<a id="tfm-006"></a>
## TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？

**L2**

### 答案

RMSNorm 按输入的均方根缩放，省去 LayerNorm 的减均值步骤，通常在根号内加入 $\epsilon$，再乘可学习的逐通道权重 $\gamma$。LN 使用中心化方差，RMSNorm 使用未经中心化的二阶矩。

两者都有对整体尺度的归一化作用，但平移性质不同：输入统一加常数时，LN 的归一化部分基本不变，RMSNorm 通常会变化。RMSNorm 的计算较简单，归约常用更高精度以减少溢出和舍入误差；实际速度与质量仍取决于模型、dtype 和融合内核，不能直接套用论文中的收益比例。

$$
\operatorname{RMSNorm}(x)=\frac{\gamma\odot x}{\sqrt{D^{-1}\sum_{i=1}^D x_i^2+\epsilon}}
$$

### 易错点

- 把 RMSNorm 说成标准化到零均值。
- 说去均值在任何模型中都完全无用。

### 追问

- 为什么 RMSNorm 常在 FP32 中计算归约？
- RMSNorm 与 L2 normalization 有何尺度差别？

<a id="tfm-007"></a>
## TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？

**L2**

### 答案

Pre-Norm 的结构是 $y=x+F(\operatorname{Norm}(x))$，Post-Norm 是 $y=\operatorname{Norm}(x+F(x))$。Pre-Norm 的残差主路径保留恒等连接，给梯度提供直接通道，深层训练通常更稳定，模型末端一般还会添加最终归一化。

Post-Norm 的初始化分析显示部分层梯度可能较大，可解释其对 warmup 的敏感性，但该结论依赖分析条件。实际是否需要 warmup，还受深度、初始化与优化器影响；比较两者应匹配训练预算并分别调参，不能只用同一学习率下的 loss 判定优劣。

$$
\begin{aligned}y_{\rm pre}&=x+F(\operatorname{Norm}(x))\\y_{\rm post}&=\operatorname{Norm}(x+F(x))\end{aligned}
$$

### 易错点

- 说 Pre-Norm 必然不需要 warmup。
- 把更稳定训练直接等同于所有任务更好的最终效果。

### 追问

- 为什么很多 Pre-Norm 模型还有 final norm？
- 残差随深度增长如何控制？

<a id="tfm-008"></a>
## TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？

**L2**

### 答案

RoPE 按位置旋转 $Q/K$ 的成对分量，使点积显式出现相对位置差，并保持向量范数；通常不旋转 $V$。旋转子空间为偶数维 $d_r$，第 $j$ 对频率 $\theta_j=\mathrm{base}^{-2j/d_r}$，位置 $m$ 使用二维旋转 $R(m\theta_j)$。不同频率覆盖不同距离尺度，部分模型只旋转 head 的部分维度。

令 $q'_m=R_mq_m$、$k'_n=R_nk_n$，由 $R_m^\top R_n=R_{n-m}$ 可得 $(q'_m)^\top k'_n=q_m^\top R_{n-m}k_n$。这是位置依赖呈相对差，内容向量仍受上下文影响。单对分量 $(a,b)$ 变为 $(a\cos\phi-b\sin\phi,a\sin\phi+b\cos\phi)$；相邻维配对与前半/后半配对可通过置换关联，已有 checkpoint 却不能未经权重转换便替换配对约定。

Transformers v4.57.1 的 `LlamaAttention` 先旋转当前 Q/K，再写入缓存，因此历史已旋转 K 不应再次旋转。新 token 须使用正确逻辑位置的 cos/sin，分块 decode、左 padding 与 packed 样本都要核对 `position_ids`。`cache_position` 管缓存槽位，RoPE 的 `position_ids` 管相位，两者接口含义不同。滑动窗口移除旧缓存不代表逻辑时间归零；若改变频率或位置缩放策略，必须核验历史 K 的一致性，并重算或使用明确支持该变化的缓存实现。

$$
\begin{aligned}\theta_j&=\mathrm{base}^{-2j/d_r}\\R(\phi)&=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}\\q'_m&=R_mq_m,\quad k'_n=R_nk_n\\(q'_m)^\top k'_n&=q_m^\top R_{n-m}k_n\end{aligned}
$$

### 易错点

- 直接把位置向量加到 embedding 来描述 RoPE，或对缓存 K 重复应用旋转。
- 把 RoPE 的相对点积性质当成任意距离单调衰减、任意长度可靠外推的保证。

### 追问

- 用整段 prefill 与逐 token cache decode 比较 logits 时，哪些位置和数值条件必须一致？
- RoPE 的 base、旋转维度、缩放方案改变后，现有 KV cache 能否复用？

<a id="tfm-009"></a>
## TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？

**L2**

### 答案

增大 `max_position_embeddings` 只放宽输入长度配置，不能保证模型处理训练范围外的位置和长距离依赖。若从 $L$ 扩到 $L'$，线性位置插值用 $m'=mL/L'$ 将位置压回训练区间，通常需继续训练。

压缩位置也会降低局部位置分辨率，倍率越大越要检查短距离性能。位置方案调整与长样本训练解决的问题不同；评测应覆盖多种长度和证据位置，并同时报告短文本表现、长检索正确率、KV 占用与 TTFT。

$$
m'=m\frac{L}{L'}
$$

### 易错点

- 把 RoPE 的数学可计算性当作可靠外推保证。
- 只测一个 needle case 就声称长上下文全面有效。

### 追问

- 位置插值会怎样影响近邻 token？
- 如何排除模型只依赖文档开头或结尾？

<a id="tfm-010"></a>
## TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？

**L2**

### 答案

FFN 对每个 token 独立做通道变换：原始 Transformer 用升维线性层、ReLU 和降维线性层。按行向量约定，$\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2$，`x:[D]`、`W_1:[D,m]`、`W_2:[m,D]`；对 `X:[B,T,D]` 沿最后一轴执行，输出形状不变。原始 base 模型取 $D=512,m=2048$。

Position-wise 表示同一层各 token 使用相同参数，不直接混合不同 token；不同层通常各有参数。跨 token 信息由 attention 等模块汇聚，FFN 增加各位置的非线性组合与容量。若去掉非线性，两层仿射可合并为 $x(W_1W_2)+(b_1W_2+b_2)$，单纯增加中间宽度没有同样的表达收益。

现代 SwiGLU 常写为 $(\operatorname{SiLU}(xW_g)\odot xW_u)W_d$，包含 gate/up/down 三个矩阵，bias 由模型决定；它增加逐元素门控，并非只替换激活名称。忽略 bias，原 FFN 参数约为 $2Dm$，SwiGLU 约为 $3Dm$；若匹配原来 $m=4D$ 的 $8D^2$ 预算，SwiGLU 中间宽度约取 $8D/3$，再按硬件对齐。宽度、激活与融合内核都影响容量、激活显存和吞吐。

$$
\begin{aligned}\operatorname{FFN}(x)&=\operatorname{ReLU}(xW_1+b_1)W_2+b_2\\\operatorname{SwiGLU}(x)&=(\operatorname{SiLU}(xW_g)\odot xW_u)W_d\\N_{\rm FFN}&\approx2Dm,\quad N_{\rm SwiGLU}\approx3Dm\end{aligned}
$$

### 易错点

- 漏写第二次线性变换或 bias，并将原论文激活写成所有现代 LLM 都使用的配置。
- 说 position-wise 参数不共享，或把 FFN 的同位置变换误认为能直接读取别的位置。

### 追问

- 若无 bias 和非线性，两层 FFN 的等效矩阵秩受什么限制？
- 为什么同参数预算下 SwiGLU 的中间宽度常小于 $4D$？

<a id="tfm-011"></a>
## TFM-011 · Transformer 一层的时间、空间复杂度如何估算？

**L2**

### 答案

设序列长度为 $T$、隐藏维度为 $d$，一层 Transformer 的主要计算是 $O(Td^2+T^2d)$，同时包含线性投影/FFN 与注意力的两两交互。QKV/O 投影约需 $4Td^2$ 次乘加，FFN 按中间维度另算；$QK^\top$ 与 $AV$ 合计约 $2T^2d$ 次乘加。按 FLOPs 统计时，一次乘加通常算 2 FLOPs。

朴素多头注意力分数约保存 $BhT^2$ 个元素，实际峰值还包括其他激活和工作区。FlashAttention 可避免完整中间矩阵的显存存储，但精确稠密注意力的算术量仍为二次。短序列、大隐藏维度时线性层可能占主导，长序列时二次项更明显。

$$
\begin{aligned}C_{\rm layer}&=O(Td^2+T^2d)\\M_{\rm attention}&=O(BhT^2)\end{aligned}
$$

### 易错点

- 只报 $O(T^2)$ 而不说明 d、投影与 FFN。
- 把 FlashAttention 说成改变了精确注意力的二次算术复杂度。

### 追问

- 使用 KV cache 后单步注意力复杂度如何变化？
- 为什么 FLOPs 更少的实现可能实际更慢？

<a id="tfm-012"></a>
## TFM-012 · 残差连接为什么能帮助深层模型训练？

**L1**

### 答案

残差连接 $y=x+F(x)$ 让子层学习输入表示的修正，并为信号和梯度提供直接路径。两支形状须一致；若改用投影支路 $y=P(x)+F(x)$，直接支路的 Jacobian 为 $J_P$。残差相加不会像 concat 一样扩大特征维，$F(x)\approx0$ 时可接近恒等传递。

简单块的 Jacobian 为 $I+J_F$，反传为 $\partial L/\partial x=(I+J_F)^\top\partial L/\partial y$。多层仍需连乘各块导数，若 $J_F\approx-I$ 还会相互抵消，所以恒等项并不保证梯度永远稳定。

原始 Transformer 的 PostNorm 是 $\operatorname{LN}(x+\operatorname{Dropout}(F(x)))$，完整导数还含 LN 和 dropout；attention、FFN 各有残差与归一化。常见 PreNorm 为 $x+\operatorname{Dropout}(F(\operatorname{LN}(x)))$，直接支路不经过该块 LN，通常有助于深层优化。随深度增加仍应检查残差尺度、激活方差、溢出和梯度范数，并结合初始化、学习率与最终 norm 判断稳定性。

$$
\begin{aligned}y&=x+F(x),\quad J_y=I+J_F\\y_{\rm post}&=\operatorname{LN}(x+\operatorname{Dropout}(F(x)))\\y_{\rm pre}&=x+\operatorname{Dropout}(F(\operatorname{LN}(x)))\end{aligned}
$$

### 易错点

- 声称残差能让任何深度和初始化的网络梯度恒为 1。
- 混淆 PostNorm/PreNorm，或把投影 shortcut 当成严格恒等支路。

### 追问

- PostNorm 为什么不能直接套用 $J=I+J_F$ 作为完整块导数？
- 将残差支路乘以 $\alpha$ 后，Jacobian 与激活尺度如何变化？

<a id="tfm-013"></a>
## TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？

**L1**

### 答案

离散分布满足 $H(p,q)=H(p)+D_{\rm KL}(p\|q)$，其中 $H(p)=-\sum_i p_i\log p_i$、$H(p,q)=-\sum_i p_i\log q_i$、$D_{\rm KL}(p\|q)=\sum_i p_i\log(p_i/q_i)$。展开对数即可证明。两者须定义在同一支持空间并归一化；$0\log0$ 按极限为 0，若 $p_i>0,q_i=0$，交叉熵与 KL 为 $+\infty$。

KL 非负，$p=q$ 时取零，通常不对称，也不是距离度量。固定目标 $p$ 时，最小化 CE 与最小化该方向的 KL 等价；若目标也含待训练参数且未 detach，$H(p)$ 与 $p$ 的梯度不能忽略。蒸馏还须明确 teacher/student 的方向以及是否停止 teacher 梯度。

One-hot 监督样本的目标熵为零，单样本 CE 与 KL 都为 $-\log q(y)$，但真实数据的条件熵不一定为零。软标签与标签平滑的目标熵通常大于零；固定归一化 $p$、$q=\operatorname{softmax}(z)$ 时，对 logits 的梯度为 $q-p$，对概率 $q_i$ 本身为 $-p_i/q_i$。

`CrossEntropyLoss` 通常接收原始 logits，内部做稳定的 log-softmax/NLL；`KLDivLoss` 通常接收 `input=log q,target=p`，接口顺序与数学记号不同，reduction 也须按分布单位解释。自然对数下，PPL 是有效 token 平均 NLL 的指数；不同长度 batch 要按有效 token 加权，比较时统一 tokenizer、数据与窗口。低 PPL 不保证指令遵循、事实性或偏好更好。

$$
\begin{aligned}H(p,q)&=H(p)+D_{\rm KL}(p\|q)\\\nabla_\theta H(p,q_\theta)&=\nabla_\theta D_{\rm KL}(p\|q_\theta)\quad(p\ \text{fixed})\\\operatorname{PPL}&=\exp\!\left(-\frac{1}{N_{\rm valid}}\sum_{t\in\mathcal T_{\rm valid}}\log p_\theta(x_t\mid x_{<t})\right)\end{aligned}
$$

### 易错点

- 无条件声称 CE 与 KL 完全相同，遗漏目标熵及 fixed-p 前提。
- 把 `KLDivLoss` 的输入顺序/mean 归一化当成数学 KL 的默认定义。

### 追问

- 如何从 softmax 推导对 logits 的梯度 $q-p$？
- 可训练的软目标或双向 KL 会怎样改变优化目标？

<a id="tfm-014"></a>
## TFM-014 · 输入 embedding 与输出 LM head 权重共享有什么利弊？

**L2**

### 答案

语言模型将 `H:[B,T,D]` 经 `W_vocab:[D,V]` 与 `b:[V]` 映射为 `logits:[B,T,V]`，再沿词表轴做 softmax，得到下一 token 概率。Logits 本身未归一化，交叉熵实现通常直接接收它。

输入 embedding 表 $E\in\mathbb R^{V\times D}$ 与输出 head 维度相容时，可令 $W_{\rm vocab}=E^\top$。这样省去一份约 $VD$ 的权重，输入表示与输出分类器接收共同梯度；输出 bias 仍可保留，词表分类计算量也不会随之消失。若两边词表或 hidden 维度不同，需额外映射。原始 Transformer 共享两个 embedding 层与 pre-softmax 线性变换，并在 embedding 使用时乘 $\sqrt D$。

权重 tying 不意味着标准 Transformer 的各层、各头自动共享参数：相同结构通常分别学习，fused QKV 也只是计算安排。ALBERT 的跨层共享是专门的设计，可共享 attention、FFN 或全部，默认全部共享。验证应检查参数别名、`state_dict`、词表与输出维度；共享也不等于冻结，不能保证所有任务都改善。

$$
\begin{aligned}Z&=HW_{\rm vocab}+b\\p_v&=\frac{\exp z_v}{\sum_u\exp z_u}\\W_{\rm vocab}&=E^\top,\quad \Delta N\approx VD\end{aligned}
$$

### 易错点

- 把同一层内跨位置使用 FFN、embedding/head tying 与跨层或跨头共享混为一谈。
- 对 logits 先 softmax 后再交给期望 logits 的 `CrossEntropyLoss`，或沿 hidden 轴做词表归一化。

### 追问

- 共享 embedding/head 时，padding token 对应的输出行是否一定永远没有梯度？
- ALBERT 共享参数为何省参数却仍要执行多层计算？

<a id="tfm-015"></a>
## TFM-015 · MoE 与 Dense 的参数量和计算量应怎样比较？

**L2**

### 答案

MoE 通常把部分 FFN 替换为专家集合，由 router 为每个 token 选择 top-k 专家并加权输出，以较少的激活计算提供较大的总参数容量。Dense 则通常每个 token 使用全部层参数。

总参数、激活参数与真实成本要分别统计：激活参数包含共享模块和被选专家，不能简单用总参数乘 $k/E$。路由不均衡会形成热点、容量溢出或 token 丢弃，常需辅助损失等机制。专家并行还有 all-to-all 通信，小 batch、跨节点或低利用率场景可能抵消算术收益，甚至增加时延。

### 易错点

- MoE 总参数大就必然推理更慢，或激活参数少就必然更快。
- 忽略共享模块、路由与通信成本。

### 追问

- 负载均衡会不会损害专家专门化？
- 如何区分模型 FLOPs 与实际 GPU 成本？

<a id="tfm-016"></a>
## TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？

**L1**

### 答案

Self-attention 的 $Q/K/V$ 来自同一序列；cross-attention 的 $Q$ 来自需要更新的目标表示，$K/V$ 来自另一组条件或 memory。设 $X\in\mathbb R^{L_q\times d_x}$、$C\in\mathbb R^{L_k\times d_c}$，取 $Q=XW_Q$、$K=CW_K$、$V=CW_V$。两边原始维度与长度可不同，只需投影后的 Q/K 维度都为 $d_k$；单头输出为 $L_q\times d_v$，多头经 $W_O$ 回到目标 hidden 维。

Encoder-Decoder 用 decoder 状态查询完整已知源 memory，同时 decoder 自身保持因果 self-attention，因此 cross-attention 无需三角 mask 也能保持自回归性。视觉/音频条件生成可让语言状态查询模态特征，也可让学习的 latent query 汇聚图像；查询方向决定更新哪一组表示。K/V 对应同一 memory 集合，来源不同本身不保证语义对齐。

注意力乘法约需 $O(L_qL_kd)$，投影另计，分数矩阵是 $L_q\times L_k$。Query 数决定输出 token 数，K/V 数决定条件粒度；压缩条件能省成本，也可能丢细节。固定 memory 的各层 K/V 投影可缓存，变化的 Q 仍需计算。应屏蔽条件 padding 与不可用内容，流式任务不能读取尚未到达的信息；权重大小也不保证事实正确或因果解释。

$$
\operatorname{CrossAttn}(X,C)=\operatorname{softmax}\!\left(\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}+M\right)CW_V
$$

### 易错点

- 要求 cross-attention 两边输入长度或原始 embedding 维度完全相同。
- 认为所有 cross-attention 都必须采用 decoder self-attention 的三角 mask。

### 追问

- 固定 encoder memory 的 K/V cache 与 decoder 历史 KV cache 有何区别？
- 可学习 query 压缩视觉 token 时，信息瓶颈主要出现在哪里？

<a id="tfm-017"></a>
## TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？

**L1**

### 答案

离散熵是自信息的期望：事件自信息为 $I(x)=-\log p(x)$，熵为 $H(X)=\mathbb E[I(X)]$。用 $\log_2$ 时单位是 bit，自然对数时是 nat，零概率项按 $0\log0=0$ 的极限处理。固定 $V$ 类时，$0\le H\le\log V$，确定分布取零，均匀分布达到上界；二元熵 $-a\log a-(1-a)\log(1-a)$ 在 $a=1/2$ 最大。熵由完整分布决定，不能只凭最大 token 概率比较。

条件熵为 $H(X\mid Y)=\sum_y p(y)H(p(X\mid y))$，满足链式法则 $H(X,Y)=H(Y)+H(X\mid Y)$。条件信息在平均意义上降低不确定性，但某个特定上下文的条件分布不一定比无条件分布熵更低；序列熵可按此链式展开。

真实数据熵 $H(p)$、模型输出熵 $H(q_\theta)$ 与训练交叉熵 $H(p,q_\theta)$ 要区分。模型可能对错误答案非常自信，交叉熵还含分布失配的 KL，所以降低输出熵不保证拟合正确。日志应明确过滤前后、mask、token 分布与上下文平均范围。连续变量的微分熵可以为负并随尺度变化，不能照搬离散熵的界。

$$
\begin{aligned}H(p)&=-\sum_i p_i\log p_i\\H(X\mid Y)&=\sum_y p(y)H(p(X\mid Y=y))\\H(X,Y)&=H(Y)+H(X\mid Y)\end{aligned}
$$

### 易错点

- 低熵就是准确，高熵就是幻觉，或把数据熵与模型预测熵混同。
- 把平均条件熵下降说成所有特定条件都降低不确定性。

### 追问

- 高置信度错误输出为何可以有低熵？
- 为什么连续均匀分布的微分熵会随区间长度变成负值？

<a id="tfm-018"></a>
## TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？

**L2**

### 答案

矩阵的秩是行空间或列空间的维数，可由消元主元数或非零奇异值数求得。对 $A\in\mathbb R^{m\times n}$，有 $\operatorname{rank}(A)+\dim\ker A=n$。LoRA 的低秩分解满足 $\operatorname{rank}(BA)\le\min(\operatorname{rank}A,\operatorname{rank}B)\le r$，不要求底座 $W_0$ 低秩。

方阵特征值可以为复数，满足 $Av=\lambda v,v\ne0$，手算可先解 $\det(\lambda I-A)=0$，再解特征向量。例 $A=\begin{pmatrix}2&1\\0&3\end{pmatrix}$ 的特征值为 2、3，对应向量可取 $(1,0)^\top$、$(1,1)^\top$，秩为 2。一般矩形矩阵没有这种标准特征值定义；$A^\top A$ 的谱用于求奇异值，不能称为矩形 $A$ 自身的特征值。

大矩阵通常不显式展开特征多项式：数值库可经 Hessenberg 化和 QR 等过程求 Schur 分解，复 Schur 对角给特征值，实 Schur 的 $2\times2$ 块需另解。对称/Hermitian 矩阵可选专用求解器。

若 $A=P\Lambda P^{-1}$ 可对角化，秩等于按重数计的非零特征值个数；一般方阵不能这样数。例如 $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 的特征值全零，秩却为 1，且不可对角化。任意方阵仍满足行列式非零、全部特征值非零与满秩的等价关系。

将奇异值和 $A^\top A$ 的特征值分别按降序排列，令 $p=\min(m,n)$。SVD 写为 $A=U\Sigma V^\top$，非零奇异值的平方构成 $A^\top A$ 的非零谱，其余位置补零；实对称矩阵的奇异值为 $|\lambda_i|$。浮点数值秩按 $\sigma_i>\max(\mathrm{atol},\mathrm{rtol}\,\sigma_{\max})$ 判断，阈值取决于 dtype、规模与噪声，近零不能等同于精确零。

$$
\begin{aligned}\det(\lambda I-A)&=0,\quad(A-\lambda I)v=0,\quad v\ne0\\\operatorname{rank}(A)&=\#\{i:\sigma_i(A)>0\}\\\lambda_i(A^\top A)&=\sigma_i(A)^2,\quad1\le i\le p=\min(m,n)\\\lambda_i(A^\top A)&=0,\quad p<i\le n\\A=P\Lambda P^{-1}&\implies\operatorname{rank}(A)=\#\{i:\lambda_i\ne0\}\\\operatorname{rank}(BA)&\le r\end{aligned}
$$

### 易错点

- 所有矩阵的秩都等于非零特征值个数，或给非方阵直接定义标准特征值。
- 把奇异值与一般特征值逐项等同，或把训练配置 r 当作已学更新的实际秩。

### 追问

- 为什么 $A^\top A$ 理论上可求奇异值，但数值计算可能放大条件数问题？
- 低秩近似、数值秩与 LoRA 的秩上界分别回答什么问题？

<a id="tfm-019"></a>
## TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？

**L1**

### 答案

没有位置线索且可见性同步置换的全连接 self-attention 满足 $\operatorname{Attn}(PX)=P\operatorname{Attn}(X)$：输入置换会同步置换 Q/K/V 和 score 行列，模型难以充分辨别顺序。因果 mask 能提供部分顺序信息，但实际位置方案与训练分布仍要考虑。

绝对位置可用可学习查表，也可用正弦/余弦编码并加到 embedding。原始公式对位置 $m$、频率对 $j$ 使用 $\sin(m/10000^{2j/D})$ 和 $\cos(m/10000^{2j/D})$，不同频率提供不同尺度，固定位置差可通过同频率的二维旋转联系。能计算新位置不等于可靠长度外推。

相对位置可直接改变注意力分数，例如 T5 的可学习距离桶偏置；多个距离共享一个桶参数，范围仍会限制表达。RoPE 对 Q/K 做位置旋转，让点积依赖位置差，通常不旋转 V；ALiBi 在因果分数中加 $-m_h(i-j)$，原方案各头斜率固定。这些方案作用部位不同，不能未经训练转换就替换底座。

比较时明确插入位置、训练长度、位置编号、旋转维度和频率、距离桶、mask 及缓存。扩展上下文还须评估长样本训练、插值/缩放、内核显存与长文任务表现，单纯提高配置上限不保证检索或推理效果。

$$
\begin{aligned}\operatorname{Attn}(PX)&=P\operatorname{Attn}(X)\\\operatorname{PE}(m,2j)&=\sin(m/10000^{2j/D})\\\operatorname{PE}(m,2j+1)&=\cos(m/10000^{2j/D})\\\operatorname{bias}_h(i,j)&=-m_h(i-j),\quad j\le i\end{aligned}
$$

### 易错点

- 绝对编码一定不能表达相对关系，或相对编码天然无限长度无退化。
- 将 ALiBi 写成加到 token embedding 的可训练位置向量，或把 RoPE 写成同样的加法。

### 追问

- 置换等变性与置换不变性分别是什么意思？
- 为什么无参数的正弦编码仍可能在超出训练长度时失败？

<a id="tfm-020"></a>
## TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？

**L1**

### 答案

Dropout 在训练时随机置零激活，inverted dropout 把保留值除以保留概率：$m_i\sim\operatorname{Bernoulli}(1-p)$、$y_i=m_ix_i/(1-p)$，其中 $0\le p<1$，故 $\mathbb E[y_i\mid x_i]=x_i$。它引入随机扰动，减少对共同激活模式的依赖，不会永久删除参数；单个层的期望保持也不意味着经过非线性和归一化的整个网络期望完全不变。

原始 Transformer 在各子层输出上使用 dropout，再与输入相加并做 PostNorm，即 $\operatorname{LN}(x+\operatorname{Dropout}(\operatorname{Sublayer}(x)))$；编码器和解码器的 embedding 与位置编码之和也使用 dropout，base 模型概率为 0.1。现代模型的概率与插入位置要按配置确认。

`nn.Dropout` 在 eval 模式是恒等映射，但 SDPA 的 `dropout_p` 由调用者显式控制，评估时须传 0。原论文 residual dropout 的描述也不能直接推定所有现代实现的 attention probability dropout 或 FFN 内部 dropout。

$$
\begin{aligned}m_i&\sim\operatorname{Bernoulli}(1-p),\quad0\le p<1\\y_i&=\frac{m_ix_i}{1-p}\\\mathbb E[y_i\mid x_i]&=x_i,\quad\operatorname{Var}(y_i\mid x_i)=\frac{px_i^2}{1-p}\\z&=\operatorname{LN}(x+\operatorname{Dropout}(\operatorname{Sublayer}(x)))\\h_0&=\operatorname{Dropout}(\sqrt{d_{\rm model}}\,\operatorname{Embedding}+\operatorname{PE})\end{aligned}
$$

### 易错点

- 把 p 当作保留概率，或在训练和推理阶段重复做 $1/(1-p)$ 缩放。
- 把现代实现的所有 Dropout 位置都归因于原始论文，或把 `nn.Dropout` 默认 0.5 说成 Transformer 标准值。

### 追问

- 推导 inverted dropout 的条件方差，并解释为何 p 越大扰动越强。
- 如果只调用 `model.eval()`，为什么直接调用 SDPA 时仍可能出现随机输出？

<a id="tfm-021"></a>
## TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？

**L2**

### 答案

标准点积注意力权重是输入经 Q/K 投影、点积和 softmax 动态计算的激活，不是独立可训练的注意力表。对 $X\in\mathbb R^{n\times d_{\rm model}}$，$Q=XW_Q$、$K=XW_K$、$V=XW_V$，$A=\operatorname{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M)$，$O=AV$；同一组参数在不同输入上产生不同 $A$。

任务损失经汇聚、softmax 与点积反传。令 $G=\partial L/\partial A$，则 $D_{ij}=\partial L/\partial S_{ij}=A_{ij}(G_{ij}-\sum_kA_{ik}G_{ik})$，随后 $\partial L/\partial Q=DK/\sqrt{d_k}$、$\partial L/\partial K=D^\top Q/\sqrt{d_k}$，更新投影和上游表征。模型还可能学习位置偏置等参数，但训练目标未必直接监督注意力图；固定 mask 下，被屏蔽位置的分数梯度为零。

高权重只说明该层该头的相对加权关系，$V$ 的方向与幅值、其他头、输出投影、残差和后续层都会改变最终影响。热力图可以描述局部计算，因果解释仍需消融、扰动和基线诊断，并注意扰动造成的分布变化。关于注意力解释性的研究存在不同诊断结论，其实验局限于具体模型与任务，不能概括为所有模型的注意力永远不能解释。

$$
\begin{aligned}A&=\operatorname{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M),\quad O=AV\\G&=\frac{\partial L}{\partial A}\\D_{ij}&=A_{ij}\left(G_{ij}-\sum_kA_{ik}G_{ik}\right)\\\frac{\partial L}{\partial Q}&=\frac{DK}{\sqrt{d_k}},\quad\frac{\partial L}{\partial K}=\frac{D^\top Q}{\sqrt{d_k}}\\\frac{\partial L}{\partial W_Q}&=\frac{X^\top DK}{\sqrt{d_k}}\end{aligned}
$$

### 易错点

- 说 softmax 中每个注意力权重都是独立学习参数，忽略 Q/K 与输入。
- 把注意力热力图当作最终预测的充分因果解释，或把特定 RNN 实验结论泛化到所有 LLM。

### 追问

- softmax 的行内归一化为什么会让一个位置的梯度依赖其他位置？
- 两个注意力图差异很大但输出相近，能推出哪些结论，不能推出哪些结论？

<a id="tfm-022"></a>
## TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？

**L1**

### 答案

全局 self-attention 让允许交互的远距离位置在一层内直接交换信息，最长位置通信路径为 $O(1)$；RNN 对相隔 $\Theta(n)$ 步的位置通常需经过 $\Theta(n)$ 次递归传递。短路径有利于信息和梯度传播，但是否学好依赖仍取决于数据、参数、位置编码、优化与任务。

这里的 $O(1)$ 指计算图路径，不是运行时间。序列长 $n$、宽度 $d$ 时，稠密 $QK^\top$ 和 $AV$ 计算为 $O(n^2d)$，显式多头权重存储为 $O(hn^2)$；若 $d_{\rm ff}=\Theta(d)$，投影与 FFN 使单层总计算为 $O(n^2d+nd^2)$。训练时同层 query 可并行计算，但自回归生成下一 token 仍依赖已生成历史。

直接路径也受 mask 限制：因果注意力只能读取历史，局部窗口需跨层传播，固定窗口尺度 $r$ 时远距路径通常随 $n/r$ 增长。有效上下文还受模型窗口与位置方案限制，短路径不保证长上下文准确率。

$$
\begin{aligned}\ell_{\rm global}&=O(1),\quad \ell_{\rm RNN}=O(n)\\C_{\rm attention}&=O(n^2d),\quad M_{\rm attention}=O(hn^2)\\C_{\rm layer}&=O(n^2d+nd^2)\quad(d_{\rm ff}=\Theta(d))\\\ell_{\rm local}&=O(n/r)\end{aligned}
$$

### 易错点

- 把常数通信路径说成常数时间或线性计算复杂度。
- 以全局注意力的结论描述局部窗口模型，或声称短路径自动解决所有长依赖问题。

### 追问

- 局部窗口半径 r 固定时，距离 n 的位置需要多少层才能建立通信？
- 训练时的同层并行与自回归解码的逐 token 依赖有什么区别？

<a id="tfm-023"></a>
## TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？

**L2**

### 答案

原始 Transformer 没有名为 Negative Attention 的标准组件。标准 softmax 在每行至少有一个允许位置时，有限未屏蔽 logits 的权重为正，被 $-\infty$ mask 的位置为零，行和为 1。因此低权重、负 logit 与负输出是不同概念。

Logit 是归一化前的分数，可以为负，如 $\operatorname{softmax}([-2,-1])\approx[0.269,0.731]$；整行加同一常数不改变分布，绝对正负不决定关注程度。即使权重全非负，$O_i=\sum_jA_{ij}V_j$ 的分量仍可为负，如 $0.5(-3)+0.5(1)=-1$，投影和残差也会产生负分量。

减小权重仅减少相对于其他 value 的份额，是否抑制最终答案还取决于 value、投影与目标；加负 bias 也不会直接生成负 softmax 权重。若指某篇论文的 signed attention、差分构造或同名方法，应先核对公式、系数范围与归一化定义，不能因原始 Transformer 未定义该组件就否认其他研究的命名。

$$
\begin{aligned}A_{ij}&=\begin{cases}\displaystyle\frac{\exp S_{ij}}{\sum_{k\in\mathcal J_i}\exp S_{ik}}>0,&j\in\mathcal J_i\\0,&j\notin\mathcal J_i\end{cases}\\\sum_jA_{ij}&=1,\quad\operatorname{softmax}(s+c\mathbf1)=\operatorname{softmax}(s)\\O_i&=\sum_jA_{ij}V_j,\quad0.5(-3)+0.5(1)=-1\end{aligned}
$$

### 易错点

- 把负 logit、接近零的非负权重与负 value/输出混为一谈。
- 为了纠正术语而断言全世界没有任何使用 Negative Attention 名称的论文或方法。

### 追问

- 为什么把所有 logit 同时减去最大值不会改变注意力分布？
- 如果一种方法允许聚合系数为负，它还满足 softmax 概率权重的哪些性质？

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Layer Normalization](https://arxiv.org/pdf/1607.06450)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)
- [torch.nn.LayerNorm — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)
- [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- [Transformers v4.57.1 official Llama implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/llama/modeling_llama.py)
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/pdf/2306.15595)
- [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [MIT 6.441 Chapter 1: Entropy and Divergence](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf)
- [torch.nn.KLDivLoss — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)
- [TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Stanford EE263: Eigenvectors and diagonalization](https://ee263.stanford.edu/lectures/eig.pdf)
- [Stanford EE263 Lecture 15: Symmetric matrices and SVD](https://web.stanford.edu/class/archive/ee/ee263/ee263.1082/lectures/symm.pdf)
- [torch.linalg.matrix_rank — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.matrix_rank.html)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [LAPACK Users' Guide: Eigenvalues, Eigenvectors and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409)
- [PyTorch 2.14 — torch.nn.Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)
- [Jain & Wallace (2019) — Attention is not Explanation](https://aclanthology.org/N19-1357.pdf)
- [Wiegreffe & Pinter (2019) — Attention is not not Explanation](https://aclanthology.org/D19-1002.pdf)
