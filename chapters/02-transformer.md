# Transformer、Attention 与位置编码

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [注意力机制与掩码](#topic-1)
  - [TFM-001 · 缩放点积注意力如何计算，为什么除以 √d_k？](#tfm-001)
  - [TFM-002 · 多头注意力与单头注意力有什么区别？](#tfm-002)
  - [TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？](#tfm-003)
  - [TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？](#tfm-016)
  - [TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？](#tfm-021)
  - [TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？](#tfm-023)
- [位置编码与长上下文](#topic-2)
  - [TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？](#tfm-008)
  - [TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？](#tfm-009)
  - [TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？](#tfm-019)
  - [TFM-024 · 递归、乘性、卷积与复数位置表示怎样提供顺序信息？](#tfm-024)
  - [TFM-025 · Shaw、Transformer-XL、T5、DeBERTa 与 TUPE 如何建模位置信息？](#tfm-025)
- [归一化、FFN 与残差](#topic-3)
  - [TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？](#tfm-005)
  - [TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？](#tfm-006)
  - [TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？](#tfm-007)
  - [TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？](#tfm-010)
  - [TFM-012 · 残差连接为什么能帮助深层模型训练？](#tfm-012)
  - [TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？](#tfm-020)
- [复杂度与实现机制](#topic-4)
  - [TFM-011 · Transformer 一层的时间、空间复杂度如何估算？](#tfm-011)
  - [TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？](#tfm-022)

<a id="topic-1"></a>
## 注意力机制与掩码

<a id="tfm-001"></a>
### TFM-001 · 缩放点积注意力如何计算，为什么除以 √d_k？

**L1** · 小红书

#### 答案

缩放点积注意力先用 $`Q`$ 与 $`K`$ 的匹配分数，在 key 轴做 softmax，再加权汇聚 $`V`$。设 $`Q\in\mathbb R^{L_q\times d_k}`$、$`K\in\mathbb R^{L_k\times d_k}`$、$`V\in\mathbb R^{L_k\times d_v}`$，则 $`S=QK^\top/\sqrt{d_k}+M`$、$`A=\mathrm{softmax}(S)`$，输出 $`AV\in\mathbb R^{L_q\times d_v}`$。Self-attention 常取 $`Q=XW_Q`$、$`K=XW_K`$、$`V=XW_V`$，cross-attention 的 $`Q`$ 与 $`K/V`$ 可来自不同序列。

缩放的尺度分析假设各分量零均值、单位方差，$`q_i`$ 与 $`k_i`$ 独立，且不同维的乘积互不相关：此时 $`\mathrm{Var}(q_i k_i)=1`$，$`\mathrm{Var}(q\cdot k)=d_k`$，除以 $`\sqrt{d_k}`$ 后方差为 1。实际训练后的相关性和方差未必满足这些假设。分数差过大会使分布过于尖锐，softmax 的 Jacobian $`\partial A_i/\partial S_j=A_i(\delta_{ij}-A_j)`$ 部分元素变小；这不表示所有后续梯度必然消失，softmax 与交叉熵合并时 logits 梯度仍为预测概率减目标概率。

允许的连接加 0，禁止连接加 $`-\infty`$，再做 softmax。实现常先减去每行最大有限分数以避免指数溢出；全屏蔽行须单独处理，并检查 dtype、广播和内核的 mask 约定。有限且未屏蔽的 logits 在精确计算下得到严格为正、和为 1 的权重，所以 softmax 本身没有硬稀疏；数值下溢和 mask 产生的零是另外的机制。权重适合描述加权汇聚，但不能直接当作因果重要性。

除了固定除以根号维度，也可对 Q/K 做适当归一化并设置温度、学习 logit scale，或使用加性注意力的评分网络。它们改变评分分布及参数化，并不是把原模型的分母随意删掉的等价替换。改变温度只控制分布尖锐度，不替代所有输入范数与训练稳定性问题；应重新分析方差并配套训练。

```math
\begin{aligned}\mathrm{Attention}(Q,K,V)&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V\\ \mathrm{Var}(q\cdot k)&=d_k\\ \frac{\partial p_i}{\partial s_j}&=p_i(\delta_{ij}-p_j)\end{aligned}
```

#### 易错点

- 把缩放分母写成 $`d_{\rm model}`$ 或 $`d_k`$，或省略独立/方差假设后把 $`\mathrm{Var}(q\cdot k)=d_k`$ 当普遍定律。
- 把 mask 加到已归一化权重后不重归一化，或把 softmax 说成自动只选少量 token。

#### 追问

- 如果 Q/K 做 L2 归一化，点积分布与温度参数该怎样重新分析？
- cross-attention 的 $`L_q\ne L_k`$ 时，各中间矩阵与输出是什么形状？

<a id="tfm-002"></a>
### TFM-002 · 多头注意力与单头注意力有什么区别？

**L1**

#### 答案

多头注意力用多组分别学习的 $`Q/K/V`$ 投影，在不同子空间并行计算注意力，拼接各头输出，再经 $`W_O`$ 混合回模型维度。以输入 `X:[B,T,D]` 为例，第 $`i`$ 头的 $`W_Q^i,W_K^i\in\mathbb R^{D\times d_k}`$、$`W_V^i\in\mathbb R^{D\times d_v}`$，输出 `head_i:[B,T,d_v]`。原始 base 模型取 $`D=512`$、$`h=8`$、$`d_k=d_v=64`$。

工程上把各头堆为 `Q/K:[B,h,T,d_k]`、`V:[B,h,T,d_v]`，分数为 `[B,h,T,T]`。输出从 `[B,h,T,d_v]` 转置并拼接为 `[B,T,h*d_v]`，乘 `W_O:[h*d_v,D]` 得到 `[B,T,D]`；拼接增加的是特征维度。Cross-attention 的权重形状为 `[B,h,L_q,L_k]`，输出长度仍为 $`L_q`$。一次大线性层后 reshape 与逐头投影等价，并不表示各头共用相同权重。

当 $`hd_k=hd_v=D`$ 时，忽略 bias，标准 MHA 的 Q/K/V/O 总权重约为 $`4D^2`$；固定 $`D`$ 增加头数只会缩小每头维度，参数量不随头数成倍增长。不过注意力矩阵元素数随 $`hT^2`$ 变化，显存与内核效率仍需比较。各头可以学习不同匹配模式，也可能冗余，不能预设每头必定负责语法或实体等固定任务。MQA/GQA 改变了 K/V 共享方式，其参数与 KV cache 要另外估算。

```math
\begin{aligned}\mathrm{head}_i&=\mathrm{softmax}\!\left(\frac{(XW_Q^i)(XW_K^i)^\top}{\sqrt{d_k}}+M\right)XW_V^i\\ \mathrm{MHA}(X)&=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O\end{aligned}
```

#### 易错点

- 将单头维度 $`d_k`$ 与总隐藏维度 D 混用，漏写转置/拼接或 $`W_O`$ 导致 shape 不闭合。
- 把实现里的 fused QKV 层误认为不同 head 使用完全相同的投影参数。

#### 追问

- 固定 D 和 h 时，GQA 的参数量与 KV cache 怎样变化？
- 为什么 concatenation 后还需要 $`W_O`$，而不是直接相加各头输出？

<a id="tfm-003"></a>
### TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？

**L1**

#### 答案

Attention mask 在 softmax 前排除不可见的 key：允许连接加 0，禁止连接加 $`-\infty`$。将禁止位置的分数乘 0 无法屏蔽，因为 $`\exp(0)`$ 仍会得到概率；全屏蔽行在朴素 softmax 中可能产生 NaN，须核对后端处理。

因果掩码从 0 编号时允许 $`j\le i`$，包括自身。位置 $`i`$ 的 logits 预测 $`x_{i+1}`$，所以看到 $`x_i`$ 不会泄漏下一词；只加 loss mask 而没有 causal mask，仍会泄漏未来信息。Padding mask 通常排除无效 key，pad query 的输出在后续计算或损失中忽略。Loss mask 只决定哪些目标计损：prompt 的 `labels=-100` 时，合法 prompt 仍应保持可见，否则回答无法读取问题。独立 packed 样本还要要求 query/key 属于同一样本。

已有 $`P`$ 个历史 token，一次处理 $`L`$ 个新 token 时，query 长度为 $`L`$、key 长度为 $`P+L`$，第 $`i`$ 个新 query 可看 $`j\le P+i`$。不能直接套用左上对齐的矩形下三角；须检查内核的因果偏置、真实缓存位置、左 padding 和静态缓存中未填的槽位。

框架的 bool mask 与加性 mask 含义不同：PyTorch 2.14 SDPA 的 `attn_mask=True` 表示允许关注，`MultiheadAttention.key_padding_mask=True` 表示屏蔽。广播应覆盖 `[B,H,L_q,L_k]`，可用小输入验证可见性。普通 batch 可按本批最长序列 padding，长度分桶能减少浪费；位置编号须与真实 token 和模型方案一致。Padding mask 通常不自动省去 pad 的算术计算，变长内核或 packing 还需显式段边界。

```math
\begin{aligned}A&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)\\ M_{ij}&=\begin{cases}0,&\mathrm{allowed}(i,j)\\-\infty,&\text{otherwise}\end{cases}\\ \mathrm{allowed}_{\rm cached}(i,j)&\iff j\le P+i\end{aligned}
```

#### 易错点

- 把 prompt loss mask、EOS 边界或重置 `position_ids` 当成 attention 隔离。
- 不检查矩形 causal 对齐和 bool 含义，导致缓存只能关注最前面的 key 或泄漏未来。

#### 追问

- 如何用扰动另一个 packed 样本的测试证明没有跨样本信息泄漏？
- 只有一个新 query 且所有 key 均为有效历史时，还需要显式三角 mask 吗？

<a id="tfm-016"></a>
### TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？

**L1** · 腾讯

#### 答案

Self-attention 的 $`Q/K/V`$ 来自同一序列；cross-attention 的 $`Q`$ 来自需要更新的目标表示，$`K/V`$ 来自另一组条件或 memory。设 $`X\in\mathbb R^{L_q\times d_x}`$、$`C\in\mathbb R^{L_k\times d_c}`$，取 $`Q=XW_Q`$、$`K=CW_K`$、$`V=CW_V`$。两边原始维度与长度可不同，只需投影后的 Q/K 维度都为 $`d_k`$；单头输出为 $`L_q\times d_v`$，多头经 $`W_O`$ 回到目标 hidden 维。

Encoder-Decoder 用 decoder 状态查询完整已知源 memory，同时 decoder 自身保持因果 self-attention，因此 cross-attention 无需三角 mask 也能保持自回归性。视觉/音频条件生成可让语言状态查询模态特征，也可让学习的 latent query 汇聚图像；查询方向决定更新哪一组表示。K/V 对应同一 memory 集合，来源不同本身不保证语义对齐。

注意力乘法约需 $`O(L_qL_kd)`$，投影另计，分数矩阵是 $`L_q\times L_k`$。Query 数决定输出 token 数，K/V 数决定条件粒度；压缩条件能省成本，也可能丢细节。固定 memory 的各层 K/V 投影可缓存，变化的 Q 仍需计算。应屏蔽条件 padding 与不可用内容，流式任务不能读取尚未到达的信息；权重大小也不保证事实正确或因果解释。

加性attention可写s(q,k)=vᵀtanh(W_qq+W_kk+b)，点积attention则用qᵀk或带双线性投影的qᵀWk。前者在匹配网络里引入非线性，后者更容易批量组织成高效GEMM；两者都需归一化并聚合value，没有脱离维度、kernel和任务的恒定性能排名。传统RNN seq2seq的attention也是cross-attention，用decoder状态查询encoder各时刻状态，避免只依赖单一固定向量。

```math
\mathrm{CrossAttn}(X,C)=\mathrm{softmax}\!\left(\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}+M\right)CW_V
```

#### 易错点

- 要求 cross-attention 两边输入长度或原始 embedding 维度完全相同。
- 认为所有 cross-attention 都必须采用 decoder self-attention 的三角 mask。

#### 追问

- 固定 encoder memory 的 K/V cache 与 decoder 历史 KV cache 有何区别？
- 可学习 query 压缩视觉 token 时，信息瓶颈主要出现在哪里？

<a id="tfm-021"></a>
### TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？

**L2**

#### 答案

标准点积注意力权重是输入经 Q/K 投影、点积和 softmax 动态计算的激活，不是独立可训练的注意力表。对 $`X\in\mathbb R^{n\times d_{\rm model}}`$，$`Q=XW_Q`$、$`K=XW_K`$、$`V=XW_V`$，$`A=\mathrm{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M)`$，$`O=AV`$；同一组参数在不同输入上产生不同 $`A`$。

任务损失经汇聚、softmax 与点积反传。令 $`G=\partial L/\partial A`$，则 $`D_{ij}=\partial L/\partial S_{ij}=A_{ij}(G_{ij}-\sum_kA_{ik}G_{ik})`$，随后 $`\partial L/\partial Q=DK/\sqrt{d_k}`$、$`\partial L/\partial K=D^\top Q/\sqrt{d_k}`$，更新投影和上游表征。模型还可能学习位置偏置等参数，但训练目标未必直接监督注意力图；固定 mask 下，被屏蔽位置的分数梯度为零。

高权重只说明该层该头的相对加权关系，$`V`$ 的方向与幅值、其他头、输出投影、残差和后续层都会改变最终影响。热力图可以描述局部计算，因果解释仍需消融、扰动和基线诊断，并注意扰动造成的分布变化。关于注意力解释性的研究存在不同诊断结论，其实验局限于具体模型与任务，不能概括为所有模型的注意力永远不能解释。

Attention的token混合矩阵随当前输入Q/K动态变化；全连接层的W在一次前向中固定，通常不随样本即时生成。Transformer的FFN虽非线性，却在每个位置独立使用共享参数，不能独自完成跨token汇聚。若Q=K，打分矩阵为Gram矩阵，在不对称mask之前具有对称性，但softmax归一化和mask可使权重不对称，也不必退化成单位阵；不同Q/K投影允许更灵活的有向匹配。

```math
\begin{aligned}A&=\mathrm{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M),\quad O=AV\\G&=\frac{\partial L}{\partial A}\\D_{ij}&=A_{ij}\left(G_{ij}-\sum_kA_{ik}G_{ik}\right)\\\frac{\partial L}{\partial Q}&=\frac{DK}{\sqrt{d_k}},\quad\frac{\partial L}{\partial K}=\frac{D^\top Q}{\sqrt{d_k}}\\\frac{\partial L}{\partial W_Q}&=\frac{X^\top DK}{\sqrt{d_k}}\end{aligned}
```

#### 易错点

- 说 softmax 中每个注意力权重都是独立学习参数，忽略 Q/K 与输入。
- 把注意力热力图当作最终预测的充分因果解释，或把特定 RNN 实验结论泛化到所有 LLM。

#### 追问

- softmax 的行内归一化为什么会让一个位置的梯度依赖其他位置？
- 两个注意力图差异很大但输出相近，能推出哪些结论，不能推出哪些结论？

<a id="tfm-023"></a>
### TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？

**L2**

#### 答案

原始 Transformer 没有名为 Negative Attention 的标准组件。标准 softmax 在每行至少有一个允许位置时，有限未屏蔽 logits 的权重为正，被 $`-\infty`$ mask 的位置为零，行和为 1。因此低权重、负 logit 与负输出是不同概念。

Logit 是归一化前的分数，可以为负，如 $`\mathrm{softmax}([-2,-1])\approx[0.269,0.731]`$；整行加同一常数不改变分布，绝对正负不决定关注程度。即使权重全非负，$`O_i=\sum_jA_{ij}V_j`$ 的分量仍可为负，如 $`0.5(-3)+0.5(1)=-1`$，投影和残差也会产生负分量。

减小权重仅减少相对于其他 value 的份额，是否抑制最终答案还取决于 value、投影与目标；加负 bias 也不会直接生成负 softmax 权重。若指某篇论文的 signed attention、差分构造或同名方法，应先核对公式、系数范围与归一化定义，不能因原始 Transformer 未定义该组件就否认其他研究的命名。

```math
\begin{aligned}A_{ij}&=\begin{cases}\displaystyle\frac{\exp S_{ij}}{\sum_{k\in\mathcal J_i}\exp S_{ik}}\gt 0,&j\in\mathcal J_i\\0,&j\notin\mathcal J_i\end{cases}\\\sum_jA_{ij}&=1,\quad\mathrm{softmax}(s+c\mathbf1)=\mathrm{softmax}(s)\\O_i&=\sum_jA_{ij}V_j,\quad0.5(-3)+0.5(1)=-1\end{aligned}
```

#### 易错点

- 把负 logit、接近零的非负权重与负 value/输出混为一谈。
- 为了纠正术语而断言全世界没有任何使用 Negative Attention 名称的论文或方法。

#### 追问

- 为什么把所有 logit 同时减去最大值不会改变注意力分布？
- 如果一种方法允许聚合系数为负，它还满足 softmax 概率权重的哪些性质？

<a id="topic-2"></a>
## 位置编码与长上下文

<a id="tfm-008"></a>
### TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？

**L2** · 腾讯

#### 答案

RoPE 按位置旋转 $`Q/K`$ 的成对分量，使点积显式出现相对位置差，并保持向量范数；通常不旋转 $`V`$。旋转子空间为偶数维 $`d_r`$，第 $`j`$ 对频率 $`\theta_j=\mathrm{base}^{-2j/d_r}`$，位置 $`m`$ 使用二维旋转 $`R(m\theta_j)`$。不同频率覆盖不同距离尺度，部分模型只旋转 head 的部分维度。

令 $`q'_m=R_mq_m`$、$`k'_n=R_nk_n`$，由 $`R_m^\top R_n=R_{n-m}`$ 可得 $`(q'_m)^\top k'_n=q_m^\top R_{n-m}k_n`$。这是位置依赖呈相对差，内容向量仍受上下文影响。单对分量 $`(a,b)`$ 变为 $`(a\cos\phi-b\sin\phi,a\sin\phi+b\cos\phi)`$；相邻维配对与前半/后半配对可通过置换关联，已有 checkpoint 却不能未经权重转换便替换配对约定。

Transformers v4.57.1 的 `LlamaAttention` 先旋转当前 Q/K，再写入缓存，因此历史已旋转 K 不应再次旋转。新 token 须使用正确逻辑位置的 cos/sin，分块 decode、左 padding 与 packed 样本都要核对 `position_ids`。`cache_position` 管缓存槽位，RoPE 的 `position_ids` 管相位，两者接口含义不同。滑动窗口移除旧缓存不代表逻辑时间归零；若改变频率或位置缩放策略，必须核验历史 K 的一致性，并重算或使用明确支持该变化的缓存实现。

RoPE作用于attention的Q/K，保留单向量范数却改变相位与跨位置内积，因此不能把“范数不变”解释成对注意力语义完全无影响。可以在固定的二维通道对上旋转、其余通道保持恒等映射；分块矩阵仍正交，旋转部分提供相对位置，未旋转部分保留内容内积。但哪些频率参与、旋转比例与配对规则应在训练和推理间一致，不能临时删除低频而断言无损。低频相位在短距离变化较小，在长距离仍可能重要，需用位置敏感任务和不同上下文长度消融验证。

```math
\begin{aligned}\theta_j&=\mathrm{base}^{-2j/d_r}\\R(\phi)&=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}\\q'_m&=R_mq_m,\quad k'_n=R_nk_n\\(q'_m)^\top k'_n&=q_m^\top R_{n-m}k_n\end{aligned}
```

#### 易错点

- 直接把位置向量加到 embedding 来描述 RoPE，或对缓存 K 重复应用旋转。
- 把 RoPE 的相对点积性质当成任意距离单调衰减、任意长度可靠外推的保证。

#### 追问

- 用整段 prefill 与逐 token cache decode 比较 logits 时，哪些位置和数值条件必须一致？
- RoPE 的 base、旋转维度、缩放方案改变后，现有 KV cache 能否复用？
- 只旋转部分维度为何仍保持范数，推理时改变旋转子空间有什么风险？

<a id="tfm-009"></a>
### TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？

**L2**

#### 答案

增大 `max_position_embeddings` 只放宽输入长度配置，不能保证模型处理训练范围外的位置和长距离依赖。若从 $`L`$ 扩到 $`L'`$，线性位置插值用 $`m'=mL/L'`$ 将位置压回训练区间，通常需继续训练。

压缩位置也会降低局部位置分辨率，倍率越大越要检查短距离性能。位置方案调整与长样本训练解决的问题不同；评测应覆盖多种长度和证据位置，并同时报告短文本表现、长检索正确率、KV 占用与 TTFT。

```math
m'=m\frac{L}{L'}
```

#### 易错点

- 把 RoPE 的数学可计算性当作可靠外推保证。
- 只测一个 needle case 就声称长上下文全面有效。

#### 追问

- 位置插值会怎样影响近邻 token？
- 如何排除模型只依赖文档开头或结尾？

<a id="tfm-019"></a>
### TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？

**L1**

#### 答案

没有位置线索且可见性同步置换的全连接 self-attention 满足 $`\mathrm{Attn}(PX)=P\mathrm{Attn}(X)`$：输入置换会同步置换 Q/K/V 和 score 行列，模型难以充分辨别顺序。因果 mask 能提供部分顺序信息，但实际位置方案与训练分布仍要考虑。

绝对位置可用可学习查表，也可用正弦/余弦编码并加到 embedding。原始公式对位置 $`m`$、频率对 $`j`$ 使用 $`\sin(m/10000^{2j/D})`$ 和 $`\cos(m/10000^{2j/D})`$，不同频率提供不同尺度，固定位置差可通过同频率的二维旋转联系。能计算新位置不等于可靠长度外推。

相对位置可直接改变注意力分数，例如 T5 的可学习距离桶偏置；多个距离共享一个桶参数，范围仍会限制表达。RoPE 对 Q/K 做位置旋转，让点积依赖位置差，通常不旋转 V；ALiBi 在因果分数中加 $`-m_h(i-j)`$，原方案各头斜率固定。这些方案作用部位不同，不能未经训练转换就替换底座。

比较时明确插入位置、训练长度、位置编号、旋转维度和频率、距离桶、mask 及缓存。扩展上下文还须评估长样本训练、插值/缩放、内核显存与长文任务表现，单纯提高配置上限不保证检索或推理效果。

```math
\begin{aligned}\mathrm{Attn}(PX)&=P\mathrm{Attn}(X)\\\mathrm{PE}(m,2j)&=\sin(m/10000^{2j/D})\\\mathrm{PE}(m,2j+1)&=\cos(m/10000^{2j/D})\\\mathrm{bias}_h(i,j)&=-m_h(i-j),\quad j\le i\end{aligned}
```

#### 易错点

- 绝对编码一定不能表达相对关系，或相对编码天然无限长度无退化。
- 将 ALiBi 写成加到 token embedding 的可训练位置向量，或把 RoPE 写成同样的加法。

#### 追问

- 置换等变性与置换不变性分别是什么意思？
- 为什么无参数的正弦编码仍可能在超出训练长度时失败？

<a id="tfm-024"></a>
### TFM-024 · 递归、乘性、卷积与复数位置表示怎样提供顺序信息？

**L2**

#### 答案

位置表示可以通过不同结构注入顺序。递归方案由初始状态逐步生成位置向量，FLOATER 进一步用神经 ODE 描述位置上的连续演化；参数可学习，但数值求解或递推有计算成本，能生成更远位置不保证可靠外推。乘性方案用位置相关向量调制 token 表示，例如逐元素相乘，改变的是特征幅度或相位；这只是一类构造，不能据此断言一定优于加法。

卷积核的偏移提供局部相对顺序，零填充边界还可能让 CNN 学到绝对位置线索；它与显式位置表不同，平移等变性也受边界条件影响。Complex Order 用复数的模与相位同时表示词和顺序，可写为 r·exp(i(ωm+θ))，原方法研究复数网络。RoPE 则在实数 Q/K 的二维子空间旋转，也可用复数乘法解释，但不要求整个 Transformer 改成复数模型。比较时检查位置作用于输入、分数还是 Q/K，是否影响并行、缓存和长度泛化。

```math
p_{m+1}=f_\theta(p_m),\qquad \frac{dp(t)}{dt}=h_\theta(p(t),t),\qquad z_{w,j}(m)=r_{w,j}e^{\mathrm{i}(\omega_{w,j}m+\theta_{w,j})}
```

#### 易错点

- 把所有复数位置表示都等同于 RoPE。
- 把卷积零填充带来的绝对位置线索理解为无边界条件下也严格成立。

#### 追问

- 位置相乘为什么不必然得到只依赖位置差的点积？
- ODE 位置表示如何影响预计算与推理缓存？

<a id="tfm-025"></a>
### TFM-025 · Shaw、Transformer-XL、T5、DeBERTa 与 TUPE 如何建模位置信息？

**L2**

#### 答案

Shaw 的相对位置表示把截断的距离向量加入 key，并可加入 value，所以不仅改变权重，也改变被聚合的内容。Transformer-XL 使用内容与相对位置的多个打分项和全局可学习偏置，配合跨片段缓存，避免把缓存 token 的绝对位置编号机械重复；XLNet 使用这一结构，relative shift 是高效对齐相对分数的实现技巧。

T5 在注意力 logit 上加可学习距离桶偏置，近距离区分细，远距离合并，value 通常保持内容表示。T5 的常见实现不在 logits 上再次除以根号维度，尺度由投影初始化等约定处理；不能把标准 attention 的缩放机械套进每个模型实现。DeBERTa 将内容与位置解耦，核心打分包含内容—内容、内容—位置、位置—内容三项；预训练的 enhanced mask decoder 再利用绝对位置，它的“decoder”不表示常规自回归生成解码器。TUPE 则分别计算内容相关性与位置相关性，减少两者的混合交互，并特殊处理 CLS；TUPE-A 使用解耦的绝对位置相关性，TUPE-R 在此基础上再加相对位置偏置，不应把 TUPE 的所有版本都称为相对位置编码。分桶、截断和相对旋转是不同设计，能计算任意距离不意味着对任意长文本都保留分辨率或任务精度。

```math
s_{ij}^{\mathrm{T5}}=q_i^{\top}k_j+b_{\mathrm{bucket}(j-i)},\qquad s_{ij}^{\mathrm{DeBERTa}}\propto (q_i^c)^{\top}k_j^c+(q_i^c)^{\top}k^r_{\delta(i,j)}+(q^r_{\delta(j,i)})^{\top}k_j^c
```

#### 易错点

- 认为相对位置只能修改 logit，不能修改 value。
- 把 DeBERTa enhanced mask decoder 当成 Encoder-Decoder 生成架构。

#### 追问

- 距离桶合并会损失哪些长距离信息？
- Transformer-XL 缓存与普通 decoder KV cache 的训练语义有什么区别？

<a id="topic-3"></a>
## 归一化、FFN 与残差

<a id="tfm-005"></a>
### TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？

**L1**

#### 答案

LayerNorm 对每个 token 的 hidden 维计算均值与方差，再用 $`\epsilon`$ 和可学习的 $`\gamma,\beta`$ 做标准化与仿射变换。对 `X:[B,T,D]`、`normalized_shape=D`，均值和方差形状为 `[B,T,1]`，$`\gamma,\beta`$ 为 `[D]`。方差使用分母 $`D`$ 的总体形式；$`\epsilon`$ 加在方差内，避免分母为零并影响小方差时的尺度。

归一化只覆盖指定的特征轴；若 `normalized_shape` 包含多个轴，统计范围也随之扩大。仿射或 bias 可由具体架构关闭。同一层各 token 共享这组参数，但不同层通常分别学习。LN 在训练与推理时都依据当前 token 重算统计，不依赖 batch size 或运行均值，适合变长、在线和自回归解码。

BatchNorm 通常按通道聚合 batch 与其他指定轴，训练时用批统计，常规推理用估计的总体或运行统计。小 batch、分布变化、序列位置差异和 padding 混入会影响估计；变长序列也能使用 BN，但须明确统计轴和有效位置处理。LN 有助于稳定激活与优化，训练是否稳定还依赖残差、Pre/PostNorm、初始化和学习率，不能只由 LN 推出梯度永不爆炸或消失。

以NCHW图像张量为例，BN对每个channel跨N/H/W统计，IN对每个样本每个channel跨H/W统计，GN对每个样本在一个channel组内跨组channel/H/W统计；GN不依赖batch大小。Transformer常见LN是每个token在hidden轴统计，而不是把整个序列当一条归一化轴。所谓“选择哪种Norm”先指定张量布局和normalized_shape，不能只背名称。

```math
\begin{aligned}\mu_{bt}&=\frac1D\sum_{d=1}^D x_{btd}\\\sigma^2_{bt}&=\frac1D\sum_{d=1}^D(x_{btd}-\mu_{bt})^2\\\mathrm{LN}(x)_{btd}&=\gamma_d\frac{x_{btd}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}+\beta_d\end{aligned}
```

#### 易错点

- 对 `[B,T,D]` 在 T 轴归一化，或把每 token 均值误写成跨 batch 均值。
- 省略 $`\epsilon/\gamma/\beta`$、混用 D 与 $`D-1`$，或宣称 BN 一定不能处理任何变长输入。

#### 追问

- `LN(D)` 与 `LN((T,D))` 的输出相同吗，为什么？
- RMSNorm 去掉了哪些计算，是否仍把每个 token 的均值归零？

<a id="tfm-006"></a>
### TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？

**L2**

#### 答案

RMSNorm 按输入的均方根缩放，省去 LayerNorm 的减均值步骤，通常在根号内加入 $`\epsilon`$，再乘可学习的逐通道权重 $`\gamma`$。LN 使用中心化方差，RMSNorm 使用未经中心化的二阶矩。

两者都有对整体尺度的归一化作用，但平移性质不同：输入统一加常数时，LN 的归一化部分基本不变，RMSNorm 通常会变化。RMSNorm 的计算较简单，归约常用更高精度以减少溢出和舍入误差；实际速度与质量仍取决于模型、dtype 和融合内核，不能直接套用论文中的收益比例。

pRMSNorm使用部分特征估计均方根以降低归一化开销，核心仍不做均值中心化；是否采用取决于采样误差、kernel和模型验证，不能把标准RMSNorm的公式直接标作pRMSNorm。

```math
\mathrm{RMSNorm}(x)=\frac{\gamma\odot x}{\sqrt{D^{-1}\sum_{i=1}^D x_i^2+\epsilon}}
```

#### 易错点

- 把 RMSNorm 说成标准化到零均值。
- 说去均值在任何模型中都完全无用。

#### 追问

- 为什么 RMSNorm 常在 FP32 中计算归约？
- RMSNorm 与 L2 normalization 有何尺度差别？

<a id="tfm-007"></a>
### TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？

**L2**

#### 答案

Pre-Norm 的结构是 $`y=x+F(\mathrm{Norm}(x))`$，Post-Norm 是 $`y=\mathrm{Norm}(x+F(x))`$。Pre-Norm 的残差主路径保留恒等连接，给梯度提供直接通道，深层训练通常更稳定，模型末端一般还会添加最终归一化。

Post-Norm 的初始化分析显示部分层梯度可能较大，可解释其对 warmup 的敏感性，但该结论依赖分析条件。实际是否需要 warmup，还受深度、初始化与优化器影响；比较两者应匹配训练预算并分别调参，不能只用同一学习率下的 loss 判定优劣。

DeepNorm在Post-LN风格中引入与深度相关的残差放大及初始化缩放，例如LN(αx+F(x))，并配套特定网络深度/架构的参数初始化规则。它不是把Pre-Norm挪回Post-Norm即可获得相同稳定性；α/β依赖encoder/decoder设置，公式需按论文配方核对。

```math
\begin{aligned}y_{\rm pre}&=x+F(\mathrm{Norm}(x))\\y_{\rm post}&=\mathrm{Norm}(x+F(x))\end{aligned}
```

#### 易错点

- 说 Pre-Norm 必然不需要 warmup。
- 把更稳定训练直接等同于所有任务更好的最终效果。

#### 追问

- 为什么很多 Pre-Norm 模型还有 final norm？
- 残差随深度增长如何控制？

<a id="tfm-010"></a>
### TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？

**L2**

#### 答案

FFN 对每个 token 独立做通道变换：原始 Transformer 用升维线性层、ReLU 和降维线性层。按行向量约定，$`\mathrm{FFN}(x)=\mathrm{ReLU}(xW_1+b_1)W_2+b_2`$，`x:[D]`、`W_1:[D,m]`、`W_2:[m,D]`；对 `X:[B,T,D]` 沿最后一轴执行，输出形状不变。原始 base 模型取 $`D=512,m=2048`$。

Position-wise 表示同一层各 token 使用相同参数，不直接混合不同 token；不同层通常各有参数。跨 token 信息由 attention 等模块汇聚，FFN 增加各位置的非线性组合与容量。若去掉非线性，两层仿射可合并为 $`x(W_1W_2)+(b_1W_2+b_2)`$，单纯增加中间宽度没有同样的表达收益。

现代 SwiGLU 常写为 $`(\mathrm{SiLU}(xW_g)\odot xW_u)W_d`$，包含 gate/up/down 三个矩阵，bias 由模型决定；它增加逐元素门控，并非只替换激活名称。忽略 bias，原 FFN 参数约为 $`2Dm`$，SwiGLU 约为 $`3Dm`$；若匹配原来 $`m=4D`$ 的 $`8D^2`$ 预算，SwiGLU 中间宽度约取 $`8D/3`$，再按硬件对齐。宽度、激活与融合内核都影响容量、激活显存和吞吐。

```math
\begin{aligned}\mathrm{FFN}(x)&=\mathrm{ReLU}(xW_1+b_1)W_2+b_2\\\mathrm{SwiGLU}(x)&=(\mathrm{SiLU}(xW_g)\odot xW_u)W_d\\N_{\rm FFN}&\approx2Dm,\quad N_{\rm SwiGLU}\approx3Dm\end{aligned}
```

#### 易错点

- 漏写第二次线性变换或 bias，并将原论文激活写成所有现代 LLM 都使用的配置。
- 说 position-wise 参数不共享，或把 FFN 的同位置变换误认为能直接读取别的位置。

#### 追问

- 若无 bias 和非线性，两层 FFN 的等效矩阵秩受什么限制？
- 为什么同参数预算下 SwiGLU 的中间宽度常小于 $`4D`$？

<a id="tfm-012"></a>
### TFM-012 · 残差连接为什么能帮助深层模型训练？

**L1** · 小红书

#### 答案

残差连接 $`y=x+F(x)`$ 让子层学习输入表示的修正，并为信号和梯度提供直接路径。两支形状须一致；若改用投影支路 $`y=P(x)+F(x)`$，直接支路的 Jacobian 为 $`J_P`$。残差相加不会像 concat 一样扩大特征维，$`F(x)\approx0`$ 时可接近恒等传递。

简单块的 Jacobian 为 $`I+J_F`$，反传为 $`\partial L/\partial x=(I+J_F)^\top\partial L/\partial y`$。多层仍需连乘各块导数，若 $`J_F\approx-I`$ 还会相互抵消，所以恒等项并不保证梯度永远稳定。

原始 Transformer 的 PostNorm 是 $`\mathrm{LN}(x+\mathrm{Dropout}(F(x)))`$，完整导数还含 LN 和 dropout；attention、FFN 各有残差与归一化。常见 PreNorm 为 $`x+\mathrm{Dropout}(F(\mathrm{LN}(x)))`$，直接支路不经过该块 LN，通常有助于深层优化。随深度增加仍应检查残差尺度、激活方差、溢出和梯度范数，并结合初始化、学习率与最终 norm 判断稳定性。

原始 encoder 每层有 2 条围绕子层的残差，分别对应 self-attention 与 FFN；原始 encoder-decoder 中的 decoder 每层有 3 条，分别对应 masked self-attention、cross-attention 与 FFN。GPT 类 decoder-only 通常不含 cross-attention，所以常见块为 2 条。说“Transformer 有几个残差”要先指定每层还是全网以及具体架构，不能把 decoder-only 和原始 decoder 混算。

```math
\begin{aligned}y&=x+F(x),\quad J_y=I+J_F\\y_{\rm post}&=\mathrm{LN}(x+\mathrm{Dropout}(F(x)))\\y_{\rm pre}&=x+\mathrm{Dropout}(F(\mathrm{LN}(x)))\end{aligned}
```

#### 易错点

- 声称残差能让任何深度和初始化的网络梯度恒为 1。
- 混淆 PostNorm/PreNorm，或把投影 shortcut 当成严格恒等支路。

#### 追问

- PostNorm 为什么不能直接套用 $`J=I+J_F`$ 作为完整块导数？
- 将残差支路乘以 $`\alpha`$ 后，Jacobian 与激活尺度如何变化？

<a id="tfm-020"></a>
### TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？

**L1**

#### 答案

Dropout 在训练时随机置零激活，inverted dropout 把保留值除以保留概率：$`m_i\sim\mathrm{Bernoulli}(1-p)`$、$`y_i=m_ix_i/(1-p)`$，其中 $`0\le p<1`$，故 $`\mathbb E[y_i\mid x_i]=x_i`$。它引入随机扰动，减少对共同激活模式的依赖，不会永久删除参数；单个层的期望保持也不意味着经过非线性和归一化的整个网络期望完全不变。

原始 Transformer 在各子层输出上使用 dropout，再与输入相加并做 PostNorm，即 $`\mathrm{LN}(x+\mathrm{Dropout}(\mathrm{Sublayer}(x)))`$；编码器和解码器的 embedding 与位置编码之和也使用 dropout，base 模型概率为 0.1。现代模型的概率与插入位置要按配置确认。

`nn.Dropout` 在 eval 模式是恒等映射，但 SDPA 的 `dropout_p` 由调用者显式控制，评估时须传 0。原论文 residual dropout 的描述也不能直接推定所有现代实现的 attention probability dropout 或 FFN 内部 dropout。

```math
\begin{aligned}m_i&\sim\mathrm{Bernoulli}(1-p),\quad0\le p\lt 1\\y_i&=\frac{m_ix_i}{1-p}\\\mathbb E[y_i\mid x_i]&=x_i,\quad\mathrm{Var}(y_i\mid x_i)=\frac{px_i^2}{1-p}\\z&=\mathrm{LN}(x+\mathrm{Dropout}(\mathrm{Sublayer}(x)))\\h_0&=\mathrm{Dropout}(\sqrt{d_{\rm model}}\,\mathrm{Embedding}+\mathrm{PE})\end{aligned}
```

#### 易错点

- 把 p 当作保留概率，或在训练和推理阶段重复做 $`1/(1-p)`$ 缩放。
- 把现代实现的所有 Dropout 位置都归因于原始论文，或把 `nn.Dropout` 默认 0.5 说成 Transformer 标准值。

#### 追问

- 推导 inverted dropout 的条件方差，并解释为何 p 越大扰动越强。
- 如果只调用 `model.eval()`，为什么直接调用 SDPA 时仍可能出现随机输出？

<a id="topic-4"></a>
## 复杂度与实现机制

<a id="tfm-011"></a>
### TFM-011 · Transformer 一层的时间、空间复杂度如何估算？

**L2** · 腾讯

#### 答案

设序列长度为 $`T`$、隐藏维度为 $`d`$，一层 Transformer 的主要计算是 $`O(Td^2+T^2d)`$，同时包含线性投影/FFN 与注意力的两两交互。QKV/O 投影约需 $`4Td^2`$ 次乘加，FFN 按中间维度另算；$`QK^\top`$ 与 $`AV`$ 合计约 $`2T^2d`$ 次乘加。按 FLOPs 统计时，一次乘加通常算 2 FLOPs。

朴素多头注意力分数约保存 $`BhT^2`$ 个元素，实际峰值还包括其他激活和工作区。FlashAttention 可避免完整中间矩阵的显存存储，但精确稠密注意力的算术量仍为二次。短序列、大隐藏维度时线性层可能占主导，长序列时二次项更明显。

```math
\begin{aligned}C_{\rm layer}&=O(Td^2+T^2d)\\M_{\rm attention}&=O(BhT^2)\end{aligned}
```

#### 易错点

- 只报 $`O(T^2)`$ 而不说明 d、投影与 FFN。
- 把 FlashAttention 说成改变了精确注意力的二次算术复杂度。

#### 追问

- 使用 KV cache 后单步注意力复杂度如何变化？
- 为什么 FLOPs 更少的实现可能实际更慢？

<a id="tfm-022"></a>
### TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？

**L1** · 腾讯

#### 答案

全局 self-attention 让允许交互的远距离位置在一层内直接交换信息，最长位置通信路径为 $`O(1)`$；RNN 对相隔 $`\Theta(n)`$ 步的位置通常需经过 $`\Theta(n)`$ 次递归传递。短路径有利于信息和梯度传播，但是否学好依赖仍取决于数据、参数、位置编码、优化与任务。

这里的 $`O(1)`$ 指计算图路径，不是运行时间。序列长 $`n`$、宽度 $`d`$ 时，稠密 $`QK^\top`$ 和 $`AV`$ 计算为 $`O(n^2d)`$，显式多头权重存储为 $`O(hn^2)`$；若 $`d_{\rm ff}=\Theta(d)`$，投影与 FFN 使单层总计算为 $`O(n^2d+nd^2)`$。训练时同层 query 可并行计算，但自回归生成下一 token 仍依赖已生成历史。

直接路径也受 mask 限制：因果注意力只能读取历史，局部窗口需跨层传播，固定窗口尺度 $`r`$ 时远距路径通常随 $`n/r`$ 增长。有效上下文还受模型窗口与位置方案限制，短路径不保证长上下文准确率。

```math
\begin{aligned}\ell_{\rm global}&=O(1),\quad \ell_{\rm RNN}=O(n)\\C_{\rm attention}&=O(n^2d),\quad M_{\rm attention}=O(hn^2)\\C_{\rm layer}&=O(n^2d+nd^2)\quad(d_{\rm ff}=\Theta(d))\\\ell_{\rm local}&=O(n/r)\end{aligned}
```

#### 易错点

- 把常数通信路径说成常数时间或线性计算复杂度。
- 以全局注意力的结论描述局部窗口模型，或声称短路径自动解决所有长依赖问题。

#### 追问

- 局部窗口半径 r 固定时，距离 n 的位置需要多少层才能建立通信？
- 训练时的同层并行与自回归解码的逐 token 依赖有什么区别？

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)
- [Layer Normalization](https://arxiv.org/pdf/1607.06450)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)
- [torch.nn.LayerNorm — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)
- [Group Normalization](https://arxiv.org/abs/1803.08494)
- [Instance Normalization](https://arxiv.org/abs/1607.08022)
- [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- [Transformers v4.57.1 official Llama implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/llama/modeling_llama.py)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/pdf/2306.15595)
- [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409)
- [PyTorch 2.14 — torch.nn.Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)
- [Jain & Wallace (2019) — Attention is not Explanation](https://aclanthology.org/N19-1357.pdf)
- [Wiegreffe & Pinter (2019) — Attention is not not Explanation](https://aclanthology.org/D19-1002.pdf)
- [Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/abs/2003.09229)
- [Encoding word order in complex embeddings](https://arxiv.org/abs/1912.12333)
- [How Much Position Information Do Convolutional Neural Networks Encode?](https://arxiv.org/abs/2001.08248)
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/pdf/2006.03654)
- [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595)
- [XLNet: Generalized Autoregressive Pretraining](https://arxiv.org/abs/1906.08237)
- [T5 original Mesh TensorFlow attention implementation](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/attention.py)
