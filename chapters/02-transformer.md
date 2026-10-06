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

**L1** · 小红书 / 阿里巴巴

#### 答案

缩放点积注意力先比较查询 Q 与候选键 K，再把匹配程度转成权重，用这些权重汇总内容 V。可以把它理解为查资料：Q 是当前问题，K 是各条资料的检索特征，V 是真正要取回的内容。若有 L_q 个查询、L_k 个候选，Q 和 K 的特征宽度都为 d_k，匹配矩阵就是 L_q×L_k；V 宽度为 d_v，输出就是 L_q×d_v。softmax 沿候选 key 轴做，每个查询分别分配权重。

除以 √d_k 是为了控制分数尺度。假设各分量独立、零均值、单位方差，d_k 个乘积相加后，点积方差为 d_k；除以平方根后，方差回到 1 左右。否则维度越大，分数容易相差很大，让 softmax 过于尖锐、部分导数变小。这个推导有独立性和方差前提，不能当作训练后所有 Q/K 的普遍性质。公式导数中的 p 是概率、s 是分数，δ 在两个下标相同时为 1，否则为 0；softmax 局部导数变小不等于所有梯度消失，例如与交叉熵组合后的 logits 梯度仍可写成预测概率减目标。

M 是可见性掩码：允许位置加 0，禁止位置在 softmax 前加负无穷。稳定实现还会减去每行最大有限分数，避免指数溢出；全屏蔽行要单独处理。有限且未屏蔽的分数在精确计算下都有正权重，所以 softmax 本身不是硬选择少数位置。也可以通过 Q/K 归一化、温度或加性评分改变尺度，但那是新的参数化，需要配套训练，不能随意删掉原模型的缩放项。

```math
\begin{aligned}\mathrm{Attention}(Q,K,V)&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V\\ \mathrm{Var}(q\cdot k)&=d_k\\ \frac{\partial p_i}{\partial s_j}&=p_i(\delta_{ij}-p_j)\end{aligned}
```

#### 易错点

- 缩放用 √d_k，不是 d_k 或模型宽度；方差推导还需要独立性等前提。
- 掩码应在 softmax 前加入；softmax 本身也不会硬筛选少量 token。

#### 追问

- Q/K 做 L2 归一化后，如何重新分析点积分布与温度？
- 交叉注意力查询和条件长度不同时，分数与输出各是什么形状？

<a id="tfm-002"></a>
### TFM-002 · 多头注意力与单头注意力有什么区别？

**L1**

#### 答案

多头注意力让同一段输入经过多组不同的查询、键和值投影，在不同特征子空间里分别寻找关系，再把结果合起来。单头只有一套加权汇总方式，多头能同时提供多种匹配模式；但不能预先规定某头一定负责语法、另一头一定负责实体，训练后也可能出现冗余。

输入若是 B×T×D，B 是批大小、T 是长度、D 是模型宽度，h 个头通常各用 D/h 的宽度。实现先做一次大投影，再拆成 B×h×T×d_k，每个头计算 T×T 的匹配和加权结果。随后把头维转回特征维，拼成 B×T×h d_v，再乘输出矩阵 W_O 回到 B×T×D。例如 D=512、h=8 时，每头常用 64 维；拼接增加的是特征宽度，不是序列长度。一次大投影只是在计算上合并，各头仍有各自参数。

固定 D 且 h d_k=h d_v=D 时，Q、K、V、O 四张矩阵忽略 bias 共约 4D² 参数，增加头数并不会把参数量成倍增加，因为每头变窄了。显式注意力矩阵却有约 BhT² 个元素，头数仍影响显存和内核效率。交叉注意力分数是 L_q×L_k，输出长度由查询数决定。MQA、GQA 会共享部分 K/V，其参数和缓存需要另外计算，不能照搬标准多头结论。

```math
\begin{aligned}\mathrm{head}_i&=\mathrm{softmax}\!\left(\frac{(XW_Q^i)(XW_K^i)^\top}{\sqrt{d_k}}+M\right)XW_V^i\\ \mathrm{MHA}(X)&=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O\end{aligned}
```

#### 易错点

- 每头宽度与总宽度不同，拆头、转置、拼接和输出投影要让形状闭合。
- 融合 QKV 投影只是计算安排，不代表不同头使用相同参数。

#### 追问

- 固定模型宽度和查询头数，减少 GQA 的 KV 头怎样影响参数与缓存？
- 拼接后为什么还需要输出投影，直接相加各头会改变什么？

<a id="tfm-003"></a>
### TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？

**L1**

#### 答案

causal mask 控制能否看未来，padding mask 排除填充内容，loss mask 决定哪些目标计入损失，三者作用不同。注意力掩码要在 softmax 前把禁止位置加成负无穷，直接乘零仍会留下 exp(0) 的概率；全行都被屏蔽时，朴素计算可能产生 NaN，需要明确后端处理。

因果注意力允许位置 i 读取 j≤i，包括自身，因为位置 i 的输出通常预测下一 token。Padding mask 通常屏蔽无效 key，填充 query 的输出再由后续计算或损失忽略。训练回答时把 prompt 标签设成 −100，只是不计算 prompt 的目标损失，回答仍应能读取问题；仅有 loss mask 而没有因果 mask 会泄漏未来。把多个独立样本拼进同一序列时，还要阻止跨样本注意力，EOS 或重置位置编号都不能自动完成隔离。

缓存已有 P 个 token，一次处理 L 个新 token，第 i 个新查询应能看 j≤P+i 的 key。例如已有 5 个历史 token，新块第一个查询也应能读取这 5 个历史位置。此时 L×(P+L) 的掩码不能机械套左上角三角阵；还要检查左 padding、真实缓存位置和未填槽位。PyTorch SDPA 的布尔 True 表示允许，MultiheadAttention 的 key_padding_mask True 却表示屏蔽，必须按接口核对。Padding 通常只改变可见性，并不自动省去计算，可用长度分桶、变长内核或带段边界的 packing 减少浪费。

```math
\begin{aligned}A&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)\\ M_{ij}&=\begin{cases}0,&\mathrm{allowed}(i,j)\\-\infty,&\text{otherwise}\end{cases}\\ \mathrm{allowed}_{\rm cached}(i,j)&\iff j\le P+i\end{aligned}
```

#### 易错点

- 只屏蔽 prompt 损失、加入 EOS 或重置位置编号，都不能保证独立样本的注意力隔离。
- 布尔语义与矩形因果对齐要查接口，否则可能误屏蔽历史或泄露未来。

#### 追问

- 怎样通过改动另一个 packed 样本，验证当前样本输出不受影响？
- 单个新查询的所有 key 都是有效历史时，是否还需要显式三角掩码？

<a id="tfm-016"></a>
### TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？

**L1** · 腾讯 / 深势科技

#### 答案

self-attention 在同一组表示内部交换信息，cross-attention 让一组表示查询另一组条件信息。前者的 Q/K/V 来自同一序列；后者的 Q 来自需要更新的目标序列 X，K/V 来自条件或记忆 C。Q 决定“谁在提问”，K 决定“怎样匹配”，V 提供取回的内容，因此查询方向决定更新哪一组表示。

若 X 有 L_q 个位置、C 有 L_k 个位置，两边原始宽度可以不同，投影后的 Q/K 宽度只需同为 d_k，单头输出就是 L_q×d_v。比如 20 个语言 token 查询 100 个图像特征，分数矩阵是 20×100，输出仍有 20 个语言位置。编码器—解码器生成中，目标 self-attention 保持因果，cross-attention 可以读取完整已知源文本，因此不必再套目标的三角掩码；源 padding 和流式尚未到达的信息仍要屏蔽。

匹配与汇总约为 O(L_qL_kd)，投影另算。固定条件的各层 K/V 可以缓存，变化的查询仍要计算；压缩条件能降低成本，也会失去细节。加性注意力通过小网络计算匹配，点积注意力容易组织成高效矩阵乘法，二者都归一化后汇总 V。来源不同本身并不保证跨模态对齐，权重更大也不能证明内容正确；这些仍取决于表示、训练和后续计算。

```math
\mathrm{CrossAttn}(X,C)=\mathrm{softmax}\!\left(\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}+M\right)CW_V
```

#### 易错点

- 两组输入可以不同长度和原始宽度，只需投影后查询与键维度匹配。
- 交叉注意力是否屏蔽取决于条件可用性，不必套用目标三角掩码。

#### 追问

- 固定源条件 K/V 与不断增长的目标历史缓存，复用方式有何不同？
- 用少量可学习查询压缩图像特征，信息瓶颈出现在哪里？

<a id="tfm-021"></a>
### TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？

**L2**

#### 答案

注意力权重是模型根据当前输入动态算出的结果，训练真正更新的是 Q/K/V 投影等参数，并不是为每对 token 存一张独立的可训练权重表。输入 X 先得到 Q、K、V，Q 与 K 的点积经过行内 softmax 得到 A，再用 A 汇总 V。同一套参数遇到不同句子，就会产生不同注意力图。

训练损失从输出汇总传到 A，再通过 softmax 和点积传回 Q/K 及上游表示。公式中的 G 是损失对 A 的梯度，D 是对归一化前分数的梯度；某一候选得分的变化会重新分配整行概率，因此它的梯度也依赖同一行其他候选。固定掩码禁止的位置没有分数梯度。若 Q=K，未掩码的分数矩阵会对称，但每行归一化和不对称掩码仍可能让权重不对称，也不必变成单位矩阵。

高权重只说明该层该头分给某个 V 的份额大，最终影响还取决于 V 的方向和幅度、其他头、输出投影、残差与后续层。比如某位置权重很大但 V 接近零，实际贡献仍可能很小。热力图适合描述局部计算，若要解释最终决策，应配合消融、扰动和基线，并留意扰动改变数据分布的问题。不能简单断言权重等于因果重要性，也不能把某个模型的实验推广成所有注意力永远无法解释。

```math
\begin{aligned}A&=\mathrm{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M),\quad O=AV\\G&=\frac{\partial L}{\partial A}\\D_{ij}&=A_{ij}\left(G_{ij}-\sum_kA_{ik}G_{ik}\right)\\\frac{\partial L}{\partial Q}&=\frac{DK}{\sqrt{d_k}},\quad\frac{\partial L}{\partial K}=\frac{D^\top Q}{\sqrt{d_k}}\\\frac{\partial L}{\partial W_Q}&=\frac{X^\top DK}{\sqrt{d_k}}\end{aligned}
```

#### 易错点

- 注意力权重是输入与投影动态计算的激活，不是独立参数表。
- 热力图不足以证明最终因果作用，特定实验结论也不能推广到全部模型。

#### 追问

- 行内 softmax 归一化，为什么让某分数的梯度依赖其他候选？
- 两张注意力图很不同但输出接近，说明了什么，又不能说明什么？

<a id="tfm-023"></a>
### TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？

**L2**

#### 答案

“Negative Attention”不是原始 Transformer 的标准组件，遇到这个说法要先确定它指某篇论文的方法，还是把低权重、负分数和负输出混在了一起。标准 softmax 注意力在至少有一个可见位置时，有限未屏蔽分数产生正权重，屏蔽位置权重为零，每行权重和为 1，并不会直接产生负概率。

负 logit 只是归一化前的分数为负。例如 [−2,−1] 的 softmax 约为 [0.269,0.731]，两个权重都为正；整行加同一个常数也不改变分布，所以分数绝对正负没有直接意义。减去最大值能提高数值稳定性，正是利用了这个性质。权重小表示相对分配得少，不自动意味着抑制了最终某个答案。

负输出则完全可能，因为 V 的分量可以为负。例如两个值是 −3 和 1，权重各为 0.5，汇总就是 −1；输出投影和残差还会进一步改变结果。若某方法真的允许负聚合系数，例如带符号或差分构造，就要重新核对其公式、系数范围和归一化性质，而不能继续把它全当作概率分布。回答这类题的重点是先把三个计算阶段讲清楚，再针对具体方法讨论。

```math
\begin{aligned}A_{ij}&=\begin{cases}\displaystyle\frac{\exp S_{ij}}{\sum_{k\in\mathcal J_i}\exp S_{ik}}\gt 0,&j\in\mathcal J_i\\0,&j\notin\mathcal J_i\end{cases}\\\sum_jA_{ij}&=1,\quad\mathrm{softmax}(s+c\mathbf1)=\mathrm{softmax}(s)\\O_i&=\sum_jA_{ij}V_j,\quad0.5(-3)+0.5(1)=-1\end{aligned}
```

#### 易错点

- 负 logit、很小的正权重和负输出属于不同计算阶段。
- 原始 Transformer 没定义某名称，不代表没有其他论文使用它。

#### 追问

- 所有分数减去最大值，为什么不会改变注意力权重？
- 聚合系数允许为负后，哪些 softmax 概率性质不再成立？

<a id="topic-2"></a>
## 位置编码与长上下文

<a id="tfm-008"></a>
### TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？

**L2** · 腾讯

#### 答案

RoPE 把位置变成查询 Q 和键 K 的旋转角度，使它们的点积能表达相对距离；它通常不把位置向量加到输入，也不旋转 V。做法是在偶数维的旋转子空间里把通道两两配对，每对用不同频率 θ，位置 m 对应角度 mθ。二维向量 [a,b] 旋转后变成 [a cosφ−b sinφ, a sinφ+b cosφ]，长度保持不变。

关键是查询在 m 处旋转、键在 n 处旋转，点积中的两次旋转可以合成 n−m 的相对旋转。公式中的 R_m 是所有通道对旋转组成的块矩阵，d_r 是旋转维数，base 决定频率范围。例如两个词整体往后移动相同位置，相对旋转不变；但内容向量本身仍受上下文影响，不能据此声称整个模型完全平移不变，或距离越大每个点积都单调衰减。

使用 KV cache 时，要先按当前逻辑位置旋转新 Q/K，再把已经旋转的 K 写入缓存，历史 K 不应再旋转一次。新 token 的位置不能每步重置为零；缓存槽位与 RoPE 相位位置也不是同一个概念。左 padding、分块输入和滑动窗口都要核对真实逻辑位置，删掉旧缓存不自动让时间归零。

部分模型只旋转一部分维度，剩余部分做恒等变换，整体仍保持范数，但旋转比例、频率和配对规则必须与训练一致。相邻配对和前后半配对在数学上可通过置换关联，已有权重却不能直接换约定。改变 base、位置缩放或旋转子空间可能使旧缓存不一致，需要重算或明确支持的实现；可靠长上下文还需要训练与任务评价。

```math
\begin{aligned}\theta_j&=\mathrm{base}^{-2j/d_r}\\R(\phi)&=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}\\q'_m&=R_mq_m,\quad k'_n=R_nk_n\\(q'_m)^\top k'_n&=q_m^\top R_{n-m}k_n\end{aligned}
```

#### 易错点

- RoPE 旋转 Q/K，不是把位置向量加到 embedding；缓存 K 也不能重复旋转。
- 相对位置点积不保证所有距离单调衰减或任意长度可靠外推。

#### 追问

- 比较 prefill 和逐 token 解码 logits 时，权重、位置与数值条件要怎样保持一致？
- 改变 base、旋转维度或位置缩放后，历史缓存还能直接复用吗？
- 只旋转部分通道为什么仍保持范数，推理时临时改变这部分有什么风险？

<a id="tfm-009"></a>
### TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？

**L2**

#### 答案

只把 max_position_embeddings 改大，通常只是让程序允许更长输入，并没有教会模型理解新的位置范围和更长的依赖。可学习位置表甚至可能没有对应行；RoPE 虽然能计算新位置，超出训练范围的相位和注意力模式也可能失效。能运行到某长度，与在该长度可靠找证据、做推理，是不同要求。

位置插值的一种方法是把新位置 m 映射成 mL/L′，L 是训练长度、L′ 是目标长度。例如从 4096 扩到 8192，原来的位置 6000 会用相当于 3000 的位置相位，把范围压回模型见过的区间。这通常要配合长样本继续训练；压缩也把邻近位置拉得更近，降低局部位置分辨率，倍率越大越需要检查短文本和局部关系是否退化。其他频率缩放方法也需要按具体方案验证。

我会同时检查位置实现、训练数据、注意力内核和缓存容量，再用多种长度、多种证据位置评测。不能只测一个“针藏在长文里”的例子，还应测多证据整合、跨段推理及开头、中间、结尾的差异，并报告短文本效果、首 token 延迟和 KV 显存。位置设计改善长度适配，长文本训练改善使用长信息的能力，两者不能相互替代。

```math
m'=m\frac{L}{L'}
```

#### 易错点

- RoPE 能计算训练外位置，不证明模型能正确使用对应长文本。
- 单个长文检索样例不足以证明检索与推理能力全面有效。

#### 追问

- 位置插值压缩位置后，会怎样影响相邻 token 的区分？
- 怎样测试模型是否只偏向文档开头或结尾的信息？

<a id="tfm-019"></a>
### TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？

**L1**

#### 答案

位置编码让 Transformer 区分“有哪些词”之外的“词按什么顺序出现”。如果全连接注意力没有任何位置线索，并同步置换可见关系，交换输入顺序只会同步交换输出，难以区分“人咬狗”和“狗咬人”。公式里的 P 是置换矩阵，这叫置换等变；因果掩码能提供部分顺序线索，但并不自动替代所有位置设计。

绝对位置可以是每个位置一行的可学习向量，也可以是多频率正弦、余弦向量，再加到 token 表示。公式中的 m 是位置、j 是频率对、D 是维度；低频变化慢，高频变化快，可以覆盖不同距离尺度。同频率下固定位置偏移能通过二维旋转关联，所以绝对编码也能帮助表达相对关系。

相对编码直接描述两个位置的距离，例如 T5 按距离桶给注意力分数加可学习偏置。RoPE 旋转 Q/K，让点积显式出现相对位置差；ALiBi 给因果注意力分数加 −m_h(i−j)，m_h 是每个头的固定斜率，越远通常受到越大负偏置。它们作用在不同环节，不能给已有 checkpoint 随便替换而期待等价。选择和扩展时，要一起核对训练长度、位置编号、频率或距离桶、mask 与缓存；能算出更远位置不保证长文质量，也不保证无限外推。

```math
\begin{aligned}\mathrm{Attn}(PX)&=P\mathrm{Attn}(X)\\\mathrm{PE}(m,2j)&=\sin(m/10000^{2j/D})\\\mathrm{PE}(m,2j+1)&=\cos(m/10000^{2j/D})\\\mathrm{bias}_h(i,j)&=-m_h(i-j),\quad j\le i\end{aligned}
```

#### 易错点

- 绝对表示也能帮助表达相对关系，相对方案也不保证无限长度无退化。
- ALiBi 改变分数，RoPE 旋转 Q/K，二者都不是同一种输入位置加法。

#### 追问

- 输入换顺序后输出同步换顺序，和输出完全不变有什么区别？
- 正弦编码没有长度表上限，为什么训练外长文本仍可能失败？

<a id="tfm-024"></a>
### TFM-024 · 递归、乘性、卷积与复数位置表示怎样提供顺序信息？

**L2**

#### 答案

顺序信息不一定只能来自位置查表，还可以通过递推、位置相乘、卷积偏移或复数相位注入。它们的共同目标是让同一个词出现在不同位置时有可区分的表示，区别在于位置作用在哪里、是否容易并行，以及超出训练长度时的行为。

递归位置表示用上一个位置向量生成下一个，公式中的 f 是可学习递推函数；FLOATER 进一步把位置看作连续变量，用神经常微分方程描述状态演化。这样可以生成新位置，却有递推或数值求解成本，也不自动保证外推。乘性表示让位置向量逐元素调节词表示，例如放大某些通道；但任意相乘并不必然使点积只依赖两个位置的差。

卷积通过核内的左右偏移区分局部顺序，零 padding 的边界还可能提供绝对位置线索；无边界条件下的平移等变性质不能直接照搬到实际图像或序列。复数位置方法用模 r 表示幅度、用 ωm+θ 表示随位置变化的相位，公式中的 i 是虚数单位。RoPE 也可用复数乘法解释，但它通常在实数 Q/K 的二维通道对上旋转，不要求整个网络用复数。比较时应看输入、匹配分数或 Q/K 哪一处改变，以及对缓存、并行和长度泛化的影响。

```math
p_{m+1}=f_\theta(p_m),\qquad \frac{dp(t)}{dt}=h_\theta(p(t),t),\qquad z_{w,j}(m)=r_{w,j}e^{\mathrm{i}(\omega_{w,j}m+\theta_{w,j})}
```

#### 易错点

- 复数顺序表示不全等于 RoPE，也不都要求整个网络使用复数。
- 卷积从零填充边界得到的位置线索，不能脱离边界条件推广。

#### 追问

- 位置向量逐元素相乘，为什么不自动产生只依赖位置差的内积？
- 递推或常微分方程位置表示，怎样影响预计算和缓存？

<a id="tfm-025"></a>
### TFM-025 · Shaw、Transformer-XL、T5、DeBERTa 与 TUPE 如何建模位置信息？

**L2**

#### 答案

这些方法都让注意力感知位置，但不能一概理解成“给分数加一个距离”。Shaw 方法把截断距离对应的向量加到 key，也可加到 value，因此既改变匹配，也可能改变取回的内容。Transformer-XL 把内容与相对位置分成多个打分项，配合跨片段记忆，使模型能继续利用前一段而不把位置简单重用；XLNet 沿用相关结构，relative shift 是分数对齐的实现技巧。

T5 给注意力 logit 加距离桶偏置，近距离分得细、远距离合并。公式中 bucket(j−i) 把相对距离映射到桶，共享一个参数，这省参数却损失远距离分辨率。它的常见实现不额外除以 √d_k，而用投影初始化等约定控制尺度，不能机械套标准缩放。

DeBERTa 将内容与位置解耦，核心打分包括内容—内容、内容—位置、位置—内容三种交互，公式里的 c、r 分别表示内容和相对位置。其 enhanced mask decoder 在预训练预测时再使用绝对位置，“decoder”不表示常规自回归解码器。TUPE 则分别计算内容相关性和位置相关性，减少混合交互，并特殊处理 CLS；TUPE-A 使用解耦绝对位置，TUPE-R 再加相对偏置。截断、分桶、旋转解决的问题与保留信息不同，选型和长文本扩展都要结合具体训练与实现验证。

```math
s_{ij}^{\mathrm{T5}}=q_i^{\top}k_j+b_{\mathrm{bucket}(j-i)},\qquad s_{ij}^{\mathrm{DeBERTa}}\propto (q_i^c)^{\top}k_j^c+(q_i^c)^{\top}k^r_{\delta(i,j)}+(q^r_{\delta(j,i)})^{\top}k_j^c
```

#### 易错点

- 相对位置不仅能修改分数，也可能修改 key 或 value。
- DeBERTa 的增强 mask decoder 不是常规编码器—自回归解码器结构。

#### 追问

- 把多个远距离合成同一桶，会丢掉什么位置差异？
- Transformer-XL 的跨段训练记忆，与生成时 KV cache 有什么不同？

<a id="topic-3"></a>
## 归一化、FFN 与残差

<a id="tfm-005"></a>
### TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？

**L1**

#### 答案

Transformer 常用 LayerNorm，因为它为每个 token 单独调节特征尺度，不依赖同批次还有哪些样本，训练和逐 token 推理也能使用同一套统计方式。输入 B×T×D 时，常见 LN(D) 只在最后的 D 个特征上计算均值和方差；每个 token 都有自己的统计量，同一层的缩放 γ 和偏移 β 则沿位置共享。

公式先减均值，再除以方差加 ε 的平方根，最后做可学习仿射变换。ε 防止分母为零，也影响小方差时的尺度；通常使用分母 D 的总体方差，而不是 D−1。例如一个 token 的全部特征同时加相同常数，中心化后这一平移会消失。若 normalized_shape 改成 (T,D)，统计范围会覆盖整个序列，已不是同一操作；不同层通常也分别学习参数。

BatchNorm 常按通道聚合同批样本的统计，训练用批统计、常规推理用运行统计。小 batch、变长文本与 padding 会使这种估计更复杂，自回归解码也很难保持训练时的条件。它并非绝对不能用于序列，而是要明确统计轴和有效位置。图像里的 IN 对每个样本每通道统计空间位置，GN 对每个样本的通道组统计，均不等于 token 级 LN。归一化有助于优化，但仍要与残差、初始化、学习率和放置位置一起判断稳定性。

```math
\begin{aligned}\mu_{bt}&=\frac1D\sum_{d=1}^D x_{btd}\\\sigma^2_{bt}&=\frac1D\sum_{d=1}^D(x_{btd}-\mu_{bt})^2\\\mathrm{LN}(x)_{btd}&=\gamma_d\frac{x_{btd}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}+\beta_d\end{aligned}
```

#### 易错点

- 常见 LN(D) 沿 hidden 轴统计，不能误用时间轴或跨 batch 均值。
- 要保留 ε、γ、β 并使用对应方差定义；BN 并非绝对不能处理变长输入。

#### 追问

- LN(D) 与 LN((T,D)) 的统计范围和输出为什么不同？
- RMSNorm 省掉哪些计算，是否还会把均值变为零？

<a id="tfm-006"></a>
### TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？

**L2** · 百度

#### 答案

RMSNorm 用一个 token 特征的均方根调节尺度，LayerNorm 则先减均值、再用中心化方差调节尺度。均方根就是先把 D 个分量平方、求平均，再开平方；公式中的 γ 是每个通道可训练的缩放，ε 防止分母过小。RMSNorm 常省去中心化和偏移，因此计算更简单。

比如输入是 [1,1]，LayerNorm 减均值后会得到零向量，而 RMSNorm 仍保留原来的同向分量。两者都能控制整体尺度，但把所有特征同时加一个常数时，LN 的中心化部分基本不变，RMSNorm 通常会变。因此不能说 RMSNorm 会把均值归零，也不能说均值在任何模型里都没有作用。忽略 ε 时，RMSNorm 与 L2 归一化还差一个 √D 的尺度因子。

归约平方和时常使用 FP32，避免低精度溢出与累计舍入误差，再转回计算类型；最终速度取决于融合内核和硬件，质量也需要模型验证。pRMSNorm 只用部分特征估计均方根，能减少统计计算，但引入采样误差，与使用全部特征的标准公式有区别。选择哪一种应同时看训练稳定性、效果和真实延迟，不能把某篇论文中的比例直接套到所有模型上。

```math
\mathrm{RMSNorm}(x)=\frac{\gamma\odot x}{\sqrt{D^{-1}\sum_{i=1}^D x_i^2+\epsilon}}
```

#### 易错点

- RMSNorm 调节均方根，不保证输出零均值。
- 省去减均值是架构取舍，不能推出中心化在任何模型里都无用。

#### 追问

- 低精度输入的平方和为什么常放到 FP32 计算？
- RMSNorm 与 L2 单位范数归一化相差什么尺度？

<a id="tfm-007"></a>
### TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？

**L2**

#### 答案

Pre-Norm 在子层计算之前归一化，Post-Norm 在残差相加之后归一化；主要区别是梯度的直接残差通道是否经过这一层归一化。Pre-Norm 写成 x+F(Norm(x))，输入 x 可以沿加法支路直接传到输出；Post-Norm 写成 Norm(x+F(x))，两条支路相加后都要经过 Norm。

可以把残差理解为保留旧表示、只学习修正。Pre-Norm 即使修正分支暂时没学好，仍有比较直接的信息和梯度通道，因此深层训练通常容易稳定；末端常再加 final norm，控制累积后的输出尺度。Post-Norm 的一些初始化分析显示局部梯度可能偏大，因此更依赖合适的 warmup 和初始化，但结论有其假设，不能说任何 Pre-Norm 都不需要 warmup，或稳定就一定有更好的最终效果。

DeepNorm 是配套的稳定化方案，在 Post-LN 风格中使用与深度相关的残差系数及初始化缩放，例如 Norm(αx+F(x))；α 等参数要按编码器、解码器配置计算，不能仅把 Norm 挪位置就声称复现了它。实际比较要匹配预算并分别调学习率，检查激活、梯度和残差尺度，才能区分结构优势与训练设置不合适造成的差异。

```math
\begin{aligned}y_{\rm pre}&=x+F(\mathrm{Norm}(x))\\y_{\rm post}&=\mathrm{Norm}(x+F(x))\end{aligned}
```

#### 易错点

- Pre-Norm 通常更易稳定，但不保证任何设置都不需要 warmup。
- 训练稳定性不等于所有任务的最终效果必然更好。

#### 追问

- Pre-Norm 各层已有归一化，为什么末端还常加 final norm？
- 残差表示随层数增长时，怎样控制它的尺度？

<a id="tfm-010"></a>
### TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？

**L2**

#### 答案

FFN 为每个 token 做非线性的特征组合，注意力负责从其他位置取信息，FFN 则进一步加工已经取到的表示。原始结构先从 D 维升到 m 维，经过 ReLU，再降回 D 维；同一层所有位置用相同参数，不同层通常各有自己的参数。输入 B×T×D，输出仍是 B×T×D，不直接混合 T 轴。

例如注意力把人物、动作和时间的信息汇到某个 token，FFN 可以组合这些特征，形成更有用的表达。公式中的 W₁ 是 D×m、W₂ 是 m×D，bias 配合各自输出维度。非线性很重要：如果去掉激活，两层仿射就能合成一层，单纯升维再降维没有同样的表达能力；无 bias 时等效矩阵 W₁W₂ 的秩也受中间维度限制。

SwiGLU 用两个升维分支，其中一个经过 SiLU 作为门，再逐元素乘另一个分支，最后降维。W_g、W_u、W_d 因此是三张矩阵，而原 FFN 只有两张；忽略 bias，参数分别约为 3Dm 和 2Dm。若原 FFN 取 m=4D、预算为 8D²，要匹配参数量，SwiGLU 的 m 约取 8D/3，再做硬件对齐。门控、宽度和融合实现一起影响容量、显存与速度，不能只比较激活函数名字。

```math
\begin{aligned}\mathrm{FFN}(x)&=\mathrm{ReLU}(xW_1+b_1)W_2+b_2\\\mathrm{SwiGLU}(x)&=(\mathrm{SiLU}(xW_g)\odot xW_u)W_d\\N_{\rm FFN}&\approx2Dm,\quad N_{\rm SwiGLU}\approx3Dm\end{aligned}
```

#### 易错点

- 原始 FFN 有两次线性变换和非线性；现代模型不必采用相同激活或 bias。
- 同一层各位置共享 FFN 参数，但它不能直接从其他位置取信息。

#### 追问

- 无 bias、无非线性时，两层 FFN 的等效矩阵秩受什么限制？
- 匹配参数量时，为什么 SwiGLU 中间宽度通常小于 4D？

<a id="tfm-012"></a>
### TFM-012 · 残差连接为什么能帮助深层模型训练？

**L1** · 小红书

#### 答案

残差连接把输入直接加到子层输出，让子层学习“需要怎样修正旧表示”，也为梯度保留直接传播通道。基本形式是 y=x+F(x)，两边形状必须一致；若 F 暂时接近零，整块就接近恒等映射。它是相加，因此不会像拼接那样扩大特征维度。

这个简单块的导数包含 I+J_F，I 来自直接支路，J_F 来自修正分支。相比只经过 F，梯度多了一条路，深层优化通常更容易。但很多层的导数仍会相乘，若修正导数恰好抵消恒等项，也可能衰减，所以不能说残差保证梯度恒为 1。改变维度时直接支路要用投影 P，它的导数是 J_P，也不再是严格恒等。

原始 Transformer 把 dropout 后的子层输出与输入相加，再做 LayerNorm；完整导数还要包含 Norm。Pre-Norm 则先归一化修正分支的输入，直接支路不经过该块 Norm。原编码器每层围绕 self-attention、FFN 各有一条残差；原编码器—解码器中的解码层还多一条 cross-attention 残差，GPT 类解码器通常是两条。实际稳定性仍要结合深度、初始化、学习率、残差尺度和 final norm 检查，而不能只数残差条数。

```math
\begin{aligned}y&=x+F(x),\quad J_y=I+J_F\\y_{\rm post}&=\mathrm{LN}(x+\mathrm{Dropout}(F(x)))\\y_{\rm pre}&=x+\mathrm{Dropout}(F(\mathrm{LN}(x)))\end{aligned}
```

#### 易错点

- 残差提供直接路径，不保证所有深度与初始化下梯度恒定。
- 投影支路不是严格恒等，Pre-Norm 与 Post-Norm 的完整导数也不同。

#### 追问

- Post-Norm 的完整块导数为什么还要乘归一化的导数？
- 给残差分支乘系数 α，会怎样改变导数与激活尺度？

<a id="tfm-020"></a>
### TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？

**L1**

#### 答案

Dropout 在训练时随机屏蔽部分激活，迫使模型减少对固定特征组合的依赖，起到正则化作用。它临时改变计算，不会永久删除对应参数。常用 inverted dropout：以 p 的概率置零，保留值除以 1−p，使每个分量在随机屏蔽下的期望仍等于原输入。

例如 p=0.2 时，每个激活有八成机会保留，保留后乘 1.25；推理时就直接使用完整激活，不再重复放大。公式中的 m 是取 0 或 1 的随机变量，x 是原激活。p 越大，条件方差 px²/(1−p) 越大，扰动越强；期望保持只是一层的结论，经过非线性和归一化后，整个网络输出期望并不一定完全保持。

原始 Transformer 在子层输出做 dropout 后再残差相加、PostNorm，也在 embedding 与位置编码之和上做 dropout，base 概率为 0.1。现代模型的实际位置和概率应看实现，不能把框架默认 0.5 当成标准。nn.Dropout 在 eval 模式变成恒等映射，但直接调用 PyTorch SDPA 时，dropout_p 由调用者控制，评估必须显式传 0。排查随机推理时，既要检查 model.eval()，也要检查这些函数级参数。

```math
\begin{aligned}m_i&\sim\mathrm{Bernoulli}(1-p),\quad0\le p\lt 1\\y_i&=\frac{m_ix_i}{1-p}\\\mathbb E[y_i\mid x_i]&=x_i,\quad\mathrm{Var}(y_i\mid x_i)=\frac{px_i^2}{1-p}\\z&=\mathrm{LN}(x+\mathrm{Dropout}(\mathrm{Sublayer}(x)))\\h_0&=\mathrm{Dropout}(\sqrt{d_{\rm model}}\,\mathrm{Embedding}+\mathrm{PE})\end{aligned}
```

#### 易错点

- p 是丢弃概率；使用训练时反向缩放后，推理不要再重复缩放。
- dropout 的位置和概率要看模型，不能把默认值或现代实现全归为原论文配置。

#### 追问

- 如何推导 dropout 方差，为什么 p 越大扰动越强？
- model.eval() 后直接调用 SDPA，为什么仍需显式设置 dropout_p=0？

<a id="topic-4"></a>
## 复杂度与实现机制

<a id="tfm-011"></a>
### TFM-011 · Transformer 一层的时间、空间复杂度如何估算？

**L2** · 腾讯

#### 答案

估算一层 Transformer 的成本，要同时计算线性层和位置之间的注意力交互，通常写成 O(Td²+T²d)，T 是序列长度、d 是隐藏宽度。只报 O(T²) 会遗漏宽度和 FFN，在短序列、大模型里，线性层可能反而占主要成本。

标准 Q/K/V/O 四次投影约有 4Td² 次乘加，FFN 要按中间宽度再算；QKᵀ 与权重乘 V 合计约有 2T²d 次乘加。若一次乘加按 2 FLOPs 统计，就还要乘 2，不能把乘加次数直接当 FLOPs。比如长度翻倍，投影成本约翻倍，注意力交互约变成四倍；隐藏宽度翻倍则会明显放大线性矩阵成本。

朴素多头实现保存约 BhT² 个分数或权重，B 是 batch、h 是头数，峰值还包含其他激活和工作区。FlashAttention 分块计算、减少显存读写并避免完整保存这个矩阵，但精确稠密注意力的算术量仍然二次增长。解码有 KV cache 后，新查询只对历史 key 做匹配，单步注意力随历史长度线性增长，投影与 FFN 另算。真实速度还受带宽、并行度、内核启动与张量形状影响，FLOPs 更低并不保证更快。

```math
\begin{aligned}C_{\rm layer}&=O(Td^2+T^2d)\\M_{\rm attention}&=O(BhT^2)\end{aligned}
```

#### 易错点

- 只说 O(T²) 会遗漏宽度、线性投影和 FFN 成本。
- FlashAttention 减少存储与读写，不改变精确稠密交互的二次算术量。

#### 追问

- KV cache 下只有一个新查询时，注意力计算如何随历史长度变化？
- 算术操作更少为什么可能在实际设备上更慢？

<a id="tfm-022"></a>
### TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？

**L1** · 腾讯

#### 答案

Transformer 的全局注意力让相距很远的两个可见位置在一层内直接通信，因此信息和梯度不必像 RNN 那样经过中间每一步。这是它适合长距离依赖的重要原因。这里说的 O(1) 是位置之间的计算图路径长度，不是整个模型的运行时间。

比如句首出现人物名字，句尾需要判断代词指谁，全局注意力可以直接读取句首；RNN 对相隔 n 步的信息通常要传过约 n 次状态更新。不过“可以直接读到”不代表“已经学会正确使用”，还取决于训练数据、参数、位置表示和优化。长文中多个相似实体竞争时，仍可能选错。

公式中 n 是长度、d 是宽度、h 是头数。稠密注意力仍有 O(n²d) 的交互计算，显式分数存储为 O(hn²)，投影与 FFN 还加入 O(nd²)。训练同层位置可并行，生成下一 token 仍依赖之前输出。因果 mask 只允许读历史，局部窗口模型则要经过多层逐段传递；固定窗口尺度 r 时，远距离路径约随 n/r 增长。因此短路径不能当作常数时间，也不能代替长上下文的实际任务验证。

```math
\begin{aligned}\ell_{\rm global}&=O(1),\quad \ell_{\rm RNN}=O(n)\\C_{\rm attention}&=O(n^2d),\quad M_{\rm attention}=O(hn^2)\\C_{\rm layer}&=O(n^2d+nd^2)\quad(d_{\rm ff}=\Theta(d))\\\ell_{\rm local}&=O(n/r)\end{aligned}
```

#### 易错点

- 常数通信路径不等于常数运行时间，稠密注意力仍有二次交互。
- 局部窗口的远距路径不同，短路径也不保证学会所有长依赖。

#### 追问

- 固定窗口尺度 r，相距 n 的位置约需要多少层建立通信？
- 同层位置并行训练，与生成时逐 token 依赖有什么区别？

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
