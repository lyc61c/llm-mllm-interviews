# Transformer 与数学基础

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

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

**L1 · 社区题目线索** · 标签：Attention / 公式 / 数值稳定

**30 秒回答**

缩放点积注意力用 Q 与 K 的匹配分数，在 key 轴归一化后加权 V。除以 √d_k 可在分量独立且方差约为 1 的假设下稳定分数尺度，避免过度尖锐的 softmax；Q/K/V 来自可学习投影，掩码加在归一化前。softmax 本身并不产生硬稀疏。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 设 Q∈R^(L_q×d_k)、K∈R^(L_k×d_k)、V∈R^(L_k×d_v)，S=QK^T/√d_k+M∈R^(L_q×L_k)，A_ij=exp(S_ij)/Σ_l exp(S_il)，输出 AV∈R^(L_q×d_v)。self-attention 常取 Q=XW_Q、K=XW_K、V=XW_V；cross-attention 的 Q 与 K/V 可来自不同序列。
- 推导缩放时，假设 q_i、k_i 零均值、单位方差，q_i 与 k_i 独立且不同维的乘积互不相关，则 E(q_i k_i)=0、Var(q_i k_i)=1，Var(Σ_i q_i k_i)=d_k；标准差随 √d_k 增长，除以它将方差缩回 1。真实训练后的相关性和方差未必满足假设，因此这是尺度分析，而非恒等保证。
- softmax Jacobian 为 ∂A_i/∂S_j=A_i(δ_ij−A_j)。分数差过大时分布会很尖、部分 Jacobian 元素很小，使沿注意力分数传播的梯度敏感；不能据此声称任何 softmax 后的梯度都必然消失，例如 softmax 与交叉熵合并后的 logits 梯度为预测概率减目标概率。
- 允许的连接加 0、禁止连接加 −∞，再做 softmax；常以每行减去最大有限分数避免 exp 溢出，因为统一平移不改变概率。全屏蔽行需特殊处理，不能认为 −∞−(−∞) 会得到合法分布；实现还须检查 dtype、广播和 kernel 的 mask 约定。
- 有限且未屏蔽的 logits 在精确数学中得到严格正、和为 1 的权重，这是可微的软选择，通常没有硬零；数值下溢或 −∞ mask 不等于 softmax 天然稀疏。它便于加权汇聚但不能直接解释因果重要性；其他打分或归一化方案属于另行设计，需要比较质量、优化和实现成本。

### 公式

```text
Attention(Q,K,V)=softmax(QK^T/√d_k+M)V；Var(q·k)=d_k（零均值、单位方差与互不相关乘积假设下）；∂p_i/∂s_j=p_i(δ_ij−p_j)
```

### 易错点

- 把缩放分母写成 d_model 或 d_k，或省略独立/方差假设后把 Var(q·k)=d_k 当普遍定律。
- 把 mask 加到已归一化权重后不重归一化，或把 softmax 说成自动只选少量 token。

### 面试官可能追问

- 如果 Q/K 做 L2 归一化，点积分布与温度参数该怎样重新分析？
- cross-attention 的 L_q≠L_k 时，各中间矩阵与输出是什么形状？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**

- [CORE-S010 · LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) · `search_snippet`：代码备考仓库摘要列出注意力和 Pretrain/SFT 训练损失；题库按主题整理，不是公司真题证据。

<a id="tfm-002"></a>
## TFM-002 · 多头注意力与单头注意力有什么区别？

**L1 · 编辑补充题** · 标签：MHA / 表示能力 / 参数量

**30 秒回答**

多头注意力用多个独立的 Q/K/V 投影，在不同子空间并行计算注意力，再拼接各头输出并经 W_O 混合回模型维度。标准 MHA 在总隐藏维度固定时，每头通常变窄；头数增加不必线性增加总参数，也不保证每头都有可解释的专门分工。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 以 self-attention 输入 X:[B,T,D] 为例，第 i 头 W_Q^i、W_K^i:[D,d_k]，W_V^i:[D,d_v]，得到 Q_i/K_i:[B,T,d_k]、V_i:[B,T,d_v]。head_i=softmax(Q_i K_i^T/√d_k+M)V_i:[B,T,d_v]；原始 base 模型 D=512、h=8、d_k=d_v=64。
- 把所有头的 Q/K/V 堆成 [B,h,T,d_k]、[B,h,T,d_k]、[B,h,T,d_v]，分数和权重为 [B,h,T,T]；输出先为 [B,h,T,d_v]，转置并拼接为 [B,T,h d_v]，乘 W_O:[h d_v,D] 后得到 [B,T,D]。拼接不是在序列轴把 token 数扩大 h 倍。
- cross-attention 把 Q 的长度换成 L_q、K/V 长度换成 L_k，权重形状 [B,h,L_q,L_k]，输出长度仍为 L_q。标准头间投影参数分别学习；工程上一次大线性层生成各头再 reshape，与逐头执行这些投影等价，不代表头间权重相同。
- 当 h d_k=h d_v=D，标准 MHA 的 Q/K/V/O 总权重约为 4D²，忽略 bias；增加 h 且固定 D 时，单头维度下降，参数量并不随 h 成倍增加。注意力矩阵的元素数则与 hT² 有关，显存和 kernel 效率不能只由参数量推断。
- 各头可学习不同匹配和汇聚模式，但头间也可能冗余，不能预设每头必定负责语法、位置或实体。MQA/GQA 改变 K/V 共享方式，参数及 KV cache 需另算；多头的 W_O 负责融合各头信息，不应直接省略。

### 公式

```text
head_i=softmax((XW_Q^i)(XW_K^i)^T/√d_k+M)(XW_V^i)；MHA(X)=Concat(head_1,…,head_h)W_O
```

### 易错点

- 将单头维度 d_k 与总隐藏维度 D 混用，漏写转置/拼接或 W_O 导致 shape 不闭合。
- 把实现里的 fused QKV 层误认为不同 head 使用完全相同的投影参数。

### 面试官可能追问

- 固定 D 和 h 时，GQA 的参数量与 KV cache 怎样变化？
- 为什么 concatenation 后还需要 W_O，而不是直接相加各头输出？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-003"></a>
## TFM-003 · 掩码注意力如何实现？causal、padding 与 loss mask 有何区别？

**L1 · 编辑补充题** · 标签：Mask / 自回归 / 实现 / AttentionMask / KVCache

**30 秒回答**

掩码注意力在 softmax 前排除不可见的 key：因果掩码禁止未来信息，padding 掩码排除补齐位置，独立样本还需跨样本隔离。loss mask 只决定哪些目标计损，合法 prompt 仍应可见；缓存推理需按 query/key 的真实位置对齐。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 令 S=QK^T/√d_k，允许连接加 0、禁止连接加 −∞，再按 key 轴做 softmax。不能把被屏蔽分数乘 0，因为 exp(0) 仍会获得概率；全屏蔽行在朴素 softmax 中可产生 NaN，应避开并核对后端行为。
- 从 0 开始编号，完整因果 self-attention 允许 key 位置 j≤query 位置 i，包括自身；i 处 logits 监督 x_{i+1}，所以看到 x_i 不会泄漏下一词。仅加 loss mask 而缺少 causal mask 会让目标从未来上下文泄漏。
- padding mask 通常排除无效 key，pad query 的输出另在后续计算/损失忽略。prompt label=−100 只屏蔽该目标的直接 loss，不能把合法 prompt 的 attention_mask 设 0，否则回答看不到问题；独立 packed 样本需同时要求样本编号相同。
- 已有 P 个历史 token、一次处理 L 个新 token 时，Q 长 L、K 长 P+L，第 i 个新 query 可看 key j≤P+i。不能直接套左上对齐的 L×(P+L) 下三角；应显式核对内核矩形因果偏置、缓存位置、左 padding 与静态缓存未填位置。
- 数学上的加性 mask 与框架 bool mask 要分开。PyTorch 2.14 SDPA 的 attn_mask=True 表示允许关注，MultiheadAttention 的 key_padding_mask=True 表示屏蔽；广播需覆盖 [B,H,L_q,L_k]，用微型输入检验可见性。
- 变长输入在普通batch中可按本批最长序列padding，用有效长度生成key mask并在loss中忽略pad；按长度分桶可减少补齐浪费。位置编号应对应真实token和模型位置方案，输入仍受上下文/位置范围约束。padding mask通常不自动跳过padding的算术计算，变长attention内核或packing需另按接口设置段边界。

### 公式

```text
A=softmax(QK^T/√d_k+M); M_ij=0 if allowed(i,j), else −∞；cached causal: allowed(i,j)⇔j≤P+i（0-based）
```

### 易错点

- 把 prompt loss mask、EOS 边界或重置 position_ids 当成 attention 隔离。
- 不检查矩形 causal 对齐和 bool 含义，导致缓存只能关注最前面的 key 或泄漏未来。

### 面试官可能追问

- 如何用扰动另一个 packed 样本的测试证明没有跨样本信息泄漏？
- 只有一个新 query 且所有 key 均为有效历史时，还需要显式三角 mask 吗？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [CORE-S075 · Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [CORE-S081 · Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-004"></a>
## TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？

**L1 · 社区题目线索** · 标签：架构 / BERT / T5 / GPT

**30 秒回答**

Encoder-Decoder 先编码源序列，再由 decoder 自回归生成目标，两边长度可不同。原 Transformer 用双向 encoder、因果 decoder 及 cross-attention。其他架构的信息组织不同，需结合训练目标、任务和成本比较。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 原 2017 Transformer 的 encoder/decoder 各由 6 层堆叠：encoder 每层含 self-attention、FFN；decoder 多了查询 encoder 输出的 cross-attention。源序列 X 长 L_src 被编码为 memory H_src，目标第 t 步依赖源输入与目标历史，两边 token 数无需相同。
- decoder 的因果 self-attention 只访问已知目标历史；cross-attention 用 decoder hidden 作 Q、encoder memory 作 K/V，可访问完整已知源句并屏蔽源 padding。源 memory 可一次编码，投影的 cross-attention K/V 可复用，目标仍按自身历史条件生成。
- seq2seq 条件概率为 p(y|x)=Π_t p(y_t|y_<t,x)，训练可用右移目标和 teacher forcing 并行求各位置 loss。原论文使用残差加 PostNorm，位置编码同时注入两边；这些是该架构的具体实现，不意味着所有后续 encoder-decoder 必须固定 6 层或同一种 normalization。
- BERT 等 encoder-only 常用双向表示和 MLM，适合理解/表示任务；GPT 类 decoder-only 以因果目标训练，可将输入与输出串成同一上下文，生成接口统一。Prefix LM 的分段可见性不等于一定具有独立 encoder 和 cross-attention；架构与训练目标不能只由名称推定。
- 选择时比较任务形态、源/目标长度、memory 复用、缓存/吞吐、训练数据与目标。encoder-only 也可经另行设计用于生成，而 decoder-only 也能做分类；不能仅按结构标签声称某架构必定更强或适用于一切任务。

### 公式

```text
H_src=Encoder(x)；p_θ(y|x)=Π_t p_θ(y_t|y_<t,H_src)；decoder cross-attention: Q=H_decW_Q，K=H_srcW_K，V=H_srcW_V
```

### 易错点

- 把 source 与 target 强制要求等长，或把 cross-attention 也无条件设置成目标三角 mask。
- 把 encoder 的双向 attention 直接移到生成中的未知目标位置而造成未来信息泄漏。

### 面试官可能追问

- 改变 source 长度时，cross-attention 权重矩阵的哪一轴变化？
- 条件 memory 的缓存与 decoder 历史 KV cache 的增长方式有何不同？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S028 · BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [CORE-S029 · Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [CORE-S037 · Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

**题目出处线索**

- [CORE-S002 · 腾讯Teg大模型暑期算法面经](https://api-cdn.nowcoder.com/feed/main/detail/f91200d9c116401090432a2d78e5f76d?sourceSSR=users) · `reported_question`：公开帖子列出架构、LoRA、量化原理；仅保留问题主题，未采纳社区答案。

<a id="tfm-005"></a>
## TFM-005 · Transformer 为什么常用 LayerNorm，而不是 BatchNorm？

**L1 · 社区题目线索** · 标签：LayerNorm / BatchNorm / 归一化

**30 秒回答**

LayerNorm 对每个 token 的 hidden 维求均值和方差，再用 ε、可学习 γ/β 标准化及仿射变换。它不依赖同批其他样本，训练、推理统计一致。BatchNorm 依赖批/空间统计，变长并非不能用，但需处理 padding、统计轴与小批稳定性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 对 X:[B,T,D] 使用 normalized_shape=D 时，μ_bt=(1/D)Σ_d X_btd，σ²_bt=(1/D)Σ_d(X_btd−μ_bt)²，Y_btd=γ_d(X_btd−μ_bt)/√(σ²_bt+ε)+β_d。均值/方差各为 [B,T,1]，γ/β 各为 [D]；这里用分母 D 的总体形式，非 D−1 的无偏样本方差。
- ε 加在方差内以保护分母并影响很小方差时的尺度；γ、β 为可训练仿射参数，可按具体架构关闭 bias 或仿射。归一化是对每个 token 的通道，不是把全部 token 混在一起；PyTorch 若 normalized_shape 包含多个轴，则统计轴也相应扩大。
- LN 的统计取当前 token 自身，常规训练与推理都重算，不依赖 batch size 或运行均值，因此便于变长、在线和自回归解码。LN 参数可在同一层的各 token 间共享，但不意味着所有层共用同一组 γ/β。
- BN 通常对每个通道聚合 batch 与其他指定轴的统计，训练时使用批统计，常规推理使用估计的总体/运行统计。小批量、数据分布变化、序列位置差异和 padding 混入都会影响估计；变长可通过明确统计轴与有效位置处理实现，所以不应称 BN 数学上无法用于序列。
- LN 稳定激活尺度和优化，但不能单独保证消除梯度爆炸/消失；PreNorm/PostNorm、残差、初始化和学习率共同影响训练。解释 LN 与 BN 时应先给输入布局及被归一化轴，而非只说“一个按层、一个按批”。

### 公式

```text
μ_bt=(1/D)Σ_d x_btd；σ²_bt=(1/D)Σ_d(x_btd−μ_bt)²；LN(x)_btd=γ_d·(x_btd−μ_bt)/√(σ²_bt+ε)+β_d
```

### 易错点

- 对 [B,T,D] 在 T 轴归一化，或把每 token 均值误写成跨 batch 均值。
- 省略 ε/γ/β、混用 D 与 D−1，或宣称 BN 一定不能处理任何变长输入。

### 面试官可能追问

- LN(D) 与 LN((T,D)) 的输出相同吗，为什么？
- RMSNorm 去掉了哪些计算，是否仍把每个 token 的均值归零？

</details>

**技术依据**

- [CORE-S021 · Layer Normalization](https://arxiv.org/pdf/1607.06450)
- [CORE-S084 · Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)
- [CORE-S087 · torch.nn.LayerNorm — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)

**题目出处线索**

- [CORE-S001 · 网易大模型应用岗面经（一面、二面）](https://www.nowcoder.com/discuss/909223288612610048?sourceSSR=dynamic) · `reported_question`：公开面经题目列表直接出现这一主题；题目已改写，答案独立整理，公司归属未独立认证。

<a id="tfm-006"></a>
## TFM-006 · RMSNorm 与 LayerNorm 的公式和性质有什么区别？

**L2 · 社区题目线索** · 标签：RMSNorm / 归一化 / 数值稳定

**30 秒回答**

RMSNorm 按均方根缩放输入，省去 LayerNorm 的减均值步骤；它保留对整体尺度变化的归一化作用，但没有同样的平移不变性。计算更简单，性能和速度收益仍取决于模型、精度与融合内核。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- RMS(x)=√(D^{-1}Σx_i²)，实现常在根号内加 epsilon，并使用可学习逐通道权重。
- LayerNorm 使用中心化方差；RMSNorm 使用未经中心化的二阶矩。
- 对输入加常数，LayerNorm 的归一化部分基本不变，RMSNorm 通常会改变。
- 归约常采用更高精度以降低溢出和舍入误差；收益不能直接照搬论文中的百分比。

### 公式

```text
RMSNorm(x)=γ⊙x/√(D^{-1}Σ_i x_i²+ε)
```

### 易错点

- 把 RMSNorm 说成标准化到零均值。
- 说去均值在任何模型中都完全无用。

### 面试官可能追问

- 为什么 RMSNorm 常在 FP32 中计算归约？
- RMSNorm 与 L2 normalization 有何尺度差别？

</details>

**技术依据**

- [CORE-S022 · Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)

**题目出处线索**

- [CORE-S006 · 商汤NLP一面](https://www.nowcoder.com/feed/main/detail/fdc049a6b3444f3abee6f22256fc64ac) · `search_snippet`：仅搜索摘要可见 RMSNorm/SwiGLU/QLoRA；正文未读到，不能核验提问完整上下文。

<a id="tfm-007"></a>
## TFM-007 · Pre-Norm 与 Post-Norm 如何影响训练稳定性？

**L2 · 社区题目线索** · 标签：PreNorm / PostNorm / 梯度

**30 秒回答**

Pre-Norm 先归一化再进入子层，残差主路径保留恒等连接，深层训练通常更容易稳定；Post-Norm 在残差相加后归一化，初始化时部分层梯度可能较大。实际是否需要 warmup，还受深度、初始化和优化器影响。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Pre-Norm 为 x+F(Norm(x))；Post-Norm 为 Norm(x+F(x))。
- Pre-Norm 残差路径给梯度提供直接通道，通常还会在模型末端加最终归一化。
- 原论文的初始化分析解释 Post-Norm 对 warmup 的敏感性，但不是所有配置的定理。
- 比较两者时应匹配训练预算与调参，不能只比较相同学习率下的 loss。

### 公式

```text
Pre: y=x+F(Norm(x)); Post: y=Norm(x+F(x))
```

### 易错点

- 说 Pre-Norm 必然不需要 warmup。
- 把更稳定训练直接等同于所有任务更好的最终效果。

### 面试官可能追问

- 为什么很多 Pre-Norm 模型还有 final norm？
- 残差随深度增长如何控制？

</details>

**技术依据**

- [CORE-S023 · On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)

**题目出处线索**

- [CORE-S007 · 说说现在到底都在问什么（长文，慎入）](https://api-cdn.nowcoder.com/discuss/925432340174606336?sourceSSR=subject) · `reported_question`：公开面经汇总直接列出 Pre/Post-Norm、RoPE 外推或首 token/吞吐权衡；个人频率不采纳。

<a id="tfm-008"></a>
## TFM-008 · RoPE 如何表达相对位置，怎样与 KV cache 正确配合？

**L2 · 社区题目线索** · 标签：RoPE / 位置编码 / 公式 / CachePosition / PositionIDs

**30 秒回答**

RoPE 按位置旋转 Q、K 的成对分量，使两者点积出现相对位置差，同时保持范数。通常不旋转 V。缓存实现可保存已旋转 K，但新 token 必须沿用正确逻辑位置，不能重复旋转历史 K；位置缩放或配对约定也必须匹配模型。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 旋转子空间维度为偶数 d_r，第 j 对频率 θ_j=base^(−2j/d_r)，在位置 m 施加二维矩阵 R(mθ_j)=[[cos,−sin],[sin,cos]]。不同频率覆盖不同距离尺度；部分模型只旋转 head 的部分维度，不能机械假定总隐藏维度就是旋转维度。
- q'_m=R_m q_m、k'_n=R_n k_n，利用 R_m^T R_n=R_{n−m} 得 q'_m^T k'_n=q_m^T R_{n−m}k_n。这里的位置依赖为相对差，但 q_m/k_n 的内容表示仍依赖上下文，不能称分数只由距离决定。
- 旋转是正交变换，单个向量范数保持；一般二维例子为 (a cosφ−b sinφ, a sinφ+b cosφ)。相邻维配对与前半/后半维配对可通过置换关联，但未经相应权重转换不能直接替换 checkpoint 实现。
- Transformers v4.57.1 的 LlamaAttention 先旋转当前 Q/K，再把 K/V 写入缓存，因此历史 K 不应再旋转。prefill 后处理第 P 个逻辑位置的新 token，应生成该位置的 cos/sin；分块 decode、左 padding 和 packed 样本均需核对模型 position_ids。
- cache_position 管理缓存写入槽位，RoPE position_ids 管理位置相位，两者在简单无 padding 场景可能相同，接口含义仍不同。滑动窗口移除旧缓存不等于重置逻辑时间；若位置缩放/频率策略改变，应核验历史已旋转 K 的一致性并重算或使用明确支持该变化的缓存实现。

### 公式

```text
θ_j=base^(−2j/d_r); q'_m=R_m q_m, k'_n=R_n k_n; (q'_m)^T k'_n=q_m^T R_{n−m}k_n
```

### 易错点

- 直接把位置向量加到 embedding 来描述 RoPE，或对缓存 K 重复应用旋转。
- 把 RoPE 的相对点积性质当成任意距离单调衰减、任意长度可靠外推的保证。

### 面试官可能追问

- 用整段 prefill 与逐 token cache decode 比较 logits 时，哪些位置和数值条件必须一致？
- RoPE 的 base、旋转维度、缩放方案改变后，现有 KV cache 能否复用？

</details>

**技术依据**

- [CORE-S025 · RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- [CORE-S075 · Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [CORE-S080 · Transformers v4.57.1 official Llama implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/llama/modeling_llama.py)

**题目出处线索**

- [CORE-S005 · 大模型算法面经+问题+答案](https://www.nowcoder.com/discuss/891322059656052736?toCommentId=22719031) · `search_snippet`：正文抓取失败，搜索摘要明确出现相关问题主题，保留摘要级证据。

<a id="tfm-009"></a>
## TFM-009 · 为什么不能只把 max_position_embeddings 改大来扩展上下文？

**L2 · 社区题目线索** · 标签：长上下文 / RoPE / PositionInterpolation

**30 秒回答**

配置变大只允许更长输入，不保证模型学会处理训练范围外的位置或长距离依赖。RoPE 位置插值把更长的位置范围压回已训练区间，通常需配合继续训练；同时要验证短文本、长检索及显存与延迟。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 若从 L 扩到 L'，线性位置插值可用 m'=mL/L'，减少超出训练位置范围的程度。
- 位置压缩会降低局部位置分辨率，扩展倍率越大越需要注意短距离性能。
- 位置编码调整与长样本训练解决不同问题；长窗口允许输入不等于有效利用全部上下文。
- 评测需覆盖不同长度及证据位置，并报告任务正确率、KV 占用、TTFT。

### 公式

```text
m'=m·L/L'
```

### 易错点

- 把 RoPE 的数学可计算性当作可靠外推保证。
- 只测一个 needle case 就声称长上下文全面有效。

### 面试官可能追问

- 位置插值会怎样影响近邻 token？
- 如何排除模型只依赖文档开头或结尾？

</details>

**技术依据**

- [CORE-S026 · Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/pdf/2306.15595)

**题目出处线索**

- [CORE-S007 · 说说现在到底都在问什么（长文，慎入）](https://api-cdn.nowcoder.com/discuss/925432340174606336?sourceSSR=subject) · `reported_question`：公开面经汇总直接列出 Pre/Post-Norm、RoPE 外推或首 token/吞吐权衡；个人频率不采纳。

<a id="tfm-010"></a>
## TFM-010 · FFN 提供什么作用？SwiGLU 为什么常调整中间维度？

**L2 · 社区题目线索** · 标签：FFN / SwiGLU / 参数量

**30 秒回答**

原 Transformer 的 FFN 对每个 token 独立使用两层线性变换，中间是 ReLU：先升维、加入非线性，再降回隐藏维度。它在通道方向加工表示，不直接混合不同 token。现代 LLM 常用 SwiGLU，应区分原始公式与具体配置。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 按行向量约定，FFN(x)=max(0,xW_1+b_1)W_2+b_2，其中 x:[D]、W_1:[D,m]、b_1:[m]、W_2:[m,D]、b_2:[D]。对 X:[B,T,D] 沿最后一轴执行，输出仍为 [B,T,D]；原 base 设置 D=512、m=2048，约 4 倍扩展。
- position-wise 表示同一层各 token 使用相同 FFN 参数，却不把 token 相互相加；不同层通常有不同参数。跨 token 的信息由 attention 等模块汇聚，FFN 为各位置增加非线性特征组合和容量，不能把它当成又一次序列级注意力。
- 如果拿掉所有非线性，两层仿射可合成为 x(W_1W_2)+(b_1W_2+b_2)，增加中间宽度不会产生对应的非线性表达收益。ReLU 逐元素保留正值，但不能把“任何激活在任何规模下都等价”作为结论。
- SwiGLU 常写为 (SiLU(xW_g)⊙(xW_u))W_d，使用 gate/up/down 三个矩阵，bias 是否存在看模型；相比两矩阵的 ReLU FFN，它通过逐元素门控调整中间表示，不是对原公式仅把 ReLU 名字替换成 SiLU。
- 忽略 bias 时原 FFN 参数约 2Dm、SwiGLU 约 3Dm；若与 m=4D 的两层 FFN 保持约 8D² 参数预算，三矩阵中间宽度约取 8D/3，并常为硬件友好整数对齐。宽度、激活和融合 kernel 会影响容量、激活显存和吞吐，需据具体实现比较。

### 公式

```text
FFN(x)=ReLU(xW_1+b_1)W_2+b_2；SwiGLU(x)=(SiLU(xW_g)⊙(xW_u))W_d；参数量约为 2Dm 与 3Dm（忽略 bias）
```

### 易错点

- 漏写第二次线性变换或 bias，并将原论文激活写成所有现代 LLM 都使用的配置。
- 说 position-wise 参数不共享，或把 FFN 的同位置变换误认为能直接读取别的位置。

### 面试官可能追问

- 若无 bias 和非线性，两层 FFN 的等效矩阵秩受什么限制？
- 为什么同参数预算下 SwiGLU 的中间宽度常小于 4D？

</details>

**技术依据**

- [CORE-S024 · GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

**题目出处线索**

- [CORE-S006 · 商汤NLP一面](https://www.nowcoder.com/feed/main/detail/fdc049a6b3444f3abee6f22256fc64ac) · `search_snippet`：仅搜索摘要可见 RMSNorm/SwiGLU/QLoRA；正文未读到，不能核验提问完整上下文。

<a id="tfm-011"></a>
## TFM-011 · Transformer 一层的时间、空间复杂度如何估算？

**L2 · 社区题目线索** · 标签：复杂度 / FLOPs / Attention

**30 秒回答**

完整注意力的一层计算不仅有 O(T²d) 的注意力乘法，还有 O(Td²) 的投影与 FFN。朴素实现保存 T×T 分数带来二次显存；FlashAttention 可避免完整中间矩阵，却不把精确稠密注意力算术量变成线性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 设序列长度 T、隐藏维度 d，QKV/O 投影合计约 4Td² 次乘加；FFN 另按中间维度计算。
- QK^T 与 AV 合计约 2T²d 次乘加；若按 FLOPs 计数，乘加通常按 2 FLOPs。
- 朴素多头注意力分数存储约 B·h·T² 个元素，实际峰值还含激活及工作区。
- 短序列、大隐藏维度时线性层可占主导；长序列时二次项更明显。

### 公式

```text
单层主要计算量 O(Td²+T²d)
```

### 易错点

- 只报 O(T²) 而不说明 d、投影与 FFN。
- 把 FlashAttention 说成改变了精确注意力的二次算术复杂度。

### 面试官可能追问

- 使用 KV cache 后单步注意力复杂度如何变化？
- 为什么 FLOPs 更少的实现可能实际更慢？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

**题目出处线索**

- [CORE-S009 · 大厂问什么：2025-26 算法工程师面试常见问题整理（阿里系）](https://www.nowcoder.com/discuss/848942791164981248?sourceSSR=dynamic) · `search_snippet`：搜索摘录列出复杂度、GPTQ 或投机解码主题；未采用其自报次数/职级概率，公司归属为汇编者自述。

<a id="tfm-012"></a>
## TFM-012 · 残差连接为什么能帮助深层模型训练？

**L1 · 编辑补充题** · 标签：Residual / 梯度 / 稳定训练

**30 秒回答**

残差连接把子层输出与输入相加，让模型学习已有表示的修正，并提供梯度直接路径。简单的 y=x+F(x) 有 Jacobian I+J_F，但不保证任何参数和归一化安排下梯度都稳定。原 Transformer 用 PostNorm，现代模型常见 PreNorm。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 简单残差写成 y=x+F(x)，要求相加两支形状一致；若维度变化，可用投影 y=P(x)+F(x)，此时直接支路 Jacobian 为 J_P 而非 I。残差相加与 concat 不同，不会把隐藏维度翻倍；网络可学习 F(x)≈0 来接近恒等传递。
- 对最简单可微块，J_y=I+J_F，反传为 ∂L/∂x=(I+J_F)^T∂L/∂y，直接支路帮助信号和梯度跨越子层。多层仍需连乘各块 Jacobian，若 J_F≈−I 可相互抵消；不能由出现 I 就推出梯度永不消失或爆炸。
- 原 Transformer 子层采用 y=LN(x+Dropout(Sublayer(x)))，称 PostNorm，归一化在相加后。因此实际 Jacobian 还包含 J_LN 和 dropout，不能把 I+J_F 直接当成整块的完整导数；attention 与 FFN 各有自己的残差/归一化。
- 常见 PreNorm 写成 y=x+Dropout(F(LN(x)))，直接分支不经过此块的 LN，有助于深层训练；但学习率、最终 norm、初始化与残差尺度仍影响优化，不能说所有模型都必须采用这一顺序。
- 残差也改变激活累积方式。深度增加时要关注方差、溢出与分支尺度；分析问题时将残差路径、normalization 和 optimizer 分开核验，用梯度范数及激活分布判断，而不是只按结构名称判断稳定性。

### 公式

```text
y=x+F(x) ⇒ J_y=I+J_F；PostNorm: y=LN(x+Dropout(F(x)))；PreNorm: y=x+Dropout(F(LN(x)))
```

### 易错点

- 声称残差能让任何深度和初始化的网络梯度恒为 1。
- 混淆 PostNorm/PreNorm，或把投影 shortcut 当成严格恒等支路。

### 面试官可能追问

- PostNorm 为什么不能直接套用 J=I+J_F 作为完整块导数？
- 将残差支路乘以 α 后，Jacobian 与激活尺度如何变化？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S023 · On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- [CORE-S031 · Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-013"></a>
## TFM-013 · 交叉熵、KL 散度与 perplexity 的关系和实现注意点是什么？

**L1 · 社区题目线索** · 标签：CrossEntropy / KL / PPL / Entropy / FixedTarget / Distillation

**30 秒回答**

离散分布的交叉熵满足 H(p,q)=H(p)+KL(p||q)。当目标 p 固定、只优化 q 时，最小化交叉熵等价于最小化该方向的 KL；目标熵随参数变化时不能直接等价。one-hot 的单个样本损失是 −log q(y)，PPL 是有效 token 平均 NLL 的指数。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 定义 H(p)=−Σp_i log p_i，H(p,q)=−Σp_i log q_i，KL(p||q)=Σp_i log(p_i/q_i)。展开对数即得恒等式。p、q 须是同一支持空间上的规范化分布；0 log 0 按极限为 0，p_i>0 但 q_i=0 时 KL/交叉熵为 +∞。
- KL≥0，等号在 p=q 时成立，但通常不对称且不是距离度量。若 p 固定，H(p) 为优化常数；若目标由同一可训练模型产生且未 detach，H(p) 和 p 的梯度不能忽略。蒸馏还要明确 teacher/student 方向及是否停止 teacher 梯度。
- one-hot p 在当前监督样本上的 H(p)=0，所以该样本 CE=KL=−log q(y)。这不代表真实数据的条件熵为 0；语言会有多种合理续写。软标签或标签平滑时 H(p) 往往大于 0；当 p 固定且归一化、q=softmax(z) 时，对 logits z 的梯度为 q−p，对概率 q_i 本身的梯度则为 −p_i/q_i（q_i>0）。
- CrossEntropyLoss 通常接收原始 logits，通过稳定的 log-softmax/NLL 计算；别先做 softmax。PyTorch KLDivLoss 的 input 通常是 log q、target 是 p，名称顺序与 KL(p||q) 的数学参数顺序不同；逐 token/批次的 reduction 需按真实分布单位解释。
- 自然对数下 PPL=exp(Σ有效目标 NLL/N_valid)；log2 下用 2 的幂。不同长度 batch 应按有效 token 加权，比较模型需统一 tokenizer、数据和上下文窗口；低 PPL 不等于指令遵循、事实性或偏好更好。

### 公式

```text
H(p,q)=H(p)+KL(p||q); fixed p ⇒ ∇_θ H(p,q_θ)=∇_θ KL(p||q_θ); PPL=exp[−Σ_t log p_θ(x_t|x_<t)/N_valid]
```

### 易错点

- 无条件声称 CE 与 KL 完全相同，遗漏目标熵及 fixed-p 前提。
- 把 KLDivLoss 的输入顺序/mean 归一化当成数学 KL 的默认定义。

### 面试官可能追问

- 如何从 softmax 推导对 logits 的梯度 q−p？
- 可训练的软目标或双向 KL 会怎样改变优化目标？

</details>

**技术依据**

- [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [CORE-S043 · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [CORE-S076 · MIT 6.441 Chapter 1: Entropy and Divergence](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf)
- [CORE-S082 · torch.nn.KLDivLoss — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)

**题目出处线索**

- [CORE-S010 · LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) · `search_snippet`：代码备考仓库摘要列出注意力和 Pretrain/SFT 训练损失；题库按主题整理，不是公司真题证据。

<a id="tfm-014"></a>
## TFM-014 · 输入 embedding 与输出 LM head 权重共享有什么利弊？

**L2 · 编辑补充题** · 标签：WeightTying / Embedding / 参数量

**30 秒回答**

语言模型将 hidden state 映射为词表 logits，再用 softmax 得到下一 token 概率。相容的 embedding/head 可共享权重以省参数。标准 Transformer 的各层、各头不因此自动共享参数；ALBERT 跨层共享是另行设计。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 若 hidden H:[B,T,D]、W_vocab:[D,V]、b:[V]，logits Z=HW_vocab+b:[B,T,V]，p_btv=exp(Z_btv)/Σ_u exp(Z_btu)。softmax 沿词表 V 轴做；logits 不是已经归一化的概率，交叉熵实现通常直接接收 logits 并内部计算 log-softmax。
- 输入 embedding 表 E:[V,D] 将 token id 查表为隐藏向量；weight tying 可令 W_vocab=E^T，得到 logits=hE^T+b。共享去掉一份约 VD 权重，但不会自动去掉输出 bias；如果输入/输出词表或隐藏维不同，需要另行设计映射，不能强行转置共享。
- 原 2017 Transformer 明确共享两个 embedding 层与 pre-softmax 线性变换的权重，并在 embedding 使用时乘 √D。这个特定设计应与各 attention head 的 W_Q/W_K/W_V、各层 FFN 等参数分开：层结构相同不表示参数数值相同或训练时共用同一个张量。
- 常规标准 Transformer 各层分别学习参数，多头也各有投影子块；fused QKV 计算不改变这一点。ALBERT 专门提出跨层参数共享，可只共享 attention、只共享 FFN 或全部共享，默认方案是全部共享；它是节省参数的架构选择，不能当成原 Transformer 的普遍规则。
- 共享会让输入表示与输出分类器接受共同梯度并减少参数存储，但不减少词表输出所有类别的计算量。验证时检查 state_dict/参数别名、词表大小和输出维度；参数共享不等于参数冻结，也不保证所有任务质量改善。

### 公式

```text
Z=HW_vocab+b；p_v=exp(z_v)/Σ_u exp(z_u)；weight tying: W_vocab=E^T；减少权重约 VD
```

### 易错点

- 把同一层内跨位置使用 FFN、embedding/head tying 与跨层或跨头共享混为一谈。
- 对 logits 先 softmax 后再交给期望 logits 的 CrossEntropyLoss，或沿 hidden 轴做词表归一化。

### 面试官可能追问

- 共享 embedding/head 时，padding token 对应的输出行是否一定永远没有梯度？
- ALBERT 共享参数为何省参数却仍要执行多层计算？

</details>

**技术依据**

- [CORE-S030 · Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [CORE-S086 · ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-015"></a>
## TFM-015 · MoE 与 Dense 的参数量和计算量应怎样比较？

**L2 · 社区题目线索** · 标签：MoE / 路由 / 负载均衡

**30 秒回答**

MoE 为一个 token 只激活少数专家，以较低激活计算量提供更大的总参数容量；Dense 通常每个 token 使用全部层参数。总参数、激活参数和真实推理成本必须分开，路由通信、专家负载与硬件布局可能抵消算术收益。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 典型 MoE 将部分 FFN 换成专家集合，router 选择 top-k，并对专家输出加权。
- 激活参数包括共享模块及选中专家，不等于简单用总参数乘 k/E。
- 不均衡路由会产生热点、容量溢出或 token 丢弃，需要辅助损失等机制。
- 专家并行有 all-to-all 通信；小 batch、跨节点或低利用率下时延可能较差。

### 易错点

- MoE 总参数大就必然推理更慢，或激活参数少就必然更快。
- 忽略共享模块、路由与通信成本。

### 面试官可能追问

- 负载均衡会不会损害专家专门化？
- 如何区分模型 FLOPs 与实际 GPU 成本？

</details>

**技术依据**

- [CORE-S027 · Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)

**题目出处线索**

- [CORE-S005 · 大模型算法面经+问题+答案](https://www.nowcoder.com/discuss/891322059656052736?toCommentId=22719031) · `search_snippet`：正文抓取失败，搜索摘要明确出现相关问题主题，保留摘要级证据。

<a id="tfm-016"></a>
## TFM-016 · Cross-attention 与 self-attention 有何区别，Q/K/V 从哪里来？

**L1 · 编辑补充题** · 标签：CrossAttention / QKV / EncoderDecoder / 多模态

**30 秒回答**

Self-attention 的 Q/K/V 来自同一序列；cross-attention 的 Q 来自需要更新的目标表示，K/V 来自可访问的另一组条件或 memory。输出长度由 Q 决定，两边长度与原始特征维度可不同；投影后 Q/K 维度需匹配，是否因果取决于信息可用性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 令目标 X∈R^(L_q×d_x)、条件 C∈R^(L_k×d_c)，Q=XW_Q、K=CW_K、V=CW_V。只需投影后的 Q/K 都有 d_k 维，V 可为 d_v 维；单头结果形状为 L_q×d_v，多头拼接后经 W_O 回目标隐藏维度。
- Encoder-Decoder 中 decoder hidden states 形成 Q，encoder 输出形成 K/V；每个 decoder 位置可访问完整的已知源序列。decoder 自身的 self-attention 仍要因果掩码，因此 cross-attention 可非三角并不破坏输出自回归性。
- 视觉/音频条件生成中可让语言状态查询模态特征，也可让学习的 latent queries 汇聚图像；方向决定被更新的表示。K/V 长度相同并对应同一 memory 集合，来源相异并不自动意味着任意拼接都语义对齐。
- 仅注意力的两个主要乘法约为 O(L_q L_k d) 量级，投影另计；分数矩阵为 L_q×L_k，不要求方阵。Q 数决定输出 token 数，K/V 数决定可读取的条件粒度；压缩 K/V 可省成本但可能丢细节。
- 条件 memory 固定时，每层的条件 K/V 投影可缓存复用；Q 随目标变化仍需计算。必须屏蔽条件 padding/不可授权内容，流式任务不能看尚未到达的模态信息；attention 权重本身不保证事实正确或因果解释。

### 公式

```text
CrossAttn(X,C)=softmax((XW_Q)(CW_K)^T/√d_k+M) · (CW_V); output length=L_q
```

### 易错点

- 要求 cross-attention 两边输入长度或原始 embedding 维度完全相同。
- 认为所有 cross-attention 都必须采用 decoder self-attention 的三角 mask。

### 面试官可能追问

- 固定 encoder memory 的 K/V cache 与 decoder 历史 KV cache 有何区别？
- 可学习 query 压缩视觉 token 时，信息瓶颈主要出现在哪里？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-017"></a>
## TFM-017 · 熵衡量什么？离散熵、条件熵与模型输出熵怎样区分？

**L1 · 编辑补充题** · 标签：Entropy / ConditionalEntropy / 不确定性 / 信息论

**30 秒回答**

离散熵是分布下自信息的期望，衡量平均不确定性：集中到一个结果时为零，固定 V 个类别下均匀分布达到 log V。条件熵还对上下文分布取平均。模型输出熵描述模型自身分布，不能直接当作准确率、事实性或真实数据的固有不确定性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 事件自信息 I(x)=−log p(x)，熵 H(X)=E[I(X)]。log2 使用 bits，自然对数使用 nats；零概率项按 0 log 0=0 的极限处理。有限 V 类时 0≤H≤log V，均匀分布取上界，确定分布取下界。
- 二元分布 p=(a,1−a) 的熵为 −a log a−(1−a)log(1−a)，a=1/2 最大，a=0 或 1 为零。熵由完整分布决定，不能只比较最大 token 的概率判断所有分布熵高低。
- H(X|Y)=Σ_y p(y)H(p(X|y))，满足 H(X,Y)=H(Y)+H(X|Y)。条件信息在平均意义下降低熵 H(X|Y)≤H(X)，不表示每个特定 y 都一定比无条件分布更低；语言序列熵可用条件熵链式展开。
- 真实条件熵 H(p) 与模型输出熵 H(q_θ) 不同。模型对错误续写也可能非常自信、熵很低；训练交叉熵 H(p,q_θ) 包含失配 KL，不能通过单独降低 H(q_θ) 就保证拟合真实分布。
- 训练/评测日志应明确 token 分布、过滤前后、mask 和对上下文的平均范围。离散 token 熵通常非负，而连续变量的微分熵 h(X)=−∫p log p 可为负且随尺度变化，不能机械套用离散熵界。

### 公式

```text
H(p)=−Σ_i p_i log p_i; H(X|Y)=E_Y[H(p(X|Y))]; H(X,Y)=H(Y)+H(X|Y)
```

### 易错点

- 低熵就是准确，高熵就是幻觉，或把数据熵与模型预测熵混同。
- 把平均条件熵下降说成所有特定条件都降低不确定性。

### 面试官可能追问

- 高置信度错误输出为何可以有低熵？
- 为什么连续均匀分布的微分熵会随区间长度变成负值？

</details>

**技术依据**

- [CORE-S076 · MIT 6.441 Chapter 1: Entropy and Divergence](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2243edffb30f57181ed97dcb77691580_MIT6_441S16_chapter_1.pdf)
- [CORE-S072 · TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-018"></a>
## TFM-018 · 矩阵的秩与特征值如何计算，和奇异值有什么关系？

**L2 · 编辑补充题** · 标签：Rank / Eigenvalues / SVD / LoRA / 线性代数

**30 秒回答**

秩可用消元主元数或非零奇异值数计算；方阵特征值是 det(λI−A)=0 的根，大矩阵通常用数值分解。两者不能一般地互相计数；可对角化时，非零特征值按重数计数等于秩。矩形矩阵适用 SVD，浮点数值秩还依赖容差。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- A∈R^(m×n) 的 rank(A)=dim range(A)，等于行秩/列秩，满足 rank(A)+dim ker(A)=n。可由消元的主元个数或 SVD 得到；LoRA 的 BA 满足 rank(BA)≤min(rank A,rank B)≤r，原始 W_0 本身不必低秩。
- 方阵特征向量满足 Av=λv 且 v≠0，特征值可为复数；一般非方阵没有标准的 Av=λv 特征值定义。A^T A/A A^T 的特征值可以用来求奇异值，但它们不是矩形 A 自身的特征值。
- 手算先求特征多项式 det(λI−A)，令其为零求 λ，再解 (A−λI)v=0 得非零特征向量。例如 A=[[2,1],[0,3]]，多项式为 (λ−2)(λ−3)，特征值为 2、3，对应向量可取 (1,0)、(1,1)；消元有两个主元，所以秩为 2。该例为独立手算。
- 大规模浮点计算通常不显式展开高次特征多项式再求根；通用数值库可先化为 Hessenberg 形，再以 QR 等过程得到 Schur 分解。复 Schur 三角矩阵的对角给特征值；实 Schur 的 2×2 块还需求该块特征值。对称/Hermitian 问题可选专用求解器；谱分解、消元与 SVD 应按任务和数值稳定性选择。
- 若 A=PΛP^−1 可对角化，则 rank(A)=rank(Λ)，等于按重数计算的非零特征值个数。实对称/复 Hermitian 矩阵可酉对角化；一般方阵未必可对角化，不能无条件以特征值数量数秩。
- 反例 A=[[0,1],[0,0]]，两个特征值均为 0，但 rank(A)=1，且只有一个独立特征向量，因此不能对角化。对任意方阵仍有 det(A)≠0⇔所有特征值非零⇔满秩，但这只判断是否满秩，不给一般缺秩数值。
- SVD A=UΣV^T，σ_i≥0，rank(A)=#{σ_i>0}；非零奇异值的平方正好构成 A^T A 的非零谱，矩形情况下还须按 A^T A 的维数补齐零特征值。实对称 A 的奇异值为 |λ_i|。浮点实现按 σ_i>max(atol,rtol·σ_max) 判断数值秩，阈值依 dtype/规模/噪声，近零不等于数学上的精确零。

### 公式

```text
det(λI−A)=0; (A−λI)v=0, v≠0; rank(A)=#{σ_i(A)>0}; σ_i²=λ_i(A^T A)；A=PΛP^−1 时 rank(A)=#{λ_i≠0}（计重数）；rank(BA)≤r
```

### 易错点

- 所有矩阵的秩都等于非零特征值个数，或给非方阵直接定义标准特征值。
- 把奇异值与一般特征值逐项等同，或把训练配置 r 当作已学更新的实际秩。

### 面试官可能追问

- 为什么 A^T A 理论上可求奇异值，但数值计算可能放大条件数问题？
- 低秩近似、数值秩与 LoRA 的秩上界分别回答什么问题？

</details>

**技术依据**

- [CORE-S077 · Stanford EE263: Eigenvectors and diagonalization](https://ee263.stanford.edu/lectures/eig.pdf)
- [CORE-S078 · Stanford EE263 Lecture 15: Symmetric matrices and SVD](https://web.stanford.edu/class/archive/ee/ee263/ee263.1082/lectures/symm.pdf)
- [CORE-S083 · torch.linalg.matrix_rank — PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.matrix_rank.html)
- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [SUP-P001 · LAPACK Users' Guide: Eigenvalues, Eigenvectors and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-019"></a>
## TFM-019 · Transformer 为什么需要位置编码？绝对、相对、RoPE 与 ALiBi 怎样比较？

**L1 · 编辑补充题** · 标签：PositionalEncoding / Sinusoidal / RelativeBias / RoPE / ALiBi

**30 秒回答**

无位置线索的全可见 self-attention 对输入排列具有等变性，无法充分辨别序列顺序。位置机制可在输入加绝对编码、在注意力分数加相对偏置，或旋转 Q/K。它们改变的部位和长度扩展条件不同，不能凭无参数或相对形式保证长上下文能力。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 无位置编码的全连接 self-attention 对输入 token 置换 P 满足 Attn(PX)=P Attn(X)，因为 Q/K/V 同步置换、score 行列同步置换。因果 mask 等结构可提供部分顺序信息，但不能因此忽略实际模型所使用的位置机制与训练分布。
- 绝对位置可用学习查表 e_m，或原论文的正弦/余弦：PE(m,2j)=sin(m/10000^(2j/D))、PE(m,2j+1)=cos(m/10000^(2j/D))，与 token embedding 相加；m 是从 0 开始的位置、j 为频率对编号，维度为 D。不同维频率不同，固定位置差可通过二维旋转联系同频率的 sin/cos；可计算新位置不代表可靠长度外推。
- 相对机制按 query/key 偏移调整匹配，例如 T5 在 score 上加可学习的距离桶标量，分桶把多个距离映射到同一参数。它与直接加绝对 embedding 不同，但桶范围、可见性和训练长度仍影响能力。
- RoPE 对 Q/K 分量施加位置相关旋转，使点积显式依赖位置差，通常不旋转 V；ALiBi 在因果注意力分数加 −m_h(i−j) 的距离偏置，原方案 head 斜率固定。两者不是同一种位置向量，也不应未经训练转换便替换已有底座。
- 比较时记录插入位置、训练长度、位置编号、旋转维度/频率或距离桶、mask 与缓存实现。扩长上下文须考察数据、缩放/插值、kernel/显存和长文任务；改配置上限只解决可运行范围，不能保证检索/推理效果。

### 公式

```text
无位置且可见性同步置换：Attn(PX)=P Attn(X)；PE(m,2j)=sin(m/10000^(2j/D))，PE(m,2j+1)=cos(m/10000^(2j/D))；ALiBi causal bias_h(i,j)=−m_h(i−j), j≤i
```

### 易错点

- 绝对编码一定不能表达相对关系，或相对编码天然无限长度无退化。
- 将 ALiBi 写成加到 token embedding 的可训练位置向量，或把 RoPE 写成同样的加法。

### 面试官可能追问

- 置换等变性与置换不变性分别是什么意思？
- 为什么无参数的正弦编码仍可能在超出训练长度时失败？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S025 · RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- [CORE-S029 · Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [CORE-S079 · Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-020"></a>
## TFM-020 · Dropout 如何正则化？原始 Transformer 把它放在哪里？

**L1 · 编辑补充题** · 标签：Dropout / 正则化 / 原始Transformer / Post-LN

**30 秒回答**

Dropout 在训练时随机置零激活，inverted dropout 将保留值除以保留概率，评估时为恒等映射。原始 Transformer 在各子层输出的残差相加前，以及词嵌入与位置编码之和上使用 Dropout；基础模型丢弃概率为 0.1。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 对固定输入 x，采样 m~Bernoulli(1-p)，训练输出 m⊙x/(1-p)，其中 0≤p<1，因此 E[y|x]=x。它给激活引入随机扰动，减少对特定共同激活模式的依赖；不是永久删除参数。
- 原始论文 §5.4 的 residual dropout 位于每个子层输出上，之后与子层输入相加并归一化。结合原始 Post-LN 结构可写为 LN(x+Dropout(Sublayer(x)))，涵盖编码器的注意力/FFN 与解码器的相应子层。
- 原始论文还对编码器和解码器中 embedding 与 positional encoding 的和做 Dropout；论文基础模型 p_drop=0.1。现代模型的概率与插入位置应按具体配置和代码回答。
- PyTorch nn.Dropout 在 eval 模式为恒等映射。单个 Dropout 保持条件期望不意味着包含非线性、归一化等运算的整个网络在训练/推理时输出期望完全相同。
- 原始论文上述段落未逐一规定现代实现中的 attention probability dropout 或 FFN 内部 dropout。PyTorch SDPA 的 dropout_p 显式控制 softmax 后的 dropout，评估时调用者须传 0，不能把该函数与 nn.Dropout 的模式切换行为混为一谈。

### 公式

```text
m_i∼Bernoulli(1−p), 0≤p<1；y_i=m_i x_i/(1−p)，E[y_i|x_i]=x_i，Var(y_i|x_i)=p x_i²/(1−p)。原始 Post-LN：z=LN(x+Dropout(Sublayer(x)))；输入：Dropout(√d_model·Embedding+PE)。
```

### 易错点

- 把 p 当作保留概率，或在训练和推理阶段重复做 1/(1-p) 缩放。
- 把现代实现的所有 Dropout 位置都归因于原始论文，或把 nn.Dropout 默认 0.5 说成 Transformer 标准值。

### 面试官可能追问

- 推导 inverted dropout 的条件方差，并解释为何 p 越大扰动越强。
- 如果只调用 model.eval()，为什么直接调用 SDPA 时仍可能出现随机输出？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [NEWTFM-P01 · PyTorch 2.14 — torch.nn.Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)
- [NEWTFM-P04 · PyTorch 2.14 — scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-021"></a>
## TFM-021 · Attention 权重如何学到？权重较大就能解释模型决策吗？

**L2 · 编辑补充题** · 标签：Attention / QK / 反向传播 / 可解释性 / 张量形状

**30 秒回答**

标准点积注意力的权重由输入经 Q/K 投影、点积和 softmax 动态计算，是激活而非独立参数表。任务损失通过这些运算更新投影矩阵和上游表征。权重大只表示该层该头的相对加权关系，不能自动等同于对最终预测的因果贡献。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 省略 batch/head 维，对自注意力 X∈R^(n×d_model)，Q=XW_Q、K=XW_K∈R^(n×d_k)，V=XW_V∈R^(n×d_v)。S=QKᵀ/√d_k 为 n×n，A=softmax_row(S+M)，O=AV 为 n×d_v；同一组参数面对不同输入产生不同 A。
- 反向传播先经过 O=AV，再经过行 softmax 和点积。若 G=∂L/∂A，则 D_ij=∂L/∂S_ij=A_ij(G_ij−Σ_k A_ikG_ik)，进而 ∂L/∂Q=DK/√d_k、∂L/∂K=DᵀQ/√d_k，并更新 W_Q/W_K 与上游输入。
- 学习的是生成权重的函数，包括投影参数、上游表征及具体模型可能带有的位置偏置等参数；不能把每次前向得到的 n×n 注意力矩阵当成与样本无关的可训练参数表。训练目标也未必直接监督注意力图。
- 高 A_ij 不单独确定最终影响：V_j 的方向/幅值、其他头、输出投影、残差与后续层都会改变结果。注意力热力图可以描述局部计算，但因果归因需要明确定义和额外诊断。
- Jain & Wallace 与 Wiegreffe & Pinter 的研究呈现了注意力解释性的限制及诊断争论，实验范围是特定 NLP 模型/任务。可结合消融、扰动和基线检验提出解释，并留意扰动改变输入分布；不能据标题概括为任何模型的注意力永远不能解释。

### 公式

```text
Q=XW_Q，K=XW_K，V=XW_V；A=softmax_row(QKᵀ/√d_k+M)，O=AV。令 G=∂L/∂A，D_ij=A_ij(G_ij−Σ_k A_ikG_ik)，则 ∂L/∂Q=DK/√d_k，∂L/∂K=DᵀQ/√d_k，∂L/∂W_Q=XᵀDK/√d_k。固定 mask，仅考虑允许位置；被屏蔽位置梯度为 0。
```

### 易错点

- 说 softmax 中每个注意力权重都是独立学习参数，忽略 Q/K 与输入。
- 把注意力热力图当作最终预测的充分因果解释，或把特定 RNN 实验结论泛化到所有 LLM。

### 面试官可能追问

- softmax 的行内归一化为什么会让一个位置的梯度依赖其他位置？
- 两个注意力图差异很大但输出相近，能推出哪些结论，不能推出哪些结论？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [NEWTFM-P04 · PyTorch 2.14 — scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [NEWTFM-P02 · Jain & Wallace (2019) — Attention is not Explanation](https://aclanthology.org/N19-1357.pdf)
- [NEWTFM-P03 · Wiegreffe & Pinter (2019) — Attention is not not Explanation](https://aclanthology.org/D19-1002.pdf)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-022"></a>
## TFM-022 · Transformer 为什么适合建模长距离依赖？O(1) 路径意味着什么？

**L1 · 编辑补充题** · 标签：长距离依赖 / 复杂度 / Self-Attention / 计算图 / 因果Mask

**30 秒回答**

全局自注意力让允许交互的远距离位置在一层内直接交换信息，计算图路径不随距离增长；RNN 通常需沿时间步传递。O(1) 指路径长度，不代表常数运行时间：稠密注意力的两两计算随序列长度平方增长，短路径也不保证模型真正学好长依赖。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 原始论文 §4 比较的是位置间通信路径。全局 self-attention 中，一个 query 可直接聚合任意允许位置的 value，最长路径为 O(1)；相隔 Θ(n) 时间步的 RNN 位置通常需经过 Θ(n) 次递归状态传递。
- 短路径使信息和梯度不必逐步跨越长时间链，但只提供结构上的便利。是否学到有效依赖仍取决于训练数据、参数、位置编码、优化以及具体任务；不能把路径复杂度当作长上下文准确率保证。
- n 为序列长度、d 为模型宽度，多头总维度与 d 同阶时，稠密 QKᵀ 和 AV 的总计算为 O(n²d)，常规显式注意力矩阵存储为每头 O(n²)。加上投影与 d_ff=Θ(d) 的 FFN，一层计算为 O(n²d+nd²)。
- O(1) 的逐层位置通信也不等于整段自回归生成只需一步。训练/已知输入的同层 query 可以并行计算，生成下一个 token 仍依赖先前已生成的 token。
- 上述直接路径受 mask 限制：因果注意力能让当前位置直接读取允许的历史位置，不能读取未来；局部窗口注意力不能直接连接任意远处位置，需要跨层传播。有效上下文还受模型窗口与位置方案限制。

### 公式

```text
全局 self-attention：最大位置通信路径 O(1)，稠密注意力计算 O(n²d)，显式权重存储 O(hn²)；RNN：最大路径 O(n)。若 d_ff=Θ(d)，Transformer 单层总计算 O(n²d+nd²)。局部固定窗口的传播路径随 n/r 增长，通常为 O(n/r)。
```

### 易错点

- 把常数通信路径说成常数时间或线性计算复杂度。
- 以全局注意力的结论描述局部窗口模型，或声称短路径自动解决所有长依赖问题。

### 面试官可能追问

- 局部窗口半径 r 固定时，距离 n 的位置需要多少层才能建立通信？
- 训练时的同层并行与自回归解码的逐 token 依赖有什么区别？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="tfm-023"></a>
## TFM-023 · 所谓 Negative Attention 是什么？低权重、负 logit 与负输出有何区别？

**L2 · 编辑补充题** · 标签：Negative Attention / Softmax / Logit / 术语辨析 / Mask

**30 秒回答**

原始 Transformer 没有名为 Negative Attention 的标准组件，softmax 注意力权重非负。低权重只表示相对分配较少；负 logit 经 softmax 仍可得到正权重；负输出可来自有符号的 V 或后续投影。遇到同名论文方法，应先核对它的具体定义。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 对至少有一个允许位置的行，标准 softmax 权重 A_ij=exp(S_ij)/Σ_k exp(S_ik) 非负，允许且有限的 logit 在实数精确计算下得到正权重；mask=-∞ 的位置权重为 0。低权重不能直接称作带负系数的注意力。
- logit 是归一化前的分数，可以为负。例如 softmax([-2,-1])≈[0.269,0.731]。整行加常数不改变 softmax，因此分数绝对正负并不能单独刻画权重或关注程度。
- O_i=Σ_j A_ijV_j 的向量分量可以为负，即使 A 全部非负。例如 A=[0.5,0.5]、标量 V=[-3,1] 得到 O=-1；多头输出投影、残差等运算也会产生正负分量。
- 降低某个权重通常减少它相对于其他 value 的份额，不能在未指定 V、投影及最终目标时断言这是对最终答案的抑制。加性负 bias 降低分数也不是直接生成负 softmax 权重。
- 若面试官指某篇论文中的 signed attention、差分构造或其他同名方法，应要求给出公式并核对系数范围及归一化方式。这里纠正的是把原始 Transformer 的低权重一概称为 Negative Attention；不能据原论文未定义该组件断言所有研究中都不存在相关命名。

### 公式

```text
标准 softmax（dropout 前，每行至少一个允许位置）：若 j 被允许，A_ij=exp(S_ij)/Σ_{k∈允许位置}exp(S_ik)>0；被 mask 的 j 则 A_ij=0，Σ_jA_ij=1。softmax(s+c·1)=softmax(s)。O_i=Σ_j A_ijV_j 的分量不要求非负；例如 0.5·(-3)+0.5·1=-1。
```

### 易错点

- 把负 logit、接近零的非负权重与负 value/输出混为一谈。
- 为了纠正术语而断言全世界没有任何使用 Negative Attention 名称的论文或方法。

### 面试官可能追问

- 为什么把所有 logit 同时减去最大值不会改变注意力分布？
- 如果一种方法允许聚合系数为负，它还满足 softmax 概率权重的哪些性质？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [NEWTFM-P04 · PyTorch 2.14 — scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
