# 模型架构、MoE 与模型家族

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [架构范式与参数](#topic-1)
  - [TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？](#tfm-004)
  - [TFM-014 · LM head 如何产生 token 概率，输入输出权重共享有什么利弊？](#tfm-014)
  - [ARC-007 · Prefix LM 与 Causal LM 的 attention mask 有何区别，KV cache 有什么限制？](#arc-007)
  - [ARC-010 · 怎样由 config 估算 Transformer 参数量，12Ld² 为什么只是近似？](#arc-010)
  - [ARC-012 · 自回归与非自回归生成有什么区别，为什么并行输出可能牺牲质量？](#arc-012)
  - [ARC-015 · NAS 怎样搜索模型架构，DARTS 与权重共享有哪些取舍？](#arc-015)
- [预训练模型与家族对比](#topic-2)
  - [ARC-001 · BERT 的 MLM 与 NSP 怎么训练，15% 和 80/10/10 代表什么？](#arc-001)
  - [ARC-002 · BERT 的 token、segment、position embedding 为何相加？512 是数学上限吗？](#arc-002)
  - [ARC-003 · RoBERTa、ALBERT 与 SpanBERT 分别改进了 BERT 的什么？](#arc-003)
  - [ARC-004 · XLNet 的排列语言建模和双流注意力是什么，是否把输入词序打乱？](#arc-004)
  - [ARC-005 · GLM 的自回归空白填充、二维位置编码与 ChatGLM 系列怎样理解？](#arc-005)
  - [ARC-006 · LLaMA 1、Llama 2 与 Llama 3 的结构和训练配方有哪些关键变化？](#arc-006)
  - [ARC-011 · T5 与 BART 怎样做去噪预训练，与 BERT MLM 有何区别？](#arc-011)
  - [ARC-022 · DeepSeek-V2、V3、R1 有何区别？它们还是 Transformer 吗？](#arc-022)
  - [ARC-025 · 原始 Qwen3 文本模型相较 Qwen2.5 有哪些结构和后训练变化？](#arc-025)
- [MoE 路由、训练与压缩](#topic-3)
  - [TFM-015 · MoE 与 Dense 的参数量和计算量应怎样比较？](#tfm-015)
  - [ARC-008 · MoE 路由、Top-k、capacity factor 与 token dropping 分别做什么？](#arc-008)
  - [ARC-009 · MoE 的负载均衡损失与 router z-loss 有何区别？](#arc-009)
  - [ARC-013 · MoE 怎样用于模型压缩，WideNet 与 MoEBERT 的思路有什么不同？](#arc-013)
  - [ARC-014 · MoE 微调为什么容易过拟合，专家一定按语言或领域自动分工吗？](#arc-014)
  - [ARC-023 · DeepSeek-V3 的负载平衡、MTP 和 FP8 分别解决什么问题？](#arc-023)
- [推理模型与 MLA](#topic-4)
  - [ARC-021 · MLA 怎样压缩 KV Cache？和 MQA/GQA、RoPE 有什么关系？](#arc-021)
  - [ARC-024 · DeepSeek-R1-Zero、R1 与蒸馏模型的训练流程分别是什么？](#arc-024)

<a id="topic-1"></a>
## 架构范式与参数

<a id="tfm-004"></a>
### TFM-004 · Encoder-only、Decoder-only 与 Encoder-Decoder 怎样选择？

**L1** · 腾讯 / 小红书

#### 答案

理解与表示任务常从 Encoder-only 入手，自回归生成常用 Decoder-only，源输入与目标输出分工明确的条件生成可以用 Encoder-Decoder。三者区别主要在可见性与信息流，不是某类结构绝对只能做一种任务。BERT 类编码器通常双向看上下文；GPT 类解码器只读前缀，把问答、示例和输出放进同一序列。

Encoder-Decoder 先把源文本 x 编成 memory，再让解码器根据源信息与已生成目标逐步预测 y。公式中的概率乘积表示每一步都依赖目标历史和源表示；训练把正确目标右移，用 teacher forcing 并行计算各位置。解码器用自身状态做 Q、源 memory 做 K/V，能读取完整已知源文本，同时目标 self-attention 仍保持因果。源和目标可以不同长度，固定源的编码及各层 cross-attention K/V 可以复用。

原始 Transformer 两边各 6 层、使用 PostNorm，编码层有 self-attention 与 FFN，解码层还多 cross-attention；现代结构无需固定这些配置。Decoder-only 的优势是训练目标、连续文本数据和生成接口统一，也容易配合 KV cache；Encoder-Decoder 在长源输入、源表示复用或条件生成中仍有价值。Prefix LM 的分段可见性并不等于独立编码器。最终应按源目标长度、任务效果、训练数据、缓存与吞吐选型，训练并行也不能混同于生成步骤并行。

```math
\begin{aligned}H_{\rm src}&=\mathrm{Encoder}(x)\\p_\theta(y\mid x)&=\prod_t p_\theta(y_t\mid y_{\lt t},H_{\rm src})\\Q&=H_{\rm dec}W_Q,\quad K=H_{\rm src}W_K,\quad V=H_{\rm src}W_V\end{aligned}
```

#### 易错点

- 源和目标不必等长，源条件注意力也不必使用目标三角掩码。
- 生成未知目标时，直接双向读取目标会泄露未来答案。

#### 追问

- 源输入变长时，交叉注意力分数矩阵哪一轴改变？
- 固定源缓存与目标历史缓存的长度如何分别变化？

<a id="tfm-014"></a>
### TFM-014 · LM head 如何产生 token 概率，输入输出权重共享有什么利弊？

**L2** · 腾讯

#### 答案

LM head 把每个位置的隐藏表示映射成词表中所有 token 的分数，再用 softmax 得到下一 token 概率。若 H 是 B×T×D，输出矩阵为 D×V，logits 就是 B×T×V；V 是词表大小，softmax 必须沿 V 轴计算。logits 本身没有归一化，训练交叉熵通常直接接收它。

输入 embedding 表 E 是 V×D，维度与词表相容时，可以令输出矩阵等于 Eᵀ，这叫权重共享或 tying。原本两张矩阵变成一张，约省 VD 个参数，输入查表与输出分类的梯度共同更新它；bias 仍可单独存在，词表打分计算也不会消失。例如一个词既作为上下文输入，又作为预测目标，会从两个角色获得训练信号。两边词表或维度不同时则需要额外映射。

共享可能节省参数并改善统计利用，也可能限制两种角色的独立表达，要看任务。它不等于冻结，也不等于所有层和头都共享；ALBERT 的跨层共享是另外的专门设计，虽然省参数，多层计算仍要执行。实现可检查参数是否真正指向同一对象、state_dict 与词表尺寸。padding 行即使输入端不更新，也可能从输出分类端得到梯度，因此不能只凭输入 embedding 的 padding 设置断言这一行永远不变。

```math
\begin{aligned}Z&=HW_{\rm vocab}+b\\p_v&=\frac{\exp z_v}{\sum_u\exp z_u}\\W_{\rm vocab}&=E^\top,\quad \Delta N\approx VD\end{aligned}
```

#### 易错点

- 跨位置参数共享、输入输出 tying 和跨层共享是不同设计。
- CrossEntropyLoss 通常接收 logits；重复 softmax 或沿 hidden 轴归一都不正确。

#### 追问

- 输入 embedding 的 padding 行不更新时，输出共享分类器还可能给它梯度吗？
- ALBERT 共享参数省存储，为什么仍需要多层计算？

<a id="arc-007"></a>
### ARC-007 · Prefix LM 与 Causal LM 的 attention mask 有何区别，KV cache 有什么限制？

**L2**

#### 答案

Causal LM 与 Prefix LM 都能给定前缀后生成，区别是前缀内部能否双向读取。Causal LM 的每个位置只看自身和之前位置；Prefix LM 的固定前缀内部彼此可见，生成部分能看完整前缀及此前生成的后缀。普通 GPT 接 prompt 生成，并不因此变成 Prefix LM。

设前缀结束位置为 p，公式允许前缀查询读取 j≤p，后缀查询读取前缀和 j≤i 的历史，其他位置加负无穷。例如已知一段问题，Prefix LM 可让问题开头的表示读到问题结尾，再生成答案；Causal LM 中开头的表示不能读结尾，但答案位置仍能读取完整问题。loss 是否只计算答案，是监督范围的选择，与这两种 attention mask 不等价。

固定前缀计算完成后，两者都能缓存其 K/V，再增量生成后缀。但若向双向前缀增加仍属于前缀的新内容，旧位置表示可能因新上下文改变，通常需要重算对应缓存；因果前缀的旧位置不依赖后面新增内容，更容易复用。训练还要指定前缀划分、位置编号和损失范围，实际选型应匹配预训练目标及条件生成任务，而不只根据是否有 prompt 判断。

```math
M_{ij}=\begin{cases}0,&i\le p,\ j\le p\\0,&i\gt p,\ (j\le p\text{ or }j\le i)\\-\infty,&\text{otherwise}\end{cases}
```

#### 易错点

- 因果语言模型同样能读取给定 prompt 并生成后缀。
- 双向前缀内容改变后，旧前缀缓存可能不再对应新表示。

#### 追问

- 前缀固定后，Prefix LM 为什么仍能增量缓存生成后缀？
- 前缀双向掩码与只计算回答损失，为什么不是同一操作？

<a id="arc-010"></a>
### ARC-010 · 怎样由 config 估算 Transformer 参数量，12Ld² 为什么只是近似？

**L2**

#### 答案

参数量应从具体矩阵形状累加，12Ld² 只是标准多头注意力配普通 FFN 的粗估。标准 Q/K/V/O 四张矩阵约 4d²，若 FFN 中间宽度是 4d，两张矩阵再有 8d²，所以单层约 12d²，L 层后还要加词表、归一化和 bias。

公式中 h_q 是查询头数、h_kv 是 KV 头数、d_h 是每头宽度。Q 输出 h_qd_h，K/V 各输出 h_kvd_h，O 再映射回 d，注意力权重总量约为 2d(h_q+h_kv)d_h。GQA 少一些 KV 头，这部分会变小；SwiGLU 用三张 FFN 矩阵，参数约 3dd_ff；MoE 则要计全部专家和 router，而不是只计这次选中的专家。例如头数保持不变、词表翻倍，主体层参数不变，但向量表成本会翻倍。

输入输出共享时，n_table 只计一张独立表；跨层共享也要按独立存储去重，不能看 state_dict 名称就重复累加。最终可检查配置和实际张量形状核实。参数数、文件大小、训练显存和 FLOPs 是不同指标：量化还包含 scale 等元数据，训练还存梯度、优化器和激活，因此“参数乘几个字节”只能估算部分成本，不能直接当部署或训练内存。

```math
\begin{aligned}P_{\rm attn}&=2d(h_q+h_{\rm kv})d_h\\P_{\rm FFN}&=2dd_{\rm ff}\ (\mathrm{MLP}),\quad3dd_{\rm ff}\ (\mathrm{SwiGLU})\\P_{\rm total}&\approx L(P_{\rm attn}+P_{\rm FFN})+n_{\rm table}|V|d+P_{\rm other}\end{aligned}
```

#### 易错点

- 12Ld² 的形状假设不适用于任意 GQA、SwiGLU 或 MoE。
- 共享张量在多个路径出现时，只应计一次独立参数。

#### 追问

- MoE 的总容量与每 token 激活参数，怎样分别估算？
- 4 bit 量化文件为什么还要计 scale 等额外存储？

<a id="arc-012"></a>
### ARC-012 · 自回归与非自回归生成有什么区别，为什么并行输出可能牺牲质量？

**L2**

#### 答案

自回归生成一步依赖前面已生成内容，非自回归生成尝试同时预测多个位置，以减少串行等待。自回归的概率分解本身是精确链式法则；最简单的非自回归模型先预测长度 T，再假设各位置在给定输入和长度后独立，这是一种更强的近似。

例如一句话有两种合理表达：“今天去学校”和“今天在家学习”。各位置独立选高概率词时，可能把不同表达拼成不一致的结果；问题不是每个位置都错，而是没有充分协调它们。非自回归方法因此常加入潜变量、词数分配、教师蒸馏，或者多轮遮蔽再修正、插入与编辑，让位置之间重新协调。这些方案也不等于一次前向精确采样原自回归模型的联合分布。

自回归模型训练时有正确历史，可通过 teacher forcing 并行计算整句各位置损失，常规推理才逐 token 进行。非自回归虽然减少轮数，仍可能有长度预测、编码和多轮修正成本。比较要报告相同任务质量、输出长度、迭代数、batch 与硬件下的端到端延迟。固定格式、低延迟任务可能接受适当质量折中，自由文本则更依赖强输出相关性，不能只看“并行”二字判断优劣。

```math
p_{\rm AR}(y\mid x)=\prod_t p(y_t\mid x,y_{\lt t}),\qquad p_{\rm NAT}(y\mid x)=p(T\mid x)\prod_{t=1}^{T}p(y_t\mid x,T)
```

#### 易错点

- 自回归训练可以并行，不等于常规自回归生成可以跳过输出依赖。
- 简单非自回归独立假设通常不与原自回归联合分布精确等价。

#### 追问

- 同一输入有多种合理表达时，独立预测为什么容易拼出不一致结果？
- 多轮修正生成应怎样比较轮数、质量和端到端延迟？

<a id="arc-015"></a>
### ARC-015 · NAS 怎样搜索模型架构，DARTS 与权重共享有哪些取舍？

**L2**

#### 答案

NAS 是自动搜索模型结构，需要先明确允许改变什么、怎样找候选、怎样评价候选。搜索空间可以包含层数、宽度、算子和连接方式，搜索策略可用随机、进化、强化学习或梯度方法；固定结构下只调学习率通常属于超参数优化，两者可以联合但概念不同。

DARTS 把一条边上的多个候选算子用 softmax 权重混合，得到可微结构，再学习模型权重 w 和架构权重 α。公式的内层用训练集拟合 w，外层用验证集选 α，最后再把混合结构离散化。比如同一位置同时候选卷积与跳连，搜索阶段可给它们不同份额，最终只保留选中的操作。混合结构的分数却不等于离散结构的最终效果。

权重共享的 supernet 让多个候选借用同一组训练权重，减少逐个从头训练的成本，但候选之间会相互影响，共享状态下的排名可能与独立训练不一致。最终模型应按统一协议重训或校准，再用独立测试集评测，并计入搜索和重训总成本。若目标是某 GPU 的延迟或显存，应直接测目标设备或用可靠约束模型；FLOPs 与真实时延不一一对应，不能只靠一个代理数字声称找到了最优结构。

```math
\min_{\alpha}\mathcal L_{\mathrm{val}}(w^*(\alpha),\alpha),\quad w^*(\alpha)=\arg\min_w\mathcal L_{\mathrm{train}}(w,\alpha),\qquad \bar o(x)=\sum_{o\in\mathcal O}\mathrm{softmax}(\alpha)_o\,o(x)
```

#### 易错点

- 连续混合架构的验证分数，不等于离散候选重训后的测试分数。
- 不要反复用测试集挑结构，搜索与重训算力也要计入成本。

#### 追问

- 多个候选共享训练权重，为什么可能改变独立训练后的排名？
- 怎样把目标 GPU 的延迟和显存约束加入搜索？

<a id="topic-2"></a>
## 预训练模型与家族对比

<a id="arc-001"></a>
### ARC-001 · BERT 的 MLM 与 NSP 怎么训练，15% 和 80/10/10 代表什么？

**L1** · 小红书

#### 答案

BERT 通过遮蔽词预测学习双向上下文，通过下一句预测学习句对关系。MLM，就是遮蔽语言模型，先选择约 15% 的 WordPiece 位置作为预测目标；在这些被选位置里，80% 换成 MASK，10% 换成随机词，10% 保留原词。因此不是全部 15% 都变成 MASK，三种情况都要预测原来的 token。

例如“我今天喝咖啡”中选中“咖啡”，模型可能看到 MASK、另一个词或原词，再结合左右内容预测“咖啡”。混入随机词和保留原词，是为了减轻预训练常有 MASK、实际任务通常没有 MASK 的差异，不是一个理论上最优的固定比例。公式里的 M 是被选目标位置，波浪号 x 是扰动后的输入；未被选位置不直接计算 MLM 损失，但可以作为上下文。这里的“遮蔽”是换输入内容，注意力仍然双向，与阻止看未来的因果掩码不同。

NSP 把约一半句对设成真实相邻片段，另一半设成随机片段，用 CLS 表示判别是否接续，再与 MLM 损失组合。CLS 经过任务训练可以汇总信息，但不天然是最好的通用句向量。BERT 可接分类、实体标注或问答起止位置头，原始双向 MLM 却不能直接等价当作左到右生成器；后续模型去掉 NSP，也需要结合新的数据组织和训练配方评价。

```math
\mathcal L_{\rm MLM}=-\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x),\qquad\mathcal L_{\rm BERT}=\mathcal L_{\rm MLM}+\mathcal L_{\rm NSP}
```

#### 易错点

- 15% 是被选作预测目标的比例，其中只有约八成换成 MASK。
- MLM 扰动输入 token，不能用因果 attention mask 代替。

#### 追问

- 被选位置里保留原词的 10%，为什么不等于取消整体去噪任务？
- 后续模型去掉 NSP 后，怎样仍学习有用的文本表示？

<a id="arc-002"></a>
### ARC-002 · BERT 的 token、segment、position embedding 为何相加？512 是数学上限吗？

**L2**

#### 答案

BERT 把词、句段和位置这三类信息相加，是为了让每个输入位置同时携带“是什么词、属于哪段、在什么位置”，并保持隐藏维度不变。原始输入由 WordPiece 向量 E、A/B 句段向量 S 和可学习位置向量 P 相加，再做 LayerNorm 与 dropout；它们都在同一个 d 维空间中共同学习。

比如句对任务里，“问题”和“候选答案”可以使用不同 segment 标记，SEP 负责分隔，CLS 常用于句级输出；单句任务通常所有 token 使用同一种 segment。相加不会强迫网络把三种信息混为一谈，参数和后续层可以学习如何使用各部分。它与 RoPE 在 Q/K 上旋转的设计不同，不能把两者的实现方式互换着描述。

512 是原始 checkpoint 的位置表与训练长度配置，不是 Transformer 的数学上限，也不是 512 个中文词。超长输入可以截断、滑窗分块后聚合，或扩位置表示并继续训练，直接补随机位置不保证效果。问答分块要保留 token 到原文的偏移，设置重叠并处理跨块答案；句向量还要验证池化与训练目标。选择方法时也要计算长序列注意力和激活成本，程序接受更长输入并不等于模型已能有效使用它。

```math
h_i^{(0)}=\mathrm{LN}(E_{x_i}+S_{a_i}+P_i),\qquad E_{x_i},S_{a_i},P_i\in\mathbb R^d
```

#### 易错点

- 512 是原 checkpoint 的配置，不是 Transformer 的理论长度上限。
- token 数不等于空格词数，也不总等于汉字数。

#### 追问

- 单句任务不需要区分 A/B 时，怎样处理 token-type 向量？
- 滑窗问答的答案跨块时，怎样设置重叠和合并结果？

<a id="arc-003"></a>
### ARC-003 · RoBERTa、ALBERT 与 SpanBERT 分别改进了 BERT 的什么？

**L2**

#### 答案

RoBERTa 主要改善训练配方，ALBERT 主要减少独立参数，SpanBERT 主要加强连续片段表示。它们都从 BERT 的双向编码思路出发，但解决的瓶颈不同，不能把提升统统归因于同一个新结构。

RoBERTa 使用更多数据与训练、更大 batch、动态遮蔽，去掉 NSP，并采用 byte-level BPE。它提醒我们，比较架构时也要匹配训练条件。ALBERT 先把词映射到较小的 e 维，再投影到 d 维隐藏空间，embedding 参数从 |V|d 变为 |V|e+ed；还可在深度方向共享 Transformer 参数。它的 SOP 把连续片段顺序颠倒作为负例，让两个片段仍属于同一主题，减少只靠主题差异完成 NSP 的捷径。

SpanBERT 遮蔽连续 span，并让两端表示加内部相对位置预测被遮蔽 token，迫使边界包含片段信息。例如问答要抽取一整段人名或日期，这种目标比只记单个位置更贴近跨度任务。ALBERT 共享参数省存储，但仍要执行多层计算，不代表 FLOPs 同比下降；SpanBERT 也不同于 T5 自回归生成被删片段。最终应按参数、算力、数据和任务共同比较。

```math
P_{\rm embed}^{\rm BERT}=|V|d,\qquad P_{\rm embed}^{\rm ALBERT}=|V|e+ed,\quad e\ll d
```

#### 易错点

- ALBERT 共享参数减少存储，不保证计算量同比下降。
- 去掉 NSP 的模型也可能有不同替代目标和训练数据组织。

#### 追问

- SOP 使用同一主题的倒序片段，为什么比随机主题负例更难取巧？
- 从片段两端预测内部内容，为什么有助于跨度表示？

<a id="arc-004"></a>
### ARC-004 · XLNet 的排列语言建模和双流注意力是什么，是否把输入词序打乱？

**L2**

#### 答案

XLNet 的排列语言建模改变的是预测位置的先后顺序，不是把句子的词序真的打乱。它对多种概率因子分解顺序训练，让同一个位置有机会利用左边和右边内容，而位置本身仍对应原句。例如原句位置是 1、2、3，某次可以按 3、1、2 预测，但词仍保留原位置关系。

公式中的 z 是一个位置排列，x_z_t 是这一轮待预测的词，条件只含排列中已经出现的内容。若预测位置的表示直接包含它自己的 token，模型会看见答案。因此 XLNet 使用双流注意力：content stream 携带位置与内容，供后续已知上下文使用；query stream 只携带目标位置信息，读取已知位置的 content，预测时不读自己的真实内容。两条流的区别是为避免标签泄露，而不是两套完全无关的语言模型。

XLNet 还继承 Transformer-XL 的跨片段记忆与相对位置设计，旧段状态常停止梯度，在新段中作为额外上下文。它与生成时不断追加历史 K/V 的缓存有不同训练语义。排列采样、只预测部分位置与记忆长度都是计算折中；这种目标改善一些 MLM 限制，却不自动解决所有长依赖或生成质量问题。实现时首先要验证两条流的可见关系和目标是否泄露。

```math
\mathcal L=-\mathbb E_{z\sim\mathcal Z_T}\sum_{t=1}^{T}\log p_\theta(x_{z_t}\mid x_{z_{\lt t}})
```

#### 易错点

- 排列语言建模改变概率分解顺序，不是打乱原始文本位置。
- 预测目标的 query stream 不能读取该位置真实 token 内容。

#### 追问

- 双流里 content 与 query 各能读取哪些位置与内容？
- 跨段记忆持续使用旧内容时，为什么相对位置更合适？

<a id="arc-005"></a>
### ARC-005 · GLM 的自回归空白填充、二维位置编码与 ChatGLM 系列怎样理解？

**L2**

#### 答案

原始 GLM 通过自回归填补文本中的空白，把双向理解和条件生成结合起来。它删去若干连续片段，用 mask 标记缺口，让未删文本彼此双向可见；生成缺失片段时，则可以读取未删文本及此前生成内容。它的可见性设计比普通因果语言模型前面加一条 prompt 更具体。

例如原文“他在北京工作”删掉“北京”，模型读取“他在 MASK 工作”，再逐步生成缺失内容。二维位置分别表示原文缺口的锚点，以及这个缺失片段内部生成到了第几步。公式里的 A 是已知部分、S 是待生成片段，S_j 依赖 A 和之前生成的 S；anchor 定位空白，j 定位片段内部。训练还可改变多个 span 的生成顺序，使模型学习不同填空条件。

ChatGLM 是沿 GLM 思路发展的对话系列，但各代架构与训练有变化，不能把初代配置套到全部名字上。初代 ChatGLM-6B 侧重中英对话，ChatGLM2 引入 MQA 等效率改动，ChatGLM3 扩展工具和代码相关能力。实际比较要明确 checkpoint，并查看 tokenizer、配置和注意力实现，确认二维位置、激活与 mask 是否还采用原方案；模型家族名本身不足以推断计算公式。

```math
p_\theta(S\mid A)=\prod_{j}p_\theta(S_j\mid A,S_{\lt j}),\qquad\mathrm{pos}(S_j)=(\mathrm{anchor}(S),j)
```

#### 易错点

- 初代 ChatGLM 的二维位置或组件不能套到全部后续版本。
- GLM 填空中未删文本可双向读取，不能全部描述成普通因果掩码。

#### 追问

- 多个缺失片段的生成顺序，可以怎样区别于原文顺序？
- 怎样从 attention 实现确认模型采用因果或前缀可见性？

<a id="arc-006"></a>
### ARC-006 · LLaMA 1、Llama 2 与 Llama 3 的结构和训练配方有哪些关键变化？

**L2**

#### 答案

LLaMA 1、Llama 2 和 Llama 3 都以自回归 Decoder-only Transformer 为基础，区别既包括注意力和 tokenizer，也包括数据与后训练，不能只用一个模块解释能力变化。它们常见的组合是 Pre-Norm、RMSNorm、RoPE 和 SwiGLU：先控制尺度，用旋转位置表示顺序，再用门控前馈层加工特征。

LLaMA 1 使用 SentencePiece tokenizer，重视较充分的 next-token 预训练。Llama 2 增加数据与上下文长度，并提供经过监督微调、偏好训练的 Chat 版本；GQA 只用于其中较大型号，原始 7B、13B 与 70B 的 KV 头配置不能混为一谈。2024 年首发的 Llama 3 8B/70B 使用约 128K 词表、GQA 和更大规模的精细处理数据；首发窗口为 8K，后来的 Llama 3.1 等才扩到 128K。

更大词表可以把某些语言切得更短，却增加约 |V|d 的向量表成本；公式中的 KV 缓存随层数 L、长度 T、KV 头数 n_kv 和每头宽度 d_h 增长，所以 GQA 能减少这一部分负担。效果还取决于混合数据、去重和后训练。回答具体型号时，应以该 checkpoint 的配置为准，并分开说明结构、训练长度和对话能力，不能按家族名给所有版本同一组数字。

```math
P_{\rm vocab}=|V|d\quad(\text{one table}),\qquad M_{\rm KV}\propto L\,T\,n_{\rm kv}\,d_h
```

#### 易错点

- Llama 2 不同尺寸的 KV 头配置不同；首发 Llama 3 窗口也不是后来版本的 128K。
- 对话后训练和主体架构改动，需要分别说明。

#### 追问

- 词表扩大为什么同时可能缩短序列、增加参数与输出计算？
- 怎样从具体型号 config 核实查询头和 KV 头数量？

<a id="arc-011"></a>
### ARC-011 · T5 与 BART 怎样做去噪预训练，与 BERT MLM 有何区别？

**L2**

#### 答案

T5 和 BART 都把受扰动文本交给编码器，再让自回归解码器生成目标；BERT MLM 则主要在输入中被选位置直接分类预测原 token。它们都利用去噪，但监督的位置、输出格式和目标长度不同，不能因为都有 mask 就当作同一种任务。

T5 常把连续片段替换成不同 sentinel，也就是标记各缺口的特殊 token，目标按顺序生成这些标记和对应缺失片段。比如“我住在北京，喜欢咖啡”删去两个片段，目标只串起被删内容与标记。BART 则从扰动后的文本重建完整原文，研究过片段填空、句子顺序打乱等噪声。公式中的 corrupt(x) 是坏掉的输入，y 是对应模型规定的目标，每一步读取目标历史与编码器信息。

BERT 的双向编码器可以融合两侧上下文，但原始 MLM 没有像 BART 那样的自回归目标解码器，不能直接等价用于持续生成。T5/BART 适合翻译、摘要等输入输出分离任务，并能缓存固定源表示和目标历史。遮蔽比例、平均片段长度和 sentinel 数都会影响目标长度及成本，实际比较应同时考虑质量与计算，Decoder-only 的普及并不意味着这些结构失去价值。

```math
\mathcal L_{\rm denoise}=-\sum_{t=1}^{|y|}\log p_\theta(y_t\mid y_{\lt t},\mathrm{corrupt}(x))
```

#### 易错点

- T5 生成缺失片段，BART 通常重建完整文本，目标格式不同。
- 编码器—解码器也能复用源表示和目标历史缓存。

#### 追问

- 遮蔽比例固定时，改变平均片段长度怎样影响 sentinel 数与目标长度？
- BART 的生成解码器与原始 BERT 的 MLM 分类头有什么不同？

<a id="arc-022"></a>
### ARC-022 · DeepSeek-V2、V3、R1 有何区别？它们还是 Transformer 吗？

**L2**

#### 答案

DeepSeek-V2、V3 和原始 R1 仍属于自回归 Transformer 家族，区别主要在效率结构与训练路线，而不是另起一种完全无关的网络。它们保留因果注意力、残差和前馈模块，MLA 改变注意力的键值表示，MoE 改变前馈计算的分配方式。这里比较的是 2024 年 V2、2024 年底 V3 与 2025 年初 R1 的原始版本。

V2 结合 MLA 和 DeepSeekMoE，减少 KV 缓存与单 token 激活计算，同时保留较大总容量。V3 延续这两种结构，加入动态负载平衡、多 token 预测训练，以及 FP8 混合精度和计算通信重叠等效率设计；原始 V3 总参数约 671B、每 token 激活约 37B。37B 是当次计算使用量，并不意味着部署只需要存这部分权重。

R1 的重点是基于 V3-Base 的推理后训练。R1-Zero 探索直接做推理强化学习，R1 加入冷启动和筛选数据等阶段，改善推理与可读性。R1-Distill-Qwen、R1-Distill-Llama 则是其他底座学习教师输出的学生，名称里有 R1 也不自动继承教师 MLA 或 MoE。例如蒸馏 14B 模型的注意力结构应查它自己的 config。比较时应同时明确 checkpoint、底座、结构、后训练和部署成本，避免只报模型家族名。

#### 易错点

- DeepSeek 家族仍可采用 Transformer；R1 蒸馏学生不能按教师结构描述。
- 激活参数只表示当次调用量，不能当成全部权重容量。

#### 追问

- Dense 与 MoE 应怎样分别比较容量、FLOPs、通信与常驻内存？
- R1-Distill-Qwen-14B 的注意力结构，应该查哪一个 checkpoint 配置？

<a id="arc-025"></a>
### ARC-025 · 原始 Qwen3 文本模型相较 Qwen2.5 有哪些结构和后训练变化？

**L2**

#### 答案

原始 Qwen3 文本系列延续 Transformer 主体，并同时调整注意力稳定性、专家设置与后训练，使同一模型能兼顾思考和快速回答。这里指 2025 年 4 月发布、5 月技术报告中的版本，后续带日期或 Instruct、Thinking 后缀的 checkpoint 需要另看配置。

Dense 版继续使用 GQA、SwiGLU、RoPE 与 Pre-RMSNorm，去掉 QKV bias，并加入 QK-Norm，也就是在匹配前归一化查询和键的特征，帮助控制注意力分数尺度。MoE 版有 128 个路由专家，每 token 选 8 个，不沿用 Qwen2.5-MoE 的共享专家设置；总容量与激活计算仍要分开。例如专家很多并不意味着每个 token 都执行全部专家，也不意味着部署只保存选中的八个。

旗舰后训练依次包含长推理冷启动、推理 RL、思考模式融合与通用 RL。模式融合用数据和模板让同一 checkpoint 学会思考与非思考，也能结合推理预算控制，小模型还可从强模型蒸馏。原始文本模型用一维 RoPE，没有图像的时间、高、宽网格；Qwen3-VL 的位置和视觉结构是额外设计。多模态能力需要视觉编码器、连接器、位置方案与相应训练，不能靠改名称或随意加入图像向量得到。

#### 易错点

- 文本 Qwen3 的 QK-Norm 与思考模式，不等于 Qwen3-VL 的视觉位置结构。
- 不同发布日期的 Qwen3 checkpoint，思考模式行为可能不同。

#### 追问

- 归一化 Q/K 后，注意力分数尺度怎样变化？
- 思考模式的正确率收益，应怎样与额外 token 和延迟一起比较？

<a id="topic-3"></a>
## MoE 路由、训练与压缩

<a id="tfm-015"></a>
### TFM-015 · MoE 与 Dense 的参数量和计算量应怎样比较？

**L2** · 阿里巴巴

#### 答案

MoE 用很多专家扩展模型总容量，但每个 token 只调用少数专家，因此总参数量和每步实际计算要分开比较。常见做法是把 FFN 换成专家集合，由 router，也就是路由网络，给专家打分，选 top-k 后把它们的输出加权汇总；注意力与其他共享部分通常仍是稠密计算。

如果共享部分参数为 P_shared，每个专家为 P_exp，共 E 个专家，总参数约为 P_shared+E P_exp，单 token 激活约为 P_shared+k P_exp，路由与其他开销另外计算。比如 8 个专家每次选 2 个，不代表整个模型计算恰好是同总参数稠密模型的四分之一，因为注意力、共享层、通信与数据搬运都没有按同样比例减少。总参数仍要存储，训练的梯度和优化器状态也不会凭空消失。

收益还取决于路由能否均衡使用专家。容量限制可能让过载专家丢弃或重分配 token，辅助损失与路由正则帮助稳定分配；分布式专家通常需要 all-to-all 通信，专家 batch 太小又影响矩阵效率。Switch 常用 top-1，Mixtral 使用 top-2 等配方，各模型还有共享专家、归一化权重和容量策略差异。比较 Dense 与 MoE，应统一训练数据、预算和硬件，分别报告总参数、激活参数、FLOPs、显存及延迟，不能把激活参数直接当作实际速度。

![MoE 单 token 的 Top-2 专家路由](../assets/moe-routing.svg)

示例为 4 个专家中激活 2 个；真实系统还需要处理负载、容量与通信。

#### 易错点

- 总参数更大不必然更慢，激活参数更少也不必然更快。
- 共享模块、路由、数据分发和通信成本都要纳入比较。

#### 追问

- 均衡专家负载与让专家形成有用特化，为什么可能发生冲突？
- 理论 FLOPs 与真实 GPU 时间和成本，应怎样分别报告？

<a id="arc-008"></a>
### ARC-008 · MoE 路由、Top-k、capacity factor 与 token dropping 分别做什么？

**L2** · 阿里巴巴

#### 答案

MoE 路由决定一个 token 使用哪些专家，Top-k 控制调用数量，capacity 控制每个专家能接收多少分支，token dropping 处理超出容量的分支。专家通常是多个 FFN，router 根据 token 隐状态打分，选中专家计算后按门权重汇总。Switch 常用 Top-1，其他方案可选多个专家；选中概率是否重新归一化也因实现而异。

公式中的 T 是本次路由的 token 数、E 是专家数、k 是每 token 分支数，Tk/E 是平均分支负载，capacity factor 乘它后得到每专家预算 C。比如 100 个 token 选一个专家、共 10 个专家，平均每个接收 10 个；factor=1.2 时容量约为 12，但热门专家仍可能超过。传统实现会丢弃溢出专家分支、回退或保留残差，这不是把该 token 从原序列删掉。提高容量能减少溢出，却增加缓冲区和计算。

Dropless 方案避免丢分支，但仍要处理负载不均、显存和调度；某专家过热可能让其他设备等待。Top-k 的离散索引选择不是处处可微，路由学习通常通过被选门权重和辅助目标获得梯度。选型要同时看主任务质量、负载、溢出率、通信与吞吐；总专家参数表示存储容量，不等于每 token 的激活计算。

```math
\begin{aligned}p(x)&=\mathrm{softmax}(W_rx)\\y&=\sum_{e\in\mathrm{TopK}(p,k)}\tilde p_e(x)E_e(x)\\C&=\left\lceil\mathrm{capacity\_factor}\cdot\frac{Tk}{E}\right\rceil\end{aligned}
```

#### 易错点

- 总专家参数不能直接推导单 token FLOPs，共享部分也要计算。
- 溢出通常丢专家分支，不是从序列中删除该 token。

#### 追问

- 不丢分支的 MoE 为什么仍可能有过载专家和等待？
- Top-1 保留原 gate 概率与重归一到 1，尺度和梯度有什么差别？

<a id="arc-009"></a>
### ARC-009 · MoE 的负载均衡损失与 router z-loss 有何区别？

**L3** · 阿里巴巴

#### 答案

负载均衡损失鼓励 token 分散到专家，router z-loss 控制路由分数的数值尺度，二者解决不同问题。前者避免少数专家拥堵、其他专家闲置，使容量和通信更可控；后者有助于减少路由 logits 过大带来的不稳定，并不直接要求专家使用量均匀。

Switch 的均衡项把每个专家实际接收比例 f_e，与全体 token 给它的平均概率 P_e 相乘后求和。E 是专家数，α 是损失强度，f 来自离散选择，梯度主要经过 P。它会抑制概率过分集中，但 α 太大也可能压制对某类输入合适的专家特化。这里 f 的公式按 Top-1 定义，改成 Top-k 时必须重新明确计数与归一化，不能直接混用其他均衡公式。

ST-MoE 的 z-loss 对每个 token 的 logsumexp 平方加惩罚，β 控制强度，z 是 router logits。把一行 logits 同时加常数，softmax 概率不变，logsumexp 却会变化，正说明它与负载分布不是同一目标。训练应同时记录主损失、专家负载、溢出率和 logits 尺度，必要时用 FP32 做敏感的路由归一化；辅助损失低不等于质量好，合理均衡与有用分工之间仍需任务验证。

```math
\begin{aligned}\mathcal L_{\rm balance}&=\alpha E\sum_{e=1}^{E}f_eP_e\\f_e&=\frac1T\sum_t\mathbf1\{\mathrm{route}(t)=e\},\quad P_e=\frac1T\sum_t p_e(x_t)\\\mathcal L_z&=\frac\beta T\sum_t\left(\log\sum_e e^{z_{t,e}}\right)^2\end{aligned}
```

#### 易错点

- z-loss 控制路由数值尺度，与均衡专家使用量不是同一个目标。
- Top-1 的接收比例公式改成 Top-k 时，要重新明确计数与归一化。

#### 追问

- 同一行 logits 加常数不改变 softmax，为什么仍会改变 z-loss？
- 过强均衡约束为什么可能妨碍有用的专家特化？

<a id="arc-013"></a>
### ARC-013 · MoE 怎样用于模型压缩，WideNet 与 MoEBERT 的思路有什么不同？

**L2**

#### 答案

MoE 可以减少每个 token 使用的计算，也能与共享参数、蒸馏结合做压缩，但稀疏调用不自动让全部权重更小。WideNet 主要是在深度方向复用参数、在宽度方向引入专家容量，MoEBERT 则从已有 BERT 适配出较小专家并蒸馏知识，压缩路径不同。

WideNet 共享 Transformer 参数，仍让不同深度使用独立归一化参数，使各层能在共享计算基础上保留一些适应性。可以把它理解为同一套加工设备重复使用，各步仍能调整自己的尺度。MoEBERT 根据 FFN 神经元的重要性组织多个较小专家，再用层级蒸馏保留教师的表示与行为；推理只选一个专家，降低激活计算。它不是独立训练多个完整 BERT 后投票。

评价时，应分别算总参数、单 token 激活参数、常驻权重显存、路由开销和真实延迟。共享减少独立存储，路由减少调用范围，这两种机制不能混为一谈；专家太小、batch 太小或数据被路由得很散时，调度开销可能抵消 FLOPs 收益。公平比较还需要在相近质量下，使用相同数据、蒸馏预算和硬件，而不能只凭理论稀疏比例判断压缩是否成功。

#### 易错点

- 激活计算减少，不代表全部专家权重存储也减少。
- 跨层共享复用参数，token 路由选择计算，两者不是同一机制。

#### 追问

- 主体参数共享、各层 Norm 独立，怎样帮助不同深度适应？
- 比较 Dense 与 MoE 学生蒸馏时，哪些预算和质量条件要对齐？

<a id="arc-014"></a>
### ARC-014 · MoE 微调为什么容易过拟合，专家一定按语言或领域自动分工吗？

**L2** · 阿里巴巴

#### 答案

MoE 微调可能更容易过拟合，因为专家容量大，而小数据集给各专家的更新既少又不均，路由分布也可能从预训练状态明显改变。表现通常是训练很快变好，验证改善有限；这时既要看训练验证差距，也要看专家负载与跨域效果，不能只盯主损失。

我会先检查数据与评测，再比较学习率、训练步数、batch 和专家内部 dropout，并尝试冻结部分专家、更新共享子层或全量更新。均衡损失帮助分配计算，z-loss 控制路由尺度，但都不能代替泛化控制。Switch 与 ST-MoE 对 dropout 和参数子集微调有不同结果，说明这些是模型与任务相关的选择，不能变成“永远冻结专家”或“只更新专家就最好”的规则。expert dropout 也常指专家内部隐藏激活的随机屏蔽，不一定是删除整个专家。

专家不天然对应人定义的语言或领域。某些研究观察到编码器专家的句法或语义偏好，解码器却未必相同，多语言输入也可能共享专家。例如某专家频繁接收中文标点，只说明路由偏好，不能立即称为中文理解专家。应结合 token 类型统计、负载、消融影响和跨域泛化，区分有用特化、路由偏置与少数专家垄断的塌缩。

#### 易错点

- expert dropout 常是内部激活屏蔽，不一定随机删除整个专家。
- 专家不天然对应人类领域，增加专家数也不保证下游效果更好。

#### 追问

- 为什么只更新专家可能不如更新少量共享或非专家参数？
- 怎样用负载与消融区分有用特化、路由偏置和专家塌缩？

<a id="arc-023"></a>
### ARC-023 · DeepSeek-V3 的负载平衡、MTP 和 FP8 分别解决什么问题？

**L3**

#### 答案

DeepSeek-V3 的负载平衡解决专家拥堵，MTP 增加未来 token 的训练信号，FP8 降低部分计算与通信成本，三者针对不同瓶颈。专家负载不均时，一些设备忙、另一些等待，所以 V3 给路由选择分数加入动态偏置，过载专家偏置降低、欠载专家提高，帮助分散分配。

这个偏置用于 top-k 选择，实际汇总专家输出的权重仍来自原始亲和分数，使平衡调度与内容权重分开。它减少对强均衡损失的依赖，却仍使用很小的序列级辅助均衡项，因此不能把 auxiliary-loss-free 理解成总训练目标完全没有辅助项。MTP，即多 token 预测，在主下一词目标外增加模块预测更后的词，让表征获得更长预测关系；推理可移除这些模块，也可作为投机解码的草稿，但最终仍需验证，提速取决于接受率和实现。

FP8 用 8 位浮点格式提高部分矩阵运算和传输效率，配合细粒度 scale 保持数值范围；敏感操作仍使用较高精度，不能说权重、梯度、优化器和归约全是 FP8。MoE 通信、DualPipe 调度及计算通信重叠也影响最终效率。实际复现要分别监控数值误差、路由负载与等待时间，打开一个低精度开关并不等于获得整套系统收益。

#### 易错点

- auxiliary-loss-free 不代表总训练目标完全没有辅助项。
- MTP 预测草稿不等于无需验证就一次输出多个最终 token。

#### 追问

- 路由选择偏置与实际专家聚合权重分开，有什么作用？
- 怎样分别监控 FP8 数值误差、溢出和通信等待？

<a id="topic-4"></a>
## 推理模型与 MLA

<a id="arc-021"></a>
### ARC-021 · MLA 怎样压缩 KV Cache？和 MQA/GQA、RoPE 有什么关系？

**L3**

#### 答案

MLA 通过为每个 token 缓存一份低维的键值联合表示，减少保存所有头 K/V 的成本。MQA、GQA 是让多个查询头共享较少 KV 头，MLA 则从共同的 latent，也就是压缩向量 c，映射出各头不同的内容键值，因此不能把它简单等同于一个普通 KV 头。

公式中的 W_D 把隐藏状态 h 压到 d_c 维，W_UK 和 W_UV 再产生各头内容 K/V。推理时可以把内容 key 的上投影移到查询侧：qᵀW_UK c=(W_UKᵀq)ᵀc；value 的上投影也可与输出投影结合，于是直接在 latent 空间汇总，不必长期缓存展开后的每头键值。但若普通 RoPE 的位置旋转夹在投影之间，矩阵一般不能交换，这种吸收就被阻碍。DeepSeek 因此把内容与位置分开，再缓存一份共享的 RoPE key。

若 B 是序列数、L 是层数、T 是长度、s 是每元素字节数，理想缓存约 BLT(d_c+d_R)s，d_R 是额外位置键宽度；标准 MHA 约为 2BLTHd_hs。块表、量化元数据和工作区另算。缓存减少有利于并发和带宽，实际速度仍取决于专用内核与并行布局；朴素展开实现可能产生额外工作区。这是训练过的结构设计，不是给任意已训练 MHA 缓存做无损压缩。

```math
\begin{aligned}c_t&=W_Dh_t,\quad k_{t,i}^{C}=W_{UK,i}c_t,\quad v_{t,i}=W_{UV,i}c_t\\(q_{t,i}^{C})^\top k_{j,i}^{C}&=(W_{UK,i}^{\top}q_{t,i}^{C})^\top c_j\\M_{\mathrm{MLA}}&\approx BLT(d_c+d_R)s\end{aligned}
```

#### 易错点

- MLA latent 是联合压缩表示，不是普通单 KV 头，也不是任意 MHA 缓存的无损 SVD。
- 必须计入额外位置 key；缓存压缩比例不等于端到端提速比例。

#### 追问

- 位置相关旋转夹在投影之间，为什么会阻止矩阵吸收？
- 朴素展开所有 K/V 时，工作区、显存峰值与带宽会怎样变化？

<a id="arc-024"></a>
### ARC-024 · DeepSeek-R1-Zero、R1 与蒸馏模型的训练流程分别是什么？

**L2**

#### 答案

R1-Zero 直接从预训练底座做推理强化学习，R1 在此基础上组织冷启动、强化学习和筛选数据训练，蒸馏版则让较小底座学习教师的输出。这里的“Zero 不先做 SFT”只描述它的后训练路线，不代表模型没有语言预训练。

原始 R1-Zero 从 DeepSeek-V3-Base 开始，用 GRPO 对同一问题生成多条答案，并根据相对奖励更新；数学结果或代码测试提供正确性信号，格式规则提供结构信号。它能形成较长推理，但也出现可读性、重复和语言混杂问题。R1 先用少量长推理样本做冷启动监督微调，再做推理 RL；随后筛选较好的生成样本并加入一般任务数据，形成约 800K 样本，重新从 V3-Base 做两轮 SFT，最后用 RL 同时覆盖推理与通用偏好。

冷启动提供可读的行为起点，结果奖励鼓励探索，筛选训练巩固高质量输出，各阶段不能随意互换。原始学生蒸馏配方使用 Qwen/Llama 底座做 SFT，没有在这些学生上追加同等 RL，也不继承教师的专家数量或缓存结构。学习完整筛选推理轨迹与只学最终答案可能带来不同效果；评价应同时看正确率、推理长度、延迟和语言质量，并检查奖励是否只让模型学会格式而没有真正解题。

#### 易错点

- R1-Zero 不先做冷启动 SFT，不代表底座没有预训练。
- 蒸馏学生不自动继承教师的 MLA、专家结构或完整 RL 历史。

#### 追问

- 怎样确认奖励确实提高解题正确性，而不仅让格式合规？
- 学习筛选推理轨迹与只学习最终答案，可能带来哪些差异？

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/html/1810.04805v2)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/html/1907.11692v1)
- [ALBERT: A Lite BERT](https://arxiv.org/html/1909.11942v6)
- [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
- [XLNet: Generalized Autoregressive Pretraining](https://arxiv.org/abs/1906.08237)
- [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)
- [ChatGLM2-6B 官方仓库](https://github.com/THUDM/ChatGLM2-6B)
- [ChatGLM3 官方仓库](https://github.com/THUDM/ChatGLM3)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- [Mixtral of Experts](https://arxiv.org/html/2401.04088v1)
- [Transformers MixtralTopKRouter reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281)
- [Go Wider Instead of Deeper](https://arxiv.org/abs/2107.11817)
- [MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation](https://arxiv.org/html/2204.07675v2)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/html/1806.09055v2)
- [Weight-Sharing Neural Architecture Search: A Battle to Shrink the Optimization Gap](https://arxiv.org/abs/2008.01475)
- [DeepSeek-V2 Technical Report](https://arxiv.org/html/2405.04434v5)
- [DeepSeek-V3 official repository](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek-R1 official repository](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)
- [DeepSeek-R1 Technical Report](https://arxiv.org/html/2501.12948v1)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [Qwen3 official repository](https://github.com/QwenLM/Qwen3)
