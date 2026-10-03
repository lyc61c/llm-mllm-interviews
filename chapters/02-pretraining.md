# 预训练、数据与优化

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 题目

- [PRE-001 · 自回归预训练的 next-token loss 怎样计算？](#pre-001)
- [PRE-002 · BPE、Unigram 与 SentencePiece 分别是什么？](#pre-002)
- [PRE-003 · 词表越大越好吗，扩词表有哪些代价？](#pre-003)
- [PRE-004 · Scaling Law 描述的是什么，能直接预测下游能力吗？](#pre-004)
- [PRE-005 · Chinchilla 的结论是什么，tokens≈20×参数是硬规则吗？](#pre-005)
- [PRE-006 · 大规模预训练语料应怎样清洗？](#pre-006)
- [PRE-007 · 精确去重与近似去重有哪些方法和边界？](#pre-007)
- [PRE-008 · 怎样检测与减少 benchmark contamination？](#pre-008)
- [PRE-009 · 代码、网页、多语言等预训练数据如何配比？](#pre-009)
- [PRE-010 · SFT 数据 packing 怎样提高效率并保持跨样本隔离？](#pre-010)
- [PRE-011 · 继续预训练与 SFT 的数据和目标有什么区别？](#pre-011)
- [PRE-012 · AdamW 与 Adam 加 L2 正则为什么不等价？](#pre-012)
- [PRE-013 · FP16 与 BF16 的差异是什么，混合精度为何有用？](#pre-013)
- [PRE-014 · warmup、学习率衰减和梯度裁剪各解决什么问题？](#pre-014)
- [PRE-015 · perplexity 怎样计算才可公平比较？](#pre-015)

<a id="pre-001"></a>
## PRE-001 · 自回归预训练的 next-token loss 怎样计算？

**L1**

### 答案

自回归语言模型按概率链式法则分解 $p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t})$，第一项以 BOS 或空历史为条件；完整可变长序列通常还需 EOS 或约定结束机制。这不表示 token 相互独立，有 prompt 时可同样写条件响应的分解。训练最大化对数似然，以有效目标平均 next-token 交叉熵为损失。

Teacher forcing 使用真实历史，已知整条序列后可一次前向计算各位置 logits，再用 causal mask 排除未来；位置间仍有依赖，只是矩阵运算可以并行。比如 `input_ids=[a,b,c]`，a/b 位置分别预测 b/c。常见模型在 loss 内部完成 shift，collator 不应重复移动标签；没有额外目标时最后 c 的 logits 不计损，有 EOS 时可监督结束。

Padding、无效跨样本边界及配方不监督的 prompt 目标通过 loss mask 忽略，并按有效目标数归一化；prompt 标签不计损不等于 prompt 在 attention 中不可见。生成通常先 prefill 再逐步 decode，下一输入由已生成 token 决定，无法预先知道完整历史。KV cache 复用旧 K/V，投机解码可批量验证候选改善吞吐，但仍保持自回归条件关系；真实历史与生成历史的差异也可能造成错误累积。

$$
\begin{aligned}p_\theta(x_{1:T})&=\prod_{t=1}^T p_\theta(x_t\mid x_{<t})\\\mathcal L&=-\frac1{N_{\rm valid}}\sum_{t\in\mathcal T_{\rm valid}}\log p_\theta(x_t\mid x_{<t})\end{aligned}
$$

### 易错点

- 在 collator 和模型内部各 shift 一次，或者忘记监督 EOS/有效 token 的具体约定。
- 把并行 teacher forcing 等同于生成所有 token 相互独立，或把 prompt loss mask 当成 attention mask。

### 追问

- 有 prompt-response 边界时，哪个位置的 logit 监督首个 response token？
- 为什么 KV cache 改善计算成本却不把标准逐 token 生成变成一次前向？

<a id="pre-002"></a>
## PRE-002 · BPE、Unigram 与 SentencePiece 分别是什么？

**L1**

### 答案

BPE 从较小单元出发，逐步合并高频相邻对，学习 merge 规则并按规则编码；子词表示能减少未知词问题。Unigram 为候选切分建立概率模型，逐步删减候选词表，用动态规划找高概率切分，也可采样切分。

SentencePiece 是支持 BPE、Unigram 等模型的训练工具，不是与它们并列的单一算法。它可直接从原始文本训练，处理空白与归一化，减少对语言特定预分词的依赖。字符、byte 和 Unicode 归一化设定都会影响可逆性、序列长度及代码字符串表现。

### 易错点

- 把 SentencePiece 当作独立于 BPE、Unigram 的分词算法。
- 认为 token 总是一个汉字或一个英文单词。

### 追问

- byte-level tokenizer 为什么通常能覆盖未知字符？
- 分词可逆性与文本归一化有什么冲突？

<a id="pre-003"></a>
## PRE-003 · 词表越大越好吗，扩词表有哪些代价？

**L2**

### 答案

更大的词表可能降低同一文本的 token 数，改善部分语言的压缩率，但也增大 embedding、LM head 和 logits 计算。若输入/输出不共享权重，两套矩阵约有 $2Vd$ 参数，共享后约为 $Vd$，bias 另算；稀有或新增 token 出现太少时也难以学到稳定表示。

选词表应同时评估覆盖、压缩率、训练充分度和显存，在多语言、代码、数字与罕见字符上统计 tokens/byte 或 tokens/字符。继续训练时改变 tokenizer 还会改变序列分布，必须检查旧 token ID、初始化与新增行的训练，不能只看总 token 数减少。

### 易错点

- 把 tokenizer 扩词表当作无需训练的免费优化。
- 直接把更少 token 等同于更好的语义理解。

### 追问

- 什么时候值得为中文新增 token？
- 如何避免新增 token 的 embedding 未被 PEFT 更新？

<a id="pre-004"></a>
## PRE-004 · Scaling Law 描述的是什么，能直接预测下游能力吗？

**L2**

### 答案

Scaling Law 是模型损失随参数、训练数据与计算规模变化的经验规律，常以 held-out loss 为因变量，以参数量 $N$、训练 token 数 $D$ 和 FLOPs $C$ 为资源变量。它能帮助小实验外推和算力分配，但系数依赖实验范围、数据、架构与优化，并不保证全部下游指标或新能力服从同一曲线。

应在若干规模和训练长度上拟合，再保留额外规模验证预测，不能只报告拟合优度。数据质量、重复曝光、多语言配比和优化未收敛都可能改变曲线。预算还应包含推理成本，训练计算最优不一定是模型生命周期成本最优。

### 易错点

- 把经验幂律解释成任何规模都成立的理论定理。
- 把训练 loss 曲线直接换成准确率或安全性曲线。

### 追问

- 为什么小模型拟合可能无法预测大模型行为？
- 训练预算固定时如何做 isoFLOP 实验？

<a id="pre-005"></a>
## PRE-005 · Chinchilla 的结论是什么，tokens≈20×参数是硬规则吗？

**L2**

### 答案

Chinchilla 在其训练设置和预算范围内发现，计算最优的参数量与训练 token 数应近似同步增长；常说的每参数约 20 个 token 是该拟合下的量级参考，数据质量、目标与推理预算变化后需重新估计。

Dense Transformer 训练常以 $C\approx6ND$ 粗估 FLOPs，长上下文注意力等计算应另加。在固定预算下扫描参数量并相应调整数据量，再比较训练末端的验证损失；更大的模型若数据不足，并不总优于较小且训练充分的模型。若上线调用量很大，训练小模型更久可能更划算，因此不能只优化一次训练成本。

$$
C\approx6ND
$$

### 易错点

- 把 20 tokens/parameter 当成达到能力上限的通用阈值。
- 直接将 dense 估算套到总参数很大的稀疏 MoE。

### 追问

- 70B 与 7B 在相同训练预算下怎样比较？
- 什么时候会选择过度训练较小模型？

<a id="pre-006"></a>
## PRE-006 · 大规模预训练语料应怎样清洗？

**L2**

### 答案

预训练清洗先确定目标语料分布，再完成解析、语言识别、质量过滤、去重、评测污染与敏感信息处理，并保留数据血缘和抽样审计。HTML 正文提取应去除导航、乱码和模板，再做语言与文档级检查。

规则和模型打分可以互补，对高低评分及不同语言、领域都应抽检。去重后重新统计唯一 token、来源配比与重复曝光，记录规则版本；用保留率、领域覆盖、小模型实验和固定验证集衡量改动。过强过滤可能删除少数语言或专业内容，不能仅为降低训练 loss 调整清洗策略。

### 易错点

- 高 perplexity 文本全是低质量，低 perplexity 文本全是高质量。
- 使用黑箱过滤器却无法追踪被删除的领域。

### 追问

- 如何量化清洗造成的分布偏移？
- 为什么代码或数学文本需要单独制定过滤规则？

<a id="pre-007"></a>
## PRE-007 · 精确去重与近似去重有哪些方法和边界？

**L2**

### 答案

精确去重使用规范化文档或片段哈希找完全重复，成本较低，但难以发现只改标题或局部段落的副本。近似去重可用 shingles、MinHash 等生成相似候选，再做更细比较，需同时评估误报和漏报。

公共模板可在段落或子串层面去重，保留高质量代表并记录映射。去重顺序影响后续采样权重，先按质量挑代表能避免随机留下差版本。去重有助于减少重复曝光、记忆和评测重叠，但不能保证完全消除泄漏；阈值过严也会损失有价值的重复结构。

### 易错点

- 同一文章的网址不同就视为独立数据。
- 把语义相似的不同事实问答全部删掉。

### 追问

- 如何在大规模语料上控制近似去重成本？
- 为什么 dedup 之后验证 loss 可能上升但可信度提高？

<a id="pre-008"></a>
## PRE-008 · 怎样检测与减少 benchmark contamination？

**L2**

### 答案

Benchmark contamination 指训练集包含测试题、答案或高度相似信息，使评测分数无法可靠表示泛化。应建立题干与答案的可追踪清单，通过文档、片段和 n-gram 重叠检索发现候选，并人工复核翻译、改写、镜像站与题解，不能只删除完全相同字符串或原网址。

训练/测试切分尽量按来源、实体、时间或任务组隔离，再用受控新题验证。报告可检测污染比例与清洁子集分数；未检出只能说明检测方法没有发现，不能证明完全无污染。

### 易错点

- 把预训练见过某领域事实与见过完整测试题一概混称同一污染。
- 根据高分反推模型必然泄漏。

### 追问

- 闭源模型无法审计训练集时怎么补充评估？
- 污染检测为什么需要保留误报边界？

<a id="pre-009"></a>
## PRE-009 · 代码、网页、多语言等预训练数据如何配比？

**L2**

### 答案

预训练配比应围绕目标能力和质量设计，而非按原始文件数量采样。先统一有效 token、唯一 token 与曝光轮次的统计，再通过小代理模型、领域验证 loss 和下游任务实验调整权重，并检验结论能否迁移到大模型。

提高某类数据比例会挤占其他能力的预算，过采样稀缺高质量数据也要监控重复与记忆。DoReMi 用代理模型优化域权重，说明配比可以学习，但迁移仍有条件。各主要领域应保留独立验证集，同时检查总体效果和弱势语言表现。

### 易错点

- 高 loss 的领域永远应该加权更多，忽略噪声或任务难度。
- 把论文给出的配比直接用在不同数据和目标上。

### 追问

- 如何设计配比 ablation 避免算力不公平？
- 数据配比与课程学习有什么区别？

<a id="pre-010"></a>
## PRE-010 · SFT 数据 packing 怎样提高效率并保持跨样本隔离？

**L2**

### 答案

Packing 把多个短样本装入一条训练行，减少 padding。先应用 chat template、分词并标记回答监督区间，再连同 `input_ids`、`labels`、样本编号和边界一起装箱。BFD 等策略可减少空位；截断或拆分会改变保留内容，应统计丢弃的回答 token，而不只报告吞吐。

独立 SFT 样本的可见性必须同时满足同一样本与因果条件 $j\le i$，并排除 padding。可以用块对角 causal mask，或给支持变长 attention 的后端显式段边界。EOS 和把 `position_ids` 归零都不会自动隔离通用 attention。

普通全行 label shift 会让前一段末 logits 预测下一段首 token。独立样本目标应把后一段首 label 设为 `ignore_index`，或逐段 shift 并在段尾补 ignore；BOS/EOS 是否监督要按段定义，不能粗暴删除全部 EOS loss。Prompt labels 忽略而首 answer label 保留，末 prompt logits 才能预测首回答；合法 prompt 仍需作为上下文。

连续语料 concatenate-then-split 可有意允许跨文档上下文，这与独立 SFT 目标不同。可对比未 packing 的逐段有效 NLL 与梯度，并扰动另一段检查当前段 logits 是否改变；比较时要控制内核、dropout 和浮点次序，再评估显存与吞吐收益。

式中 $s_t$ 是样本编号，$v_j$ 标记有效 key，$m_t$ 选择监督目标，$\mathcal H_t$ 是同一样本的有效历史。

$$
\begin{aligned}\operatorname{allowed}(i,j)&=\mathbf1[s_i=s_j]\,\mathbf1[j\le i]\,\mathbf1[v_j=1]\\\mathcal L&=-\frac{\sum_t m_t\log p_\theta(x_t\mid x_{\mathcal H_t})}{\sum_t m_t}\\\mathcal H_t&=\{j:j<t,\ s_j=s_t,\ v_j=1\}\end{aligned}
$$

### 易错点

- EOS、loss mask 或重置 `position_ids` 任一项就能阻断跨样本 attention。
- 隔离 attention 后仍保留跨边界 shifted label，或在不同实现中重复 shift。

### 追问

- 如何处理长对话拆分时需要继承的历史 prompt？
- 为何 packing 开启后吞吐提高却可能改变梯度加权？

<a id="pre-011"></a>
## PRE-011 · 继续预训练与 SFT 的数据和目标有什么区别？

**L1**

### 答案

继续预训练通常用领域原始文本延续语言建模目标，以适应术语、知识与语言分布；SFT 使用指令、对话或结构化输入输出，学习任务行为与格式。领域无标注数据充足且分布差异大时可考虑 DAPT，任务输出明确时可先尝试 SFT，也可以串联两者。

领域适应不保证指令能力，应分别记录领域 loss、任务正确率与通用回归，检查能力退化。CPT 可以使用 PEFT，SFT 也可更新全部参数，训练目标与参数更新方法是两个独立选择。

### 易错点

- 把继续预训练简单定义为全参数微调，SFT 简单定义为 LoRA。
- 认为任何新增知识都必须继续预训练。

### 追问

- 领域知识变化很快时为什么还会考虑 RAG？
- CPT 之后 SFT 数据如何补足交互格式？

<a id="pre-012"></a>
## PRE-012 · AdamW 与 Adam 加 L2 正则为什么不等价？

**L2**

### 答案

Adam 中的 L2 项作为 loss 梯度进入自适应一阶、二阶矩和预条件缩放；AdamW 的 weight decay 则直接作用于参数，与梯度更新解耦。因此二者通常不等价，不能因都使用 `weight_decay` 字段就当作同一算法。普通 SGD 在适当系数下的等价关系也不能直接迁移到 Adam。

训练时应明确参数组、衰减规则与学习率；bias 和 norm 是否衰减由配方决定。学习率影响每步衰减量，比较配置还需匹配训练步数与调度。

### 易错点

- 说 AdamW 只是 Adam 的别名。
- 无依据断言所有 embedding、norm 必须采用同一种衰减配置。

### 追问

- 为什么相同 `weight_decay` 在不同训练时长下效果不同？
- LoRA 参数是否应该使用权重衰减？

<a id="pre-013"></a>
## PRE-013 · FP16 与 BF16 的差异是什么，混合精度为何有用？

**L1**

### 答案

FP16 和 BF16 都是 16 bit：FP16 使用 1 位符号、5 位指数、10 位尾数，BF16 为 1、8、7。BF16 指数范围接近 FP32，FP16 尾数精度更细，却更容易发生梯度下溢和激活溢出。

混合精度让合适的算子使用低精度，敏感归约保持高精度；`autocast` 按算子选 dtype，并非把所有张量转成 half。FP16 常用 loss scaling 缓解梯度下溢，BF16 较少需要，但仍可能有数值问题。速度取决于硬件与内核，应持续检查 NaN、Inf 和梯度。

### 易错点

- BF16 精度在所有数值范围都高于 FP16。
- loss scaling 能修复前向激活溢出。

### 追问

- 为什么部分 norm 或 softmax 使用 FP32？
- FP16 的梯度裁剪前应做什么？

<a id="pre-014"></a>
## PRE-014 · warmup、学习率衰减和梯度裁剪各解决什么问题？

**L2**

### 答案

Warmup 缓慢增加训练初期的学习率，给优化器状态和激活尺度建立提供缓冲；需要多长取决于模型尺度、初始化、优化器与数据。原 Transformer 的 Noam 策略先线性增长，再按步数平方根倒数衰减：更新步 $s\ge1$、模型维度 $D$、warmup 步 $w$ 时，$\eta(s)=cD^{-1/2}\min(s^{-1/2},sw^{-3/2})$。交点 $s=w$ 的峰值为 $c/\sqrt{Dw}$；原实验 $w=4000$，现代配方也可用 cosine 或 linear，不能把历史设置当作通用规则。

Global norm clipping 将全部参数梯度的总范数 $G=\sqrt{\sum_p\|g_p\|_2^2}$ 限制在 $C>0$，对所有梯度统一乘 $\min(1,C/G)$；$G=0$ 时保持零。这样保留方向，与逐元素 clamp 或各张量独立裁剪不同。

应在完整梯度累积之后、AMP unscale 之后裁剪，再做 `optimizer.step`；scheduler 按真实参数更新计步。分布式分片要检查全局范数聚合，非有限梯度须先检测和处理，裁剪不能修复 NaN/Inf。它限制当前梯度，却不保证 Adam 更新范数或后续 loss，动量、二阶矩和衰减仍会影响结果。持续触发时应记录裁剪前范数、比例、loss 与 AMP 跳步，再排查数据、学习率和数值。

$$
\begin{aligned}\eta(s)&=cD^{-1/2}\min(s^{-1/2},sw^{-3/2}),\quad s\ge1\\G&=\sqrt{\sum_p\|g_p\|_2^2}\\g'_p&=\begin{cases}g_p\min(1,C/G),&G>0\\0,&G=0\end{cases}\end{aligned}
$$

### 易错点

- 按每个参数各自 clip，却宣称等价于 global norm；在 AMP unscale 前套用原阈值。
- 把原始 Noam 配方写成所有 Transformer/LLM 固定规则，或使用 `step=0` 导致负幂未定义。

### 追问

- 若 warmup 步数翻 4 倍且其他因子不变，Noam 峰值怎样变化？
- 为何梯度范数已被裁到 C，Adam 的参数更新范数仍可能大于 $\eta C$？

<a id="pre-015"></a>
## PRE-015 · perplexity 怎样计算才可公平比较？

**L1**

### 答案

PPL 是有效 token 平均负对数似然的指数，测量文本预测分布，不直接等同于对话质量。公平比较要统一语料、tokenizer、上下文窗口与计损范围；不同 tokenizer 的 token-level PPL 不直接可比，可辅以 bits-per-byte 等统一口径。

固定窗口模型评估长文时，互不重叠切块会使块开头缺少历史，通常抬高 PPL。滑动窗口能补足上下文，但重叠部分应只作条件，避免目标重复计数，并考虑模型内部 label shift 后的有效数量。还需结合领域 loss、指令遵循、事实性与任务正确率判断能力。

$$
\operatorname{PPL}=\exp\!\left(\frac{\sum_{t\in\mathcal T_{\rm valid}}\operatorname{NLL}_t}{N_{\rm valid}}\right)
$$

### 易错点

- 先算各 batch PPL 再等权平均。
- 对 prompt、padding 和目标回答采用不一致的计损策略。

### 追问

- 为什么更短的评测 stride 可能得到更低 PPL？
- 如何处理超长语料的首个 token？

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [TRL Reducing Memory Usage — packing and padding-free](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)
- [PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Automatic Mixed Precision package — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/amp.html)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- [torch.nn.utils.clip_grad_norm_ — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)
