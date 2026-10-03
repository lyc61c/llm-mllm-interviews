# 预训练、数据与优化

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

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

**L1 · 社区题目线索** · 标签：CLM / TeacherForcing / Loss

**30 秒回答**

自回归模型把序列概率分解为每个 token 在历史条件下的概率乘积，用真实历史的下一 token 交叉熵训练。teacher forcing 下，已知整条目标序列，可用因果掩码并行计算各位置；生成时下一位置依赖已生成历史，通常逐步执行，KV cache 减少重复计算。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 联合概率按 p_θ(x_1,…,x_T)=Π_{t=1}^T p_θ(x_t|x_<t) 分解，第一项以 BOS/空历史为条件；要建模完整的可变长序列，通常包含 EOS 或约定结束机制。这里是概率链式法则，绝不表示 token 相互独立；有 prompt 时可对条件响应写同样的乘积。
- 训练最大化对数似然，L=−Σ_t log p_θ(x_t|x_<t)/N_valid。teacher forcing 使用真实历史 token，因此所有输入已知，可以一次前向为各位置算 logits，再用 causal mask 排除未来；位置间仍有信息依赖，只是矩阵运算能并行。
- 按 next-token 约定，input_ids=[a,b,c] 的位置 a/b 的 logits 分别监督 b/c；常见模型 loss 内部完成 labels 右侧目标对齐，因此 collator 不应重复 shift。没有额外目标时最后 c 的 logit 不计损；有 EOS 标签则可监督序列结束，具体由数据与接口决定。
- pad、无效跨样本边界或不训练的 prompt 目标使用 ignore_index 等 loss mask，并按有效目标数归一化；忽略 prompt 标签不等于把 prompt 从 attention 上下文屏蔽。padding/样本隔离的 attention mask 与目标计损 mask 必须各自检查。
- 通常生成先 prefill prompt，再逐个 decode：第 t 步生成的 token 决定下一步输入及条件分布，无法像 teacher forcing 一样事先获知所有历史。KV cache 复用旧 K/V，speculative decoding 等可批量验证候选改善吞吐，但没有取消目标分布的自回归条件关系；真实历史与生成历史的分布差异也可能引起错误累积。

### 公式

```text
p_θ(x_1:T)=Π_{t=1}^T p_θ(x_t|x_<t)；L=−Σ_{t∈有效目标} log p_θ(x_t|x_<t)/N_valid
```

### 易错点

- 在 collator 和模型内部各 shift 一次，或者忘记监督 EOS/有效 token 的具体约定。
- 把并行 teacher forcing 等同于生成所有 token 相互独立，或把 prompt loss mask 当成 attention mask。

### 面试官可能追问

- 有 prompt-response 边界时，哪个位置的 logit 监督首个 response token？
- 为什么 KV cache 改善计算成本却不把标准逐 token 生成变成一次前向？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S032 · torch.nn.CrossEntropyLoss — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [CORE-S037 · Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [CORE-S043 · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [CORE-S075 · Transformers v4.57.1: Caching and cache position](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [CORE-S081 · Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)

**题目出处线索**

- [CORE-S008 · 知乎问答：生成语言模型微调与预训练如何计算 loss](https://www.zhihu.com/en/answer/3335594083) · `search_snippet`：知乎搜索摘要讨论预训练与微调 loss/label shift；属于技术问答主题，非公司面经，正文安全验证受阻。

<a id="pre-002"></a>
## PRE-002 · BPE、Unigram 与 SentencePiece 分别是什么？

**L1 · 社区题目线索** · 标签：Tokenizer / BPE / Unigram

**30 秒回答**

BPE 从较小单元开始按规则合并高频相邻对；Unigram 用子词概率模型选择或采样切分，并逐步删减候选词表。SentencePiece 是可训练这些子词模型的工具，不是与 BPE、Unigram 并列的单一算法。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- BPE 学习一系列 merge 规则，编码时按学习规则执行，子词能降低未知词问题。
- Unigram 对候选切分使用概率评分，可用动态规划寻找高概率切分，也可做采样。
- SentencePiece 可以从原始文本训练，处理空白和归一化，减少对语言特定预分词的依赖。
- 字符、byte 与 Unicode 归一化设定会影响可逆性、长度及代码字符串表现。

### 易错点

- 把 SentencePiece 当作独立于 BPE、Unigram 的分词算法。
- 认为 token 总是一个汉字或一个英文单词。

### 面试官可能追问

- byte-level tokenizer 为什么通常能覆盖未知字符？
- 分词可逆性与文本归一化有什么冲突？

</details>

**技术依据**

- [CORE-S035 · SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- [CORE-S067 · Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf)

**题目出处线索**

- [CORE-S004 · LLMs_Interview：目录与更新记录](https://github.com/threeneedone/LLMs_Interview) · `reported_topic`：README 目录提供分词与评测备考主题；此题将主题具体化，不是已认证公司真题。

<a id="pre-003"></a>
## PRE-003 · 词表越大越好吗，扩词表有哪些代价？

**L2 · 社区题目线索** · 标签：词表 / 参数量 / 多语言

**30 秒回答**

增大词表可能减少同一文本的 token 数，改善某些语言的压缩率，却增大 embedding、LM head 和 logits 计算；新增 token 也可能缺少训练。选词表需在覆盖、长度、学习充分度及显存之间评估，不能只看压缩率。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 不共享权重时两套 V×d 矩阵约有 2Vd 参数；共享后约 Vd，bias 另算。
- 词表变大会降低某些稀有 token 的出现次数，难学出稳定表示。
- 评估不同语言、代码、数字、罕见字符的 tokens/byte 或 tokens/字符。
- 继续训练时改变 tokenizer 会改变序列分布，需检查旧 token ID、初始化和新增行训练。

### 易错点

- 把 tokenizer 扩词表当作无需训练的免费优化。
- 直接把更少 token 等同于更好的语义理解。

### 面试官可能追问

- 什么时候值得为中文新增 token？
- 如何避免新增 token 的 embedding 未被 PEFT 更新？

</details>

**技术依据**

- [CORE-S035 · SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- [CORE-S030 · Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

**题目出处线索**

- [CORE-S011 · 大模型常考面试题 100 道（第 51～75 道）](https://www.nowcoder.com/discuss/866256452975943680) · `search_snippet`：搜索摘要出现 tokenizer、Scaling Law、去重；只保留题库主题，未核验其大厂或高频说法。

<a id="pre-004"></a>
## PRE-004 · Scaling Law 描述的是什么，能直接预测下游能力吗？

**L2 · 社区题目线索** · 标签：ScalingLaw / 算力预算 / Loss

**30 秒回答**

Scaling Law 是给定实验范围内，语言模型损失随模型规模、数据量和计算量变化的经验规律。它可以帮助小规模实验外推和分配算力，但系数依赖数据、架构和训练设置，也不保证所有下游指标或新能力严格服从同一规律。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先明确因变量是 held-out loss，资源变量常为参数 N、训练 token D、FLOPs C。
- 用若干规模和训练长度拟合，保留额外规模做预测验证，避免只报告拟合优度。
- 数据质量、重复、多语言配比和优化未收敛都可能改变拟合曲线。
- 预算讨论还要考虑推理成本；训练 compute-optimal 不一定生命周期成本最优。

### 易错点

- 把经验幂律解释成任何规模都成立的理论定理。
- 把训练 loss 曲线直接换成准确率或安全性曲线。

### 面试官可能追问

- 为什么小模型拟合可能无法预测大模型行为？
- 训练预算固定时如何做 isoFLOP 实验？

</details>

**技术依据**

- [CORE-S033 · Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [CORE-S034 · Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

**题目出处线索**

- [CORE-S011 · 大模型常考面试题 100 道（第 51～75 道）](https://www.nowcoder.com/discuss/866256452975943680) · `search_snippet`：搜索摘要出现 tokenizer、Scaling Law、去重；只保留题库主题，未核验其大厂或高频说法。

<a id="pre-005"></a>
## PRE-005 · Chinchilla 的结论是什么，tokens≈20×参数是硬规则吗？

**L2 · 编辑补充题** · 标签：Chinchilla / ComputeOptimal / 预算

**30 秒回答**

Chinchilla 的实验表明，在其训练设置和预算范围内，计算最优的模型参数与训练 token 应近似同步增长。常说的每参数约 20 个 token 是特定拟合下的量级参考；数据质量、目标和推理预算变化时应重新估计。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 同一 FLOPs 预算下较大模型并非总是更优，数据不足会导致模型训练不充分。
- dense Transformer 训练常用 C≈6ND 粗估，长上下文注意力等项会带来额外成本。
- 固定 C，扫描 N 并相应调整 D，比较各自训练末端的验证损失。
- 如果上线推理调用很多，训练较小模型更久可能更划算，不能只优化训练成本。

### 公式

```text
dense 粗估：C≈6ND；实际需补入注意力及其他计算
```

### 易错点

- 把 20 tokens/parameter 当成达到能力上限的通用阈值。
- 直接将 dense 估算套到总参数很大的稀疏 MoE。

### 面试官可能追问

- 70B 与 7B 在相同训练预算下怎样比较？
- 什么时候会选择过度训练较小模型？

</details>

**技术依据**

- [CORE-S034 · Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-006"></a>
## PRE-006 · 大规模预训练语料应怎样清洗？

**L2 · 编辑补充题** · 标签：数据质量 / RefinedWeb / 清洗

**30 秒回答**

先定义目标语料分布，再做解析、语言识别、质量过滤、去重、污染与敏感信息处理，并保留血缘及抽样审计。过滤阈值应通过保留率、领域覆盖和小模型实验验证，过强过滤可能删除少数语言或专业内容。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 从 HTML 提取正文并去导航、乱码、模板文本，再做语言及文档级质量检查。
- 规则和模型打分互补；对高低评分、不同语言和领域都做人工抽检。
- 去重后再统计唯一 token、来源配比和重复曝光，记录规则及版本。
- 保留一份固定验证集评估改动，避免清洗策略只迎合训练 loss。

### 易错点

- 高 perplexity 文本全是低质量，低 perplexity 文本全是高质量。
- 使用黑箱过滤器却无法追踪被删除的领域。

### 面试官可能追问

- 如何量化清洗造成的分布偏移？
- 为什么代码或数学文本需要单独制定过滤规则？

</details>

**技术依据**

- [CORE-S036 · The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-007"></a>
## PRE-007 · 精确去重与近似去重有哪些方法和边界？

**L2 · 社区题目线索** · 标签：Dedup / MinHash / 数据

**30 秒回答**

精确去重用规范化文本或片段哈希找完全重复；近似去重可用 shingles、MinHash 等找高度相似文本。去重能减少重复曝光、记忆和评测重叠，但不能保证完全消除泄漏；阈值过严也会伤害有价值的重复结构。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 文档哈希成本低，但无法检测只改标题或局部段落的重复。
- 近似方法先生成候选再做更精细比较，需评估误报及漏报。
- 长公共模板可做段落或子串去重，保留高质量代表版本并记录映射。
- 去重顺序影响采样权重，先按质量选代表可避免随机保留较差版本。

### 易错点

- 同一文章的网址不同就视为独立数据。
- 把语义相似的不同事实问答全部删掉。

### 面试官可能追问

- 如何在大规模语料上控制近似去重成本？
- 为什么 dedup 之后验证 loss 可能上升但可信度提高？

</details>

**技术依据**

- [CORE-S038 · Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [CORE-S036 · The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)

**题目出处线索**

- [CORE-S011 · 大模型常考面试题 100 道（第 51～75 道）](https://www.nowcoder.com/discuss/866256452975943680) · `search_snippet`：搜索摘要出现 tokenizer、Scaling Law、去重；只保留题库主题，未核验其大厂或高频说法。

<a id="pre-008"></a>
## PRE-008 · 怎样检测与减少 benchmark contamination？

**L2 · 编辑补充题** · 标签：数据污染 / 评测可信度 / Leakage

**30 秒回答**

评测污染是训练数据包含测试题、答案或高度相似信息，使分数不能可靠表示泛化。应检查跨集合重叠、答案模式与时间来源，并用受控新题验证；仅删除完全相同字符串不足以排除翻译、改写或题解泄漏。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 建立 benchmark 文本及答案的可追踪清单，按文档、片段及 n-gram 做候选重叠检索。
- 对题干变体、镜像站和带答案解释的页面人工复核，避免只查原网址。
- 切分尽量按文档来源、实体、时间或任务组隔离，不只随机分行。
- 报告可检测污染比例和清洁子集分数；未检出不等于已证明无污染。

### 易错点

- 把预训练见过某领域事实与见过完整测试题一概混称同一污染。
- 根据高分反推模型必然泄漏。

### 面试官可能追问

- 闭源模型无法审计训练集时怎么补充评估？
- 污染检测为什么需要保留误报边界？

</details>

**技术依据**

- [CORE-S037 · Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [CORE-S038 · Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-009"></a>
## PRE-009 · 代码、网页、多语言等预训练数据如何配比？

**L2 · 编辑补充题** · 标签：DataMixture / DoReMi / 多语言

**30 秒回答**

数据配比应围绕目标能力和数据质量设计，而不是直接按各来源原始数量采样。可以用小代理模型、领域验证损失与下游指标做配比实验，再验证能否迁移到大模型；提高某类占比还会挤占其他能力的训练预算。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先统一统计口径，如有效 token、唯一 token 与曝光轮次，避免按文件数配比。
- 过采样稀缺高质量数据需监控重复率及记忆，不是无限重复。
- DoReMi 用代理模型优化域权重，说明数据配比可学习，但迁移效果仍有条件。
- 为主要领域保留独立验证集，同时观察总体效果和弱势语言表现。

### 易错点

- 高 loss 的领域永远应该加权更多，忽略噪声或任务难度。
- 把论文给出的配比直接用在不同数据和目标上。

### 面试官可能追问

- 如何设计配比 ablation 避免算力不公平？
- 数据配比与课程学习有什么区别？

</details>

**技术依据**

- [CORE-S039 · DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-010"></a>
## PRE-010 · SFT 数据 packing 怎样提高效率并保持跨样本隔离？

**L2 · 编辑补充题** · 标签：Packing / Mask / 数据管线 / SFT / BlockDiagonal / BoundaryLabels

**30 秒回答**

packing 将多个短样本装进一条训练行，减少 padding 浪费。独立 SFT 样本应保留各自上下文与监督边界，使用块对角因果可见性或显式变长序列边界；EOS 和重置位置都不自动隔离。还须屏蔽跨样本的下一词标签并检验目标一致性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先应用 chat template、分词并构造回答监督区间，再将 input_ids、labels、样本编号和边界一起 packing。BFD 等装箱策略减少空位；截断与拆分长样本会改变保留内容，应统计被丢弃的回答 token，而非只报告 token 吞吐。
- 若保持样本独立，mask 允许条件为 same_sample(i,j) 且 j≤i，并排除 padding。可显式构造块对角 causal mask，或向支持变长 attention 的后端传每段边界；不能只把 position_ids 归零便认为通用 attention 会自动识别边界。
- 第一个样本 EOS 后紧接第二个样本首 token 时，普通全行 shift 会让前者末 logit 预测后者首 token。若训练目标是独立样本，应把第二段首 token 的 labels 设 ignore_index，或使用按段 shift/末尾补 ignore 的实现；BOS/EOS 是否监督也按每段定义。
- 回答 mask 仍只保留监督目标：末 prompt 位置的 logit 预测首 answer，因此保留首 answer 的 label，而 prompt labels 置 −100；合法 prompt 仍作为 K/V 上下文。position_ids、有效 loss mask 与序列边界必须共同校验，不能给所有边界粗暴删除 EOS loss。
- 连续语料的 concatenate-then-split 可有意允许跨文档上下文，这和独立 SFT 目标不同。验证时对比未 packing 的每段有效 NLL/梯度，并扰动另一段检查本段 logits 是否变化；兼容内核、dropout 和浮点次序后再比较数值及显存收益。

### 公式

```text
allowed(i,j)=1[sample_i=sample_j]·1[j≤i]·1[key_j valid]; L=−Σ_t m_t log p(x_t|same-sample history)/Σ_t m_t
```

### 易错点

- EOS、loss mask 或重置 position_ids 任一项就能阻断跨样本 attention。
- 隔离 attention 后仍保留跨边界 shifted label，或在不同实现中重复 shift。

### 面试官可能追问

- 如何处理长对话拆分时需要继承的历史 prompt？
- 为何 packing 开启后吞吐提高却可能改变梯度加权？

</details>

**技术依据**

- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S072 · TRL SFT Trainer — loss, labels and packing](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [CORE-S073 · TRL Reducing Memory Usage — packing and padding-free](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)
- [CORE-S074 · PyTorch SDPA — masks, shapes and GQA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [CORE-S081 · Transformers v4.57.1 official ForCausalLMLoss](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/loss/loss_utils.py)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-011"></a>
## PRE-011 · 继续预训练与 SFT 的数据和目标有什么区别？

**L1 · 编辑补充题** · 标签：CPT / DAPT / SFT

**30 秒回答**

继续预训练通常沿用语言建模目标，用领域原始文本调整模型的语言与知识分布；SFT 用输入和目标回答教任务行为或输出格式。它们可以串联，但领域适应不保证指令能力，需要分别验证领域收益及通用能力退化。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 继续预训练以领域正文等自监督数据为主，SFT 以指令、对话、结构化输入输出为主。
- 领域内无标注数据丰富、术语和分布差异大时可考虑 DAPT；任务输出明确时先尝试 SFT。
- 选择顺序时记录领域 loss、任务准确率和通用回归，而非只看训练 loss。
- CPT 也可做 PEFT，SFT 也可更新全部参数，目标与更新方法是两个维度。

### 易错点

- 把继续预训练简单定义为全参数微调，SFT 简单定义为 LoRA。
- 认为任何新增知识都必须继续预训练。

### 面试官可能追问

- 领域知识变化很快时为什么还会考虑 RAG？
- CPT 之后 SFT 数据如何补足交互格式？

</details>

**技术依据**

- [CORE-S040 · Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [CORE-S044 · TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-012"></a>
## PRE-012 · AdamW 与 Adam 加 L2 正则为什么不等价？

**L2 · 编辑补充题** · 标签：AdamW / WeightDecay / 优化

**30 秒回答**

Adam 的自适应预条件会改变 L2 项经过梯度更新时的缩放；AdamW 把权重衰减与梯度更新解耦，因此二者通常不等价。实际训练需明确参数组、学习率和衰减设定，不要把 weight_decay 字段自动当成同一算法。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 普通 SGD 下适当缩放的 L2 与 weight decay 可等价，不能无条件迁移到 Adam。
- AdamW 的衰减直接作用于参数，而不是作为 loss 梯度再进入一二阶矩。
- bias、norm 参数是否衰减取决于训练配方，需明确参数组规则。
- 学习率改变会影响每步衰减量，比较配置应匹配训练步数及调度。

### 易错点

- 说 AdamW 只是 Adam 的别名。
- 无依据断言所有 embedding、norm 必须采用同一种衰减配置。

### 面试官可能追问

- 为什么相同 weight_decay 在不同训练时长下效果不同？
- LoRA 参数是否应该使用权重衰减？

</details>

**技术依据**

- [CORE-S041 · Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-013"></a>
## PRE-013 · FP16 与 BF16 的差异是什么，混合精度为何有用？

**L1 · 编辑补充题** · 标签：BF16 / FP16 / AMP

**30 秒回答**

FP16 和 BF16 都占 16 bit，但 BF16 的指数范围接近 FP32，FP16 的尾数更细。混合精度让适合的算子使用低精度、敏感归约保持高精度；FP16 常需 loss scaling，BF16 较少需要，但仍可能出现数值问题。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- FP16 为 1 位符号、5 位指数、10 位尾数；BF16 为 1、8、7。
- FP16 的有限范围可能造成梯度下溢或激活溢出；缩放 loss 主要缓解梯度下溢。
- autocast 按算子选精度，不等于把所有张量无差别转成 half。
- 是否更快取决于硬件原生支持及内核；训练稳定还需监控 NaN、Inf 和梯度。

### 易错点

- BF16 精度在所有数值范围都高于 FP16。
- loss scaling 能修复前向激活溢出。

### 面试官可能追问

- 为什么部分 norm 或 softmax 使用 FP32？
- FP16 的梯度裁剪前应做什么？

</details>

**技术依据**

- [CORE-S042 · Automatic Mixed Precision package — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/amp.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-014"></a>
## PRE-014 · warmup、学习率衰减和梯度裁剪各解决什么问题？

**L2 · 编辑补充题** · 标签：Warmup / LearningRate / GradientClipping

**30 秒回答**

Warmup 逐步增加初期学习率，裁剪限制异常梯度的总范数。原 Transformer 的 Noam 策略先线性增长、再按步数平方根倒数衰减，这是历史配方。使用 loss scaling 时先 unscale 再裁剪，按 optimizer 更新计调度步，不能把裁剪当成 NaN 修复。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 原论文公式为 η(s)=D^(−1/2)·min(s^(−1/2),s·w^(−3/2))，s 为从 1 开始的更新步，D=d_model、w=warmup_steps；原实验 w=4000。s≤w 时线性增长，s>w 后按 s^(−1/2) 衰减，连续交点的峰值为 1/√(Dw)；工程也可另乘倍率 c。
- warmup 为早期大更新提供缓冲，模型尺度、初始化、优化器状态和数据都可能影响所需长度。现代配方也可用 cosine、linear 等 schedule；原论文 warmup/逆平方根策略不意味着每个大模型训练都应固定 4000 步。
- 将全部参数梯度拼成一向量，L2 global norm G=√(Σ_p||g_p||²_2)；阈值 C>0，若 G>0 则 g_p←g_p·min(1,C/G)，G=0 时保持零。它按同一个比例缩放所有梯度，保留方向；不同于每个元素 clamp，也不同于逐张量单独裁剪。
- 应在完整 gradient accumulation 结束、AMP loss scaling 的梯度已 unscale 后裁剪，再调用 optimizer.step；scheduler 通常跟实际参数更新步计数，不能误将 microbatch 当更新。非有限梯度需先检测/处理，clip 不能可靠地修复 NaN/Inf，分布式分片需核对全局范数是否聚合完整。
- clip 限制梯度的当前范数，不能直接保证 Adam 的实际参数更新范数或以后每步 loss；动量、二阶矩及 weight decay 仍改变更新。持续触发裁剪可能是学习率/数据/数值问题，需要记录裁剪前范数、比例、loss 与 AMP 跳步，避免仅以“未报错”判断训练稳定。

### 公式

```text
η(s)=c·D^(−1/2)·min(s^(−1/2),s·w^(−3/2)), s≥1；G=√Σ_p||g_p||²_2；g_p←g_p·min(1,C/G)（G>0）
```

### 易错点

- 按每个参数各自 clip，却宣称等价于 global norm；在 AMP unscale 前套用原阈值。
- 把原始 Noam 配方写成所有 Transformer/LLM 固定规则，或使用 step=0 导致负幂未定义。

### 面试官可能追问

- 若 warmup 步数翻 4 倍且其他因子不变，Noam 峰值怎样变化？
- 为何梯度范数已被裁到 C，Adam 的参数更新范数仍可能大于 ηC？

</details>

**技术依据**

- [CORE-S020 · Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [CORE-S023 · On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- [CORE-S042 · Automatic Mixed Precision package — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/amp.html)
- [CORE-S085 · torch.nn.utils.clip_grad_norm_ — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="pre-015"></a>
## PRE-015 · perplexity 怎样计算才可公平比较？

**L1 · 社区题目线索** · 标签：PPL / 评测 / 长文本

**30 秒回答**

perplexity 是有效 token 平均负对数似然的指数，测的是文本预测分布，不等于对话质量。比较要统一语料、tokenizer、上下文和计损范围；固定窗口模型评长文可用滑动窗口，让每个目标获得更充分且一致的历史。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 长文本直接分成互不重叠块，会让块开头缺乏上下文，通常抬高 PPL。
- 滑动窗口的重叠部分只当上下文，避免目标重复计数；内部 label shift 要计入有效数量。
- 不同 tokenizer 的 token-level PPL 不直接可比，可辅以 bits-per-byte 等统一口径。
- 同时看领域 loss、指令遵循、事实性及任务正确率，避免把低 PPL 当成万能指标。

### 公式

```text
PPL=exp(Σ有效目标 NLL / 有效目标数)
```

### 易错点

- 先算各 batch PPL 再等权平均。
- 对 prompt、padding 和目标回答采用不一致的计损策略。

### 面试官可能追问

- 为什么更短的评测 stride 可能得到更低 PPL？
- 如何处理超长语料的首个 token？

</details>

**技术依据**

- [CORE-S043 · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)

**题目出处线索**

- [CORE-S004 · LLMs_Interview：目录与更新记录](https://github.com/threeneedone/LLMs_Interview) · `reported_topic`：README 目录提供分词与评测备考主题；此题将主题具体化，不是已认证公司真题。
