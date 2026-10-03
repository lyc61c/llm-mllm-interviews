# 视觉语言与图文多模态

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [VLM-001 · CLIP 的结构和双向对比损失是什么？](#vlm-001)
- [VLM-002 · CLIP 如何做零样本分类？提示模板有什么作用？](#vlm-002)
- [VLM-003 · 图文检索很好，为什么关系和计数题仍可能答错？](#vlm-003)
- [VLM-004 · SigLIP 与 CLIP 的损失主要区别是什么？](#vlm-004)
- [VLM-005 · BLIP 的 ITC、ITM、LM 三个目标各做什么？](#vlm-005)
- [VLM-006 · BLIP 的 CapFilt 为什么同时需要 captioner 和 filter？](#vlm-006)
- [VLM-007 · BLIP-2 的 Q-Former 为什么能连接冻结的视觉编码器和 LLM？](#vlm-007)
- [VLM-008 · Q-Former 与两层 MLP 连接器怎样选，MLP 已普遍淘汰 Q-Former 吗？](#vlm-008)
- [VLM-009 · 原始 LLaVA 的两阶段训练分别更新哪些模块？](#vlm-009)
- [VLM-010 · VLM 指令微调的 labels 应怎样掩码？](#vlm-010)
- [VLM-011 · LLaVA-1.5 相比原始 LLaVA 的关键改进是什么？](#vlm-011)
- [VLM-012 · Qwen2-VL 的动态分辨率解决了什么问题？](#vlm-012)
- [VLM-013 · M-RoPE 如何统一文本、图像和视频位置？](#vlm-013)
- [VLM-014 · Qwen2.5-VL 的视觉编码器和时间建模有哪些变化？](#vlm-014)
- [VLM-015 · InternVL 的动态切图与全局缩略图各有什么作用？](#vlm-015)
- [VLM-016 · OCR/文档问答差，怎样判断是视觉瓶颈还是语言瓶颈？](#vlm-016)
- [VLM-017 · 单图 VLM 怎样扩展到多页文档问答？](#vlm-017)
- [VLM-018 · 视觉幻觉怎样定义、评测和缓解？](#vlm-018)
- [VLM-019 · 多模态模型看起来不看图，如何排查？](#vlm-019)
- [VLM-020 · VLM 能力如何评估，为什么不能只报一个榜单分数？](#vlm-020)
- [VLM-021 · 微调 VLM 时，视觉骨干、连接器和 LLM 该怎样冻结？](#vlm-021)
- [VLM-022 · 交错图文和多图输入如何保持图像与指代关系？](#vlm-022)
- [VLM-023 · Grounding 与普通图像问答有什么区别？](#vlm-023)
- [VLM-024 · 多模态预训练和 SFT 的数据配比怎样设计？](#vlm-024)
- [VLM-025 · 多模态检索模型与生成式 VLM 问答怎样分工？](#vlm-025)
- [VLM-026 · 以 LLaVA-1.5-7B 与 InternVL2.5-8B 为例，除骨干和训练之外有哪些差异？](#vlm-026)

<a id="vlm-001"></a>
## VLM-001 · CLIP 的结构和双向对比损失是什么？

**L1 · 社区题目线索** · 标签：CLIP / 对比学习

**30 秒回答**

CLIP 用图像和文本双编码器把两种输入映射到同一空间，以归一化特征相似度区分匹配图文对。一个 batch 同时做图到文和文到图分类，学习的是跨模态匹配表征，并非直接训练文本生成器。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 正样本通常是同一索引的图文对，其他配对作为负样本。
- 温度控制 softmax 的尖锐程度；原论文用可训练对数尺度。
- 双向交叉熵取平均，实际数据可能出现同义描述造成的假负例。

### 公式

```text
s_ij = <v_i,t_j>/τ；L = (CE(s,diag) + CE(s^T,diag))/2，其中 v、t 为单位向量。
```

### 易错点

- 不要把 CLIP 的文本编码器当作生成式 LLM。

### 面试官可能追问

- 大 batch 为什么有帮助，又有什么代价？

</details>

**技术依据**

- [MM-S010 · Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/html/2103.00020v1)

**题目出处线索**

- [MM-S001 · 淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) · `reported_question`：公开面经明确要求介绍 CLIP；此处将其拆成架构与目标。

<a id="vlm-002"></a>
## VLM-002 · CLIP 如何做零样本分类？提示模板有什么作用？

**L1 · 编辑补充题** · 标签：CLIP / 零样本

**30 秒回答**

把候选类别写成自然语言模板，经文本编码器得到类别向量，再与图像向量计算相似度并排序即可分类。模板影响文本表示和训练分布匹配，因此需验证集选择或模板集成，输出分数也不天然等于校准后的置信度。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 例如把类别名嵌入描述照片的句子，而非只用孤立标签。
- 图像与文本特征归一化后计算点积，候选类别变动会影响 softmax。
- 多模板可先聚合类别特征，再统一归一化与打分。

### 易错点

- 零样本不代表类别概念从未出现在预训练数据。

### 面试官可能追问

- 如何评估跨域类别和相似类别？

</details>

**技术依据**

- [MM-S011 · OpenAI CLIP 官方实现](https://github.com/openai/CLIP)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-003"></a>
## VLM-003 · 图文检索很好，为什么关系和计数题仍可能答错？

**L2 · 编辑补充题** · 标签：组合性 / 计数

**30 秒回答**

整体图文匹配分数高，不等于模型可靠理解主体、关系、顺序和数量。应构建只交换关系或属性的最小对照样本，看分数能否区分，再分别检查视觉细节、局部表示、训练监督和语言先验的影响。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- Winoground 用同词不同组合的图文配对检验组合性。
- 计数要区分遮挡、小目标、重复目标与语言猜测。
- 整体检索和关系问答应分开评估，不能用一个均值替代。

### 易错点

- 不要由单例错误推断所有 VLM 都缺乏该能力。

### 面试官可能追问

- 怎样生成不引入答案泄漏的对照集？

</details>

**技术依据**

- [MM-S012 · Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-004"></a>
## VLM-004 · SigLIP 与 CLIP 的损失主要区别是什么？

**L2 · 编辑补充题** · 标签：SigLIP / 对比学习

**30 秒回答**

CLIP 用 batch 内 softmax 比较图文相似度，SigLIP 对每个图文配对分别做 sigmoid 二分类，无需为损失归一化获取完整相似度分布。这改变了正负配对和通信的组织方式，但仍需要处理负样本与计算预算。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 正对和负对分别贡献二分类损失。
- 小 batch 与大 batch 的效果取决于样本数、负正比例和训练设置。
- 损失改变不意味着编码器自动具备生成或 grounding 能力。

### 易错点

- 不能说 SigLIP 完全不需要负样本或跨卡通信。

### 面试官可能追问

- 逐对损失如何实现分块计算？

</details>

**技术依据**

- [MM-S013 · Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-005"></a>
## VLM-005 · BLIP 的 ITC、ITM、LM 三个目标各做什么？

**L1 · 社区题目线索** · 标签：BLIP / 预训练目标

**30 秒回答**

BLIP 用图文对比 ITC 学整体语义空间，用图文匹配 ITM 学融合后的细粒度对应，用语言建模 LM 学基于图像生成描述。三种目标支持理解与生成任务，差别体现在编码交互和监督形式，而非三个名称的机械相加。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- ITC 分别编码图像与文本后做对比学习。
- ITM 通过跨模态交互判断图文是否匹配，可使用困难负例。
- LM 用因果文本注意力预测后续文本，视觉信息作为条件。

### 易错点

- ITM 是配对匹配，不是对比 softmax 的另一个名字。

### 面试官可能追问

- 三种目标的 attention mask 有何不同？

</details>

**技术依据**

- [MM-S014 · BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/html/2201.12086v2)

**题目出处线索**

- [MM-S001 · 淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) · `reported_question`：面经明确问 BLIP 的三个损失；仅抽该主题。

<a id="vlm-006"></a>
## VLM-006 · BLIP 的 CapFilt 为什么同时需要 captioner 和 filter？

**L2 · 社区题目线索** · 标签：BLIP / 数据清洗

**30 秒回答**

网络图文对可能含无关文字，captioner 根据图像补充描述，filter 判断原始和生成描述是否匹配图像。生成和过滤共同提升训练信号质量；生成描述也会犯错，因此需要独立过滤和人工抽检，不能把合成数据视为可靠标签。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- captioner 负责补充图像条件描述，filter 负责筛掉错配。
- 保留描述多样性与提高精确性之间需要权衡。
- 工程中进一步做去重、质量分层和域外抽检。

### 易错点

- 过滤器与生成器同源时可能共享错误偏好。

### 面试官可能追问

- 如何测量过滤后的长尾覆盖损失？

</details>

**技术依据**

- [MM-S014 · BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/html/2201.12086v2)

**题目出处线索**

- [MM-S001 · 淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) · `reported_question`：面经同一道 BLIP 问题涉及数据清洗，此题展开 CapFilt。

<a id="vlm-007"></a>
## VLM-007 · BLIP-2 的 Q-Former 为什么能连接冻结的视觉编码器和 LLM？

**L1 · 社区题目线索** · 标签：BLIP-2 / Q-Former / 可学习查询 / 两阶段预训练

**30 秒回答**

Q-Former 是带可学习查询的 Transformer：查询先自注意力交互，再跨注意力读取视觉特征，输出固定数量表示并投影为 LLM 软提示。BLIP-2 用两阶段训练对齐冻结骨干；原论文用 32 个查询，这不是所有实现的固定值。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 设视觉特征为 N×d_v，查询为 M×d_q；cross-attention 的 Q 来自查询，K/V 来自视觉特征，输出为 M×d_q，再映射到 LLM 维度。输出查询数 M 不随输入 patch 数 N 直接增长。
- 原始 Q-Former 从 BERT-base 初始化共享自注意力层，每隔一个 Transformer block 插入视觉 cross-attention。查询会相互交互，不能把它理解为对每个 patch 独立做投影。
- 第一阶段用 ITC、ITM、ITG 学图文表示：ITC 隔离 query 与文本，ITM 允许两者双向交互；ITG 用混合掩码，query 之间双向可见且不看文本，文本可看全部 query 和已生成文本。query 子图不是 causal；第二阶段以 LLM 生成目标更新 Q-Former 与投影层。
- 冻结视觉骨干和 LLM 指其参数不更新；仍需让梯度经过冻结 LLM 回到连接器。把 LLM 前向包进 no_grad 会切断所需训练路径。
- 原始 BLIP-2 生成阶段的视觉查询未直接接收用户指令；InstructBLIP 把指令送入 Q-Former，实现指令相关特征提取。不能把这一改动倒填为所有 BLIP-2 的默认机制。

### 公式

```text
Z=QFormer(Q_learned,X)∈R^{M×d_q}; V_soft=Z·W_proj∈R^{M×d_LLM}
```

### 易错点

- 可学习 query 是模型参数向量，不等于用户问题文本、检测框或 LLM 每步的 query。
- 32 是原始 BLIP-2 实验设置；更多查询会改变瓶颈与 LLM 成本，不能当成架构必须值。

### 面试官可能追问

- ITC、ITM、ITG 的 query/text 可见关系为什么不同？
- 冻结 LLM 时，如何验证连接器仍有非零梯度？

</details>

**技术依据**

- [MM-S015 · BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/html/2301.12597v3)
- [MM-S068 · InstructBLIP 官方模型文档](https://huggingface.co/docs/transformers/model_doc/instructblip)

**题目出处线索**

- [MM-S001 · 淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) · `reported_question`：面经问 BLIP2 的改进，且明确提到 Q-Former。

<a id="vlm-008"></a>
## VLM-008 · Q-Former 与两层 MLP 连接器怎样选，MLP 已普遍淘汰 Q-Former 吗？

**L2 · 社区题目线索** · 标签：连接器 / 架构比较 / Q-Former / MLP / Resampler / Token预算

**30 秒回答**

两层 MLP 逐 patch 映射，结构简单且不主动压成少量查询；Q-Former 先交互提取再压缩，减少 LLM token 负担。选型取决于局部细节、冻结策略和预算。公开论文与实现显示多种路径并存，不能据此宣称业界普遍淘汰 Q-Former。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- LLaVA-1.5 的两层连接器为 Linear→GELU→Linear，N 个视觉 token 映射后仍是 N 个；连接器本身不新增跨 token 混合，视觉编码器此前已可通过注意力融合上下文。这保留 patch 粒度，不等于保证原始像素信息无损。
- Q-Former 用 M 个查询汇聚 N 个视觉特征，增加查询自注意力与 cross-attention 成本，却可减小后续 LLM 的序列、prefill 与 KV 预算。比较应计入视觉编码、连接器和 LLM 的端到端训练/推理开销。
- 需要读小字、局部定位且 token 预算充足时，可先试逐 patch MLP；多图/视频或 LLM 输入预算紧张时，可试查询压缩。固定 M 可能形成细节瓶颈，但压缩不必然损害 OCR，必须在对应任务、分辨率与数据上消融。
- MLP 是简洁的有效基线；BLIP-2 的两阶段表征学习服务于冻结骨干，InstructBLIP 用指令感知 Q-Former，MiniCPM-V 4.5 用 3D-Resampler 压缩图像/视频。实例说明设计路径并存，不能证明商业部署占比或普遍淘汰趋势。
- Resampler 是更宽泛的特征重采样设计；MiniCPM 的可学习查询、位置编码和 cross-attention 不等于 BLIP-2 的 BERT 初始化、共享文本分支及 ITC/ITM/ITG 目标。不要把所有固定 token 压缩器都称为 Q-Former。

### 公式

```text
逐 token MLP: N×d_v→N×d_LLM；查询桥接: N×d_v→M×d_q→M×d_LLM
```

### 易错点

- 不同视觉骨干、数据和冻结策略的整模型成绩不能全部归因连接器，也不能外推成“MLP 一定更强”。
- MLP 本身通常不压 token；若系统另有 pooling/pixel shuffle，压缩应归于那些操作。

### 面试官可能追问

- 同一视觉骨干下，怎样分别控制视觉 token 数与总 FLOPs 做比较？
- 若查询压缩使 OCR 退化，你会调整查询数、切图、训练数据还是连接器？

</details>

**技术依据**

- [MM-S015 · BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/html/2301.12597v3)
- [MM-S017 · Improved Baselines with Visual Instruction Tuning](https://arxiv.org/html/2310.03744v2)
- [MM-S068 · InstructBLIP 官方模型文档](https://huggingface.co/docs/transformers/model_doc/instructblip)
- [MM-S069 · MiniCPM-V 4.5 技术报告](https://arxiv.org/html/2509.18154v1)
- [MM-S070 · MiniCPM-V 4.5 Resampler 固定提交官方实现](https://huggingface.co/openbmb/MiniCPM-V-4_5/blob/bf6f912c8f3d652bda9431e4ec0b8de07c902205/resampler.py)
- [MM-S076 · LLaVA 官方多模态投影器构建代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

**题目出处线索**

- [MM-S001 · 淘天面经：大模型（知乎英语索引页）](https://www.zhihu.com/en/article/26593966342) · `reported_question`：面经明确比较 BLIP2 Q-Former 与 LLaVA MLP 的利弊。

<a id="vlm-009"></a>
## VLM-009 · 原始 LLaVA 的两阶段训练分别更新哪些模块？

**L1 · 编辑补充题** · 标签：LLaVA / 训练阶段

**30 秒回答**

原始 LLaVA 先冻结视觉编码器和 LLM，仅训练投影层做特征对齐；再保持视觉编码器冻结，更新投影层和 LLM 学多模态指令跟随。必须说明具体版本，因为后续 VLM 的视觉解冻和阶段划分可能不同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 第一阶段图文描述数据把视觉特征接入语言嵌入空间。
- 第二阶段用对话、详细描述和推理指令学习回答格式与任务。
- 两阶段职责不同，指令效果不能只靠增大连接器获得。

### 易错点

- 原论文说 end-to-end 不代表所有模块都解冻。

### 面试官可能追问

- 域内视觉分布变化时是否应解冻视觉骨干？

</details>

**技术依据**

- [MM-S016 · Visual Instruction Tuning](https://arxiv.org/html/2304.08485v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-010"></a>
## VLM-010 · VLM 指令微调的 labels 应怎样掩码？

**L2 · 编辑补充题** · 标签：SFT / Loss Mask

**30 秒回答**

常见对话式 VLM 只对 assistant 的文本答案及结束标记计算自回归损失，用户问题、系统提示、视觉占位和 padding 用忽略标签。关键是处理视觉 token 展开后的序列对齐与因果移位，错位会造成无声的训练失败。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先生成最终多模态序列，再确保 labels 与实际 token 位置一致。
- 检查每个样本有效监督 token 数，防止答案全被屏蔽。
- 多轮历史是否监督应遵循数据配方，不能照搬纯文本模板。

### 易错点

- attention mask 与 loss mask 控制不同事情。

### 面试官可能追问

- 如何用单样本打印核对输入与目标？

</details>

**技术依据**

- [MM-S016 · Visual Instruction Tuning](https://arxiv.org/html/2304.08485v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-011"></a>
## VLM-011 · LLaVA-1.5 相比原始 LLaVA 的关键改进是什么？

**L1 · 编辑补充题** · 标签：LLaVA-1.5 / 数据

**30 秒回答**

LLaVA-1.5 通过两层 MLP 连接器、更高分辨率视觉输入，以及覆盖 VQA、OCR、区域问答的任务数据改善能力。答案格式提示也参与配方，面试时应把架构、分辨率、数据和监督分别解释，避免只背一个模型名。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 连接器从线性映射扩展为 MLP。
- CLIP 视觉骨干采用 336px 配置，提高可见细节。
- 增加学术任务数据和简短答案格式约束，减少任务输出冲突。

### 易错点

- 性能进步不意味着原始模型视觉骨干整体解冻。

### 面试官可能追问

- 格式提示为什么会影响开放问答表现？

</details>

**技术依据**

- [MM-S017 · Improved Baselines with Visual Instruction Tuning](https://arxiv.org/html/2310.03744v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-012"></a>
## VLM-012 · Qwen2-VL 的动态分辨率解决了什么问题？

**L2 · 编辑补充题** · 标签：Qwen2-VL / 视觉token

**30 秒回答**

固定尺寸缩放会损失小字并扭曲长宽比，动态分辨率让输入尺寸映射成可变长度视觉序列，再对局部 token 合并控制 LLM 成本。细节与成本的权衡仍存在，因此部署应限制像素和视觉 token 上限并按任务评估。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- ViT 使用二维位置表达处理不同空间大小。
- 邻近 2×2 视觉 token 经合并减少进入 LLM 的序列长度。
- 图像越大通常 token 越多，不能只按图像张数估算显存。

### 易错点

- 动态分辨率不是无限分辨率，也不等于完全不缩放。

### 面试官可能追问

- OCR 与自然图片如何设置不同预算？

</details>

**技术依据**

- [MM-S018 · Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/html/2409.12191v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-013"></a>
## VLM-013 · M-RoPE 如何统一文本、图像和视频位置？

**L2 · 编辑补充题** · 标签：M-RoPE / 位置编码

**30 秒回答**

Qwen2-VL 将旋转位置编码的维度分配给时间、高度和宽度。文本三者使用相同位置索引，图像固定时间并区分空间位置，视频还引入帧时间位置；它提供坐标结构，不会自动解决图文对齐或任意长度外推。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 三组位置分别作用于对应的通道子空间。
- 混合模态输入需连续安排位置编号，保持边界可解释。
- 后续版本可改变时间编码定义，回答需说明版本。

### 易错点

- Qwen2-VL 帧序号与 Qwen2.5-VL 绝对时间不能混写。

### 面试官可能追问

- 多图与视频的时间位置为何应区分？

</details>

**技术依据**

- [MM-S018 · Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/html/2409.12191v2)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-014"></a>
## VLM-014 · Qwen2.5-VL 的视觉编码器和时间建模有哪些变化？

**L2 · 社区题目线索** · 标签：Qwen2.5-VL / 窗口注意力

**30 秒回答**

Qwen2.5-VL 在视觉编码器多数层用窗口注意力，同时保留少量全局注意力；引入动态 FPS 采样和对绝对时间对齐的位置编码。这降低局部视觉计算并增强时间表达，但模型整体计算不因此变成纯线性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 窗口层在固定窗口大小下对 patch 数近似线性。
- 四层全局注意力仍建模跨窗口信息并产生全局成本。
- 合并视觉特征后送入语言模型，LLM 长度成本也需计算。

### 易错点

- 不要把整个 ViT 或整个 VLM 的复杂度都写成 O(n)。

### 面试官可能追问

- 窗口大小与小目标识别如何权衡？

</details>

**技术依据**

- [MM-S019 · Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)

**题目出处线索**

- [MM-S007 · 字节多模态算法面试题详解](https://zhuanlan.zhihu.com/p/2023042484121413435) · `search_snippet`：搜索片段含 Qwen2.5-VL 结构问法；答案以作者技术报告校对。

<a id="vlm-015"></a>
## VLM-015 · InternVL 的动态切图与全局缩略图各有什么作用？

**L2 · 编辑补充题** · 标签：InternVL / 动态切图

**30 秒回答**

动态切图把高分辨率图像按合适长宽比分成多个固定大小 tile，保留局部细节；全局缩略图补充整体布局。输入代价随 tile 数增加，必须维护同一原图的 tile 顺序与数量，裁剪坐标也要能映射回原图。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 官方示例用长宽比匹配网格，再裁成固定尺寸块。
- 缩略图帮助模型理解各局部块之间的整体关系。
- max_num 限制切块预算；版本和实现可能使用不同规则。

### 易错点

- 不能把每个 tile 误当作一张独立原图。

### 面试官可能追问

- 表格跨 tile 边界时如何减少结构损失？

</details>

**技术依据**

- [MM-S020 · InternVL2 Quick Start 官方文档](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-016"></a>
## VLM-016 · OCR/文档问答差，怎样判断是视觉瓶颈还是语言瓶颈？

**L2 · 社区题目线索** · 标签：OCR / DocVQA

**30 秒回答**

先用裁剪或提高分辨率检查字是否可辨，再把可靠 OCR 文本与布局作为对照输入。若文本对照能答而图片不能，优先检查视觉表示和接入；若两者都错，检查问题语义、布局关系、检索范围及监督。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 分开报告文字识别、字段关联和推理准确率。
- 比较大小字、旋转、扫描噪声和多列排版切片。
- 核对预处理方向、归一化和裁剪是否损坏文本。

### 易错点

- 外部 OCR 同样有误差，不能直接当完美真值。

### 面试官可能追问

- 图表数值读对但单位读错如何构造评测？

</details>

**技术依据**

- [MM-S021 · DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398)

**题目出处线索**

- [MM-S006 · ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) · `reported_question`：公开题目列表包含 document understanding/layout。

<a id="vlm-017"></a>
## VLM-017 · 单图 VLM 怎样扩展到多页文档问答？

**L3 · 社区题目线索** · 标签：多页文档 / 检索

**30 秒回答**

多页文档需要同时解决相关页定位与页面内读取。可先检索候选页，再结合页码、视觉与文本布局回答；长文档也可使用分层聚合。应保留答案所在页的证据，分别衡量检索召回和最终问答准确率。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 多页直接拼接会增加 token、干扰页和位置混淆。
- 页级检索后做细粒度读取，评估跨页关系是否被截断。
- 训练样本需包含页标识、无答案页和跨页问题。

### 易错点

- 检索错页时不能靠生成答案掩盖错误。

### 面试官可能追问

- 怎样处理答案跨两页的表格？

</details>

**技术依据**

- [MM-S022 · Hierarchical multimodal transformers for Multi-Page DocVQA](https://arxiv.org/abs/2212.05935)

**题目出处线索**

- [MM-S006 · ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) · `reported_question`：公开题目明确提出单图问答有效、多页文档失败的场景。

<a id="vlm-018"></a>
## VLM-018 · 视觉幻觉怎样定义、评测和缓解？

**L2 · 社区题目线索** · 标签：视觉幻觉 / POPE

**30 秒回答**

视觉幻觉是回答中的对象、属性或关系不受图像支持。POPE 通过对象存在性问答衡量一部分幻觉，开放描述还需事实核对。缓解要覆盖数据、视觉证据、训练与解码，不能只依赖更保守的输出风格。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 按对象、属性、关系分别建立标注和失败切片。
- 同时关注错误肯定和过度否定，避免只会回答不知道。
- 检查真实图像条件、语言先验和数据共现偏差。

### 易错点

- POPE 分数不是所有视觉幻觉的完整度量。

### 面试官可能追问

- 更短的描述为何可能降低幻觉指标却损失信息？

</details>

**技术依据**

- [MM-S023 · Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)

**题目出处线索**

- [MM-S006 · ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) · `reported_question`：题目列表包含 VLM 生成事实错误图像描述的排查场景。

<a id="vlm-019"></a>
## VLM-019 · 多模态模型看起来不看图，如何排查？

**L3 · 社区题目线索** · 标签：图像依赖 / 消融

**30 秒回答**

用同一问题替换、遮挡和打乱图像，观察答案对视觉证据是否敏感，再检查视觉输入是否真正进入模型以及梯度是否到达连接器。若链路正常，再检验文本捷径和数据偏差；只看注意力热图不足以证明因果依赖。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先核对占位符、pixel_values、视觉 token 数和 mask。
- 对照只用文本、正确图像和错配图像的准确率。
- 检查连接器梯度、视觉特征变化和标签是否泄漏答案。

### 易错点

- 错配图像不改变答案也可能是问题本来不依赖图像。

### 面试官可能追问

- 视觉对比解码为何可能降低语言先验干扰？

</details>

**技术依据**

- [MM-S024 · Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
- [MM-S026 · LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA)

**题目出处线索**

- [MM-S006 · ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) · `reported_question`：公开题目明确提出模型忽略图像只按文本描述的问题。

<a id="vlm-020"></a>
## VLM-020 · VLM 能力如何评估，为什么不能只报一个榜单分数？

**L2 · 编辑补充题** · 标签：MMBench / 评估

**30 秒回答**

多模态任务涵盖感知、OCR、空间关系、推理和指令遵循，一个平均分会隐藏弱项。应固定模型版本、输入预算和评测协议，报告任务切片、误差类型、延迟与成本，并用人工抽检确认自动评分可靠性。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 选择题可测标准化能力，开放问答需更细事实标注。
- 控制答案格式、随机性、分辨率与测试集污染。
- 业务评估加入无答案、歧义图像和跨域样本。

### 易错点

- 不同提示或图像预算下的榜单分数不宜直接横比。

### 面试官可能追问

- 模型裁判如何减少长度偏好与视觉遗漏？

</details>

**技术依据**

- [MM-S025 · MMBench: Is Your Multi-modal Model an All-around Player?](https://arxiv.org/abs/2307.06281)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-021"></a>
## VLM-021 · 微调 VLM 时，视觉骨干、连接器和 LLM 该怎样冻结？

**L3 · 社区题目线索** · 标签：微调 / 冻结策略

**30 秒回答**

先判断差距来自视觉域、跨模态对齐还是任务指令，再选择更新模块。只训练连接器成本低，LLM LoRA 能调整任务行为，视觉解冻适合视觉分布明显变化。应用消融验证收益，并防止少量域数据破坏原有能力。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 按参数组记录学习率、梯度与可训练参数数量。
- 先验证连接器和格式，再扩大更新范围。
- 加入通用图文与纯文本回归集测能力遗忘。

### 易错点

- 冻结参数不等于禁止该模块向前层传递梯度。

### 面试官可能追问

- 新视觉骨干替换后能直接复用旧连接器吗？

</details>

**技术依据**

- [MM-S026 · LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA)

**题目出处线索**

- [MM-S006 · ai-engineering-interview-questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/README.md) · `reported_question`：公开题目列表包含 vision-language model fine-tuning。

<a id="vlm-022"></a>
## VLM-022 · 交错图文和多图输入如何保持图像与指代关系？

**L2 · 编辑补充题** · 标签：多图 / Flamingo

**30 秒回答**

多图输入要保留图像边界、顺序和指代标记，让文本能引用对应图像。交错图文训练提供这种关联监督；推理时应核对占位符与视觉块一一对应，训练和测试的图数分布也会影响多图比较能力。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 图像边界明确，页面或图片编号与原始数据保持一致。
- 训练应包含比较、跨图关联和局部引用任务。
- token 压缩后仍要保存每张图的归属信息。

### 易错点

- 按一张图训练的模型不会自动可靠支持任意张图。

### 面试官可能追问

- 交换两图顺序后正确答案应如何变化？

</details>

**技术依据**

- [MM-S027 · Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [MM-S020 · InternVL2 Quick Start 官方文档](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-023"></a>
## VLM-023 · Grounding 与普通图像问答有什么区别？

**L2 · 编辑补充题** · 标签：Grounding / 坐标

**30 秒回答**

普通问答生成文本，grounding 还要把语言中的实体或短语绑定到图像区域，例如输出框坐标。它需要区域对应监督和坐标协议，评测应检查框的定位与语义；调整尺寸、裁剪或切图后必须正确反变换坐标。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 明确输出坐标是像素、归一化数值还是离散位置 token。
- 同时核对类别/短语与边界框，不只检查输出格式。
- 用 IoU、召回和复杂空间关系切片衡量定位。

### 易错点

- 合法坐标并不代表框对应了正确实体。

### 面试官可能追问

- 如何处理多个相同物体的指代表达？

</details>

**技术依据**

- [MM-S028 · Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/abs/2306.14824)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-024"></a>
## VLM-024 · 多模态预训练和 SFT 的数据配比怎样设计？

**L3 · 编辑补充题** · 标签：数据配比 / 能力平衡

**30 秒回答**

配比应围绕目标能力和数据质量选择，而非简单按原始样本量混合。自然图像、OCR、文档、区域问答和纯文本的 token 成本与难度不同，要先建立分任务评测，再通过受控采样与消融寻找能力和成本的平衡。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 记录去重后样本数、视觉 token 数和有效答案 token 数。
- 低质量海量描述可能压过稀缺细粒度监督。
- 持续检查纯文本回归、视觉细节和域外泛化。

### 易错点

- 不能把单篇论文的配比当跨模型最优配方。

### 面试官可能追问

- 如何区分数据质量收益与训练步数收益？

</details>

**技术依据**

- [MM-S029 · Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-025"></a>
## VLM-025 · 多模态检索模型与生成式 VLM 问答怎样分工？

**L2 · 编辑补充题** · 标签：ColPali / 检索 / RAG

**30 秒回答**

检索模型把问题和视觉文档映射成可评分表示，负责找证据；生成式 VLM 结合检索页和问题生成答案。视觉多向量可保留页面局部信息，但索引与匹配成本更高；系统应分别衡量检索召回、答案正确和引用定位。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 单向量适合紧凑索引，多向量支持局部匹配。
- 保留页面原图、页码与区域信息供读取和溯源。
- 用无关页和跨页题检验召回及证据充分性。

### 易错点

- 会生成图像描述不代表天然适合大规模向量检索。

### 面试官可能追问

- 扫描文档为什么可能从视觉检索获益？

</details>

**技术依据**

- [MM-S030 · ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="vlm-026"></a>
## VLM-026 · 以 LLaVA-1.5-7B 与 InternVL2.5-8B 为例，除骨干和训练之外有哪些差异？

**L2 · 编辑补充题** · 标签：LLaVA-1.5 / InternVL2.5 / 动态切图 / PixelShuffle / 空间结构 / Token预算 / 视频

**30 秒回答**

以 LLaVA-1.5-7B 与 InternVL2.5-8B 为例，前者通常将单图补方后送入 336px 编码器，保留 576 个 patch token；后者动态切成 448px 块，先空间重排压至每块 256 token，再加缩略图或按帧组织输入。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 输入组织：本次 LLaVA-1.5-7B 配置为 image_aspect_ratio=pad、336px、14px patch；InternVL2.5 官方示例按长宽比选择网格切 448px tile，max_num=12 限局部块数，多块时另加全局 thumbnail，因此总块数可为 13。
- 连接器：前者保留去 CLS 后的 24×24=576 patch，经两层 MLP 映射；后者去 CLS 后将 32×32 网格按 ratio=0.5 重排为 16×16，邻接 2×2 特征进入通道，再经 LayerNorm 与两层 MLP，得到每块 256 token。
- 空间结构：InternVL 代码将 tile 按网格行优先排列并在末尾追加 thumbnail，pixel shuffle 保留邻域分组；LLM 最终接收的仍是一维 token 序列。跨 tile 表格/对象关系不能仅凭“保留二维结构”保证，需要全局视图与训练支持。
- 预算：InternVL 的视觉 token 约为 256×实际块数，不含边界/文本 token；12 个局部块加缩略图为 3328，而非整张原图固定 256。切图提高可见细节，也增加视觉编码、LLM prefill 与 KV 成本；比较应固定任务和实际输入预算。
- 多图/视频：InternVL2.5 quickstart 通过 num_patches_list 标记每张图/每帧的块数，并给出抽帧、Frame 编号示例；LLaVA-1.5 基础版以单图为主。不能将 LLaVA-NeXT/OneVision 的高分辨率或视频能力倒填到 1.5，也不能把抽帧接口等同专用时序编码。

### 公式

```text
LLaVA-1.5: (336/14)^2=576；InternVL2.5 每tile: (448/14)^2×0.5^2=256；总量=256×(n_tiles+1_{thumbnail且n_tiles>1})
```

### 易错点

- 本题版本、checkpoint、预处理与 token 数要一起报；不能以两个系列名概括所有后续版本。
- InternVL 函数名是 pixel_shuffle，但 ratio=0.5 实际做空间缩小、通道增大的重排，不是图像超分辨率上采样。

### 面试官可能追问

- 为什么 thumbnail 不计入 max_num=12，却必须计入视觉 token 与显存？
- 给定相同 OCR 任务预算，怎样消融 tile 数、分辨率与空间压缩率？

</details>

**技术依据**

- [MM-S017 · Improved Baselines with Visual Instruction Tuning](https://arxiv.org/html/2310.03744v2)
- [MM-S071 · InternVL2.5 Quick Start 官方文档](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html)
- [MM-S072 · InternVL2.5-8B modeling_internvl_chat 官方实现](https://huggingface.co/OpenGVLab/InternVL2_5-8B/blob/main/modeling_internvl_chat.py)
- [MM-S073 · InternVL2.5-8B 官方 checkpoint 配置](https://huggingface.co/OpenGVLab/InternVL2_5-8B/blob/main/config.json)
- [MM-S074 · LLaVA-1.5-7B 官方 checkpoint 配置](https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json)
- [MM-S075 · LLaVA 官方 CLIP 视觉特征选择代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py)
- [MM-S076 · LLaVA 官方多模态投影器构建代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
