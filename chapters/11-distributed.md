# 分布式训练与显存工程

[返回仓库首页](../README.md) · [来源目录](../SOURCES.md) · [阅读全部题目](../exports/questions.jsonl)

L1 基础 · L2 进阶 · L3 深入。难度为编辑判断；社区线索不等于雇主确认真题。所有答案为独立整理。

## 题目导航

- [DST-001 · 数据并行 DDP 每个 step 做了什么？](#dst-001)
- [DST-002 · DDP 梯度累积如何保持与大 batch 等价？](#dst-002)
- [DST-003 · 大模型训练显存怎样估算和优化？以 7B Adam 为例](#dst-003)
- [DST-004 · ZeRO-1/2/3 各切分什么？理想状态显存是多少？](#dst-004)
- [DST-005 · FSDP 与 ZeRO-3 有什么联系和区别？](#dst-005)
- [DST-006 · Megatron 的 MLP 张量并行为什么先列切再行切？](#dst-006)
- [DST-007 · 流水线并行的 bubble 从哪里来，怎样降低？](#dst-007)
- [DST-008 · Megatron 的 Sequence Parallelism 与 TP 怎样配合？](#dst-008)
- [DST-009 · Context Parallelism 如何训练更长上下文？](#dst-009)
- [DST-010 · MoE 的专家并行有哪些通信与负载问题？](#dst-010)
- [DST-011 · AllReduce、ReduceScatter、AllGather 怎样对应？](#dst-011)
- [DST-012 · DDP 怎样重叠反向计算与梯度通信？](#dst-012)
- [DST-013 · 激活 checkpointing 为什么省显存，有哪些正确性条件？](#dst-013)
- [DST-014 · BF16 与 FP16 混合精度训练为什么表现不同？](#dst-014)
- [DST-015 · CPU/NVMe Offload 适合哪些场景，为什么可能变慢？](#dst-015)
- [DST-016 · 分布式训练怎样做到可靠断点续训？](#dst-016)
- [DST-017 · DistributedSampler、set_epoch 和 drop_last 怎么用？](#dst-017)
- [DST-018 · 训练挂在 NCCL collective 上，如何定位？](#dst-018)
- [DST-019 · MFU、HFU 与 GPU utilization 有什么区别？](#dst-019)
- [DST-020 · 多模态训练吞吐波动，怎样做 profiling 与负载平衡？](#dst-020)

<a id="dst-001"></a>
## DST-001 · 数据并行 DDP 每个 step 做了什么？

**L1 · 社区题目线索** · 标签：DDP / 数据并行

**30 秒回答**

DDP 让各进程持有模型副本并处理各自数据，反向传播期间聚合梯度，各副本执行相同优化器更新。它不会自动分割数据集，也通常不切分模型状态；需要一致初始化、数据采样和 collective 顺序才能正确训练。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 训练程序负责为各 rank 提供不同的数据子集。
- 梯度同步后，各 rank 用同一结果更新本地参数。
- 广播 buffers 与梯度归约是不同机制。

### 易错点

- DDP 不会让每张卡只持有一部分模型参数。

### 面试官可能追问

- 为什么单卡可训练，多卡却可能 OOM？

</details>

**技术依据**

- [MM-S050 · PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)

**题目出处线索**

- [MM-S002 · 大模型基础架构岗面经（二）](https://www.nowcoder.com/discuss/656279057671155712) · `reported_question`：公开基础架构面经明确问对数据并行的理解。

<a id="dst-002"></a>
## DST-002 · DDP 梯度累积如何保持与大 batch 等价？

**L2 · 编辑补充题** · 标签：梯度累积 / Loss归一化

**30 秒回答**

等大小 microbatch 时，按累积步数缩放损失并只在最后同步，可得到同一有效 batch 的平均梯度。变长文本应按有效监督 token 总数加权，否则短序列被放大；优化器步数、调度器和梯度裁剪都要按更新边界处理。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- DDP no_sync 需同时包住前向和反向过程。
- 每组累积结束后再裁剪、step 与清梯度。
- dropout、BatchNorm 或随机性可能使其与整批计算不严格一致。

### 易错点

- 框架若已归一化损失，再除一次会使梯度过小。

### 面试官可能追问

- 各 rank 有效 token 数不同时如何得到全局均值？

</details>

**技术依据**

- [MM-S050 · PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-003"></a>
## DST-003 · 大模型训练显存怎样估算和优化？以 7B Adam 为例

**L1 · 社区题目线索** · 标签：显存估算 / Adam / Microbatch / GradientAccumulation / ZeRO / FlashAttention / Streaming

**30 秒回答**

先将显存拆为权重、梯度、优化器/主权重、激活与工作区；16字节每参数仅是特定混合精度Adam假设。再按瓶颈选小microbatch配累积、重计算、状态分片或offload。FlashAttention减少中间存储；数据流式加载主要解决磁盘/主存问题。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 若低精度权重和梯度各2P bytes，FP32主权重4P、Adam一二阶状态合计8P，模型状态为16P；P=7×10^9对应112GB十进制，约104.3GiB。实现若用FP32梯度、无独立主权重或不同优化器须重算。峰值另含保留激活、算子工作区、通信桶与临时聚合；allocated、reserved和设备已用量也不是同一口径。
- 固定有效batch时，减小每卡microbatch并增加累积步数，逐个前向/反向释放计算图，通常可降低激活峰值；保持microbatch不变只增加累积步数不会自动省激活，也不减少底座权重或Adam状态。等大小样本时B_eff=B_micro×A×D；变长样本按有效监督token加权，避免重复归一化。FSDP no_sync等实现还可能保留完整梯度，必须实测峰值。
- 普通DDP复制模型状态；TP切分单层矩阵与计算，PP切分不同层，ZeRO在数据并行组内依次分片optimizer/master、gradients、parameters。分片范围和峰值不同，当前模块的参数all-gather、通信桶、PP在途microbatch均可能增加临时占用。CPU/NVMe offload迁移状态以换容量，受传输与主存/存储带宽限制；详见DST-004/013/015。
- 激活checkpointing用反向重算换保留激活；标准FlashAttention以分块和在线softmax避免完整S×S注意力矩阵驻留HBM，计算精确稠密注意力，浮点次序可有差异，算术复杂度仍约O(S²)。稀疏attention改变可见连接/算法，需单独评测质量，不应与FlashAttention混称；LoRA等减少可训练状态，长序列激活仍可很大。
- 数据集streaming按需读取数据，可减少整库下载/转换及主存常驻，并需设置分片、shuffle buffer和worker；它不自动压缩模型参数、optimizer或当前batch激活。先profile峰值出现于前向、反向还是step，再比较显存、有效token吞吐与任务质量；减长度、量化、冻结或稀疏化均需说明目标/精度变化。

### 公式

```text
M_states=(2+2+4+4+4)P=16P bytes（指定dtype的Adam例）；M_peak还含activations/workspace/communication；B_eff=B_micro×A×D（等大小microbatch，D为数据并行度）
```

### 易错点

- 仅加梯度累积就声称显存下降，或用16P状态账单冒充所有实现的峰值显存。
- 把数据流式加载、TP/PP、ZeRO和稀疏attention视为同一种省显存操作，忽略各自影响的对象及通信/质量成本。

### 面试官可能追问

- microbatch已经是1而权重与optimizer仍放不下时，应优先尝试哪些状态优化？
- 怎么用profiler与峰值allocated/reserved区分激活、参数聚合和optimizer.step的OOM？

</details>

**技术依据**

- [MM-S052 · ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/html/1910.02054v3)
- [MM-S050 · PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
- [MM-S053 · DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [MM-S055 · Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4)
- [MM-S057 · Megatron Bridge Parallelisms 官方文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/parallelisms.md)
- [MM-S060 · PyTorch torch.utils.checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [CORE-S046 · LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CORE-S057 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [MM-S077 · Dataset streaming — Hugging Face Datasets](https://huggingface.co/docs/datasets/stream)
- [MM-S078 · Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/gradient_accumulation)
- [MM-S079 · Gradient synchronization — Accelerate main](https://huggingface.co/docs/accelerate/main/en/concept_guides/gradient_synchronization)

**题目出处线索**

- [MM-S003 · NLP 大模型春招记录](https://www.nowcoder.com/discuss/601149086300971008) · `search_snippet`：牛客搜索片段明确问 7B 微调 Adam 混合精度内存倍数。

<a id="dst-004"></a>
## DST-004 · ZeRO-1/2/3 各切分什么？理想状态显存是多少？

**L1 · 社区题目线索** · 标签：ZeRO / 显存

**30 秒回答**

ZeRO 在数据并行组内依次切分优化器状态、梯度和参数，减少副本冗余。显存收益要按状态 dtype 和数据并行度计算，不能简单说每升一级都减半；第三阶段还需临时收集当前计算模块的参数，峰值仍有额外成本。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 在 16P 假设下，Stage 1 为 4P+12P/D。
- Stage 2 为 2P+14P/D，Stage 3 理想持久状态为 16P/D。
- 这些不含激活、通信桶与未分片参数，也不保证每 rank 峰值相同。

### 公式

```text
D 为数据并行分片数；M1=4P+12P/D，M2=2P+14P/D，M3≈16P/D（沿用 DST-003 的 dtype 假设）。
```

### 易错点

- ZeRO 切状态冗余，与 TP 切单个矩阵计算不同。

### 面试官可能追问

- 为什么 ZeRO-3 节省显存却可能更慢？

</details>

**技术依据**

- [MM-S052 · ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/html/1910.02054v3)
- [MM-S053 · DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)

**题目出处线索**

- [MM-S002 · 大模型基础架构岗面经（二）](https://www.nowcoder.com/discuss/656279057671155712) · `reported_question`：面经明确问 DeepSpeed 各 ZeRO stage 的作用。

<a id="dst-005"></a>
## DST-005 · FSDP 与 ZeRO-3 有什么联系和区别？

**L2 · 社区题目线索** · 标签：FSDP / ZeRO

**30 秒回答**

两者都可将参数、梯度和优化器状态分片，并在需要计算时收集参数、归约后保留局部梯度。区别主要在框架集成、分片表示、通信调度和配置接口；回答 FSDP 时要指明版本，FSDP2 的 fully_shard 与旧包装 API 不同。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- FSDP2 按参数使用 DTensor 表示分片。
- 模块分组影响参数收集粒度、显存峰值和通信重叠。
- reshard 策略以驻留内存换重复收集成本。

### 易错点

- 不能认为 FSDP 与 DeepSpeed 配置可直接互换。

### 面试官可能追问

- 为什么只在根模块分片可能提高峰值显存？

</details>

**技术依据**

- [MM-S054 · PyTorch FSDP2 fully_shard 文档](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)

**题目出处线索**

- [MM-S004 · llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) · `reported_topic`：公开 Infra 题目集合讨论 FSDP 收集参数与分片梯度；公司/频率未采纳。

<a id="dst-006"></a>
## DST-006 · Megatron 的 MLP 张量并行为什么先列切再行切？

**L2 · 社区题目线索** · 标签：张量并行 / Megatron

**30 秒回答**

按矩阵乘法记号，第一层权重沿输出维度切分，各卡独立得到并激活部分中间特征；第二层沿输入维度切分，各卡得到输出的部分和，再归约求和。这样可避免在两次 GEMM 之间收集中间特征，具体通信依布局调整。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 非线性通常可逐元素作用于各分片输出。
- 第二层局部乘积需要相加，不是简单拼接。
- 反向也有对应通信，SP 等方案可改变 collective 布局。

### 公式

```text
W1=[W1_1,…,W1_D]，H_i=φ(XW1_i)；W2 按对应输入维分块，Y=Σ_i H_i W2_i。
```

### 易错点

- 框架权重存储可能转置，必须明确所谓行和列。

### 面试官可能追问

- attention 中 QKV 与输出投影怎样对应切分？

</details>

**技术依据**

- [MM-S055 · Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4)

**题目出处线索**

- [MM-S003 · NLP 大模型春招记录](https://www.nowcoder.com/discuss/601149086300971008) · `search_snippet`：牛客片段出现张量并行与 ZeRO 区别；此题延伸层内切分细节。

<a id="dst-007"></a>
## DST-007 · 流水线并行的 bubble 从哪里来，怎样降低？

**L2 · 社区题目线索** · 标签：流水线并行 / Bubble

**30 秒回答**

流水线把不同层分配到不同阶段，启动和排空时部分卡在等待，形成 bubble。用多个 microbatch、合理前后向调度、虚拟阶段和负载平衡可降低浪费，但更多 microbatch、通信和激活驻留也有成本，不能只按层数平均切分。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- GPipe 与 1F1B 的激活驻留和调度不同。
- 阶段耗时受层结构、输入长度和通信影响。
- 理想等时阶段的近似只用于直觉估算。

### 公式

```text
理想 fill-drain 且 p 个等时阶段、m 个 microbatch：bubble 占比约 (p−1)/(m+p−1)；实际依调度和负载而变。
```

### 易错点

- 增大 microbatch 个数不一定增大每个 microbatch 的 batch size。

### 面试官可能追问

- 视觉编码器放在首阶段为何易失衡？

</details>

**技术依据**

- [MM-S056 · Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

**题目出处线索**

- [MM-S002 · 大模型基础架构岗面经（二）](https://www.nowcoder.com/discuss/656279057671155712) · `reported_question`：面经明确问流水线并行理解。

<a id="dst-008"></a>
## DST-008 · Megatron 的 Sequence Parallelism 与 TP 怎样配合？

**L2 · 编辑补充题** · 标签：序列并行 / SP

**30 秒回答**

Megatron SP 将部分原来在 TP 各卡重复的 LayerNorm、Dropout 等操作沿序列维切分，降低激活驻留，并在进入需全序列或张量切分的区域时转换布局。它通常依赖 TP，不等于把所有长上下文 attention 任意分给各卡。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 作用重点是 TP 之外重复的逐 token 操作。
- 布局转换通常使用 all-gather/reduce-scatter。
- 说明具体框架语义，其他文献也会用序列并行称不同方案。

### 易错点

- SP 和 CP 名称相似，但在 Megatron 中切分范围不同。

### 面试官可能追问

- SP 为什么不需要切开一条序列的语义？

</details>

**技术依据**

- [MM-S057 · Megatron Bridge Parallelisms 官方文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/parallelisms.md)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-009"></a>
## DST-009 · Context Parallelism 如何训练更长上下文？

**L3 · 编辑补充题** · 标签：上下文并行 / CP

**30 秒回答**

CP 把一条序列的 token 及各层激活分到多张卡，每卡处理局部 query，同时交换或收集 attention 所需的其他 key/value。它降低每卡激活压力，但要正确实现全局因果 mask、位置和通信，不能只在局部片段做独立 attention。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 跨分片 attention 仍需访问允许范围的 KV。
- 因果场景可用交错分配改善不同位置的计算失衡。
- 结合 packing 时需要维护样本边界，禁止跨样本注意力。

### 易错点

- 将长文本切段分别前向不是等价 CP。

### 面试官可能追问

- CP 与重计算分别解决哪种显存占用？

</details>

**技术依据**

- [MM-S057 · Megatron Bridge Parallelisms 官方文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/parallelisms.md)
- [MM-S058 · Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-010"></a>
## DST-010 · MoE 的专家并行有哪些通信与负载问题？

**L3 · 编辑补充题** · 标签：专家并行 / MoE

**30 秒回答**

专家并行把不同专家放在不同卡，router 按 token 选择专家后进行分发，专家计算完成再汇集结果。瓶颈包括 token 不均衡、all-to-all 传输和小批 GEMM，需结合负载约束、分组计算与拓扑，稀疏激活不代表免费扩展。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 记录每专家 token 分布与溢出/丢弃比例。
- 共享专家和稠密部分仍有独立的并行布局。
- 通信与计算要在同一时间线观察，避免只统计活跃参数。

### 易错点

- EP 大小不总能独立乘到总卡数上，取决于分组映射。

### 面试官可能追问

- 增加 top-k 为什么影响效果、显存和通信？

</details>

**技术依据**

- [MM-S058 · Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-011"></a>
## DST-011 · AllReduce、ReduceScatter、AllGather 怎样对应？

**L1 · 社区题目线索** · 标签：Collective / Ring AllReduce

**30 秒回答**

AllReduce 让各 rank 得到全量归约结果；ReduceScatter 归约后每卡只保留一块；AllGather 再把各块分发给所有卡。先 ReduceScatter 再 AllGather 可实现 AllReduce，分片训练则可停在局部归约结果以节省驻留。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 所有参与 rank 需按相容顺序调用相同 collective。
- 归约算子与 tensor shape/dtype 必须相容。
- ring 分两阶段传输块，但 NCCL 会按情况选择不同算法。

### 公式

```text
理想 ring、D 张卡、每卡归约数据大小 S：每卡发送量约 2(D−1)S/D，接收量相同；不含协议开销。
```

### 易错点

- AllGather 是拼接收集，不是求和归约。

### 面试官可能追问

- 小消息为什么可能更适合低延迟算法？

</details>

**技术依据**

- [MM-S059 · NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

**题目出处线索**

- [MM-S004 · llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) · `reported_topic`：Infra 题目集合列出 ring all-reduce 与 reduce-scatter/all-gather 等价关系。

<a id="dst-012"></a>
## DST-012 · DDP 怎样重叠反向计算与梯度通信？

**L2 · 社区题目线索** · 标签：通信重叠 / DDP Bucket

**30 秒回答**

DDP 把梯度组织为 bucket，某个 bucket 的梯度都就绪后即异步归约，让通信与后续反向计算重叠。bucket 大小、参数顺序和网络决定效果；过小增加启动开销，过大推迟通信，必须通过时间线观察真正被隐藏的部分。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- autograd hook 标记梯度就绪，reducer 控制同步顺序。
- 各 rank 必须使用一致 bucket collective 顺序。
- find_unused_parameters 会增加图遍历成本，应按模型需要选择。

### 易错点

- 异步 API 并不保证通信与计算实际并行。

### 面试官可能追问

- 动态图中未使用视觉分支会怎样影响同步？

</details>

**技术依据**

- [MM-S051 · PyTorch Distributed Data Parallel 设计说明](https://docs.pytorch.org/docs/2.14/notes/ddp.html)

**题目出处线索**

- [MM-S004 · llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) · `reported_topic`：Infra 集合列出 reducer/bucket/overlap 深挖题；这里概括原理。

<a id="dst-013"></a>
## DST-013 · 激活 checkpointing 为什么省显存，有哪些正确性条件？

**L2 · 社区题目线索** · 标签：激活重计算 / Checkpointing

**30 秒回答**

激活 checkpointing 保存部分边界张量，反向时重新运行前向恢复所需中间值，以额外计算换显存。选择重算粒度需同时看峰值与 step 时间，并保证随机数、控制流和状态更新一致，否则可能改变梯度或产生重复副作用。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 它省的是保留激活，不直接切分参数和 Adam 状态。
- 保留 RNG 可维持 dropout 一致性，但有开销。
- PyTorch 可重入与非可重入版本行为不同，应显式选择。

### 易错点

- 带缓存更新或依赖全局状态的函数不能盲目重算。

### 面试官可能追问

- 什么时候省显存反而能提升吞吐？

</details>

**技术依据**

- [MM-S060 · PyTorch torch.utils.checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)

**题目出处线索**

- [MM-S003 · NLP 大模型春招记录](https://www.nowcoder.com/discuss/601149086300971008) · `search_snippet`：牛客片段讨论反向重计算与 activation 显存。

<a id="dst-014"></a>
## DST-014 · BF16 与 FP16 混合精度训练为什么表现不同？

**L1 · 编辑补充题** · 标签：BF16 / FP16 / AMP

**30 秒回答**

两者都占两字节，但 BF16 指数位更宽、尾数更少，动态范围更接近 FP32；FP16 更容易出现梯度下溢，常配合 loss scaling。混合精度会按算子选择 dtype，归约和优化器状态也需单独核对，不能只看模型权重类型。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- autocast 选择算子执行精度，不是把所有运算强制成同一 dtype。
- GradScaler 缩放损失，更新前解除缩放再裁剪。
- BF16 不保证训练不会 NaN，输入、归一化和学习率仍可能有问题。

### 易错点

- BF16 范围更大不等于每个数都比 FP16 更精确。

### 面试官可能追问

- loss scaling 与模型输出 logits 温度有何区别？

</details>

**技术依据**

- [MM-S061 · PyTorch AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-015"></a>
## DST-015 · CPU/NVMe Offload 适合哪些场景，为什么可能变慢？

**L2 · 编辑补充题** · 标签：Offload / ZeRO

**30 秒回答**

Offload 将部分参数、梯度或优化器状态迁出 GPU，利用 CPU 内存和 NVMe 扩大可训练规模，但数据搬运与 CPU 优化器会增加关键路径。适合显存无法容纳的任务，应测吞吐、主存占用与传输重叠，不能把节省显存等同于加速。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 参数 offload 的阶段支持与优化器 offload 不同。
- pinned memory 与预取能帮助传输，但会占主存资源。
- 考虑 PCIe、存储带宽、线程数与缓存命中。

### 易错点

- 状态可放下不代表当前模块的完整计算也能放下。

### 面试官可能追问

- 怎样判断瓶颈在 CPUAdam 还是传输？

</details>

**技术依据**

- [MM-S053 · DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-016"></a>
## DST-016 · 分布式训练怎样做到可靠断点续训？

**L3 · 编辑补充题** · 标签：Checkpoint / 恢复

**30 秒回答**

续训需要恢复参数之外的优化器、调度器、随机数、采样进度和训练计数，分片模型还需一致的分片元数据。保存必须保证所有 rank 的状态对应同一更新边界，并验证恢复后的下一步与连续训练一致，不能只保存 rank0 的权重。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 保存模型与优化器 state_dict，并记录框架和并行配置。
- 考虑异步保存完成标记，避免把未写完检查点视为有效。
- 变化 world size 时需支持重分片并明确样本顺序语义。

### 易错点

- 能加载权重不代表优化器和数据进度已恢复。

### 面试官可能追问

- 如何测试故障发生在保存过程中？

</details>

**技术依据**

- [MM-S062 · PyTorch Distributed Checkpoint 文档](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-017"></a>
## DST-017 · DistributedSampler、set_epoch 和 drop_last 怎么用？

**L2 · 编辑补充题** · 标签：数据采样 / DDP

**30 秒回答**

DistributedSampler 为各数据并行 rank 分配样本，shuffle 时每轮需调用 set_epoch 改变顺序。不能整除时可补齐或丢尾，影响覆盖和重复；Sampler 与 DataLoader 的 drop_last 含义不同，评估需清理重复样本。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 各 rank 采用一致数据集大小与种子。
- 训练每轮开始前设置 epoch，而非只初始化一次。
- 按全局样本标识合并评测结果，清理补齐导致的重复。

### 易错点

- DDP 本身不会替你调用 set_epoch 或切分数据。

### 面试官可能追问

- IterableDataset 如何避免 worker/rank 重复读取？

</details>

**技术依据**

- [MM-S063 · PyTorch torch.utils.data 文档](https://docs.pytorch.org/docs/2.14/data.html)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。

<a id="dst-018"></a>
## DST-018 · 训练挂在 NCCL collective 上，如何定位？

**L3 · 社区题目线索** · 标签：NCCL / 故障诊断

**30 秒回答**

先收集所有 rank 日志与最早错误，确认有无某卡 OOM、异常退出或输入耗尽，再核对 collective 顺序、shape 和进程组。若程序一致，再用最小通信测试检查网络、拓扑和驱动；增加 timeout 只能延后报错，不能修复根因。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 先区分真正网络故障与某 rank 未进入通信。
- 调试模式记录 collective 序号，必要时用监测 barrier 定位缺席 rank。
- 单机与多机最小复现可分离程序、硬件和网络问题。

### 易错点

- 其他 rank 的最后一个 NCCL 等待通常不一定是最早故障。

### 面试官可能追问

- 动态视觉分支为何可能造成 collective 不一致？

</details>

**技术依据**

- [MM-S064 · NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [MM-S067 · PyTorch Distributed communication 文档](https://docs.pytorch.org/docs/2.14/distributed.html)

**题目出处线索**

- [MM-S004 · llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) · `reported_topic`：Infra 集合包含 NCCL group 冲突、FSDP 故障诊断，扩展为通用排查。

<a id="dst-019"></a>
## DST-019 · MFU、HFU 与 GPU utilization 有什么区别？

**L2 · 社区题目线索** · 标签：MFU / 性能指标

**30 秒回答**

GPU utilization 只反映设备是否忙，HFU 可包含重计算等实际执行 FLOPs，MFU 用完成训练所需的模型有效计算衡量理论峰值利用率。因此忙碌可能来自低效算子或通信等待，需要同时看 token 吞吐、精度和计算口径。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 稠密 Transformer 的 6NT 近似忽略注意力及额外模块。
- MFU 分子不把激活重计算算成额外有效模型工作。
- 峰值应匹配 dtype、稠密/稀疏口径和实际 GPU 数。

### 公式

```text
MFU = 模型有效 FLOPs/step ÷ (step_seconds × 集群理论峰值 FLOPs/s)；稠密近似 C≈6NT，仅在忽略额外项时成立。
```

### 易错点

- MoE 不能无条件按总参数量套 6NT。

### 面试官可能追问

- 长序列下为什么 attention 项不能忽略？

</details>

**技术依据**

- [MM-S065 · PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5)

**题目出处线索**

- [MM-S004 · llm-rl-infra-interview](https://github.com/XFWang522/llm-rl-infra-interview) · `reported_topic`：公开 Infra 集合列出 FLOPs 估算与 MFU/HFU/utilization 对比题。

<a id="dst-020"></a>
## DST-020 · 多模态训练吞吐波动，怎样做 profiling 与负载平衡？

**L3 · 编辑补充题** · 标签：多模态训练 / Profiling / Packing

**30 秒回答**

动态图像分辨率、视频帧数和答案长度会让各 rank 工作量不同，快卡在同步点等待慢卡。应记录每卡视觉与文本 token、数据准备、算子和通信时间，再按估计成本分桶或 packing；仅平均样本张数难以平衡训练。

<details>
<summary>展开答案、易错点与追问</summary>

### 展开要点

- 用 profiler 时间线分开数据解码、视觉编码、LLM 与通信。
- packed 样本需保留边界 mask，避免跨样本注意力与标签串扰。
- 优化后复查最大卡显存、有效监督 token 吞吐和任务配比。

### 易错点

- 总 token 相同也不保证视觉编码开销相同。

### 面试官可能追问

- 冻结视觉编码器时怎样判断预计算特征是否合适？

</details>

**技术依据**

- [MM-S066 · PyTorch Profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html)
- [MM-S019 · Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)

**题目出处线索**：为补全知识体系设计；未认定为某公司的实际提问。
