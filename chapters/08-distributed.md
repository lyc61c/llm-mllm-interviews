# 分布式训练、并行与显存

[首页](../README.md) · [全部题目](../QUESTION-INDEX.md)

## 目录

- [训练显存与状态分片](#topic-1)
  - [DST-003 · 大模型训练显存怎样估算和优化？以 7B Adam 为例](#dst-003)
  - [DST-004 · ZeRO-1/2/3 各切分什么？理想状态显存是多少？](#dst-004)
  - [DST-005 · FSDP 与 ZeRO-3 有什么联系和区别？](#dst-005)
  - [DST-015 · CPU/NVMe Offload 适合哪些场景，为什么可能变慢？](#dst-015)
- [并行策略与通信](#topic-2)
  - [DST-001 · 数据并行 DDP 每个 step 做了什么？](#dst-001)
  - [DST-002 · DDP 梯度累积如何保持与大 batch 等价？](#dst-002)
  - [DST-006 · Megatron 的 MLP 张量并行为什么先列切再行切？](#dst-006)
  - [DST-007 · 流水线并行的 bubble 从哪里来，怎样降低？](#dst-007)
  - [DST-008 · Megatron 的 Sequence Parallelism 与 TP 怎样配合？](#dst-008)
  - [DST-009 · Context Parallelism 如何训练更长上下文？](#dst-009)
  - [DST-010 · MoE 的专家并行有哪些通信与负载问题？](#dst-010)
  - [DST-011 · AllReduce、ReduceScatter、AllGather 怎样对应？](#dst-011)
  - [DST-012 · DDP 怎样重叠反向计算与梯度通信？](#dst-012)
  - [DST-021 · DP、TP、PP、SP/CP 与 EP 如何组合，为什么网络拓扑决定并行方案？](#dst-021)
  - [DST-022 · 自动并行怎样搜索或传播分片方案，GSPMD、FlexFlow 与 Alpa 有何区别？](#dst-022)
  - [DST-023 · GPipe、同步 1F1B 与异步 PipeDream 的梯度语义和权重版本怎样区分？](#dst-023)
  - [DST-025 · 怎样检查 GPU/NVLink/网络拓扑，区分算力不足和通信瓶颈？](#dst-025)
  - [DST-026 · 2D、2.5D、3D 张量并行与 DP×TP×PP 的“3D 并行”有什么区别？](#dst-026)
- [精度与激活优化](#topic-3)
  - [DST-013 · 激活 checkpointing 为什么省显存，有哪些正确性条件？](#dst-013)
  - [DST-014 · BF16 与 FP16 混合精度训练为什么表现不同？](#dst-014)
  - [DST-024 · TF32 与 FP32、FP16、BF16 有何区别，TF32 会把模型存成19位吗？](#dst-024)
- [训练故障与可靠性](#topic-4)
  - [DST-016 · 分布式训练怎样做到可靠断点续训？](#dst-016)
  - [DST-017 · DistributedSampler、set_epoch 和 drop_last 怎么用？](#dst-017)
  - [DST-018 · 训练挂在 NCCL collective 上，如何定位？](#dst-018)
  - [DST-019 · MFU、HFU 与 GPU utilization 有什么区别？](#dst-019)
  - [DST-020 · 多模态训练吞吐波动，怎样做 profiling 与负载平衡？](#dst-020)
  - [DST-027 · DataLoader、Sampler、BatchSampler 和 collate_fn 分别负责什么？](#dst-027)

<a id="topic-1"></a>
## 训练显存与状态分片

<a id="dst-003"></a>
### DST-003 · 大模型训练显存怎样估算和优化？以 7B Adam 为例

**L1** · 美团 / 深势科技

#### 答案

估算训练显存要把模型状态和中间计算分开，不能用推理权重大小直接推算训练需要几张卡。设参数量为 $`P`$，常见混合精度 Adam 有 2 字节权重、2 字节梯度、4 字节主权重和两个各 4 字节状态，合计 $`16P`$ 字节。7B 约为十进制 112 GB，约 104.3 GiB，这还不是总峰值。

峰值还包括激活、注意力工作区、通信桶和临时权重聚合。主权重是否独立保存、梯度类型和优化器实现不同，都要重算。显存 allocated 是活跃张量，reserved 是框架保留内存，设备已用量还包含其他分配，不能混为一数。固定有效 batch 时，减小每卡 microbatch、增加累积次数可降低激活峰值；只增加累积次数而不减小单批，不会自动省激活。公式里的 $`A`$ 是累积次数、$`D`$ 是数据并行度。

状态太大可用 ZeRO 或 FSDP 分片，单层矩阵太大可用张量并行，层数太多可用流水线分布，激活太大可重算。FlashAttention 避免完整注意力分数矩阵驻留，但标准稠密版本的运算复杂度仍约为序列长度平方；稀疏注意力则改变连接，需另测质量。数据流式读取减少下载和主存压力，不会自动减少模型状态或当前激活。

冻结底座的浮点 LoRA 状态粗估 $`2P+16P_a`$，理想全量化 QLoRA 约为 $`P/2+16P_a`$ 加量化元数据，$`P_a`$ 是真实适配器参数，不能把 rank 直接代入。额外解冻层、未量化层和工作区都另计。若梯度要经过冻结模块到达更早的可训练层，不能整段 no_grad，否则会断开路径。先定位溢出发生在前向、反向还是优化器更新，再按实际瓶颈优化并测吞吐。

```math
\begin{aligned} M_{\mathrm{states}}&=(2+2+4+4+4)P=16P\ \mathrm{bytes}\\ M_{\mathrm{peak}}&=M_{\mathrm{states}}+M_{\mathrm{activations}}+M_{\mathrm{workspace}}+M_{\mathrm{communication}}\\ B_{\mathrm{eff}}&=B_{\mathrm{micro}}\times A\times D \end{aligned}
```

#### 易错点

- 只增加累积次数未必减少激活，16P 账单也不是所有实现的总峰值。
- 数据流式读取、状态分片和注意力优化作用对象不同，必须各自计算成本。

#### 追问

- microbatch 已经是 1，而权重与 optimizer 仍放不下时，应优先尝试哪些状态优化？
- 怎么用 profiler 与峰值 `allocated`/`reserved` 区分激活、参数聚合和 `optimizer.step` 的 OOM？
- 怎样根据 adapter 的 target_modules 和实际 shape 计算 $`P_a`$，而不是套固定显存百分比？

<a id="dst-004"></a>
### DST-004 · ZeRO-1/2/3 各切分什么？理想状态显存是多少？

**L1** · 腾讯

#### 答案

ZeRO 通过在数据并行组内分片模型状态，减少每张卡保存同一份训练状态的浪费；阶段越高，分片对象越多，但通信和临时聚合也更复杂。Stage 1 分片优化器和主权重，Stage 2 再分片梯度，Stage 3 再分片参数。

公式采用每参数 16 字节账单：低精度参数和梯度共 4，主权重及 Adam 状态共 12。$`P`$ 是参数量，$`D`$ 是分片卡数，所以理想常驻状态是 Stage 1 的 $`4P+12P/D`$、Stage 2 的 $`2P+14P/D`$、Stage 3 的 $`16P/D`$。例如 $`D=4`$，分别约为每参数 7、5.5、4 字节，而不是每升一级就减半。这不含激活、未分片部分和通信峰值。

Stage 1、2 更新后仍要让各卡获得完整新参数，通常通过收集分片或广播实现；Stage 2 常把梯度归约成各卡负责的块。Stage 3 计算当前模块前，需要 all-gather 参数，计算后可释放，反向若已释放则还要再次收集，再用 reduce-scatter 保留本地梯度。梯度平均的缩放还需正确处理。

因此更高阶段不一定更快。理想等字节、每步完整覆盖并在前向后重新分片的常见分析里，Stage 3 有前向和反向参数收集，加梯度分散归约，流量可高于 Stage 1、2；缓存、预取和重算策略又会改变次数。它分的是状态冗余，张量并行分的是矩阵计算。选阶段要同时看能否装下、网络和真实峰值，不能只看常驻公式。

```math
\begin{aligned} M_1&=4P+\frac{12P}{D}\\ M_2&=2P+\frac{14P}{D}\\ M_3&\approx\frac{16P}{D} \end{aligned}
```

#### 易错点

- ZeRO 主要切模型状态，TP 切同层计算；常驻显存公式还需加临时聚合峰值。

#### 追问

- 为什么 ZeRO-3 节省显存却可能更慢？

<a id="dst-005"></a>
### DST-005 · FSDP 与 ZeRO-3 有什么联系和区别？

**L2**

#### 答案

FSDP 和 ZeRO-3 都能把参数、梯度与优化器状态分到多卡，需要用到某模块时再收集完整参数，以减少常驻显存。它们思想相近，但框架集成、分片表示和通信调度不同，不能直接互换配置。

FSDP 通常在模块前向前 all-gather 参数，如果前向后重新分片，反向前还要再收集，梯度则归约并保留本地分片。比如逐 Transformer 层分组，只需逐层聚合；若整个模型作为一个巨大的组，临时完整参数可能占很大显存。组太小又会增加启动和调度开销，因此分组是显存与通信的权衡。

FSDP2 的 fully_shard 使用按参数的 DTensor 分片表示，与旧的包装和展平参数接口不同，版本必须明确。Reshard 策略决定前向后是否释放完整参数：保留能减少再次收集，却占更多内存；预取也在重叠通信与提高临时驻留之间取舍。保存、加载和优化器处理要采用对应版本的机制。

两者都不能保证状态平均除卡数就是实际峰值，还要算激活、当前聚合组和通信缓冲。选择时看模型支持、训练框架、断点需求和实测吞吐，而不是仅凭名称认为一个一定优于另一个。

#### 易错点

- 不能认为 FSDP 与 DeepSpeed 配置可直接互换。

#### 追问

- 为什么只在根模块分片可能提高峰值显存？

<a id="dst-015"></a>
### DST-015 · CPU/NVMe Offload 适合哪些场景，为什么可能变慢？

**L2**

#### 答案

CPU 或 NVMe offload 把参数、梯度或优化器状态暂时移出 GPU，用主存或存储容量换训练可行性，主要适合显存装不下的任务。它可能让训练变慢，因为下一步用到的数据必须再搬回来，CPU 优化器计算也可能进入关键路径。

例如把 Adam 状态放在 CPU，可以减轻 GPU 常驻压力，但更新时需要处理梯度和参数传输。参数卸载则影响前后向取权重，支持阶段和调度与优化器卸载不同。NVMe 容量更大，带宽和延迟却通常比主存更差，需要预取与缓存，否则 GPU 容易等数据。

Pinned memory 可帮助异步传输，但占用锁页主存；线程数、NUMA、PCIe、存储吞吐和缓存命中都影响效果。应在时间线里分离传输、CPUAdam 和 GPU 计算，判断是真正隐藏了搬运，还是在关键路径等待。省显存不等于加速，也不能简单假设异步接口就自动重叠。

即使全部状态能卸载，当前模块完整参数、激活和工作区仍要在计算时放得下。要检查主存和存储余量，以及稳态吞吐；如果模型能用合适分片直接装入 GPU，卸载未必划算。选择由容量需求与可接受训练时间共同决定。

#### 易错点

- 状态可放下不代表当前模块的完整计算也能放下。

#### 追问

- 怎样判断瓶颈在 CPUAdam 还是传输？

<a id="topic-2"></a>
## 并行策略与通信

<a id="dst-001"></a>
### DST-001 · 数据并行 DDP 每个 step 做了什么？

**L1**

#### 答案

DDP 数据并行让每个进程持有一份模型，处理不同数据，再同步梯度，使各份模型做出相同更新。它主要扩大数据处理吞吐，不会自动把模型权重切成几份，也不会替训练程序分割数据集。

一轮通常先让各进程在本地数据上前向和反向，梯度准备好后做集体归约，得到跨进程平均梯度；每个进程再用相同优化器状态执行更新。例如两张卡的某个参数梯度分别为 2 和 4，等权平均得到 3，两份模型都按这个梯度更新。模型初始化、优化器状态、更新次数和通信顺序要一致，数据则通过分布式采样器等机制分配。

Buffer 广播与梯度同步是不同操作，例如某些非参数统计可能按配置广播。每张卡通常仍保留完整参数、梯度和优化器状态，多卡并不让这些状态显存按卡数下降。通信桶和更大的局部输入还可能导致单卡能跑、多卡某些配置反而溢出。

PyTorch DataParallel 常在单进程内分发输入、复制模型并集中收集输出，主卡开销容易不均；DDP 常用每 GPU 独立进程，适合多机扩展。参数服务器则让工作进程向中心或分片服务器推拉数据，可以同步或异步，有不同瓶颈和陈旧梯度语义。数据并行是策略名称，不等于某一个 API。

#### 易错点

- DDP 不会让每张卡只持有一部分模型参数。

#### 追问

- 为什么单卡可训练，多卡却可能 OOM？

<a id="dst-002"></a>
### DST-002 · DDP 梯度累积如何保持与大 batch 等价？

**L2**

#### 答案

DDP 梯度累积要把整个跨卡累积窗口看成一次大 batch，按同一个目标归一化，并且只在窗口结束更新参数。等大小小 batch 常把 loss 除以累积次数，再让 DDP 平均各卡梯度；变长文本不能简单等权平均每个小 batch 的 token loss。

例如两张卡各累积两个小批，共有 100、200、300、400 个有效目标，正确 token 平均的分母是 1000。DDP 默认对两卡梯度取平均，因此可把每卡窗口的损失和乘以 $`2/1000`$，跨卡平均后得到全局损失和除以 1000；如果框架已经完成这种缩放，就不能再除卡数或累积次数。有效 token 数应跨 rank 汇总，最后不足窗口也要使用实际数量。

中间小步用 no_sync 减少梯度通信，并让它同时包住前向和反向，最后一步正常同步。然后解除混合精度缩放、裁剪、optimizer.step、推进调度器并清零，累积期间参数保持不变。要确认各 rank 有相容的更新和通信次数，不能某卡提前结束。

在加性损失与正确归一条件下，这接近一次大 batch，但 dropout 的随机数、浮点顺序和 BatchNorm 等 batch 相关操作可能造成差别。最直接的检查是在小规模固定数据上比较梯度和一次更新结果。累积主要减少单次激活峰值，并不切分 DDP 的完整模型状态。

#### 易错点

- 框架可能已经完成累积或全局 token 缩放，重复除数会让梯度过小。

#### 追问

- 各 rank 有效 token 数不同时如何得到全局均值？

<a id="dst-006"></a>
### DST-006 · Megatron 的 MLP 张量并行为什么先列切再行切？

**L2**

#### 答案

Megatron 的前馈层先按输出特征切第一层，再按对应输入特征切第二层，是为了让中间激活留在各卡本地，避免两次矩阵乘法之间收集整个隐藏张量。这里用 $`XW`$ 的乘法约定，先明确矩阵维度，否则框架转置存储时“行切、列切”容易说反。

第一层 $`W_1=[W_{1,1},\ldots,W_{1,D}]`$ 沿输出维拼接，每卡计算自己的 $`H_i=\phi(XW_{1,i})`$，$`\phi`$ 是逐元素激活。因为非线性只依赖本地元素，不需要先合并。第二层 $`W_2`$ 沿对应输入维切，各卡计算 $`H_iW_{2,i}`$，这些结果形状相同，是最终输出的部分和，所以要归约相加，而不是拼接。

例如中间维度 8、两卡各持 4 个特征，第一层各算四维并激活，第二层各自把四维映射回同一输出空间，再把两个贡献相加。这样通信主要放在最后，而不是中间先 all-gather 大特征。反向对输入梯度仍有对应归约，并不是前后向都无通信。

注意力的 QKV 可沿头或输出特征切，输出投影再做对应行切，逻辑类似但要考虑头数和分组查询等条件。序列并行还会调整归约布局。实际权重的存储形状、偏置、激活及通信位置都需核对，不能只背一句“先列后行”。

```math
\begin{aligned} W_1&=[W_{1,1},\ldots,W_{1,D}]\\ H_i&=\phi(XW_{1,i})\\ W_2&=\begin{bmatrix}W_{2,1}\\\vdots\\W_{2,D}\end{bmatrix},\qquad Y=\sum_{i=1}^{D}H_iW_{2,i} \end{aligned}
```

#### 易错点

- 框架权重存储可能转置，必须明确所谓行和列。

#### 追问

- attention 中 QKV 与输出投影怎样对应切分？

<a id="dst-007"></a>
### DST-007 · 流水线并行的 bubble 从哪里来，怎样降低？

**L2**

#### 答案

流水线并行把模型分成多个阶段，启动时后面的卡还没拿到输入，排空时前面的卡已经没活干，这些空闲就是 bubble。把一个训练 batch 拆成多个 microbatch，让不同阶段同时处理不同数据，可以提高利用率。

公式中的 $`p`$ 是等时阶段数，$`m`$ 是 microbatch 数，理想 fill-drain 的空闲比例约为 $`(p-1)/(m+p-1)`$。例如四阶段、八个小批，约为 $`3/11`$；若小批更多，启动排空成本占比变小。这个近似要求阶段均衡且忽略部分通信，不能当所有调度的精确利用率。

GPipe 常先前向再集中反向，保存较多在途激活；同步 1F1B 在预热后交替一前向一反向，降低激活驻留。虚拟阶段和交错调度也可减少空闲，但增加调度与通信复杂度。阶段划分不应只平均层数，例如首阶段有视觉编码器时，工作量可能远高于其他阶段。

增加 microbatch 数与增加每个小批的样本数是两回事。固定总 batch 下切得太细，矩阵乘可能太小、启动开销变大；切得太粗，bubble 和激活又高。选择要测阶段耗时、传输与显存，并确认同步更新语义，不能仅靠理论 bubble 比例判断训练更快。

```math
f_{\mathrm{bubble}}\approx\frac{p-1}{m+p-1}
```

#### 易错点

- 增大 microbatch 个数不一定增大每个 microbatch 的 batch size。

#### 追问

- 视觉编码器放在首阶段为何易失衡？

<a id="dst-008"></a>
### DST-008 · Megatron 的 Sequence Parallelism 与 TP 怎样配合？

**L2**

#### 答案

Megatron 的 Sequence Parallelism 把张量并行中原本重复的逐 token 操作和激活沿序列维分开，减少每卡冗余，而不是把长文本切成互不通信的独立段。它通常与张量并行配合使用，特别针对 LayerNorm、Dropout 等不需要跨 token 信息的区域。

例如张量并行两卡原本各自保留长度 1000 的完整逐 token 激活，序列并行可让每卡在相应区域只处理 500 个位置。进入需要完整序列布局的列并行投影前，可 all-gather；离开产生部分和的区域，用 reduce-scatter 得到沿序列分片的结果。相比先 all-reduce 完整结果再各卡重复保留，驻留激活较少。

每个 token 的语义关系没有被截断，相关计算仍通过布局转换和通信完成。它不表示任意全局注意力都能独立只算本地片段，长上下文注意力更广的切分通常属于 Context Parallelism。不同系统也可能用相同名字表示不同范围，所以回答应说明框架语义。

收益取决于激活比例、张量并行度和通信重叠，仍需检查梯度、随机 dropout 和全局位置处理。SP 主要减少特定区域的重复激活，不能自动替代参数分片、激活重算或 CP。

#### 易错点

- 在 Megatron 语义中，SP 与 CP 切分范围不同，不能只凭名称互换。

#### 追问

- SP 为什么不需要切开一条序列的语义？

<a id="dst-009"></a>
### DST-009 · Context Parallelism 如何训练更长上下文？

**L3**

#### 答案

Context Parallelism 把一条长序列及各层激活分到多卡，让每张卡只保存和处理一部分上下文，同时通过通信完成全局注意力。它减少长上下文激活压力，不能把片段独立前向就声称与原模型等价。

例如两卡分担一条 16000 token 序列，每卡负责部分 query，但这些 query 仍要读取因果 mask 允许的其他位置的 key 和 value。可通过环式交换、收集等方式得到这些信息，并按正确的注意力归一合并。前向和反向都有相关通信，参数是否分片则取决于是否另结合 TP 或 FSDP 等策略。

因果序列前半段能看见的历史比后半段少，简单连续切分可能负载不均，交错分配可改善。但全局位置、因果条件和段边界必须准确；若多个独立样本打包，仍不能跨样本读取。仅把片段位置重置也不能代替边界隔离。

CP 用多卡容量和通信换每卡激活降低，重算则用额外计算换激活，两者可以组合但代价不同。实际收益受长度、通信方式和拓扑影响，要同时测显存、吞吐和与未分片参考的结果，不能只看分片长度除以卡数。

#### 易错点

- 将长文本切段分别前向不是等价 CP。

#### 追问

- CP 与重计算分别解决哪种显存占用？

<a id="dst-010"></a>
### DST-010 · MoE 的专家并行有哪些通信与负载问题？

**L3** · 阿里巴巴

#### 答案

专家并行把不同专家放到不同 GPU，路由器为每个 token 选择专家，再把 token 发过去计算，最后收回结果；主要难点是通信和负载，而不是专家参数分开就免费扩展。稀疏激活减少每个 token 用到的计算，不代表全部专家无需存储。

例如八个专家分在四卡，如果大多数 token 都选同一个专家，那张卡很忙，其他卡会等待。分发与回收常使用 all-to-all，跨机时流量和小消息开销可能很大；每个专家拿到很少 token 时，小矩阵乘利用率也差。增加 top-k 让每个 token 使用更多专家，可能提高容量，但计算、激活和传输通常都增加。

应记录每专家 token 数、负载偏斜、容量溢出或丢弃比例，配合时间线检查通信与计算。负载辅助约束、分组计算和合适的拓扑映射可以改善，但约束太强也可能影响路由质量。共享专家和稠密主干可能使用另外的并行布局。

专家组与数据、张量并行组如何嵌套依框架而定，EP 大小并不总能独立乘到总卡数上。训练与 RL 还要检查新旧策略的路由变化，避免概率计算路径不一致。选择方案需测质量、负载和带宽，不能只报告“每次只激活少量参数”。

#### 易错点

- EP 大小不总能独立乘到总卡数上，取决于分组映射。

#### 追问

- 增加 top-k 为什么影响效果、显存和通信？

<a id="dst-011"></a>
### DST-011 · AllReduce、ReduceScatter、AllGather 怎样对应？

**L1**

#### 答案

AllReduce 让所有进程拿到完整的归约结果，ReduceScatter 只让每个进程拿到其中一块，AllGather 则把各块收集到所有进程。先归约分散、再收集，就能实现 AllReduce；分片训练可以停在局部块，减少完整结果驻留。

例如两卡各有 `[1,2]` 和 `[3,4]`，求和 AllReduce 后两卡都得到 `[4,6]`。ReduceScatter 可让第一卡得到 `[4]`、第二卡得到 `[6]`；之后 AllGather 让两卡都得到 `[4,6]`。AllGather 只是按约定收集拼接，本身不做加法。DDP 常归约梯度，ZeRO/FSDP 常保留梯度分片并在计算前收集参数。

公式里的 $`D`$ 是卡数，$`S`$ 是每卡完整输入消息的字节数，理想 ring AllReduce 每卡发送约 $`2(D-1)S/D`$，接收相同，不含协议开销。例如四卡每卡完整消息 100 MB，发送约 150 MB，接收约 150 MB；若统计双向总流量，就不能再当单向口径使用。

各 rank 需按相容顺序调用通信，形状、类型和归约操作一致，否则可能挂起或出错。真实算法可能采用 ring、tree 或其他方式，小消息更受启动延迟影响，大消息更受带宽影响，不能只凭理想公式预测全部性能。

```math
V_{\mathrm{send}}\approx V_{\mathrm{recv}}\approx\frac{2(D-1)}{D}S
```

#### 易错点

- AllGather 收集拼接而不求和，通信量还要区分发送、接收与双向总量。

#### 追问

- 小消息为什么可能更适合低延迟算法？

<a id="dst-012"></a>
### DST-012 · DDP 怎样重叠反向计算与梯度通信？

**L2**

#### 答案

DDP 用梯度桶在反向还没结束时启动通信，让已就绪的梯度同步与后面的反向计算重叠。一个 bucket 是若干参数梯度的集合，只有桶内所需梯度都准备好，才开始对应归约。

自动微分 hook 标记参数梯度就绪，reducer 按一致桶顺序发起异步 AllReduce。例如后几层先产生梯度，通信它们时前面层还在反向，若硬件和网络允许，通信就能被计算部分覆盖。各 rank 必须保持相同 collective 顺序，不能哪个桶先准备好就各自随意发送不同顺序。

桶太小，通信启动频繁、消息效率低；桶太大，又要等更多梯度，启动较晚，最后留下较长通信尾巴。参数顺序、网络、算子耗时和通信资源竞争也影响重叠。异步 API 只表示调用不立即等待，不保证 GPU 时间线上真的同时运行。

Profiler 应检查桶何时启动、最后通信尾部多长，以及快慢 rank 是否等待。模型有条件分支时，要按实际需求处理未使用参数；find_unused_parameters 会遍历计算图，也有开销，不能盲目开或关。梯度累积还可用 no_sync 减少中间同步，但最终全局更新仍要正确。

#### 易错点

- 异步 API 并不保证通信与计算实际并行。

#### 追问

- 动态图中未使用视觉分支会怎样影响同步？

<a id="dst-021"></a>
### DST-021 · DP、TP、PP、SP/CP 与 EP 如何组合，为什么网络拓扑决定并行方案？

**L2**

#### 答案

混合并行是在不同维度分担模型和数据，网络拓扑决定哪些通信能承受，所以先算显存瓶颈，再安排通信频繁的组靠近。数据并行 DP 分样本，张量并行 TP 分同层矩阵，流水线 PP 分层，序列或上下文并行分 token，专家并行 EP 分专家。

基本 mesh 的 $`W=D\times T\times P`$，分别是总卡数、数据、张量和流水线维度。例如两组数据副本，每组两卡张量并行再分两段流水线，共八卡，但数据独立样本数只按 $`D=2`$ 扩大，不能再乘 TP 和 PP。全局 batch 还乘每副本 microbatch 和累积次数。CP、EP 怎样与这些轴嵌套，要看框架分组，不能把每个名称都独立相乘。

TP 常每层通信，适合放在高带宽互连内；PP 主要跨阶段传激活与反向梯度，DP 归约梯度或分片状态，EP 做 token all-to-all，CP 交换注意力上下文。它们可能争用同一链路，名义带宽不等于同时通信的有效带宽。

ZeRO 主要消除状态冗余，不替代 TP 切算子或 PP 切层。低带宽多机上，高阶段参数收集可能很贵，即使显存更省也未必更快。选 DeepSpeed、Megatron 或 FSDP 等方案，要核对算子、断点和运维支持，先测小规模基线，再按实际拓扑扩展。

```math
W=D\times T\times P,\qquad B_{\rm global}=D\times B_{\rm micro}\times N_{\rm accumulation}
```

#### 易错点

- 把张量并行度也乘进数据样本数量。
- 不考虑网络拓扑就把同一通信组跨慢链路拆散。

#### 追问

- TP通信与EP通信为什么可能争抢同一链路？
- 为什么低带宽多机ZeRO-3可能不如单机LoRA？

<a id="dst-022"></a>
### DST-022 · 自动并行怎样搜索或传播分片方案，GSPMD、FlexFlow 与 Alpa 有何区别？

**L3**

#### 答案

自动并行把计算图、设备网格、合法切分和通信成本结合，寻找能装进显存且运行较快的计划，重点是全图布局协调，而不是逐算子各选最快方案。公式里的 $`\pi`$ 是执行计划，$`M_i(\pi)`$ 是第 $`i`$ 张设备的峰值内存，约束要求每张都不超过可用容量。

Mesh-TensorFlow 用张量维度到网格轴的映射表达分布，GSPMD 从一些分片标注传播布局并插入通信，主要是通用分片编译，不意味着穷举所有硬件的全局最优。FlexFlow 在样本、算子、属性和参数这些 SOAP 维度搜索，用模拟器估计代价；Alpa 将算子内并行和算子间、流水线规划分层处理，各自搜索空间和编译路径不同。

例如某层单独切成四份最快，但下一层需要另一种布局，两层之间的大型重分片可能抵消收益。成本模型必须计入通信、工作区、在途激活和设备争用；动态 shape 和内核变化也会让估计偏离实际。

系统能减少手工设计，但计划仍需实测校准，不保证任意图和拓扑达到数学全局最优。还要看算子支持、编译时间、动态长度下是否重编，以及断点可移植性。解释方法时应分清“传播已有分片意图”和“搜索优化计划”，不能用自动二字掩盖不同能力范围。

```math
\min_{\pi}\ t_{\rm step}(\pi)\quad\text{s.t.}\quad M_i(\pi)\le M_i^{\rm available}\ \forall i
```

#### 易错点

- 分片标注传播与全局计划搜索不是同一能力。
- 计划需要计算布局转换和峰值内存，不能只加各算子局部时间。

#### 追问

- 为什么单算子最优分片不一定是全图最优？
- 动态shape如何影响自动规划的复用？

<a id="dst-023"></a>
### DST-023 · GPipe、同步 1F1B 与异步 PipeDream 的梯度语义和权重版本怎样区分？

**L3**

#### 答案

流水线的前后向顺序和参数何时更新是两件事；GPipe 与同步 1F1B 可以保持整批同一权重，异步 PipeDream 则要处理权重版本和陈旧梯度。仅看 1F1B 调度图，无法断言训练是不是异步。

GPipe 常先完成多个 microbatch 前向，再做反向，累积后在批次边界更新，保存较多在途激活。同步 1F1B 预热后交替前向与反向，某条激活较早被释放，但仍等完整 batch 完成才更新。公式里 $`g_m(\theta_k)`$ 是第 $`m`$ 个小批在同一参数 $`\theta_k`$ 下的梯度，汇总后更新；若目标是均值，还需要对应归一。

原始 PipeDream 允许各阶段异步更新，前向与反向可能跨参数版本，因此使用 weight stashing 保存对应权重。比如一条小批前向用版本 3，轮到反向时当前权重已是版本 5，需找到版本 3 计算相容局部梯度。但即使这一阶段版本对上，跨阶段和时间仍可能陈旧，不等价于同步 SGD。

同步 flush 变体在边界排空，双缓冲权重方案则减少版本保存成本，各有约束。比较要同时报告 bubble、激活、版本内存、全局 batch 和收敛质量。增加小批数能减少空闲，却可能降低矩阵乘效率或增加调度，不能只按显存或吞吐一个维度选择。

```math
\theta_{k+1}=\theta_k-\eta\sum_m g_m(\theta_k)\quad\text{(synchronous accumulation)}
```

#### 易错点

- 1F1B 描述顺序，不自动等于异步或陈旧权重。
- 保存前向权重版本只解决部分局部一致性，不保证等价同步 SGD。

#### 追问

- 1F1B为何通常能降低activation峰值？
- 增加microbatch数时bubble与kernel效率如何权衡？

<a id="dst-025"></a>
### DST-025 · 怎样检查 GPU/NVLink/网络拓扑，区分算力不足和通信瓶颈？

**L2**

#### 答案

判断算力还是通信瓶颈，要先确认 GPU 与网卡实际连接关系，再做独立链路测试和训练时间线分析。显卡利用率高、网卡标称速度大，都不能直接证明有效计算或通信效率高。

先用 nvidia-smi 检查卡型、状态和进程绑定，用 topo -m 查看 GPU、NVLink、PCIe 与 CPU 亲和性，再确认网卡、NUMA、驱动和 NCCL。分别测节点内点对点、节点间网络和 nccl-tests 的实际 collective，并扫描小到大消息。算法带宽与总线带宽口径不同，结果要按同一定义比较。

公式 $`t_{\rm comm}\approx\alpha+n/B_{\rm effective}`$ 表示启动延迟加传输时间，alpha 是延迟，$`n`$ 是字节数，$`B`$ 是有效带宽。小消息容易被启动成本支配，大消息更依赖带宽。比如同样链路，把消息从 1 KB 换成 1 GB，会暴露完全不同的瓶颈，所以只测一种大小不够。

再看训练里计算、通信和数据准备的重叠，以及最慢 rank 的等待。若矩阵计算很短而通信尾巴很长，扩大计算粒度或调整并行组可能有效；若 CPU 解码拖慢，则网络优化无用。环境报告确认安装能力，不能代替链路测量。保留拓扑、消息和版本，按证据调节而不是盲改 NCCL 环境变量。

```math
t_{\rm comm}\approx\alpha+\frac{n_{\rm bytes}}{B_{\rm effective}}
```

#### 易错点

- 把nvidia-smi的GPU utilization当成MFU。
- 只测一个消息尺寸就判断所有collective的性能。

#### 追问

- 小消息与大消息分别更受延迟还是带宽限制？
- NUMA/NIC亲和性为什么可能影响跨机吞吐？

<a id="dst-026"></a>
### DST-026 · 2D、2.5D、3D 张量并行与 DP×TP×PP 的“3D 并行”有什么区别？

**L3**

#### 答案

二维、2.5 维和三维张量并行，是把同一次矩阵计算映射到多维设备网格；DP×TP×PP 的三维混合并行则是三个不同策略的组合，两种“三维”不是同一含义。多维 TP 仍可作为混合并行中的张量并行部分。

常见二维方案用 $`q\times q`$ 网格分布矩阵块，沿不同轴广播或归约；2.5 维增加复制深度 $`c`$，用更多副本和存储减少某些通信；三维用 $`q\times q\times q`$ 网格安排输入、权重和输出。公式中对应设备数为 $`q^2`$、$`q^2c`$ 和 $`q^3`$，只描述常见方形或立方网格，不是所有算法的统一通信公式。比如 $`q=2,c=2`$，2.5 维是八设备的特定布局，半维并非物理设备的一半。

额外复制能让部分数据在本地复用，避免反复跨网格传输，但增加内存，且矩阵形状、分片整除和拓扑要匹配。实际还有布局转换和归约，维度更多不保证更快，小矩阵或不合适网络可能得不偿失。

三维混合并行里的 DP 分不同样本，TP 分同层算子，PP 分不同层，作用对象不同。回答应说明具体张量怎样分、在哪个轴通信，不能只列维度名称。框架支持和版本也要核对，不能把某个方案的理想公式直接当所有模型的收益。

```math
P_{\rm 2D}=q^2,\qquad P_{\rm 2.5D}=q^2c,\qquad P_{\rm 3D}=q^3\quad\text{for common square/cubic meshes}
```

#### 易错点

- 把3D tensor parallel等同DP×TP×PP。
- 把2.5D中的复制深度理解为半个物理维度。

#### 追问

- 额外复制为什么可以减少某些通信却增加内存？
- 矩阵shape与网格轴不整除时怎样处理？

<a id="topic-3"></a>
## 精度与激活优化

<a id="dst-013"></a>
### DST-013 · 激活 checkpointing 为什么省显存，有哪些正确性条件？

**L2**

#### 答案

激活 checkpointing 少保存前向中间值，反向需要时重算，以额外计算换更低显存；正确性要求重算得到与原计算相容的结果和梯度。它减少的是激活，不会直接切分参数或 Adam 状态。

例如连续几层只保存组入口，反向来到这里时从入口再跑一次前向，恢复需要的内部值。切分粒度决定保存多少、重算多少，越细不一定越快。如果省出的显存能让 microbatch 增大、减少通信或避免慢速卸载，虽然多做计算，最终吞吐也可能提高，仍需实测。

随机 dropout 要管理随机状态，函数里的缓存更新、计数器或其他副作用可能在重算时重复发生。控制流和设备变化也应遵守实现约束，不能把任意有状态函数直接放进检查点。保留随机数状态有成本，但跳过后结果可能与未重算不同。

PyTorch 可重入与非可重入实现有不同梯度条件和重算行为，应明确选择，冻结底座与适配器组合还要检查梯度没断。验证峰值显存、一步耗时及梯度或 loss 与参考的相容性，再根据瓶颈定粒度。保存训练文件叫断点保存，不是这里的激活重算。

#### 易错点

- 带缓存更新或依赖全局状态的函数不能盲目重算。

#### 追问

- 什么时候省显存反而能提升吞吐？

<a id="dst-014"></a>
### DST-014 · BF16 与 FP16 混合精度训练为什么表现不同？

**L1**

#### 答案

BF16 与 FP16 占同样两字节，但数值范围和刻度不同，所以训练稳定性可能不同。BF16 指数 8 位、尾数 7 位，范围接近 FP32；FP16 指数 5 位、尾数 10 位，刻度更细但范围窄，更容易出现溢出或小梯度下溢。

可以把两者想成量程和刻度不同的尺子。BF16 能覆盖大数量级，未必更精确；FP16 常用 loss scaling，把小梯度先放大，反向后再解除缩放。比如原梯度太小被舍成零，放大可能让它可表示，但如果前向激活已经溢出，缩放损失不能倒回去修复。

混合精度的 autocast 按算子选择类型，归一化、归约或优化器状态可能保留 FP32，不是所有计算都统一变成 half。使用 GradScaler 时应先解除梯度缩放，再按原阈值裁剪并更新。它不同于给 logits 除温度：前者用于数值范围，后者改变预测分布。

BF16 通常较少需要梯度缩放，但仍会有舍入误差、NaN 和 Inf；持续异常要查输入、学习率、归一和内核。速度也取决于硬件支持、矩阵形状和实现，因此选择应同时比较稳定性、任务效果和吞吐，不能单按存储位数下结论。

#### 易错点

- BF16 范围更大不等于每个数都比 FP16 更精确。

#### 追问

- loss scaling 与模型输出 logits 温度有何区别？

<a id="dst-024"></a>
### DST-024 · TF32 与 FP32、FP16、BF16 有何区别，TF32 会把模型存成19位吗？

**L2**

#### 答案

TF32 是硬件对某些 FP32 矩阵乘和卷积使用的计算模式，不会把 PyTorch 模型权重存成 19 位。输入输出通常仍是 FP32 张量、每元素四字节，只是受支持乘法的输入有效精度降低，并通常用 FP32 累加。

TF32 的指数范围接近 FP32，输入显式尾数约 10 位；符号、指数和有效尾数合起来可描述为 19 位计算格式，但它不是普通 float16 那样的存储 dtype。公式里 numel 是元素数，即使启用 TF32，FP32 张量理想数据存储仍是 $`4\,\mathrm{numel}`$ 字节。例如一亿个 FP32 参数仍约 400 MB，而不是按 19/32 压缩。

FP16 和 BF16 则既可低精度计算，也可减少张量存储和带宽；前者刻度更细但范围窄，后者范围接近 FP32但尾数少。TF32 主要加速受支持的 FP32 运算，真实收益受硬件、形状和后端开关影响。输入舍入后，乘单位矩阵也可能不逐位等于原输入。

数值敏感任务要比较误差、收敛和任务结果，必要部分保留更精确路径。框架接口和默认值随版本变化，应核对当前配置，分别说明输入精度、累加精度和输出类型。不能按过去的默认值判断所有版本行为，也不能用内部计算格式估计模型文件大小。

```math
\mathrm{storage}(X_{\rm FP32})=4\,\mathrm{numel}(X)\ \text{bytes even when TF32 computation is enabled}
```

#### 易错点

- 把TF32内部计算精度当作模型文件的存储位宽。
- 把某一版本框架的TF32默认值当作所有版本的固定行为。

#### 追问

- TF32为何可能改变X乘单位矩阵的结果？
- 低精度输入、累加精度和输出dtype为何要分别说明？

<a id="topic-4"></a>
## 训练故障与可靠性

<a id="dst-016"></a>
### DST-016 · 分布式训练怎样做到可靠断点续训？

**L3**

#### 答案

可靠断点续训要恢复“模型在哪里、优化到哪一步、下一个数据是什么”，不只恢复权重。需要保存模型、优化器、学习率调度、随机数、训练计数和数据采样进度，分片训练还要保留对应分片元数据。

例如 Adam 的动量不恢复，加载同一权重后的下一步也会不同；shuffle 顺序或 dropout 随机状态不恢复，轨迹也可能变化。各 rank 的状态应对应同一个参数更新边界，若在累积中间保存，还要明确是否保存了部分梯度和窗口进度，通常选择完整更新边界更容易复现。

保存应有完整性标记或原子发布机制，异步写入没完成的文件不能当有效断点。分布式 checkpoint 工具可处理分片，但框架版本、并行配置和模型结构仍需记录。world size 改变时，需要受支持的重分片机制，同时说明数据顺序和有效 batch 是否改变；仅 rank0 的完整权重不足以恢复其他状态。

验证可用“连续训练几步”和“保存后重启训练相同步数”比较下一批、学习率、loss 与参数；同时模拟写入中断，确认会回退到完整旧断点。若硬件、内核或并行度改变，只能承诺相应的语义恢复，未必逐 bit 一致，应明确实际保证。

#### 易错点

- 能加载权重不代表优化器和数据进度已恢复。

#### 追问

- 如何测试故障发生在保存过程中？

<a id="dst-017"></a>
### DST-017 · DistributedSampler、set_epoch 和 drop_last 怎么用？

**L2**

#### 答案

DistributedSampler 给不同数据并行 rank 分配样本索引，set_epoch 让每轮打乱顺序变化，drop_last 决定无法整除时怎样处理尾部。DDP 不会自动替程序切分数据，也不会代为调用 set_epoch。

假设有 10 个样本、三卡。Sampler 的 drop_last=True 会丢到可整除数量，每卡三条；False 可补到 12 个索引，让每卡四条，但部分样本重复。它处理的是跨 rank 数量相等。DataLoader 的 drop_last 则处理各 rank 本地不足一个 batch 的尾部，两者作用层级不同，不能只看名字相同就混用。

启用 shuffle 时，各 rank 使用相同种子和 epoch，在开始迭代前调用 set_epoch，才能得到一致的全局顺序及不同分片。数据集大小和索引含义也应稳定。评估若补齐了重复索引，要按样本标识合并并去重，否则指标可能被重复样本加权。

IterableDataset 没有普通随机索引，通常要在自身迭代里按 rank 和 worker 分片，不能直接照搬 map-style Sampler。增加 worker 不自动避免重复读取。可靠复现还需控制 worker 中增强随机性，并检查各 rank 迭代次数与通信更新相容。

#### 易错点

- DDP 本身不会替你调用 `set_epoch` 或切分数据。

#### 追问

- `IterableDataset` 如何避免 worker/rank 重复读取？

<a id="dst-018"></a>
### DST-018 · 训练挂在 NCCL collective 上，如何定位？

**L3**

#### 答案

训练停在 NCCL 通信处，首先要查是否有某个 rank 没来到同一次通信，而不是马上认为网络坏了。其他进程在等待，最早故障可能是另一张卡 OOM、输入提前耗尽或某个 Python 异常。

收集所有 rank 日志，找时间上最早错误，记录训练步、通信组和 collective 序号。核对各进程是否按相同顺序调用，张量形状、类型和参与组是否相容。例如一张卡因条件分支跳过反向，其他卡还在归约梯度，就可能一直等。必要时用支持的监测 barrier 或调试日志定位缺席进程，但这些工具本身也要按正确组使用。

程序语义确认后，再做单机和多机最小通信测试，逐步区分互连、网卡选择、驱动和硬件问题。小模型训练正常、某种消息大小异常，也可能提示通信或拓扑条件，不应只凭最后等待栈确定根因。先建立稳定最小复现，再逐项恢复数据、条件分支和并行策略。

增加 timeout 只适合本来就慢且最终会完成的操作，不能修复调用次序不一致或进程退出。盲目修改 NCCL 环境变量也可能掩盖问题。最终修复应对应明确证据，并验证完整训练的通信顺序和进程退出情况。

#### 易错点

- 其他 rank 最后等待的 NCCL 调用，不一定是最早故障发生处。

#### 追问

- 动态视觉分支为何可能造成 collective 不一致？

<a id="dst-019"></a>
### DST-019 · MFU、HFU 与 GPU utilization 有什么区别？

**L2**

#### 答案

GPU utilization 表示设备有多忙，HFU 统计实际执行的浮点计算，MFU 则衡量完成模型有效训练计算用了多少理论算力。它们口径不同，GPU 很忙并不代表有效训练效率很高。

MFU 公式里 $`C_{\rm model,step}`$ 是一次更新所需的有效模型 FLOPs，$`t_{\rm step}`$ 是耗时，$`F_{\rm peak,cluster}`$ 是整个集群对应精度的理论峰值。例如有效计算 $`10^{15}`$ 次，耗时 2 秒，集群峰值每秒 $`10^{15}`$ 次，MFU 是 50%。激活重算额外做的前向通常计入 HFU，却不当作 MFU 的新增有效工作；所以重算更多可能让 GPU 更忙，而有效 token 吞吐没有相同比例提升。

稠密 Transformer 常用 $`6NT`$ 粗估训练运算，$`N`$ 是参数量，$`T`$ 是这一步处理的 token 数，但长序列注意力、视觉模块等要补充。专家模型要按实际激活与算子计算，不能无条件代入全部专家参数。峰值还应匹配卡数、dtype 和稠密或稀疏计算口径。

低利用率可能来自数据等待，高利用率也可能来自低效算子或通信相关工作。诊断需结合 profiler、步耗时、有效 token/s 和质量，说明 FLOPs 估计范围。不能把监控面板的一项百分比直接当成 MFU，更不能通过降低数值或改变任务来只追求高数字。

```math
\begin{aligned} \mathrm{MFU}&=\frac{C_{\mathrm{model,step}}}{t_{\mathrm{step}}F_{\mathrm{peak,cluster}}}\\ C_{\mathrm{dense}}&\approx 6NT \end{aligned}
```

#### 易错点

- MoE 不能无条件按总参数量套 $`6NT`$。

#### 追问

- 长序列下为什么 attention 项不能忽略？

<a id="dst-020"></a>
### DST-020 · 多模态训练吞吐波动，怎样做 profiling 与负载平衡？

**L3**

#### 答案

多模态训练吞吐波动，常见原因是不同样本的图片分辨率、视频帧数和文本长度不同，让各卡工作量不均，最快的卡也得等最慢的卡同步。先测数据准备、视觉编码、语言模型和通信的时间，再决定分桶、打包或负载调度。

例如两张卡各处理两条样本，一卡是低分辨率图片，另一卡是高分辨率多帧视频，样本数相同却完全不均衡。记录各卡视觉 token、文本 token、帧数、有效监督数和峰值显存；总 token 一样也未必视觉编码成本一样。时间线可以区分 CPU 解码慢、GPU 算子慢和同步等待。

按估计计算成本分桶，或在多个 rank 间平衡长短和视觉负载，可以降低长尾；数据缓存、预取和批处理也需按瓶颈选择。Packing 仍要保留独立样本边界与正确标签，不能为吞吐让不同对话互相注意。优化后确认任务、长度和领域采样比例没有无意改变。

冻结视觉编码器时可以考虑预计算特征，但前提是图像处理和所需增强固定、缓存版本明确；若需要在线随机增强或训练视觉部分，缓存未必适用。最终看最大 rank 峰值、全局有效 token 吞吐和质量，不能只报告某张快卡的局部速度。

#### 易错点

- 总 token 相同也不保证视觉编码开销相同。

#### 追问

- 冻结视觉编码器时怎样判断预计算特征是否合适？

<a id="dst-027"></a>
### DST-027 · DataLoader、Sampler、BatchSampler 和 collate_fn 分别负责什么？

**L1** · 腾讯

#### 答案

DataLoader 组织整个加载流程，Sampler 决定取哪些索引，BatchSampler 决定一批有哪些索引，collate_fn 把读出的样本整理成模型输入。对可按索引访问的数据集，顺序是选索引、读取样本、组装张量，而不是让 padding 函数决定全局采样顺序。

普通 batch_size 会让 DataLoader 用 BatchSampler 包装逐条索引的 Sampler。自定义长度分桶或 token 预算时，可以直接传 batch_sampler，例如把几条短文本组合到总长度限制以内；此时它负责批次，不能再同时指定互斥的 batch_size、shuffle、sampler 和 drop_last。判断一个采样器返回单索引还是索引列表，要看协议和传入参数，不能只看类名。

公式中 $`\mathcal I`$ 是一批索引，dataset[i] 读样本，collate 将列表变成批次。它可做文本 padding、attention mask、labels 和图像视频元数据。如果关闭自动组批，collate 接收单个样本，不能默认永远是列表。多进程加载通常由主进程产生索引，worker 读数据、变换并整理，预取和 pinned memory 则帮助供给。

worker 数多不替代 DDP rank 分片。IterableDataset 自己定义迭代，每个 worker 都有副本，需在迭代里显式避免重复，不能直接套普通索引采样器。复现还要控制 epoch 和 worker 随机增强，Windows spawn 下供 worker 用的函数要可序列化，常用顶层定义。加载慢时先分离 I/O、解码、组批和传输，再决定增加 worker、缓存还是调整长度。

```math
\mathcal B=\mathrm{collate}([\mathrm{dataset}[i]\mid i\in\mathcal I]),\qquad \mathcal I\sim\mathrm{batch\_sampler}
```

#### 易错点

- Collate 负责整理读出的样本，不默认负责全局取样顺序。
- worker 多不替代 rank 分片，迭代数据集若不分片会重复读。
- 自定义 batch_sampler 后不要再设置互斥的自动组批参数。

#### 追问

- 数据加载长期比模型计算慢，怎样用profiler分离瓶颈？
- persistent_workers下如何让每个epoch的数据增强随机性符合复现要求？

## 参考资料

- [PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch：DistributedDataParallel](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/html/1910.02054v3)
- [DeepSpeed ZeRO 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4)
- [Megatron Bridge Parallelisms 官方文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/parallelisms.md)
- [PyTorch torch.utils.checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [Dataset streaming — Hugging Face Datasets](https://huggingface.co/docs/datasets/stream)
- [Performing gradient accumulation with Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/gradient_accumulation)
- [Gradient synchronization — Accelerate main](https://huggingface.co/docs/accelerate/main/en/concept_guides/gradient_synchronization)
- [PEFT: LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)
- [PEFT: Quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PyTorch FSDP2 fully_shard 文档](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch Distributed Data Parallel 设计说明](https://docs.pytorch.org/docs/2.14/notes/ddp.html)
- [PyTorch AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html)
- [PyTorch Distributed Checkpoint 文档](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)
- [PyTorch torch.utils.data 文档](https://docs.pytorch.org/docs/2.14/data.html)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [PyTorch Distributed communication 文档](https://docs.pytorch.org/docs/2.14/distributed.html)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5)
- [PyTorch Profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html)
- [Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)
- [Mesh-TensorFlow: Deep Learning for Supercomputers](https://arxiv.org/abs/1811.02084)
- [Alpa: Automating Inter- and Intra-Operator Parallelism](https://arxiv.org/abs/2201.12023)
- [GSPMD: General and Scalable Parallelization](https://arxiv.org/abs/2105.04663)
- [Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/abs/1807.05358)
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://aaronharlap.github.io/papers/sosp19-pipedream.pdf)
- [PyTorch：Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
- [PyTorch：CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
- [NVIDIA NCCL：Performance and tuning](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/performance_and_tuning.html)
- [Colossal-AI：2.5D Tensor Parallelism](https://colossalai.org/docs/features/2p5D_tensor_parallel/)
- [PyTorch官方Sampler与BatchSampler实现](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py)
