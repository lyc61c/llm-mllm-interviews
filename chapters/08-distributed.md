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

<a id="topic-1"></a>
## 训练显存与状态分片

<a id="dst-003"></a>
### DST-003 · 大模型训练显存怎样估算和优化？以 7B Adam 为例

**L1**

#### 答案

先把训练显存拆为权重、梯度、优化器/主权重、激活、工作区及通信临时状态。设参数量为 $`P`$，若低精度权重和梯度各为 $`2P`$ bytes，FP32 主权重为 $`4P`$，Adam 一二阶状态合计 $`8P`$，模型状态共 $`16P`$ bytes。$`P=7\times10^9`$ 时为十进制 112GB，约 104.3GiB；FP32 梯度、没有独立主权重或换优化器时须重算。峰值还包含激活、工作区、通信桶及临时聚合，`allocated`、`reserved` 和设备已用量也不是同一口径。

固定有效 batch 时，减小每卡 microbatch、增加累积步数，可通过逐次前后向释放计算图降低激活峰值；只增累积步数而不减 microbatch 不会自动省激活，也不减少权重或 Adam 状态。等大小样本时 $`B_{\mathrm{eff}}=B_{\mathrm{micro}}AD`$，$`A`$ 为累积步数、$`D`$ 为数据并行度；变长样本按有效监督 token 加权，避免重复归一化。FSDP 的 `no_sync` 等设置可能保留完整梯度，须实测峰值。

按瓶颈选择优化：

- 模型状态：DDP 复制状态；TP 切单层矩阵与计算，PP 切不同层，ZeRO 在数据并行组中依次分片 optimizer/master、gradients、parameters。当前模块的参数 all-gather、通信桶和 PP 在途 microbatch 仍可能提高临时峰值。CPU/NVMe offload 用传输与主存/存储带宽换容量。
- 激活与注意力：checkpointing 用反向重算换保留激活；标准 FlashAttention 通过分块、在线 softmax 避免完整 $`S\times S`$ 注意力矩阵驻留 HBM，仍是精确稠密注意力，浮点次序可不同，算术复杂度仍约 $`O(S^2)`$。稀疏注意力改变可见连接，需另测质量；LoRA 减少可训练状态，长序列激活仍可很大。
- 数据：streaming 按需读取数据，可减少整库下载、转换和主存驻留，需设置分片、shuffle buffer 与 worker；它不自动压缩模型参数、优化器或当前 batch 的激活。

先 profile OOM 出现在前向、反向还是优化器更新，再比较显存、有效 token 吞吐与任务质量。缩短序列、量化、冻结或稀疏化，都应说明目标或精度变化。

同一混合精度 Adam 账单下，全参微调模型状态约 $`16P`$。底座全部冻结时，浮点 LoRA 约为 $`2P+16P_a`$，理想全量化 QLoRA 约为 $`P/2+16P_a+M_{\mathrm{quant\ metadata}}`$ bytes；$`P_a`$ 是实际新增的可训练 adapter 参数数，不是 rank 本身。未量化冻结层、额外解冻参数须单独重算，不能把已经计入底座的权重再重复加入 adapter 账单。这些都须再加激活、工作区与通信临时状态，且 master/gradient dtype、量化层范围和优化器实现不同会改写系数。14B 全参的这项模型状态账单约 224 GB、208.6 GiB，不能用推理的 28 GB FP16 权重直接估算训练卡数。若冻结视觉或语言模块但需向可训练前层传播梯度，也不能简单整段 no_grad；是否省激活取决于计算图中的可训练位置。

```math
\begin{aligned} M_{\mathrm{states}}&=(2+2+4+4+4)P=16P\ \mathrm{bytes}\\ M_{\mathrm{peak}}&=M_{\mathrm{states}}+M_{\mathrm{activations}}+M_{\mathrm{workspace}}+M_{\mathrm{communication}}\\ B_{\mathrm{eff}}&=B_{\mathrm{micro}}\times A\times D \end{aligned}
```

#### 易错点

- 仅加梯度累积就声称显存下降，或用 $`16P`$ 状态账单冒充所有实现的峰值显存。
- 把数据流式加载、TP/PP、ZeRO 和稀疏 attention 视为同一种省显存操作，忽略各自影响的对象及通信/质量成本。

#### 追问

- microbatch 已经是 1，而权重与 optimizer 仍放不下时，应优先尝试哪些状态优化？
- 怎么用 profiler 与峰值 `allocated`/`reserved` 区分激活、参数聚合和 `optimizer.step` 的 OOM？
- 怎样根据 adapter 的 target_modules 和实际 shape 计算 $`P_a`$，而不是套固定显存百分比？

<a id="dst-004"></a>
### DST-004 · ZeRO-1/2/3 各切分什么？理想状态显存是多少？

**L1**

#### 答案

ZeRO 在数据并行组内逐步消除模型状态冗余：Stage 1 切优化器及主权重，Stage 2 再切梯度，Stage 3 再切参数。在 DST-003 的 $`16P`$ bytes 假设下，若分片数为 $`D`$，理想持久状态分别为 $`4P+12P/D`$、$`2P+14P/D`$ 和 $`16P/D`$。

收益取决于状态 dtype 和并行度，不能说每升一级都减半。以上不含激活、通信桶和未分片参数；Stage 3 还需临时收集当前计算模块的参数，因此不保证每个 rank 的峰值相同，也不保证更快。ZeRO 切状态冗余，与 TP 切单个矩阵计算不同。

```math
\begin{aligned} M_1&=4P+\frac{12P}{D}\\ M_2&=2P+\frac{14P}{D}\\ M_3&\approx\frac{16P}{D} \end{aligned}
```

#### 易错点

- ZeRO 切状态冗余，与 TP 切单个矩阵计算不同。

#### 追问

- 为什么 ZeRO-3 节省显存却可能更慢？

<a id="dst-005"></a>
### DST-005 · FSDP 与 ZeRO-3 有什么联系和区别？

**L2**

#### 答案

FSDP 与 ZeRO-3 都可分片参数、梯度和优化器状态，需要计算时收集参数，归约后保留局部梯度。主要差别在框架集成、分片表示、通信调度和配置接口，不能直接互换配置。

FSDP 应注明版本：FSDP2 的 `fully_shard` 与旧包装 API 不同，按参数用 DTensor 表示分片。模块分组影响参数收集粒度、峰值显存和通信重叠，reshard 策略则用驻留内存换重复收集成本。

#### 易错点

- 不能认为 FSDP 与 DeepSpeed 配置可直接互换。

#### 追问

- 为什么只在根模块分片可能提高峰值显存？

<a id="dst-015"></a>
### DST-015 · CPU/NVMe Offload 适合哪些场景，为什么可能变慢？

**L2**

#### 答案

CPU/NVMe offload 将部分参数、梯度或优化器状态迁出 GPU，利用主存和存储扩大可训练规模。数据搬运和 CPU 优化器可能增加关键路径，适合显存无法容纳的任务，需实测吞吐、主存占用和传输重叠。

参数与优化器 offload 的阶段支持不同。pinned memory 和预取可帮助传输，但占用主存；还要考虑 PCIe、存储带宽、线程数和缓存命中。状态能放下不代表当前模块的完整计算也能放下，省显存也不等于加速。

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

DDP 让各进程持有模型副本并处理自己的数据，反向期间聚合梯度，随后各副本用相同结果执行优化器更新。训练程序负责向各 rank 提供不同数据子集，DDP 不自动分割数据集，也通常不切分模型状态。

正确训练需保持初始化、采样及 collective 顺序一致。广播 buffers 与梯度归约是不同机制，各卡仍通常持有完整参数，不能据此期待模型状态显存按卡数下降。

PyTorch DataParallel通常在一个进程内多线程scatter输入、复制模型、并行前向再gather输出，主卡更容易承担额外开销；DDP通常每GPU独立进程，通过梯度collective同步并让各rank本地更新，适合单机与多机扩展。参数服务器则让worker向中心或分片server推/拉参数或梯度，可同步或异步；与DDP的collective模式相比，其流量、瓶颈和陈旧梯度语义不同。不能把“数据并行”一概等同于某一个PyTorch API。

#### 易错点

- DDP 不会让每张卡只持有一部分模型参数。

#### 追问

- 为什么单卡可训练，多卡却可能 OOM？

<a id="dst-002"></a>
### DST-002 · DDP 梯度累积如何保持与大 batch 等价？

**L2**

#### 答案

等大小 microbatch 时，将损失按累积步数缩放，只在最后一次同步，可得到同一有效 batch 的平均梯度。DDP 的 `no_sync` 需同时包住前向和反向；每组累积结束后才裁剪梯度、执行 `step`、推进调度器并清梯度。

变长文本应按有效监督 token 总数加权，否则短序列被放大；若框架已做归一化，再除一次会使梯度过小。dropout、BatchNorm 和其他随机性，也可能使累积与整批计算不严格一致。

#### 易错点

- 框架若已归一化损失，再除一次会使梯度过小。

#### 追问

- 各 rank 有效 token 数不同时如何得到全局均值？

<a id="dst-006"></a>
### DST-006 · Megatron 的 MLP 张量并行为什么先列切再行切？

**L2**

#### 答案

按 $`XW`$ 的矩阵乘法记号，MLP 第一层权重沿输出维度切分，每卡独立得到并逐元素激活自己的中间特征；第二层权重沿对应输入维度切分，每卡得到输出的部分和，再归约相加。这样可避免两次 GEMM 之间收集中间特征。

第二层输出须求和而非拼接，反向也存在对应通信，SP 等方案可调整 collective 布局。框架实际存储的权重可能转置，应先明确所谓行、列所指的维度。

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

流水线并行将不同层分配到不同阶段，启动和排空时部分卡等待，形成 bubble。多个 microbatch、合理前后向调度、虚拟阶段和负载平衡可减少浪费，但也会带来通信与激活驻留成本。

GPipe 与 1F1B 的调度和激活驻留不同，阶段耗时还受层结构、输入长度和通信影响，不能只按层数平均切分。理想 fill-drain 中，$`p`$ 个等时阶段处理 $`m`$ 个 microbatch 时，bubble 比例约为 $`(p-1)/(m+p-1)`$，实际依调度与负载而变；增大 microbatch 个数也不等于增大每个 microbatch 的 batch size。

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

Megatron 的 Sequence Parallelism 将原本在 TP 各卡重复的 LayerNorm、Dropout 等逐 token 操作沿序列维切分，以减少激活驻留；进入需要全序列或张量切分的区域时，通常通过 all-gather/reduce-scatter 转换布局。

它通常与 TP 配合，重点是减少 TP 之外的重复操作，并不等于将所有长上下文 attention 任意分给各卡。回答时须说明框架语义，其他文献可能用同名描述不同方案；Megatron 中 SP 与 CP 的切分范围不同。

#### 易错点

- SP 和 CP 名称相似，但在 Megatron 中切分范围不同。

#### 追问

- SP 为什么不需要切开一条序列的语义？

<a id="dst-009"></a>
### DST-009 · Context Parallelism 如何训练更长上下文？

**L3**

#### 答案

Context Parallelism 将一条序列的 token 和各层激活分到多卡，每卡计算局部 query，同时交换或收集注意力需要的其他 key/value，降低每卡激活压力。跨分片仍须访问全局 mask 允许的 KV，不能把切出的片段独立做 attention 来替代。

正确性依赖全局因果 mask、位置与通信。因果场景可用交错分配改善不同位置的计算失衡；结合 packing 时还须维护样本边界，禁止跨样本注意力。

#### 易错点

- 将长文本切段分别前向不是等价 CP。

#### 追问

- CP 与重计算分别解决哪种显存占用？

<a id="dst-010"></a>
### DST-010 · MoE 的专家并行有哪些通信与负载问题？

**L3**

#### 答案

专家并行将不同专家放到不同卡，router 按 token 选择专家后进行分发，计算完再汇集结果。主要瓶颈包括专家 token 分布不均、all-to-all 传输和小批 GEMM，可结合负载约束、分组计算与拓扑优化。

记录每专家的 token 分布及溢出/丢弃比例，在同一时间线观察通信和计算。共享专家与稠密部分还有独立并行布局，稀疏激活不代表免费扩展；EP 大小能否单独乘到总卡数上，取决于分组映射。

#### 易错点

- EP 大小不总能独立乘到总卡数上，取决于分组映射。

#### 追问

- 增加 top-k 为什么影响效果、显存和通信？

<a id="dst-011"></a>
### DST-011 · AllReduce、ReduceScatter、AllGather 怎样对应？

**L1**

#### 答案

AllReduce 使各 rank 获得全量归约结果；ReduceScatter 先归约，每卡只保留一块；AllGather 再将各块收集到所有卡。先 ReduceScatter 再 AllGather 可实现 AllReduce，分片训练则可只保留局部归约结果，减少驻留。

参与 rank 须按相容顺序调用 collective，归约算子、shape 与 dtype 也要相容。理想 ring 中，$`D`$ 卡对每卡大小为 $`S`$ 的数据做 AllReduce，每卡发送量约 $`2(D-1)S/D`$，接收量相同，未计协议开销；NCCL 会依情况选择不同算法。AllGather 是拼接收集，并非求和。

```math
V_{\mathrm{send}}\approx V_{\mathrm{recv}}\approx\frac{2(D-1)}{D}S
```

#### 易错点

- AllGather 是拼接收集，不是求和归约。

#### 追问

- 小消息为什么可能更适合低延迟算法？

<a id="dst-012"></a>
### DST-012 · DDP 怎样重叠反向计算与梯度通信？

**L2**

#### 答案

DDP 将梯度组织成 bucket，某个 bucket 的梯度全部就绪后即异步归约，让通信与后续反向计算重叠。autograd hook 标记梯度就绪，reducer 控制同步顺序，各 rank 的 collective 顺序须一致。

bucket 太小会增加启动开销，太大会推迟通信；效果还取决于参数顺序和网络，须看 profiler 时间线确认通信实际隐藏了多少。`find_unused_parameters` 增加图遍历开销，应按模型需要启用；异步 API 本身不保证实际并行。

#### 易错点

- 异步 API 并不保证通信与计算实际并行。

#### 追问

- 动态图中未使用视觉分支会怎样影响同步？

<a id="dst-021"></a>
### DST-021 · DP、TP、PP、SP/CP 与 EP 如何组合，为什么网络拓扑决定并行方案？

**L2**

#### 答案

先用参数、optimizer和激活估算显存，再用profile确定计算、通信和数据供给瓶颈。DP复制模型分处理数据，TP分同层矩阵，PP分层，SP/CP分token或上下文，EP分专家；它们可以组合，但不同维度要建立对应通信组，不能把world_size分别除多次后重复计算同一组设备。基本DP×TP×PP mesh可写W=DTP，实际CP/EP与DP是否嵌套取决于框架。

TP每层通信频繁，常放在高带宽节点内；PP跨阶段传activation和gradient，适合相对慢的节点间链路；DP通信量依赖梯度/ZeRO阶段，EP有token all-to-all，长上下文还要考虑CP环通信。ZeRO节省状态显存却不替代TP算子切分或PP层分布，因此“有ZeRO就无需混合并行”不成立。选择DeepSpeed/Megatron/FSDP等框架要看所需并行、模型算子、checkpoint和运维支持，先做小规模稳定性/吞吐基线再扩展。

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

自动并行需要把模型计算图、设备mesh、算子合法分片和通信成本结合，找满足显存约束的低时延执行计划。Mesh-TensorFlow让张量维度映射到mesh轴；GSPMD依据少量sharding标注传播分片并插入通信，不等于对所有硬件组合做穷举全局最优搜索。FlexFlow在sample/operator/attribute/parameter（SOAP）维度搜索并用模拟器估成本；Alpa把算子内并行与算子间/流水线规划分层求解。

划分改变时可能产生reshape/reshard通信，最小化单算子时间不一定最小化完整step。成本模型还受网络争用、kernel实现、dynamic shape与activation内存影响，模拟计划需实测校准。系统能降低手工设计负担，但不能保证任意图、拓扑和实际负载达到数学全局最优；支持的算子、编译耗时、计划稳定性和checkpoint可移植性同样重要。

```math
\min_{\pi}\ t_{\rm step}(\pi)\quad\text{s.t.}\quad M_i(\pi)\le M_i^{\rm available}\ \forall i
```

#### 易错点

- 把sharding传播与全局穷举最优搜索混为一谈。
- 忽略计划中的reshard通信与峰值显存。

#### 追问

- 为什么单算子最优分片不一定是全图最优？
- 动态shape如何影响自动规划的复用？

<a id="dst-023"></a>
### DST-023 · GPipe、同步 1F1B 与异步 PipeDream 的梯度语义和权重版本怎样区分？

**L3**

#### 答案

GPipe的典型调度先完成多个microbatch前向，再集中反向并累积梯度，最后同步更新；它保持一个batch使用同一权重，但保存较多在途activation。同步1F1B在warmup后交替前向/反向，仍在完整batch边界更新，因此可减少activation驻留，不必引入陈旧权重。1F1B描述调度顺序，不自动说明是否异步更新。

原始PipeDream允许流水线各stage异步推进更新，microbatch的前向和反向可能跨权重版本，weight stashing保存对应版本以保持该microbatch局部计算一致；即使局部版本匹配，仍存在跨stage/时间的陈旧梯度，不等价普通同步SGD。PipeDream-Flush等同步变体在边界flush以恢复同步语义，2BW降低保留版本的成本但也有自己的约束。比较应报告bubble、activation、权重版本内存、收敛和global batch，不能只凭调度图判断数值等价。

```math
\theta_{k+1}=\theta_k-\eta\sum_m g_m(\theta_k)\quad\text{(synchronous accumulation)}
```

#### 易错点

- 认为所有1F1B都是异步或都存在同样的weight staleness。
- 认为保存前向权重版本就保证整体等价同步SGD。

#### 追问

- 1F1B为何通常能降低activation峰值？
- 增加microbatch数时bubble与kernel效率如何权衡？

<a id="dst-025"></a>
### DST-025 · 怎样检查 GPU/NVLink/网络拓扑，区分算力不足和通信瓶颈？

**L2**

#### 答案

先核对GPU型号、数量、驱动/CUDA/NCCL及每个rank绑定，用nvidia-smi查看GPU状态、topo -m查看节点内互连，再确认NIC/NUMA亲和性和实际使用的网络接口。名义网卡速率不等于collective有效带宽，GPU utilization高也不证明有用矩阵计算或MFU高。

分别测节点内GPU点对点带宽、节点间RDMA带宽和nccl-tests的collective，扫描从小到大消息，比较all_reduce/all_gather等真实通信模式。algorithm bandwidth与bus bandwidth使用不同口径，不直接混比。再用训练timeline查看计算与通信重叠、长尾rank、数据加载及同步等待，配合MFU和step time判断瓶颈；Deepspeed环境报告可确认安装能力，不能代替实际链路测量。保留拓扑、并行组、消息大小和测试版本，避免通过盲目改NCCL环境变量掩盖根因。

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

多维张量并行把同一矩阵乘及其操作数映射到多维设备网格，沿不同轴分块、广播或归约，以改变每卡存储和通信。2D常用q×q网格分布矩阵块；2.5D增加一个复制深度，用额外存储或设备复本换更少部分通信；3D沿三维网格安排输入、权重和输出的分布与通信。具体矩阵形状、网格和算法决定公式，不是“维度越高总越快”。

DP×TP×PP的3D混合并行则是三个不同策略组合：数据、同层算子、不同层。两处“3D”完全不同，2D TP也能作为混合并行里的TP维度。多维TP可能减少特定大矩阵的通信，却增加分片约束、拓扑要求和reshard开销，小规模或不匹配shape未必值得；框架版本对这些方案的支持也不同。回答应给实际sharding与collective路径，不只背1D/2D/3D名词。

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

激活 checkpointing 只保留部分边界张量，反向时重跑前向恢复中间值，用额外计算换显存。它减少保留激活，不直接切分参数或 Adam 状态；选择重算粒度需同时评估峰值显存和 step 时间。

重算须保证随机数、控制流和状态更新一致，否则可能改变梯度或重复副作用。保留 RNG 可维持 dropout 一致性，但有开销；PyTorch 可重入与非可重入版本行为不同，应显式选择，带缓存更新或全局状态依赖的函数不能盲目重算。

#### 易错点

- 带缓存更新或依赖全局状态的函数不能盲目重算。

#### 追问

- 什么时候省显存反而能提升吞吐？

<a id="dst-014"></a>
### DST-014 · BF16 与 FP16 混合精度训练为什么表现不同？

**L1**

#### 答案

BF16 和 FP16 都占两字节，但 BF16 指数位更宽、尾数更少，动态范围接近 FP32；FP16 更易梯度下溢，常需 loss scaling。BF16 范围更大并不代表每个数更精确，也不保证训练不会 NaN。

混合精度的 `autocast` 按算子选择执行精度，不是将所有运算强制为同一 dtype；归约、优化器状态也需单独核对。使用 `GradScaler` 时先缩放损失，更新前解除缩放，再裁剪梯度；异常仍要查输入、归一化和学习率。

#### 易错点

- BF16 范围更大不等于每个数都比 FP16 更精确。

#### 追问

- loss scaling 与模型输出 logits 温度有何区别？

<a id="dst-024"></a>
### DST-024 · TF32 与 FP32、FP16、BF16 有何区别，TF32 会把模型存成19位吗？

**L2**

#### 答案

TF32是支持硬件上Tensor Core执行某些FP32矩阵乘/卷积时采用的计算精度模式，使用近似FP32指数范围和约10bit显式尾数输入精度，通常以FP32累加；FP32输入和输出tensor仍占32bit。把符号、指数与有效尾数加成19位是描述计算格式，不能据此估算PyTorch参数存储为19bit，也不能把它当作torch.float16那样常规存储dtype。

FP16尾数较细但指数范围较窄，BF16尾数较粗而指数范围接近FP32，二者可降低权重/activation存储和带宽。TF32主要加速受支持FP32运算，实际收益和误差依赖硬件、shape与后端开关；数学上乘单位矩阵也未必逐bit保持输入。数值敏感任务应比较收敛、输出误差和必要FP32路径，且查所用PyTorch版本的API，不把过去默认设置当永久规则。

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

可靠断点续训不仅恢复参数，还要恢复优化器、调度器、随机数、采样进度和训练计数；分片训练还需一致的分片元数据。各 rank 保存的状态应对应同一更新边界，并验证恢复后的下一步与连续训练一致。

保存模型和优化器 `state_dict`，记录框架及并行配置；异步保存需有完成标记，不能将未写完的 checkpoint 当作有效文件。world size 变化时须支持重分片并说明样本顺序语义，仅保存 rank0 权重不足以恢复全部训练进度。

#### 易错点

- 能加载权重不代表优化器和数据进度已恢复。

#### 追问

- 如何测试故障发生在保存过程中？

<a id="dst-017"></a>
### DST-017 · DistributedSampler、set_epoch 和 drop_last 怎么用？

**L2**

#### 答案

DistributedSampler 为各数据并行 rank 分配样本，启用 shuffle 时每轮开始前调用 `set_epoch` 改变顺序，各 rank 使用一致的数据集大小和种子。数据不能整除时，补齐或丢尾会影响覆盖与重复。

Sampler 和 DataLoader 的 `drop_last` 含义不同。评估需按全局样本标识合并结果，清理补齐产生的重复；DDP 本身不会代替训练程序调用 `set_epoch` 或分割数据。

#### 易错点

- DDP 本身不会替你调用 `set_epoch` 或切分数据。

#### 追问

- `IterableDataset` 如何避免 worker/rank 重复读取？

<a id="dst-018"></a>
### DST-018 · 训练挂在 NCCL collective 上，如何定位？

**L3**

#### 答案

先收集所有 rank 的日志和最早错误，确认有无某卡 OOM、异常退出或输入耗尽，再核对 collective 顺序、shape 和进程组。其他 rank 最后停在 NCCL 等待处，未必就是最早故障位置。

调试时记录 collective 序号，必要时用监测 barrier 定位缺席 rank。程序一致后，再做单机、多机最小通信测试，将程序问题与网络、拓扑、硬件和驱动分离；只增加 timeout 不能修复根因。

#### 易错点

- 其他 rank 的最后一个 NCCL 等待通常不一定是最早故障。

#### 追问

- 动态视觉分支为何可能造成 collective 不一致？

<a id="dst-019"></a>
### DST-019 · MFU、HFU 与 GPU utilization 有什么区别？

**L2**

#### 答案

GPU utilization 只反映设备是否忙；HFU 可包含重计算等实际执行 FLOPs；MFU 则用完成训练所需的模型有效 FLOPs，除以 step 时间与集群理论峰值的乘积。MFU 分子不把激活重计算当作额外有效工作，因此 GPU 忙也可能来自低效算子或通信等待。

稠密 Transformer 的 $`6NT`$ 近似忽略 attention 和额外模块，长序列时这些项不可忽略，MoE 也不能直接按总参数量套用。峰值须匹配 dtype、稠密/稀疏口径和实际 GPU 数，并结合 token 吞吐与训练精度解释。

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

动态图像分辨率、视频帧数及答案长度使各 rank 的工作量不同，快卡会在同步点等待慢卡。先记录每卡视觉/文本 token、数据准备、算子和通信时间，用 profiler 分离解码、视觉编码、LLM 与通信，再按估计成本分桶或 packing。

仅平均样本张数不足以平衡，总 token 数相同也不保证视觉编码成本相同。packed 样本须保留边界 mask，避免跨样本注意力和标签串扰；优化后复查最大卡显存、有效监督 token 吞吐及任务配比。

#### 易错点

- 总 token 相同也不保证视觉编码开销相同。

#### 追问

- 冻结视觉编码器时怎样判断预计算特征是否合适？

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
